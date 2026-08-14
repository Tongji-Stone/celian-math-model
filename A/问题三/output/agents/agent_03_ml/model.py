from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

# Keep sklearn/OpenMP inside the workspace sandbox's single-thread allowance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - the runner records unavailable candidates
    XGBRegressor = None

from problem3.common.data import cutoff_view, physical_static_features


SEED = 20260814
EARLY_CUTOFFS = (50, 75, 100, 125, 150)
FORECAST_HORIZON = 50


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[int], object]
    hyperparameters: dict[str, object]


def _slope_rmse(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(y) < 2 or np.allclose(x, x[0]):
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return float(slope), float(np.sqrt(np.mean(residual**2)))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


def _robust_soh(values: np.ndarray) -> np.ndarray:
    """Remove only extreme within-cutoff spikes without using future cycles."""
    y = np.asarray(values, dtype=float).copy()
    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 1e-12:
        return y
    return np.clip(y, median - 8.0 * robust_sigma, median + 8.0 * robust_sigma)


def _clean_ir(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    series = series.mask(series.le(0)).interpolate(limit_direction="both")
    return series.to_numpy(dtype=float)


def _policy_column(policy: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in policy)
    return f"policy__{safe}"


def extract_cutoff_features(
    cycles: pd.DataFrame,
    summary: pd.DataFrame,
    battery_id: int,
    cutoff: int,
    policy_levels: tuple[str, ...],
) -> tuple[dict[str, float], float]:
    view = cutoff_view(cycles, battery_id, cutoff)
    summary_row = summary.loc[summary["battery_id"].eq(battery_id)].iloc[0]
    cycle_values = view["cycle"].to_numpy(dtype=float)
    soh_raw = view["SOH"].to_numpy(dtype=float)
    soh = _robust_soh(soh_raw)
    anchor = float(soh_raw[-1])

    features: dict[str, float] = {
        "cutoff": float(cutoff),
        "SOH_raw_last": anchor,
        "SOH_last": float(soh[-1]),
        "SOH_variance": float(np.var(soh)),
    }
    for window in (10, 20, 30, 50):
        width = min(window, len(view))
        x_tail = cycle_values[-width:]
        y_tail = soh[-width:]
        slope, fit_rmse = _slope_rmse(x_tail, y_tail)
        features[f"SOH_slope_{window}"] = slope
        features[f"SOH_delta_{window}"] = float(y_tail[-1] - y_tail[0])
        features[f"SOH_linear_rmse_{window}"] = fit_rmse

    width = min(50, len(view))
    x_local = cycle_values[-width:]
    y_local = soh[-width:]
    if len(y_local) >= 3:
        quadratic, linear, _ = np.polyfit(x_local, y_local, 2)
        first_derivative = 2.0 * quadratic * x_local[-1] + linear
        curvature = 2.0 * quadratic
    else:
        first_derivative = features["SOH_slope_10"]
        curvature = 0.0
    features["SOH_quadratic_curvature"] = float(curvature)
    features["SOH_first_derivative"] = float(first_derivative)
    features["SOH_second_derivative"] = float(curvature)

    dynamic_arrays: dict[str, np.ndarray] = {}
    for column, prefix in (("IR", "IR"), ("Tavg", "Tavg"), ("chargetime", "chargetime")):
        values = view[column].to_numpy(dtype=float)
        if column == "IR":
            values = _clean_ir(values)
        dynamic_arrays[column] = values
        slope, _ = _slope_rmse(cycle_values, values)
        features[f"{prefix}_last"] = float(values[-1])
        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_slope"] = slope
        features[f"{prefix}_delta"] = float(values[-1] - values[0])
        features[f"{prefix}_std"] = float(np.std(values))

    features["corr_SOH_IR"] = _safe_corr(soh, dynamic_arrays["IR"])
    features["corr_SOH_Tavg"] = _safe_corr(soh, dynamic_arrays["Tavg"])
    features["corr_SOH_chargetime"] = _safe_corr(soh, dynamic_arrays["chargetime"])

    static = physical_static_features(summary_row)
    features.update(
        {
            "C1_physical": static["C1_effective"],
            "C1_missing_indicator": static["C1_missing_indicator"],
            "Q1": static["Q1"],
            "C2": static["C2"],
            "initial_capacity": static["initial_capacity"],
            "E1": static["E1"],
            "E2": static["E2"],
            "C1_minus_C2": static["C1_minus_C2"],
            "weighted_C_rate": static["weighted_C_rate"],
            "stage1_width": static["stage1_width"],
            "stage2_width": static["stage2_width"],
        }
    )
    target_policy = str(summary_row["policy"])
    for policy in policy_levels:
        features[_policy_column(policy)] = float(target_policy == policy)
    return features, anchor


def build_learning_table(
    cycles: pd.DataFrame,
    summary: pd.DataFrame,
    battery_ids: list[int],
    cutoffs: tuple[int, ...] = EARLY_CUTOFFS,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    train_summary = summary.loc[summary["battery_id"].isin(battery_ids)]
    policy_levels = tuple(sorted(train_summary["policy"].astype(str).unique()))
    cycle_lookup = cycles.set_index(["battery_id", "cycle"])["SOH"]
    rows: list[dict[str, object]] = []
    for battery_id in sorted(int(value) for value in battery_ids):
        policy = str(summary.loc[summary["battery_id"].eq(battery_id), "policy"].iloc[0])
        for cutoff in cutoffs:
            base, anchor = extract_cutoff_features(
                cycles, summary, battery_id, cutoff, policy_levels
            )
            for forecast_horizon in range(1, horizon + 1):
                cycle = cutoff + forecast_horizon
                y_true = float(cycle_lookup.loc[(battery_id, cycle)])
                row: dict[str, object] = {
                    "battery_id": battery_id,
                    "policy": policy,
                    "cycle": cycle,
                    "cutoff": cutoff,
                    "forecast_horizon": forecast_horizon,
                    "SOH_anchor": anchor,
                    "y_true": y_true,
                    "y_delta": y_true - anchor,
                    **base,
                }
                row["horizon_fraction"] = forecast_horizon / float(horizon)
                for window in (10, 20, 30, 50):
                    row[f"SOH_linear_delta_{window}"] = (
                        base[f"SOH_slope_{window}"] * forecast_horizon
                    )
                row["SOH_quadratic_delta"] = (
                    base["SOH_first_derivative"] * forecast_horizon
                    + 0.5
                    * base["SOH_second_derivative"]
                    * forecast_horizon**2
                )
                rows.append(row)
    return pd.DataFrame(rows)


def feature_sets(table: pd.DataFrame) -> dict[str, list[str]]:
    administrative = {
        "battery_id",
        "policy",
        "cycle",
        "SOH_anchor",
        "y_true",
        "y_delta",
    }
    policy_prefixes = (
        "C1_",
        "Q1",
        "C2",
        "initial_capacity",
        "E1",
        "E2",
        "weighted_C_rate",
        "stage1_width",
        "stage2_width",
        "policy__",
    )
    dynamic_prefixes = ("IR_", "Tavg_", "chargetime_", "corr_SOH_")
    all_features = [column for column in table.columns if column not in administrative]
    policy_features = [
        column for column in all_features if column.startswith(policy_prefixes)
    ]
    dynamic_features = [
        column for column in all_features if column.startswith(dynamic_prefixes)
    ]
    soh_features = [
        column
        for column in all_features
        if column not in set(policy_features) and column not in set(dynamic_features)
    ]
    return {
        "A_SOH_only": soh_features,
        "B_SOH_plus_dynamics": soh_features + dynamic_features,
        "C_plus_policy": soh_features + dynamic_features + policy_features,
    }


def candidate_models() -> dict[str, Candidate]:
    candidates = {
        "extra_trees": Candidate(
            name="extra_trees",
            factory=lambda seed: ExtraTreesRegressor(
                n_estimators=32,
                max_features=0.75,
                min_samples_leaf=2,
                bootstrap=False,
                random_state=seed,
                n_jobs=1,
            ),
            hyperparameters={
                "n_estimators": 32,
                "max_features": 0.75,
                "min_samples_leaf": 2,
                "bootstrap": False,
            },
        ),
        "random_forest": Candidate(
            name="random_forest",
            factory=lambda seed: RandomForestRegressor(
                n_estimators=32,
                max_features=0.75,
                min_samples_leaf=2,
                bootstrap=True,
                random_state=seed,
                n_jobs=1,
            ),
            hyperparameters={
                "n_estimators": 32,
                "max_features": 0.75,
                "min_samples_leaf": 2,
                "bootstrap": True,
            },
        ),
        "hist_gradient_boosting": Candidate(
            name="hist_gradient_boosting",
            factory=lambda seed: HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=45,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=0.1,
                random_state=seed,
            ),
            hyperparameters={
                "learning_rate": 0.05,
                "max_iter": 45,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 30,
                "l2_regularization": 0.1,
            },
        ),
    }
    if XGBRegressor is not None:
        candidates["xgboost"] = Candidate(
            name="xgboost",
            factory=lambda seed: XGBRegressor(
                objective="reg:squarederror",
                n_estimators=45,
                learning_rate=0.04,
                max_depth=3,
                min_child_weight=20,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                reg_alpha=0.0,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            ),
            hyperparameters={
                "n_estimators": 45,
                "learning_rate": 0.04,
                "max_depth": 3,
                "min_child_weight": 20,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
            },
        )
    return candidates


def make_pipeline(candidate: Candidate, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("regressor", candidate.factory(seed)),
        ]
    )
