from __future__ import annotations

import numpy as np
import pandas as pd

from .data import clean_dynamic_cutoff, cutoff_view, physical_static_features


def linear_fit(values: np.ndarray, cycles: np.ndarray | None = None) -> tuple[float, float, float]:
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float) if cycles is None else np.asarray(cycles, dtype=float)
    if len(y) < 2 or np.allclose(x, x[0]):
        return 0.0, float(y[-1]), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.sqrt(np.mean(residual**2)))


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    value = a.corr(b)
    return 0.0 if pd.isna(value) else float(value)


def extract_features(
    cycles: pd.DataFrame,
    summary: pd.DataFrame,
    battery_id: int,
    cutoff: int,
) -> dict[str, float]:
    view = clean_dynamic_cutoff(cutoff_view(cycles, battery_id, cutoff))
    row = summary.loc[summary["battery_id"].eq(battery_id)].iloc[0]
    features: dict[str, float] = {
        "battery_id": float(battery_id),
        "cutoff": float(cutoff),
        **physical_static_features(row),
    }

    soh = view["SOH_smooth"].to_numpy(dtype=float)
    features["SOH_last"] = float(soh[-1])
    features["SOH_variance"] = float(np.var(soh))
    for window in (10, 20, 30, 50):
        tail = view.tail(min(window, len(view)))
        slope, _, rmse = linear_fit(tail["SOH_smooth"].to_numpy(), tail["cycle"].to_numpy())
        features[f"SOH_slope_{window}"] = slope
        features[f"SOH_delta_{window}"] = float(tail["SOH_smooth"].iloc[-1] - tail["SOH_smooth"].iloc[0])
        features[f"SOH_linear_rmse_{window}"] = rmse

    local = view.tail(min(50, len(view)))
    x = local["cycle"].to_numpy(dtype=float)
    y = local["SOH_smooth"].to_numpy(dtype=float)
    if len(local) >= 3:
        q2, q1, _ = np.polyfit(x, y, 2)
        features["SOH_quadratic_curvature"] = float(2.0 * q2)
        features["SOH_first_derivative"] = float(2.0 * q2 * x[-1] + q1)
    else:
        features["SOH_quadratic_curvature"] = 0.0
        features["SOH_first_derivative"] = features["SOH_slope_10"]
    features["SOH_second_derivative"] = features["SOH_quadratic_curvature"]

    for column, prefix in (("IR", "IR"), ("Tavg", "Tavg"), ("chargetime", "chargetime")):
        values = view[column].astype(float)
        slope, _, _ = linear_fit(values.to_numpy(), view["cycle"].to_numpy())
        features[f"{prefix}_last"] = float(values.iloc[-1])
        features[f"{prefix}_mean"] = float(values.mean())
        features[f"{prefix}_slope"] = slope
        features[f"{prefix}_delta"] = float(values.iloc[-1] - values.iloc[0])
        features[f"{prefix}_std"] = float(values.std(ddof=0))

    features["corr_SOH_IR"] = safe_corr(view["SOH_smooth"], view["IR"])
    features["corr_SOH_Tavg"] = safe_corr(view["SOH_smooth"], view["Tavg"])
    features["corr_SOH_chargetime"] = safe_corr(view["SOH_smooth"], view["chargetime"])
    return features


def feature_frame(
    cycles: pd.DataFrame,
    summary: pd.DataFrame,
    battery_ids: list[int],
    cutoffs: list[int],
) -> pd.DataFrame:
    rows = [
        extract_features(cycles, summary, battery_id, cutoff)
        for battery_id in battery_ids
        for cutoff in cutoffs
    ]
    return pd.DataFrame(rows)
