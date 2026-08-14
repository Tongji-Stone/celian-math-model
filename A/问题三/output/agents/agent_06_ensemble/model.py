from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from problem3.common.data import cutoff_view, future_view, physical_static_features
from problem3.common.features import extract_features
from problem3.common.validation import FORECAST_HORIZON, SEED


EXPERTS = (
    "linear20",
    "linear30",
    "linear50",
    "same_policy",
    "feature_ml",
    "gp",
)
EXPERT_COLUMNS = {name: f"pred_{name}" for name in EXPERTS}
HORIZON_BINS = ("1-10", "11-20", "21-30", "31-40", "41-50")
INNER_SPLITS = 3
META_SPLITS = 3
ML_TREES = 40
ML_MAX_DEPTH = 7
ML_MIN_SAMPLES_LEAF = 8
GATE_RIDGE_ALPHA = 5.0
GATE_TEMPERATURE = 0.75
GATE_PRIOR_BLEND = 0.50
SCALE_FLOOR = 5e-5

ML_FEATURES = (
    "SOH_last",
    "SOH_variance",
    "SOH_slope_10",
    "SOH_slope_20",
    "SOH_slope_30",
    "SOH_slope_50",
    "SOH_delta_20",
    "SOH_delta_50",
    "SOH_linear_rmse_20",
    "SOH_linear_rmse_50",
    "SOH_quadratic_curvature",
    "SOH_first_derivative",
    "IR_last",
    "IR_mean",
    "IR_slope",
    "IR_delta",
    "IR_std",
    "Tavg_last",
    "Tavg_mean",
    "Tavg_slope",
    "Tavg_std",
    "chargetime_last",
    "chargetime_mean",
    "chargetime_slope",
    "chargetime_std",
    "corr_SOH_IR",
    "corr_SOH_Tavg",
    "corr_SOH_chargetime",
    "C1_effective",
    "C1_missing_indicator",
    "Q1",
    "C2",
    "initial_capacity",
    "E1",
    "E2",
    "C1_minus_C2",
    "weighted_C_rate",
)

GATE_NUMERIC = (
    "gate_slope20",
    "gate_slope50",
    "gate_curvature",
    "gate_volatility",
    "gate_ir_slope",
    "gate_temperature_slope",
    "gate_chargetime_slope",
    "gate_peer_similarity",
    "gate_C1",
    "gate_Q1",
    "gate_C2",
    "gate_weighted_C_rate",
)


def horizon_bin(horizon: int) -> str:
    return HORIZON_BINS[min((int(horizon) - 1) // 10, len(HORIZON_BINS) - 1)]


def recent_linear(history: pd.DataFrame, window: int) -> np.ndarray:
    tail = history.tail(min(window, len(history)))
    x = tail["cycle"].to_numpy(dtype=float)
    y = tail["SOH"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(int(history["cycle"].iloc[-1]) + 1, int(history["cycle"].iloc[-1]) + 51)
    return slope * future_x + intercept


def simplex_weights(prediction_matrix: np.ndarray, truth: np.ndarray) -> np.ndarray:
    matrix = np.asarray(prediction_matrix, dtype=float)
    target = np.asarray(truth, dtype=float)
    n_experts = matrix.shape[1]
    start = np.repeat(1.0 / n_experts, n_experts)

    def objective(weights: np.ndarray) -> float:
        residual = matrix @ weights - target
        return float(np.mean(residual**2) + 1e-10 * np.sum((weights - start) ** 2))

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_experts,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": 500, "ftol": 1e-14},
    )
    if result.success and np.isfinite(result.x).all():
        weights = np.clip(result.x, 0.0, 1.0)
        return weights / weights.sum()
    weights, _ = nnls(matrix, target)
    return weights / weights.sum() if weights.sum() > 0 else start


def grouped_folds(battery_ids: list[int], n_splits: int, seed_offset: int = 0):
    ids = np.asarray(sorted(int(x) for x in battery_ids), dtype=int)
    splitter = KFold(
        n_splits=min(n_splits, len(ids)),
        shuffle=True,
        random_state=SEED + seed_offset,
    )
    for train_index, validation_index in splitter.split(ids):
        yield ids[train_index].tolist(), ids[validation_index].tolist()


@dataclass
class FeatureML:
    model: ExtraTreesRegressor
    feature_columns: tuple[str, ...]


@dataclass
class AdaptiveGate:
    scaler: StandardScaler
    model: Ridge
    policy_levels: tuple[str, ...]


class ExperimentContext:
    def __init__(self, summary: pd.DataFrame, cycles: pd.DataFrame):
        self.summary = summary.copy()
        self.cycles = cycles.copy()
        self.policy_levels = tuple(sorted(summary["policy"].astype(str).unique()))
        self._feature_cache: dict[tuple[int, int], dict[str, float]] = {}
        self._gp_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def history(self, battery_id: int, cutoff: int) -> pd.DataFrame:
        return cutoff_view(self.cycles, battery_id, cutoff)

    def future(self, battery_id: int, cutoff: int) -> pd.DataFrame:
        return future_view(self.cycles, battery_id, cutoff, FORECAST_HORIZON)

    def features(self, battery_id: int, cutoff: int) -> dict[str, float]:
        key = (int(battery_id), int(cutoff))
        if key not in self._feature_cache:
            self._feature_cache[key] = extract_features(
                self.cycles, self.summary, int(battery_id), int(cutoff)
            )
        return self._feature_cache[key]

    def fit_feature_ml(self, reference_ids: list[int], cutoff: int) -> FeatureML:
        feature_rows: list[dict[str, float]] = []
        targets: list[float] = []
        for battery_id in reference_ids:
            base = self.features(battery_id, cutoff)
            history = self.history(battery_id, cutoff)
            future = self.future(battery_id, cutoff)
            anchor = float(history["SOH"].iloc[-1])
            for h, truth in enumerate(future["SOH"].to_numpy(dtype=float), start=1):
                row = {name: float(base[name]) for name in ML_FEATURES}
                row.update(
                    {
                        "horizon": h / FORECAST_HORIZON,
                        "horizon_squared": (h / FORECAST_HORIZON) ** 2,
                        "horizon_x_slope20": h * float(base["SOH_slope_20"]),
                        "horizon_x_curvature": h * h * float(base["SOH_quadratic_curvature"]),
                    }
                )
                feature_rows.append(row)
                targets.append(float(truth - anchor))
        frame = pd.DataFrame(feature_rows)
        model = ExtraTreesRegressor(
            n_estimators=ML_TREES,
            max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF,
            max_features=0.80,
            random_state=SEED + int(cutoff),
            n_jobs=1,
        )
        model.fit(frame, np.asarray(targets, dtype=float))
        return FeatureML(model=model, feature_columns=tuple(frame.columns))

    def predict_feature_ml(self, fitted: FeatureML, battery_id: int, cutoff: int) -> np.ndarray:
        base = self.features(battery_id, cutoff)
        rows: list[dict[str, float]] = []
        for h in range(1, FORECAST_HORIZON + 1):
            row = {name: float(base[name]) for name in ML_FEATURES}
            row.update(
                {
                    "horizon": h / FORECAST_HORIZON,
                    "horizon_squared": (h / FORECAST_HORIZON) ** 2,
                    "horizon_x_slope20": h * float(base["SOH_slope_20"]),
                    "horizon_x_curvature": h * h * float(base["SOH_quadratic_curvature"]),
                }
            )
            rows.append(row)
        frame = pd.DataFrame(rows).loc[:, fitted.feature_columns]
        anchor = float(self.history(battery_id, cutoff)["SOH"].iloc[-1])
        return anchor + fitted.model.predict(frame)

    def gp_forecast(self, battery_id: int, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
        key = (int(battery_id), int(cutoff))
        if key in self._gp_cache:
            return self._gp_cache[key]
        history = self.history(battery_id, cutoff).tail(min(50, cutoff))
        x = history["cycle"].to_numpy(dtype=float)
        y = history["SOH_smooth"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        mean_train = slope * x + intercept
        residual = y - mean_train
        kernel = (
            ConstantKernel(2e-7, constant_value_bounds="fixed")
            * Matern(length_scale=12.0, length_scale_bounds="fixed", nu=1.5)
            + WhiteKernel(noise_level=2e-8, noise_level_bounds="fixed")
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            optimizer=None,
            normalize_y=False,
            random_state=SEED,
        )
        gp.fit(x.reshape(-1, 1), residual)
        future_x = np.arange(cutoff + 1, cutoff + FORECAST_HORIZON + 1, dtype=float)
        residual_mean, residual_std = gp.predict(future_x.reshape(-1, 1), return_std=True)
        prediction = slope * future_x + intercept + residual_mean
        self._gp_cache[key] = prediction.astype(float), residual_std.astype(float)
        return self._gp_cache[key]

    def same_policy_forecast(
        self, battery_id: int, reference_ids: list[int], cutoff: int
    ) -> tuple[np.ndarray, int]:
        target_policy = str(
            self.summary.loc[self.summary["battery_id"].eq(battery_id), "policy"].iloc[0]
        )
        peers = self.summary.loc[
            self.summary["battery_id"].isin(reference_ids)
            & self.summary["policy"].astype(str).eq(target_policy),
            "battery_id",
        ].astype(int).tolist()
        if not peers:
            peers = list(reference_ids)
        deltas = []
        for peer_id in peers:
            peer_history = self.history(peer_id, cutoff)
            peer_future = self.future(peer_id, cutoff)
            deltas.append(
                peer_future["SOH"].to_numpy(dtype=float)
                - float(peer_history["SOH"].iloc[-1])
            )
        anchor = float(self.history(battery_id, cutoff)["SOH"].iloc[-1])
        return anchor + np.mean(np.vstack(deltas), axis=0), len(peers)

    def peer_similarity(self, battery_id: int, reference_ids: list[int], cutoff: int) -> float:
        target_policy = str(
            self.summary.loc[self.summary["battery_id"].eq(battery_id), "policy"].iloc[0]
        )
        peers = self.summary.loc[
            self.summary["battery_id"].isin(reference_ids)
            & self.summary["policy"].astype(str).eq(target_policy),
            "battery_id",
        ].astype(int).tolist()
        if not peers:
            peers = list(reference_ids)
        length = min(30, cutoff)
        target = self.history(battery_id, cutoff).tail(length)["SOH_smooth"].to_numpy(dtype=float)
        target = target - target[-1]
        distances = []
        for peer_id in peers:
            peer = self.history(peer_id, cutoff).tail(length)["SOH_smooth"].to_numpy(dtype=float)
            peer = peer - peer[-1]
            distances.append(float(np.sqrt(np.mean((target - peer) ** 2))))
        return float(min(distances)) if distances else 0.0

    def gate_features(self, battery_id: int, reference_ids: list[int], cutoff: int) -> dict[str, object]:
        base = self.features(battery_id, cutoff)
        row = self.summary.loc[self.summary["battery_id"].eq(battery_id)].iloc[0]
        static = physical_static_features(row)
        return {
            "policy": str(row["policy"]),
            "gate_slope20": float(base["SOH_slope_20"]),
            "gate_slope50": float(base["SOH_slope_50"]),
            "gate_curvature": float(base["SOH_quadratic_curvature"]),
            "gate_volatility": float(base["SOH_linear_rmse_20"]),
            "gate_ir_slope": float(base["IR_slope"]),
            "gate_temperature_slope": float(base["Tavg_slope"]),
            "gate_chargetime_slope": float(base["chargetime_slope"]),
            "gate_peer_similarity": self.peer_similarity(battery_id, reference_ids, cutoff),
            "gate_C1": float(static["C1_effective"]),
            "gate_Q1": float(static["Q1"]),
            "gate_C2": float(static["C2"]),
            "gate_weighted_C_rate": float(static["weighted_C_rate"]),
        }

    def expert_predictions(
        self,
        target_ids: list[int],
        reference_ids: list[int],
        cutoff: int,
        ml_model: FeatureML | None = None,
    ) -> pd.DataFrame:
        fitted = ml_model or self.fit_feature_ml(reference_ids, cutoff)
        rows: list[dict[str, object]] = []
        for battery_id in target_ids:
            history = self.history(battery_id, cutoff)
            future = self.future(battery_id, cutoff)
            policy = str(
                self.summary.loc[self.summary["battery_id"].eq(battery_id), "policy"].iloc[0]
            )
            same_policy, peer_count = self.same_policy_forecast(
                battery_id, reference_ids, cutoff
            )
            gp_prediction, gp_std = self.gp_forecast(battery_id, cutoff)
            predictions = {
                "linear20": recent_linear(history, 20),
                "linear30": recent_linear(history, 30),
                "linear50": recent_linear(history, 50),
                "same_policy": same_policy,
                "feature_ml": self.predict_feature_ml(fitted, battery_id, cutoff),
                "gp": gp_prediction,
            }
            gate = self.gate_features(battery_id, reference_ids, cutoff)
            truth = future["SOH"].to_numpy(dtype=float)
            for h in range(1, FORECAST_HORIZON + 1):
                result: dict[str, object] = {
                    "battery_id": int(battery_id),
                    "policy": policy,
                    "cutoff": int(cutoff),
                    "horizon": h,
                    "horizon_bin": horizon_bin(h),
                    "cycle": cutoff + h,
                    "y_true": float(truth[h - 1]),
                    "same_policy_peer_count": int(peer_count),
                    "gp_std": float(gp_std[h - 1]),
                    **gate,
                }
                for name in EXPERTS:
                    result[EXPERT_COLUMNS[name]] = float(predictions[name][h - 1])
                rows.append(result)
        return pd.DataFrame(rows)


def expert_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[EXPERT_COLUMNS[name] for name in EXPERTS]].to_numpy(dtype=float)


def fit_weight_models(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    global_weights = simplex_weights(expert_matrix(frame), frame["y_true"].to_numpy(dtype=float))
    horizon_weights = {
        label: simplex_weights(
            expert_matrix(group), group["y_true"].to_numpy(dtype=float)
        )
        for label, group in frame.groupby("horizon_bin", sort=False)
    }
    return global_weights, horizon_weights


def gate_design(frame: pd.DataFrame, policy_levels: tuple[str, ...]) -> np.ndarray:
    numeric = frame.loc[:, GATE_NUMERIC].to_numpy(dtype=float)
    policy = frame["policy"].astype(str).to_numpy()
    one_hot = np.column_stack([(policy == value).astype(float) for value in policy_levels])
    return np.column_stack([numeric, one_hot])


def battery_gate_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["battery_id", "policy", *GATE_NUMERIC]
    return predictions.loc[:, columns].drop_duplicates("battery_id").sort_values("battery_id")


def adaptive_targets(predictions: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    battery_ids = sorted(predictions["battery_id"].astype(int).unique())
    targets = []
    for battery_id in battery_ids:
        battery = predictions.loc[predictions["battery_id"].eq(battery_id)]
        values = []
        for label in HORIZON_BINS:
            group = battery.loc[battery["horizon_bin"].eq(label)]
            truth = group["y_true"].to_numpy(dtype=float)
            for expert in EXPERTS:
                prediction = group[EXPERT_COLUMNS[expert]].to_numpy(dtype=float)
                values.append(math.log(float(np.sqrt(np.mean((truth - prediction) ** 2))) + 1e-6))
        targets.append(values)
    return np.asarray(targets, dtype=float), battery_ids


def fit_adaptive_gate(predictions: pd.DataFrame) -> AdaptiveGate:
    feature_frame = battery_gate_frame(predictions)
    targets, battery_ids = adaptive_targets(predictions)
    feature_frame = feature_frame.set_index("battery_id").loc[battery_ids].reset_index()
    policy_levels = tuple(sorted(predictions["policy"].astype(str).unique()))
    design = gate_design(feature_frame, policy_levels)
    scaler = StandardScaler().fit(design)
    model = Ridge(alpha=GATE_RIDGE_ALPHA).fit(scaler.transform(design), targets)
    return AdaptiveGate(scaler=scaler, model=model, policy_levels=policy_levels)


def adaptive_weight_map(
    gate: AdaptiveGate,
    target_metadata: pd.DataFrame,
    horizon_prior: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    design = gate_design(target_metadata.iloc[[0]], gate.policy_levels)
    predicted_log_error = gate.model.predict(gate.scaler.transform(design)).reshape(
        len(HORIZON_BINS), len(EXPERTS)
    )
    weights: dict[str, np.ndarray] = {}
    for index, label in enumerate(HORIZON_BINS):
        logits = -predicted_log_error[index] / GATE_TEMPERATURE
        logits -= logits.max()
        learned = np.exp(np.clip(logits, -30.0, 30.0))
        learned /= learned.sum()
        prior = horizon_prior[label]
        combined = GATE_PRIOR_BLEND * prior + (1.0 - GATE_PRIOR_BLEND) * learned
        weights[label] = combined / combined.sum()
    return weights


def weighted_prediction_and_scale(
    frame: pd.DataFrame,
    weight_by_bin: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    point = np.empty(len(frame), dtype=float)
    scale = np.empty(len(frame), dtype=float)
    matrix = expert_matrix(frame)
    gp_index = EXPERTS.index("gp")
    for row_index, (_, row) in enumerate(frame.iterrows()):
        weights = weight_by_bin[str(row["horizon_bin"])]
        values = matrix[row_index]
        mean = float(values @ weights)
        variance = float(np.sum(weights * (values - mean) ** 2))
        variance += float((weights[gp_index] * float(row["gp_std"])) ** 2)
        point[row_index] = mean
        scale[row_index] = math.sqrt(variance + SCALE_FLOOR**2)
    return point, scale


def apply_ensemble_models(
    frame: pd.DataFrame,
    global_weights: np.ndarray,
    horizon_weights: dict[str, np.ndarray],
    adaptive_weights: dict[int, dict[str, np.ndarray]],
) -> pd.DataFrame:
    work = frame.copy()
    global_map = {label: global_weights for label in HORIZON_BINS}
    point, scale = weighted_prediction_and_scale(work, global_map)
    work["pred_global"] = point
    work["scale_global"] = scale
    point, scale = weighted_prediction_and_scale(work, horizon_weights)
    work["pred_horizon"] = point
    work["scale_horizon"] = scale
    adaptive_parts = []
    for battery_id, group in work.groupby("battery_id", sort=False):
        part = group.copy()
        point, scale = weighted_prediction_and_scale(
            part, adaptive_weights[int(battery_id)]
        )
        part["pred_adaptive"] = point
        part["scale_adaptive"] = scale
        adaptive_parts.append(part)
    return pd.concat(adaptive_parts, ignore_index=True).sort_values(
        ["battery_id", "horizon"]
    )


def inner_oof_predictions(
    context: ExperimentContext,
    outer_training_ids: list[int],
    cutoff: int,
) -> pd.DataFrame:
    parts = []
    for reference_ids, validation_ids in grouped_folds(
        outer_training_ids, INNER_SPLITS, seed_offset=cutoff
    ):
        ml_model = context.fit_feature_ml(reference_ids, cutoff)
        parts.append(
            context.expert_predictions(
                validation_ids, reference_ids, cutoff, ml_model=ml_model
            )
        )
    return pd.concat(parts, ignore_index=True).sort_values(["battery_id", "horizon"])


def meta_crossfit_predictions(inner_predictions: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(inner_predictions["battery_id"].astype(int).unique())
    parts = []
    cutoff = int(inner_predictions["cutoff"].iloc[0])
    for training_ids, validation_ids in grouped_folds(ids, META_SPLITS, seed_offset=1000 + cutoff):
        training = inner_predictions.loc[inner_predictions["battery_id"].isin(training_ids)]
        validation = inner_predictions.loc[inner_predictions["battery_id"].isin(validation_ids)]
        global_weights, horizon_weights = fit_weight_models(training)
        gate = fit_adaptive_gate(training)
        adaptive_weights = {}
        for battery_id in validation_ids:
            metadata = battery_gate_frame(
                validation.loc[validation["battery_id"].eq(battery_id)]
            )
            adaptive_weights[battery_id] = adaptive_weight_map(
                gate, metadata, horizon_weights
            )
        parts.append(
            apply_ensemble_models(
                validation, global_weights, horizon_weights, adaptive_weights
            )
        )
    return pd.concat(parts, ignore_index=True).sort_values(["battery_id", "horizon"])


def finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    clean = np.sort(np.asarray(values, dtype=float))
    rank = min(len(clean), int(math.ceil((len(clean) + 1) * coverage)))
    return float(clean[max(rank - 1, 0)])


def conformal_quantiles(
    meta_predictions: pd.DataFrame,
    variant: str,
) -> dict[tuple[str, float], float]:
    output: dict[tuple[str, float], float] = {}
    for label in HORIZON_BINS:
        group = meta_predictions.loc[meta_predictions["horizon_bin"].eq(label)].copy()
        group["score"] = (
            (group["y_true"] - group[f"pred_{variant}"]).abs()
            / group[f"scale_{variant}"].clip(lower=SCALE_FLOOR)
        )
        battery_scores = group.groupby("battery_id")["score"].max().to_numpy(dtype=float)
        for coverage in (0.90, 0.95):
            output[(label, coverage)] = finite_sample_quantile(battery_scores, coverage)
    return output


def add_intervals(
    outer_predictions: pd.DataFrame,
    meta_predictions: pd.DataFrame,
) -> pd.DataFrame:
    work = outer_predictions.copy()
    for variant in ("global", "horizon", "adaptive"):
        quantiles = conformal_quantiles(meta_predictions, variant)
        for coverage in (0.90, 0.95):
            suffix = str(int(coverage * 100))
            half_width = np.asarray(
                [
                    quantiles[(str(label), coverage)] * scale
                    for label, scale in zip(
                        work["horizon_bin"], work[f"scale_{variant}"]
                    )
                ],
                dtype=float,
            )
            work[f"lower_{suffix}_{variant}"] = work[f"pred_{variant}"] - half_width
            work[f"upper_{suffix}_{variant}"] = work[f"pred_{variant}"] + half_width
    expert_values = expert_matrix(work)
    work["expert_disagreement"] = expert_values.std(axis=1, ddof=0)
    return work


def weight_records(
    outer_target: int,
    cutoff: int,
    global_weights: np.ndarray,
    horizon_weights: dict[str, np.ndarray],
    adaptive_weights: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for expert, weight in zip(EXPERTS, global_weights):
        rows.append(
            {
                "outer_target_battery": outer_target,
                "cutoff": cutoff,
                "ensemble": "global",
                "horizon_bin": "all",
                "expert": expert,
                "weight": float(weight),
            }
        )
    for ensemble, weight_map in (
        ("horizon", horizon_weights),
        ("adaptive", adaptive_weights),
    ):
        for label in HORIZON_BINS:
            for expert, weight in zip(EXPERTS, weight_map[label]):
                rows.append(
                    {
                        "outer_target_battery": outer_target,
                        "cutoff": cutoff,
                        "ensemble": ensemble,
                        "horizon_bin": label,
                        "expert": expert,
                        "weight": float(weight),
                    }
                )
    return rows


def run_nested_backtest(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    train_ids: list[int],
    cutoffs: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = ExperimentContext(summary, cycles)
    prediction_parts = []
    all_weight_rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        for outer_target in sorted(train_ids):
            outer_training = [battery_id for battery_id in train_ids if battery_id != outer_target]
            inner_predictions = inner_oof_predictions(context, outer_training, cutoff)
            global_weights, horizon_weights = fit_weight_models(inner_predictions)
            gate = fit_adaptive_gate(inner_predictions)
            outer_ml = context.fit_feature_ml(outer_training, cutoff)
            outer_experts = context.expert_predictions(
                [outer_target], outer_training, cutoff, ml_model=outer_ml
            )
            target_metadata = battery_gate_frame(outer_experts)
            target_adaptive = adaptive_weight_map(gate, target_metadata, horizon_weights)
            meta_predictions = meta_crossfit_predictions(inner_predictions)
            outer_ensembles = apply_ensemble_models(
                outer_experts,
                global_weights,
                horizon_weights,
                {outer_target: target_adaptive},
            )
            outer_ensembles = add_intervals(outer_ensembles, meta_predictions)
            outer_ensembles["outer_training_battery_count"] = len(outer_training)
            outer_ensembles["inner_group_folds"] = INNER_SPLITS
            outer_ensembles["meta_group_folds"] = META_SPLITS
            prediction_parts.append(outer_ensembles)
            all_weight_rows.extend(
                weight_records(
                    outer_target,
                    cutoff,
                    global_weights,
                    horizon_weights,
                    target_adaptive,
                )
            )
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["cutoff", "battery_id", "horizon"]
    )
    weights = pd.DataFrame(all_weight_rows).sort_values(
        ["cutoff", "outer_target_battery", "ensemble", "horizon_bin", "expert"]
    )
    return predictions, weights
