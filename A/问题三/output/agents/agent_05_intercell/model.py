from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from problem3.common.data import clean_dynamic_cutoff, cutoff_view, future_view


TRAJECTORY_CONFIGS = [
    {"L": length, "tau_scale": tau_scale}
    for length in (20, 30, 50)
    for tau_scale in (0.75, 1.5)
]

SLOPE_CONFIGS = [
    {"L": length, "tau_scale": tau_scale, "clip_low": low, "clip_high": high}
    for length in (20, 30, 50)
    for tau_scale in (0.75, 1.5)
    for low, high in ((0.50, 1.50), (0.75, 1.25), (0.25, 2.00))
]

MULTIVARIATE_CONFIGS = [
    {"L": length, "tau_scale": tau_scale, "dynamic_weight": dynamic_weight}
    for length in (20, 30, 50)
    for tau_scale in (0.75, 1.5)
    for dynamic_weight in (0.25, 0.50, 1.00)
]

SOFT_POLICY_CONFIGS = [
    {"L": length, "tau_scale": tau_scale, "strategy_tau": strategy_tau}
    for length in (20, 30, 50)
    for tau_scale in (0.75, 1.5)
    for strategy_tau in (0.50, 1.00, 2.00)
]

CONFIG_GRIDS = {
    "trajectory": TRAJECTORY_CONFIGS,
    "slope": SLOPE_CONFIGS,
    "multivariate": MULTIVARIATE_CONFIGS,
    "soft_policy": SOFT_POLICY_CONFIGS,
}


@dataclass(frozen=True)
class CutoffData:
    cutoff: int
    battery_ids: np.ndarray
    policies: np.ndarray
    static_strategy: np.ndarray
    histories: dict[str, np.ndarray]
    last_soh: np.ndarray
    future_soh: np.ndarray
    future_delta: np.ndarray

    @property
    def n_batteries(self) -> int:
        return int(len(self.battery_ids))


def _effective_strategy(summary_row: pd.Series) -> np.ndarray:
    c1 = float(summary_row["C2"] if pd.isna(summary_row["C1"]) else summary_row["C1"])
    return np.asarray([c1, float(summary_row["Q1"]), float(summary_row["C2"])], dtype=float)


def build_cutoff_data(
    cycles: pd.DataFrame,
    summary: pd.DataFrame,
    train_ids: list[int],
    cutoff: int,
    horizon: int = 50,
) -> CutoffData:
    ids = np.asarray(sorted(int(x) for x in train_ids), dtype=int)
    histories: dict[str, list[np.ndarray]] = {
        "SOH_smooth": [],
        "IR": [],
        "Tavg": [],
        "chargetime": [],
    }
    policies: list[str] = []
    static_strategy: list[np.ndarray] = []
    last_soh: list[float] = []
    future_soh: list[np.ndarray] = []

    indexed_summary = summary.set_index("battery_id")
    for battery_id in ids:
        history = clean_dynamic_cutoff(cutoff_view(cycles, int(battery_id), cutoff))
        future = future_view(cycles, int(battery_id), cutoff, horizon)
        summary_row = indexed_summary.loc[int(battery_id)]
        for column in histories:
            histories[column].append(history[column].to_numpy(dtype=float))
        policies.append(str(summary_row["policy"]))
        static_strategy.append(_effective_strategy(summary_row))
        last_soh.append(float(history["SOH"].iloc[-1]))
        future_soh.append(future["SOH"].to_numpy(dtype=float))

    history_arrays = {name: np.vstack(values) for name, values in histories.items()}
    last_array = np.asarray(last_soh, dtype=float)
    future_array = np.vstack(future_soh)
    return CutoffData(
        cutoff=int(cutoff),
        battery_ids=ids,
        policies=np.asarray(policies, dtype=object),
        static_strategy=np.vstack(static_strategy),
        histories=history_arrays,
        last_soh=last_array,
        future_soh=future_array,
        future_delta=future_array - last_array[:, None],
    )


def exact_policy_indices(data: CutoffData, target_index: int, reference_indices: np.ndarray) -> np.ndarray:
    exact = reference_indices[data.policies[reference_indices] == data.policies[target_index]]
    return exact if len(exact) else reference_indices


def _relative_soh_distance(
    data: CutoffData,
    target_index: int,
    peer_indices: np.ndarray,
    length: int,
) -> np.ndarray:
    values = data.histories["SOH_smooth"][:, -length:]
    target = values[target_index] - values[target_index, -1]
    peers = values[peer_indices] - values[peer_indices, -1, None]
    return np.sqrt(np.mean((peers - target[None, :]) ** 2, axis=1))


def _kernel_weights(distances: np.ndarray, tau_scale: float) -> np.ndarray:
    if len(distances) == 1:
        return np.ones(1, dtype=float)
    positive = distances[distances > 1e-12]
    base = float(np.median(positive)) if len(positive) else 1.0
    tau = max(base * float(tau_scale), 1e-12)
    log_weights = -(distances / tau) ** 2
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    return weights / total if np.isfinite(total) and total > 0 else np.repeat(1.0 / len(weights), len(weights))


def _degradation_rates(data: CutoffData, length: int) -> np.ndarray:
    values = data.histories["SOH_smooth"][:, -length:]
    x = np.arange(values.shape[1], dtype=float)
    x_centered = x - float(np.mean(x))
    denominator = float(np.sum(x_centered**2))
    slopes = (values @ x_centered) / denominator
    return np.maximum(-slopes, 1e-7)


def _reference_scale(values: np.ndarray, reference_indices: np.ndarray) -> float:
    scale = float(np.std(values[reference_indices]))
    return max(scale, 1e-9)


def _multivariate_distance(
    data: CutoffData,
    target_index: int,
    peer_indices: np.ndarray,
    reference_indices: np.ndarray,
    length: int,
    dynamic_weight: float,
) -> np.ndarray:
    soh = data.histories["SOH_smooth"][:, -length:]
    target_soh = soh[target_index] - soh[target_index, -1]
    peer_soh = soh[peer_indices] - soh[peer_indices, -1, None]
    reference_soh = soh[reference_indices] - soh[reference_indices, -1, None]
    soh_scale = max(float(np.std(reference_soh)), 1e-9)
    components = [np.mean(((peer_soh - target_soh[None, :]) / soh_scale) ** 2, axis=1)]

    for column in ("IR", "Tavg", "chargetime"):
        values = data.histories[column][:, -length:]
        scale = _reference_scale(values, reference_indices)
        difference = (values[peer_indices] - values[target_index][None, :]) / scale
        components.append(np.mean(difference**2, axis=1))

    dynamic_mean = np.mean(np.vstack(components[1:]), axis=0)
    return np.sqrt(components[0] + float(dynamic_weight) * dynamic_mean)


def _strategy_distance(
    data: CutoffData,
    target_index: int,
    peer_indices: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    reference = data.static_strategy[reference_indices]
    scale = np.std(reference, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    difference = (data.static_strategy[peer_indices] - data.static_strategy[target_index]) / scale
    return np.sqrt(np.mean(difference**2, axis=1))


def predict_cohort(
    data: CutoffData,
    target_index: int,
    reference_indices: np.ndarray,
    method: str,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    reference_indices = np.asarray(reference_indices, dtype=int)
    if target_index in set(reference_indices.tolist()):
        raise ValueError("Target battery cannot appear in its reference pool")
    if not len(reference_indices):
        raise ValueError("Reference pool is empty")

    exact = exact_policy_indices(data, target_index, reference_indices)
    exact_count = int(np.sum(data.policies[reference_indices] == data.policies[target_index]))
    config = {} if config is None else config

    if method == "exact_policy":
        peers = exact
        weights = np.repeat(1.0 / len(peers), len(peers))
        scaled_delta = data.future_delta[peers]
    elif method == "trajectory":
        peers = exact
        distances = _relative_soh_distance(data, target_index, peers, int(config["L"]))
        weights = _kernel_weights(distances, float(config["tau_scale"]))
        scaled_delta = data.future_delta[peers]
    elif method == "slope":
        peers = exact
        length = int(config["L"])
        distances = _relative_soh_distance(data, target_index, peers, length)
        weights = _kernel_weights(distances, float(config["tau_scale"]))
        rates = _degradation_rates(data, length)
        ratios = rates[target_index] / rates[peers]
        ratios = np.clip(ratios, float(config["clip_low"]), float(config["clip_high"]))
        scaled_delta = data.future_delta[peers] * ratios[:, None]
    elif method == "multivariate":
        peers = exact
        distances = _multivariate_distance(
            data,
            target_index,
            peers,
            reference_indices,
            int(config["L"]),
            float(config["dynamic_weight"]),
        )
        weights = _kernel_weights(distances, float(config["tau_scale"]))
        scaled_delta = data.future_delta[peers]
    elif method == "soft_policy":
        peers = reference_indices
        trajectory_distance = _relative_soh_distance(
            data, target_index, peers, int(config["L"])
        )
        positive = trajectory_distance[trajectory_distance > 1e-12]
        base = float(np.median(positive)) if len(positive) else 1.0
        tau = max(base * float(config["tau_scale"]), 1e-12)
        strategy_distance = _strategy_distance(data, target_index, peers, reference_indices)
        strategy_tau = max(float(config["strategy_tau"]), 1e-12)
        log_weights = -(trajectory_distance / tau) ** 2 - (strategy_distance / strategy_tau) ** 2
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        weights /= float(np.sum(weights))
        scaled_delta = data.future_delta[peers]
    else:
        raise ValueError(f"Unknown cohort method: {method}")

    prediction = data.last_soh[target_index] + weights @ scaled_delta
    diagnostics: dict[str, float | int] = {
        "exact_peer_count": exact_count,
        "reference_count": int(len(reference_indices)),
        "used_peer_count": int(len(peers)),
        "effective_peer_count": float(1.0 / np.sum(weights**2)),
    }
    return np.asarray(prediction, dtype=float), diagnostics


def select_config_nested(
    data: CutoffData,
    outer_target_index: int,
    method: str,
    configs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    outer_training = np.asarray(
        [index for index in range(data.n_batteries) if index != outer_target_index], dtype=int
    )
    scores: list[tuple[float, int]] = []
    for config_index, config in enumerate(configs):
        squared_error = 0.0
        point_count = 0
        for inner_target in outer_training:
            inner_reference = outer_training[outer_training != inner_target]
            prediction, _ = predict_cohort(
                data, int(inner_target), inner_reference, method, config
            )
            squared_error += float(np.sum((prediction - data.future_soh[inner_target]) ** 2))
            point_count += int(len(prediction))
        scores.append((float(np.sqrt(squared_error / point_count)), config_index))

    best_score, best_index = min(scores, key=lambda item: (item[0], item[1]))
    return dict(configs[best_index]), {
        "inner_RMSE": float(best_score),
        "candidate_count": int(len(configs)),
    }
