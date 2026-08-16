from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import cutoff_view, future_view, load_data, split_battery_ids
from .metrics import detailed_metric_tables, rmse
from .validation import EARLY_CUTOFFS, FORECAST_HORIZON, leave_one_battery_out


LINEAR_WINDOWS = [10, 20, 30, 50]


def persistence(history: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> np.ndarray:
    return np.repeat(float(history["SOH"].iloc[-1]), horizon)


def recent_linear(
    history: pd.DataFrame,
    window: int,
    horizon: int = FORECAST_HORIZON,
) -> np.ndarray:
    tail = history.tail(min(window, len(history)))
    x = tail["cycle"].to_numpy(dtype=float)
    y = tail["SOH"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future_cycles = np.arange(int(history["cycle"].iloc[-1]) + 1, int(history["cycle"].iloc[-1]) + horizon + 1)
    return slope * future_cycles + intercept


def same_policy_mean_future_delta(
    cycles: pd.DataFrame,
    summary: pd.DataFrame,
    target_id: int,
    reference_ids: list[int],
    cutoff: int,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, int]:
    target_policy = summary.loc[summary["battery_id"].eq(target_id), "policy"].iloc[0]
    peers = summary.loc[
        summary["battery_id"].isin(reference_ids) & summary["policy"].eq(target_policy),
        "battery_id",
    ].astype(int).tolist()
    if not peers:
        peers = list(reference_ids)
    deltas = []
    for peer_id in peers:
        peer_history = cutoff_view(cycles, peer_id, cutoff)
        peer_future = future_view(cycles, peer_id, cutoff, horizon)
        deltas.append(peer_future["SOH"].to_numpy() - float(peer_history["SOH"].iloc[-1]))
    target_last = float(cutoff_view(cycles, target_id, cutoff)["SOH"].iloc[-1])
    return target_last + np.mean(np.vstack(deltas), axis=0), len(peers)


def select_linear_window_nested(
    cycles: pd.DataFrame,
    inner_ids: list[int],
    cutoff: int,
    horizon: int = FORECAST_HORIZON,
) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for window in LINEAR_WINDOWS:
        true_parts = []
        pred_parts = []
        for battery_id in inner_ids:
            history = cutoff_view(cycles, battery_id, cutoff)
            future = future_view(cycles, battery_id, cutoff, horizon)
            true_parts.append(future["SOH"].to_numpy())
            pred_parts.append(recent_linear(history, window, horizon))
        scores[window] = rmse(np.concatenate(true_parts), np.concatenate(pred_parts))
    best_window = min(scores, key=lambda window: (scores[window], window))
    return int(best_window), scores


def run_baseline_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, cycles = load_data()
    train_ids, _ = split_battery_ids(summary)
    rows: list[dict[str, object]] = []
    for cutoff in EARLY_CUTOFFS:
        cache: dict[int, dict[str, object]] = {}
        squared_errors = {window: {} for window in LINEAR_WINDOWS}
        for battery_id in train_ids:
            history = cutoff_view(cycles, battery_id, cutoff)
            future = future_view(cycles, battery_id, cutoff, FORECAST_HORIZON)
            truth = future["SOH"].to_numpy()
            linear_predictions = {
                window: recent_linear(history, window) for window in LINEAR_WINDOWS
            }
            cache[battery_id] = {
                "history": history,
                "future": future,
                "truth": truth,
                "linear": linear_predictions,
            }
            for window, prediction in linear_predictions.items():
                squared_errors[window][battery_id] = float(np.sum((truth - prediction) ** 2))
        for reference_ids, target_id in leave_one_battery_out(train_ids):
            history = cache[target_id]["history"]
            future = cache[target_id]["future"]
            policy = summary.loc[summary["battery_id"].eq(target_id), "policy"].iloc[0]
            predictions = {
                "pred_persistence": persistence(history),
                **{
                    f"pred_linear_{window}": cache[target_id]["linear"][window]
                    for window in LINEAR_WINDOWS
                },
            }
            inner_scores = {
                window: float(
                    np.sqrt(
                        sum(squared_errors[window][battery_id] for battery_id in reference_ids)
                        / (len(reference_ids) * FORECAST_HORIZON)
                    )
                )
                for window in LINEAR_WINDOWS
            }
            best_window = min(inner_scores, key=lambda window: (inner_scores[window], window))
            predictions["pred_linear_nested"] = predictions[f"pred_linear_{best_window}"]
            cohort_pred, peer_count = same_policy_mean_future_delta(
                cycles, summary, target_id, reference_ids, cutoff
            )
            predictions["pred_same_policy"] = cohort_pred
            for h, (_, truth_row) in enumerate(future.iterrows(), start=1):
                row: dict[str, object] = {
                    "battery_id": target_id,
                    "policy": policy,
                    "cutoff": cutoff,
                    "horizon": h,
                    "cycle": cutoff + h,
                    "y_true": float(truth_row["SOH"]),
                    "nested_best_window": best_window,
                    "nested_inner_scores": json.dumps(inner_scores, sort_keys=True),
                    "same_policy_peer_count": peer_count,
                }
                for name, values in predictions.items():
                    row[name] = float(values[h - 1])
                rows.append(row)
    prediction_df = pd.DataFrame(rows)
    columns = {
        "Persistence": "pred_persistence",
        "Linear-10": "pred_linear_10",
        "Linear-20": "pred_linear_20",
        "Linear-30": "pred_linear_30",
        "Linear-50": "pred_linear_50",
        "Linear-Nested": "pred_linear_nested",
        "Same-policy mean delta": "pred_same_policy",
    }
    overall, policy, horizon = detailed_metric_tables(prediction_df, columns)
    return prediction_df, overall, policy, horizon


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    predictions, overall, policy, horizon = run_baseline_backtest()
    predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)
    overall.to_csv(output_dir / "baseline_metrics.csv", index=False)
    policy.to_csv(output_dir / "baseline_policy_metrics.csv", index=False)
    horizon.to_csv(output_dir / "baseline_horizon_metrics.csv", index=False)
    print(overall.sort_values(["cutoff", "overall_RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()
