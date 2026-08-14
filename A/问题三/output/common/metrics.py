from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON_BINS = [0, 10, 20, 30, 40, 50]
HORIZON_LABELS = ["1-10", "11-20", "21-30", "31-40", "41-50"]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def summarize_predictions(
    predictions: pd.DataFrame,
    prediction_col: str,
    model_name: str,
) -> dict[str, float | str]:
    work = predictions.dropna(subset=[prediction_col, "y_true"]).copy()
    per_battery = work.groupby("battery_id").apply(
        lambda g: rmse(g["y_true"].to_numpy(), g[prediction_col].to_numpy()),
        include_groups=False,
    )
    return {
        "model": model_name,
        "n_batteries": int(work["battery_id"].nunique()),
        "n_points": int(len(work)),
        "MAE": mae(work["y_true"].to_numpy(), work[prediction_col].to_numpy()),
        "overall_RMSE": rmse(work["y_true"].to_numpy(), work[prediction_col].to_numpy()),
        "mean_battery_RMSE": float(per_battery.mean()),
        "median_battery_RMSE": float(per_battery.median()),
        "worst_battery_RMSE": float(per_battery.max()),
    }


def detailed_metric_tables(
    predictions: pd.DataFrame,
    prediction_columns: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    for cutoff, cutoff_df in predictions.groupby("cutoff"):
        for model, column in prediction_columns.items():
            row = summarize_predictions(cutoff_df, column, model)
            row["cutoff"] = int(cutoff)
            overall_rows.append(row)
            for policy, group in cutoff_df.groupby("policy"):
                policy_rows.append(
                    {
                        "cutoff": int(cutoff),
                        "model": model,
                        "policy": policy,
                        "n_batteries": int(group["battery_id"].nunique()),
                        "RMSE": rmse(group["y_true"].to_numpy(), group[column].to_numpy()),
                        "MAE": mae(group["y_true"].to_numpy(), group[column].to_numpy()),
                    }
                )
            binned = cutoff_df.assign(
                horizon_bin=pd.cut(
                    cutoff_df["horizon"],
                    HORIZON_BINS,
                    labels=HORIZON_LABELS,
                    include_lowest=True,
                )
            )
            for label, group in binned.groupby("horizon_bin", observed=True):
                horizon_rows.append(
                    {
                        "cutoff": int(cutoff),
                        "model": model,
                        "horizon_bin": str(label),
                        "RMSE": rmse(group["y_true"].to_numpy(), group[column].to_numpy()),
                        "MAE": mae(group["y_true"].to_numpy(), group[column].to_numpy()),
                    }
                )
    return pd.DataFrame(overall_rows), pd.DataFrame(policy_rows), pd.DataFrame(horizon_rows)
