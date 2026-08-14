from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from problem3.common.data import load_data, split_battery_ids
from problem3.final.final_model import (
    CUTOFF,
    HORIZON,
    eol_sensitivity,
    forecast_short_horizon,
    frozen_power_weight,
    hybrid_cv_calibration,
)


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = Path(__file__).resolve().parent


def metric_row(frame: pd.DataFrame, prediction: str, scope: str, group: str) -> dict[str, object]:
    error = frame[prediction].to_numpy(dtype=float) - frame["y_true"].to_numpy(dtype=float)
    battery_rmse = (
        frame.assign(squared_error=error**2)
        .groupby("battery_id")["squared_error"]
        .mean()
        .pow(0.5)
    )
    return {
        "model": "Final Power101+Linear50 hybrid",
        "cutoff": CUTOFF,
        "scope": scope,
        "group": group,
        "n_batteries": int(frame["battery_id"].nunique()),
        "n_points": int(len(frame)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "mean_battery_RMSE": float(battery_rmse.mean()),
        "median_battery_RMSE": float(battery_rmse.median()),
        "worst_battery_RMSE": float(battery_rmse.max()),
    }


def build_metrics(cv: pd.DataFrame) -> pd.DataFrame:
    rows = [metric_row(cv, "SOH_pred", "overall", "all")]
    for label, group in cv.groupby("horizon_bin"):
        rows.append(metric_row(group, "SOH_pred", "horizon", str(label)))
    for policy, group in cv.groupby("policy"):
        rows.append(metric_row(group, "SOH_pred", "policy", str(policy)))
    return pd.DataFrame(rows)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    summary, cycles = load_data()
    train_ids, test_ids = split_battery_ids(summary)
    assert len(train_ids) == 40 and len(test_ids) == 9
    test_cycles = cycles.loc[cycles["battery_id"].isin(test_ids)]
    coverage = test_cycles.groupby("battery_id")["cycle"].agg(["min", "max", "count"])
    assert coverage["min"].eq(1).all()
    assert coverage["max"].eq(CUTOFF).all()
    assert coverage["count"].eq(CUTOFF).all()

    cv, interval_widths = hybrid_cv_calibration()
    power_weight = frozen_power_weight()
    prediction_parts = []
    eol_rows = []
    eol_details = []
    for battery_id in test_ids:
        history = cycles.loc[
            cycles["battery_id"].eq(battery_id) & cycles["cycle"].le(CUTOFF)
        ].sort_values("cycle")
        policy = str(summary.loc[summary["battery_id"].eq(battery_id), "policy"].iloc[0])
        prediction_parts.append(
            forecast_short_horizon(
                history,
                battery_id,
                policy,
                interval_widths,
                power_weight,
            )
        )
        eol_summary, eol_detail = eol_sensitivity(history, battery_id, policy)
        eol_rows.append(eol_summary)
        eol_details.append(eol_detail)

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["battery_id", "cycle"])
    eol = pd.DataFrame(eol_rows).sort_values("battery_id")
    eol_detail = pd.concat(eol_details, ignore_index=True).sort_values(
        ["battery_id", "model", "fit_start"]
    )
    final_metrics = build_metrics(cv)

    required_predictions = predictions[
        ["battery_id", "cycle", "SOH_pred", "SOH_lower", "SOH_upper", "model"]
    ]
    required_eol = eol[
        [
            "battery_id",
            "EOL_cycle_pred",
            "EOL_lower",
            "EOL_upper",
            "EOL_linear",
            "EOL_power",
            "EOL_alternative",
            "model_disagreement",
            "model",
            "uncertainty_interpretation",
            "bootstrap_repetitions_per_fit",
        ]
    ]
    required_predictions.to_csv(FINAL_DIR / "predictions_151_200.csv", index=False)
    predictions.to_csv(FINAL_DIR / "prediction_components.csv", index=False)
    required_eol.to_csv(FINAL_DIR / "eol_predictions.csv", index=False)
    eol_detail.to_csv(FINAL_DIR / "eol_sensitivity_detail.csv", index=False)
    cv.to_csv(FINAL_DIR / "cv_predictions.csv", index=False)
    final_metrics.to_csv(FINAL_DIR / "metrics.csv", index=False)

    config = {
        "seed": 20260815,
        "short_horizon": {
            "cutoff": CUTOFF,
            "horizon": HORIZON,
            "power_weight": power_weight,
            "linear_weight": 1.0 - power_weight,
            "power_window": "last min(101, N) cycles; cycle 50-150 at N=150",
            "linear_window": 50,
            "interval_half_width_by_horizon_bin": interval_widths,
            "interval_note": "90% battery-grouped empirical max-residual interval; not a strict IID guarantee",
        },
        "eol": {
            "threshold": 0.8,
            "families": ["linear", "power", "exponential"],
            "fit_starts": [1, 20, 50, 80],
            "bootstrap_repetitions_per_fit": 60,
            "note": "no SOH<=0.8 supervision; reported bounds are a model-window sensitivity envelope",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "test_batteries_used_for_training_or_tuning": False,
    }
    (FINAL_DIR / "model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert len(required_predictions) == 9 * 50
    assert required_predictions.groupby("battery_id").size().eq(50).all()
    assert required_predictions[["SOH_pred", "SOH_lower", "SOH_upper"]].notna().all().all()
    assert (required_predictions["SOH_lower"] <= required_predictions["SOH_pred"]).all()
    assert (required_predictions["SOH_pred"] <= required_predictions["SOH_upper"]).all()
    assert set(required_predictions["battery_id"]) == set(test_ids)
    assert len(required_eol) == 9

    from problem3.final.figures import generate_all_figures
    from problem3.final.report_builder import write_report

    generate_all_figures(summary, cycles, predictions, cv, eol, eol_detail)
    write_report(summary, cycles, predictions, cv, eol, final_metrics)
    print(final_metrics.to_string(index=False))
    print(required_eol.to_string(index=False))
    print(f"Wrote {len(required_predictions)} short-horizon rows and {len(required_eol)} EOL rows")


if __name__ == "__main__":
    main()
