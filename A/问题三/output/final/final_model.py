from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from problem3.agents.agent_02_physics.model import (
    BASE_MODELS,
    crossing_cycle,
    fit_degradation_model,
    forecast,
    moving_block_bootstrap_crossing_interval,
    robust_asof_soh,
)


ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = ROOT / "problem3" / "judge"
SEED = 20260815
CUTOFF = 150
HORIZON = 50
EOL_THRESHOLD = 0.80
EOL_FIT_STARTS = (1, 20, 50, 80)
EOL_BOOTSTRAP_REPETITIONS = 60
EOL_BOOTSTRAP_BLOCK = 8
HORIZON_BINS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50))


def horizon_bin(horizon: int) -> str:
    for lower, upper in HORIZON_BINS:
        if lower <= int(horizon) <= upper:
            return f"{lower}-{upper}"
    raise ValueError(f"Unsupported horizon: {horizon}")


def finite_sample_quantile(values: np.ndarray, coverage: float = 0.90) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) == 0:
        return float("nan")
    rank = int(np.ceil((len(ordered) + 1) * coverage))
    return float(ordered[min(max(rank, 1), len(ordered)) - 1])


def frozen_power_weight() -> float:
    config = json.loads((JUDGE_DIR / "frozen_model.json").read_text(encoding="utf-8"))
    return float(config["power_weight"])


def recent_linear_forecast(history: pd.DataFrame, horizon: int = HORIZON) -> np.ndarray:
    tail = history.sort_values("cycle").tail(50)
    x = tail["cycle"].to_numpy(dtype=float)
    y = tail["SOH"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future_cycles = np.arange(int(history["cycle"].max()) + 1, int(history["cycle"].max()) + horizon + 1)
    return slope * future_cycles + intercept


def power_forecast(history: pd.DataFrame, horizon: int = HORIZON) -> tuple[np.ndarray, object]:
    ordered = history.sort_values("cycle")
    fit_start = max(1, int(ordered["cycle"].max()) - 100)
    window = ordered.loc[ordered["cycle"].ge(fit_start)].copy()
    cleaned = robust_asof_soh(window)
    fitted = fit_degradation_model(
        window["cycle"].to_numpy(dtype=float), cleaned, "power"
    )
    future_cycles = np.arange(int(ordered["cycle"].max()) + 1, int(ordered["cycle"].max()) + horizon + 1)
    return forecast(fitted, future_cycles), fitted


def hybrid_cv_calibration() -> tuple[pd.DataFrame, dict[str, float]]:
    cv = pd.read_csv(JUDGE_DIR / "hybrid_cv_predictions.csv")
    cv["horizon_bin"] = cv["horizon"].map(horizon_bin)
    cv["abs_error"] = (cv["SOH_pred"] - cv["y_true"]).abs()
    battery_max = (
        cv.groupby(["battery_id", "horizon_bin"], as_index=False)["abs_error"]
        .max()
    )
    widths = {
        label: finite_sample_quantile(group["abs_error"].to_numpy(), 0.90)
        for label, group in battery_max.groupby("horizon_bin")
    }
    cv["interval_half_width"] = cv["horizon_bin"].map(widths)
    cv["SOH_lower"] = cv["SOH_pred"] - cv["interval_half_width"]
    cv["SOH_upper"] = cv["SOH_pred"] + cv["interval_half_width"]
    return cv, widths


def forecast_short_horizon(
    history: pd.DataFrame,
    battery_id: int,
    policy: str,
    interval_widths: dict[str, float],
    power_weight: float,
) -> pd.DataFrame:
    if int(history["cycle"].max()) != CUTOFF or len(history) != CUTOFF:
        raise ValueError(f"Test battery {battery_id} must contain exactly cycles 1-{CUTOFF}")
    power_prediction, _ = power_forecast(history)
    linear_prediction = recent_linear_forecast(history)
    prediction = power_weight * power_prediction + (1.0 - power_weight) * linear_prediction
    rows = []
    for horizon in range(1, HORIZON + 1):
        label = horizon_bin(horizon)
        half_width = float(interval_widths[label])
        rows.append(
            {
                "battery_id": int(battery_id),
                "cycle": CUTOFF + horizon,
                "SOH_pred": float(prediction[horizon - 1]),
                "SOH_lower": float(prediction[horizon - 1] - half_width),
                "SOH_upper": float(prediction[horizon - 1] + half_width),
                "model": f"{power_weight:.3f}*Power101+{1-power_weight:.3f}*Linear50",
                "policy": policy,
                "horizon": horizon,
                "pred_power": float(power_prediction[horizon - 1]),
                "pred_linear50": float(linear_prediction[horizon - 1]),
                "interval_method": "battery-grouped empirical 90% max-residual by 10-cycle bin",
            }
        )
    return pd.DataFrame(rows)


def eol_sensitivity(
    history: pd.DataFrame,
    battery_id: int,
    policy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    detail_rows = []
    ordered = history.sort_values("cycle")
    for fit_start in EOL_FIT_STARTS:
        window = ordered.loc[ordered["cycle"].ge(fit_start)].copy()
        x = window["cycle"].to_numpy(dtype=float)
        y = robust_asof_soh(window)
        for model_index, model in enumerate(BASE_MODELS):
            fitted = fit_degradation_model(x, y, model)
            point = crossing_cycle(fitted, EOL_THRESHOLD)
            lower, upper, valid = moving_block_bootstrap_crossing_interval(
                fitted,
                x,
                y,
                EOL_THRESHOLD,
                seed=SEED + int(battery_id) * 1009 + fit_start * 17 + model_index,
                repetitions=EOL_BOOTSTRAP_REPETITIONS,
                block_length=EOL_BOOTSTRAP_BLOCK,
                interval=0.90,
            )
            detail_rows.append(
                {
                    "battery_id": int(battery_id),
                    "policy": policy,
                    "fit_start": int(fit_start),
                    "model": model,
                    "EOL_cycle": float(point),
                    "conditional_lower_90": float(lower),
                    "conditional_upper_90": float(upper),
                    "bootstrap_valid": int(valid),
                    "fit_RMSE": float(fitted.fit_rmse),
                    "fit_success": bool(fitted.success),
                }
            )
    detail = pd.DataFrame(detail_rows)
    finite_points = detail.loc[np.isfinite(detail["EOL_cycle"]), "EOL_cycle"]
    family = (
        detail.loc[np.isfinite(detail["EOL_cycle"])]
        .groupby("model")["EOL_cycle"]
        .median()
    )
    finite_lowers = detail.loc[np.isfinite(detail["conditional_lower_90"]), "conditional_lower_90"]
    finite_uppers = detail.loc[np.isfinite(detail["conditional_upper_90"]), "conditional_upper_90"]
    family_values = family.to_numpy(dtype=float)
    summary = {
        "battery_id": int(battery_id),
        "EOL_cycle_pred": float(finite_points.median()) if len(finite_points) else np.nan,
        "EOL_lower": float(finite_lowers.min()) if len(finite_lowers) else np.nan,
        "EOL_upper": float(finite_uppers.max()) if len(finite_uppers) else np.nan,
        "EOL_linear": float(family.get("linear", np.nan)),
        "EOL_power": float(family.get("power", np.nan)),
        "EOL_alternative": float(family.get("exponential", np.nan)),
        "model_disagreement": float(np.max(family_values) - np.min(family_values)) if len(family_values) else np.nan,
        "model": "median(linear,power,exponential; fit starts 1/20/50/80)",
        "uncertainty_interpretation": "model-window bootstrap sensitivity envelope; not calibrated confidence interval",
        "bootstrap_repetitions_per_fit": EOL_BOOTSTRAP_REPETITIONS,
    }
    return summary, detail
