from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = PROJECT_ROOT / "battery_summary.csv"
CYCLE_PATH = PROJECT_ROOT / "cycle_train.csv"

STATIC_COLUMNS = ["C1", "Q1", "C2", "initial_capacity", "policy"]
DYNAMIC_COLUMNS = ["capacity", "SOH", "SOH_smooth", "chargetime", "IR", "Tavg"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_PATH)
    cycles = pd.read_csv(CYCLE_PATH)
    validate_source_data(summary, cycles)
    return summary, cycles


def validate_source_data(summary: pd.DataFrame, cycles: pd.DataFrame) -> None:
    required_summary = {
        "battery_id",
        "policy",
        "C1",
        "Q1",
        "C2",
        "initial_capacity",
        "prediction_test",
    }
    required_cycles = {"battery_id", "cycle", "policy", *DYNAMIC_COLUMNS}
    missing_summary = required_summary.difference(summary.columns)
    missing_cycles = required_cycles.difference(cycles.columns)
    if missing_summary or missing_cycles:
        raise ValueError(
            f"Missing columns: summary={sorted(missing_summary)}, cycles={sorted(missing_cycles)}"
        )
    if summary["battery_id"].duplicated().any():
        raise ValueError("battery_summary.csv has duplicate battery_id values")
    if cycles.duplicated(["battery_id", "cycle"]).any():
        raise ValueError("cycle_train.csv has duplicate (battery_id, cycle) rows")
    if not set(cycles["battery_id"]).issubset(set(summary["battery_id"])):
        raise ValueError("cycle_train.csv contains unknown battery_id values")
    policy_check = cycles.merge(
        summary[["battery_id", "policy"]], on="battery_id", suffixes=("_cycle", "_summary")
    )
    if (policy_check["policy_cycle"] != policy_check["policy_summary"]).any():
        raise ValueError("Policy labels disagree between source files")


def split_battery_ids(summary: pd.DataFrame) -> tuple[list[int], list[int]]:
    train_ids = summary.loc[summary["prediction_test"].eq(0), "battery_id"].astype(int).tolist()
    test_ids = summary.loc[summary["prediction_test"].eq(1), "battery_id"].astype(int).tolist()
    return train_ids, test_ids


def cutoff_view(cycles: pd.DataFrame, battery_id: int, cutoff: int) -> pd.DataFrame:
    view = cycles.loc[
        cycles["battery_id"].eq(battery_id) & cycles["cycle"].le(cutoff)
    ].sort_values("cycle")
    if view.empty or int(view["cycle"].max()) < cutoff:
        raise ValueError(f"Battery {battery_id} does not contain cutoff cycle {cutoff}")
    return view.copy()


def future_view(
    cycles: pd.DataFrame, battery_id: int, cutoff: int, horizon: int = 50
) -> pd.DataFrame:
    view = cycles.loc[
        cycles["battery_id"].eq(battery_id)
        & cycles["cycle"].gt(cutoff)
        & cycles["cycle"].le(cutoff + horizon)
    ].sort_values("cycle")
    if len(view) != horizon:
        raise ValueError(
            f"Battery {battery_id} has {len(view)} future rows at cutoff={cutoff}; expected {horizon}"
        )
    return view.copy()


def physical_static_features(summary_row: pd.Series) -> dict[str, float]:
    c1_missing = float(pd.isna(summary_row["C1"]))
    # 80PER_3_6C is a single-stage 3.6C-to-80% policy.  C1 is structurally
    # absent, so the effective first-stage rate is C2, not a population mean.
    c1_effective = float(summary_row["C2"] if c1_missing else summary_row["C1"])
    q1 = float(summary_row["Q1"])
    c2 = float(summary_row["C2"])
    stage2_width = max(0.0, 80.0 - q1)
    weighted_rate = (c1_effective * q1 + c2 * stage2_width) / 80.0
    return {
        "C1_effective": c1_effective,
        "C1_missing_indicator": c1_missing,
        "Q1": q1,
        "C2": c2,
        "initial_capacity": float(summary_row["initial_capacity"]),
        "E1": c1_effective * q1,
        "E2": c2 * stage2_width,
        "C1_minus_C2": c1_effective - c2,
        "weighted_C_rate": weighted_rate,
        "stage1_width": q1,
        "stage2_width": stage2_width,
    }


def clean_dynamic_cutoff(view: pd.DataFrame) -> pd.DataFrame:
    clean = view.copy()
    clean.loc[clean["IR"].le(0), "IR"] = np.nan
    clean["IR"] = clean["IR"].interpolate(limit_direction="both")
    return clean
