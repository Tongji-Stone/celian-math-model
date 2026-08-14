from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from problem3.common.data import load_data, split_battery_ids
from problem3.common.metrics import mae, rmse

from problem3.agents.agent_03_ml.model import (
    SEED,
    build_learning_table,
    candidate_models,
    feature_sets,
    make_pipeline,
)


OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_METRICS = PROJECT_ROOT / "problem3" / "common" / "baseline_metrics.csv"
INNER_SPLITS = 1
PERMUTATION_REPEATS = 1
INNER_HORIZONS = (1, 10, 20, 30, 40, 50)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _composite_rmse(y_true: np.ndarray, y_pred: np.ndarray, cutoff: np.ndarray) -> float:
    overall = rmse(y_true, y_pred)
    focus = cutoff == 150
    focus_rmse = rmse(y_true[focus], y_pred[focus])
    return 0.5 * overall + 0.5 * focus_rmse


def select_configuration(
    train_table: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[str, dict[str, float]]:
    candidates = candidate_models()
    selection_table = train_table.loc[
        train_table["forecast_horizon"].isin(INNER_HORIZONS)
    ].reset_index(drop=True)
    groups = selection_table["battery_id"].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=INNER_SPLITS, test_size=0.25, random_state=seed
    )
    scores: dict[str, float] = {}
    for candidate_name, candidate in candidates.items():
        fold_scores: list[float] = []
        for fold, (fit_index, validation_index) in enumerate(
            splitter.split(selection_table, groups=groups)
        ):
            estimator = make_pipeline(candidate, seed + fold)
            estimator.fit(
                selection_table.iloc[fit_index][features],
                selection_table.iloc[fit_index]["y_delta"],
            )
            prediction = estimator.predict(selection_table.iloc[validation_index][features])
            fold_scores.append(
                _composite_rmse(
                    selection_table.iloc[validation_index]["y_delta"].to_numpy(),
                    prediction,
                    selection_table.iloc[validation_index]["cutoff"].to_numpy(),
                )
            )
        scores[candidate_name] = float(np.mean(fold_scores))
    best = min(scores, key=lambda name: (scores[name], name))
    return best, scores


def outer_safe_permutation_importance(
    estimator,
    train_table: pd.DataFrame,
    validation_table: pd.DataFrame,
    features: list[str],
    baseline_prediction: np.ndarray,
    seed: int,
) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    y_true = validation_table["y_delta"].to_numpy()
    baseline_rmse = rmse(y_true, baseline_prediction)
    rows: list[dict[str, float | str]] = []
    for feature in features:
        increases: list[float] = []
        validation_values = validation_table[feature].to_numpy()
        training_values = train_table[feature].to_numpy()
        for _ in range(PERMUTATION_REPEATS):
            changed = validation_table[features].copy()
            if np.unique(validation_values).size > 1:
                changed[feature] = rng.permutation(validation_values)
            else:
                changed[feature] = rng.choice(
                    training_values, size=len(validation_table), replace=True
                )
            changed_prediction = estimator.predict(changed)
            increases.append(rmse(y_true, changed_prediction) - baseline_rmse)
        rows.append(
            {
                "feature": feature,
                "baseline_outer_RMSE": baseline_rmse,
                "RMSE_increase_mean": float(np.mean(increases)),
                "RMSE_increase_std": float(np.std(increases)),
            }
        )
    return rows


def run_nested_backtest(
    table: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    battery_ids: list[int],
    outer_start: int = 0,
    outer_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    selected_outer_ids = sorted(battery_ids)[outer_start:]
    if outer_limit is not None:
        selected_outer_ids = selected_outer_ids[:outer_limit]

    for outer_number, target_id in enumerate(selected_outer_ids, start=1):
        train_table = table.loc[table["battery_id"].ne(target_id)].reset_index(drop=True)
        validation_table = table.loc[table["battery_id"].eq(target_id)].reset_index(drop=True)
        selected, inner_scores = select_configuration(
            train_table, feature_groups["C_plus_policy"], SEED + target_id
        )
        candidate = candidate_models()[selected]
        selection_rows.append(
            {
                "outer_battery_id": target_id,
                "selection_feature_set": "C_plus_policy",
                "selected_config": selected,
                "selected_inner_composite_RMSE": inner_scores[selected],
                "all_inner_scores": json.dumps(inner_scores, sort_keys=True),
            }
        )
        for ablation, features in feature_groups.items():
            estimator = make_pipeline(candidate, SEED + target_id)
            estimator.fit(train_table[features], train_table["y_delta"])
            delta_prediction = estimator.predict(validation_table[features])
            fold_output = validation_table[
                [
                    "battery_id",
                    "policy",
                    "cutoff",
                    "forecast_horizon",
                    "cycle",
                    "SOH_anchor",
                    "y_true",
                    "y_delta",
                ]
            ].copy()
            fold_output["ablation"] = ablation
            fold_output["y_pred"] = fold_output["SOH_anchor"] + delta_prediction
            fold_output["pred_delta"] = delta_prediction
            fold_output["selected_config"] = selected
            fold_output["inner_composite_RMSE"] = inner_scores[selected]
            predictions.append(fold_output)
            if ablation == "C_plus_policy":
                for importance in outer_safe_permutation_importance(
                    estimator,
                    train_table,
                    validation_table,
                    features,
                    delta_prediction,
                    SEED + 1000 * target_id + len(ablation),
                ):
                    importance_rows.append(
                        {
                            "outer_battery_id": target_id,
                            "ablation": ablation,
                            "selected_config": selected,
                            **importance,
                        }
                    )
        print(f"outer fold {outer_number}/{len(selected_outer_ids)}: battery {target_id}", flush=True)
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(importance_rows),
        pd.DataFrame(selection_rows),
    )


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    for (ablation, cutoff), group in predictions.groupby(["ablation", "cutoff"]):
        per_battery = group.groupby("battery_id").apply(
            lambda frame: rmse(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()),
            include_groups=False,
        )
        overall_rows.append(
            {
                "ablation": ablation,
                "cutoff": int(cutoff),
                "n_batteries": int(group["battery_id"].nunique()),
                "n_points": int(len(group)),
                "MAE": mae(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
                "overall_RMSE": rmse(
                    group["y_true"].to_numpy(), group["y_pred"].to_numpy()
                ),
                "mean_battery_RMSE": float(per_battery.mean()),
                "median_battery_RMSE": float(per_battery.median()),
                "worst_battery_RMSE": float(per_battery.max()),
            }
        )
        for policy, policy_group in group.groupby("policy"):
            policy_rows.append(
                {
                    "ablation": ablation,
                    "cutoff": int(cutoff),
                    "policy": policy,
                    "n_batteries": int(policy_group["battery_id"].nunique()),
                    "MAE": mae(
                        policy_group["y_true"].to_numpy(),
                        policy_group["y_pred"].to_numpy(),
                    ),
                    "RMSE": rmse(
                        policy_group["y_true"].to_numpy(),
                        policy_group["y_pred"].to_numpy(),
                    ),
                }
            )
        horizon_group = group.assign(
            horizon_bin=pd.cut(
                group["forecast_horizon"],
                bins=[0, 10, 20, 30, 40, 50],
                labels=["1-10", "11-20", "21-30", "31-40", "41-50"],
                include_lowest=True,
            )
        )
        for horizon_bin, bin_group in horizon_group.groupby("horizon_bin", observed=True):
            horizon_rows.append(
                {
                    "ablation": ablation,
                    "cutoff": int(cutoff),
                    "horizon_bin": str(horizon_bin),
                    "MAE": mae(
                        bin_group["y_true"].to_numpy(), bin_group["y_pred"].to_numpy()
                    ),
                    "RMSE": rmse(
                        bin_group["y_true"].to_numpy(), bin_group["y_pred"].to_numpy()
                    ),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(policy_rows), pd.DataFrame(horizon_rows)


def aggregate_importance(raw_importance: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_importance.groupby(["ablation", "feature"], as_index=False)
        .agg(
            mean_RMSE_increase=("RMSE_increase_mean", "mean"),
            median_RMSE_increase=("RMSE_increase_mean", "median"),
            std_across_outer_folds=("RMSE_increase_mean", "std"),
            positive_fold_fraction=("RMSE_increase_mean", lambda values: float((values > 0).mean())),
            n_outer_folds=("outer_battery_id", "nunique"),
        )
        .sort_values(["ablation", "mean_RMSE_increase"], ascending=[True, False])
    )


def paired_ablation_effects(predictions: pd.DataFrame) -> list[dict[str, object]]:
    comparisons = (
        ("A_SOH_only", "B_SOH_plus_dynamics", "dynamic_feature_effect"),
        ("B_SOH_plus_dynamics", "C_plus_policy", "policy_feature_effect"),
    )
    rows: list[dict[str, object]] = []
    for cutoff, cutoff_group in predictions.groupby("cutoff"):
        per_battery = cutoff_group.groupby(["battery_id", "ablation"]).apply(
            lambda frame: rmse(
                frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
            ),
            include_groups=False,
        ).unstack()
        for source, target, effect in comparisons:
            difference = (per_battery[target] - per_battery[source]).to_numpy()
            rng = np.random.default_rng(SEED + int(cutoff) + len(effect))
            bootstrap = np.asarray(
                [
                    np.mean(rng.choice(difference, len(difference), replace=True))
                    for _ in range(5000)
                ]
            )
            rows.append(
                {
                    "effect": effect,
                    "cutoff": int(cutoff),
                    "source_ablation": source,
                    "target_ablation": target,
                    "mean_paired_battery_RMSE_change": float(np.mean(difference)),
                    "median_paired_battery_RMSE_change": float(np.median(difference)),
                    "target_improved_fraction": float(np.mean(difference < 0)),
                    "bootstrap_95pct_lower": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_95pct_upper": float(np.quantile(bootstrap, 0.975)),
                }
            )
    return rows


def build_metrics_payload(
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    policy: pd.DataFrame,
    horizon: pd.DataFrame,
    selections: pd.DataFrame,
    runtime_seconds: float,
) -> dict[str, object]:
    baseline = pd.read_csv(BASELINE_METRICS)
    nested_baseline = baseline.loc[baseline["model"].eq("Linear-Nested")]
    candidate_specs = {
        name: candidate.hyperparameters for name, candidate in candidate_models().items()
    }
    cutoff_rows = overall.sort_values(["ablation", "cutoff"]).to_dict(orient="records")
    policy_150 = policy.loc[policy["cutoff"].eq(150)].to_dict(orient="records")
    horizon_150 = horizon.loc[horizon["cutoff"].eq(150)].to_dict(orient="records")
    baseline_rows = nested_baseline[
        ["cutoff", "MAE", "overall_RMSE", "mean_battery_RMSE", "median_battery_RMSE", "worst_battery_RMSE"]
    ].to_dict(orient="records")
    selected_counts = dict(Counter(selections["selected_config"]))
    return {
        "agent": "agent_03_ml",
        "task": "short-horizon SOH forecasting only",
        "target": "raw SOH delta relative to the cutoff-cycle raw SOH",
        "seed": SEED,
        "outer_validation": "Leave-One-Battery-Out over 40 non-test batteries",
        "inner_validation": (
            "one deterministic GroupShuffleSplit holdout by battery_id per outer fold "
            f"on a prespecified horizon grid {list(INNER_HORIZONS)}"
        ),
        "inner_selection_score": "0.5 * overall RMSE + 0.5 * cutoff-150 RMSE",
        "ablation_design": (
            "Each outer fold selects one configuration on C_plus_policy; the same "
            "configuration is then refit on A/B/C to isolate feature-set effects."
        ),
        "cutoffs": [50, 75, 100, 125, 150],
        "forecast_horizon": 50,
        "candidate_hyperparameters": candidate_specs,
        "selected_configuration_counts": selected_counts,
        "paired_ablation_effects": paired_ablation_effects(predictions),
        "metrics_by_cutoff": cutoff_rows,
        "policy_metrics_cutoff_150": policy_150,
        "horizon_metrics_cutoff_150": horizon_150,
        "public_linear_nested_baseline": baseline_rows,
        "runtime_seconds": runtime_seconds,
        "parameter_count": "not fixed; nonparametric tree ensembles, hyperparameters reported above",
        "versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "xgboost": _package_version("xgboost"),
        },
        "leakage_controls": [
            "prediction_test batteries excluded from table construction and all selection",
            "outer and inner folds grouped by battery_id",
            "all dynamic features recomputed only from cycle <= cutoff",
            "battery_summary dynamic means never used",
            "provided SOH_smooth not used; robust SOH processing is fit within each cutoff history",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outer-start",
        type=int,
        default=0,
        help="Zero-based first outer fold, used only for resumable batches.",
    )
    parser.add_argument(
        "--outer-limit",
        type=int,
        default=None,
        help="Diagnostic-only limit; omit for the required full 40-battery backtest.",
    )
    parser.add_argument(
        "--batch-tag",
        type=str,
        default="",
        help="Suffix for resumable batch outputs, for example batch0.",
    )
    parser.add_argument(
        "--combine-batches",
        action="store_true",
        help="Combine all batch-tagged outputs and recompute final artifacts.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    if args.combine_batches:
        prediction_files = sorted(OUTPUT_DIR.glob("predictions_batch*.csv"))
        importance_files = sorted(
            OUTPUT_DIR.glob("feature_importance_outer_folds_batch*.csv")
        )
        selection_files = sorted(OUTPUT_DIR.glob("nested_selections_batch*.csv"))
        metrics_files = sorted(OUTPUT_DIR.glob("metrics_batch*.json"))
        if not prediction_files or not importance_files or not selection_files:
            raise FileNotFoundError("No complete batch artifacts found")
        predictions = pd.concat(
            [pd.read_csv(path) for path in prediction_files], ignore_index=True
        )
        raw_importance = pd.concat(
            [pd.read_csv(path) for path in importance_files], ignore_index=True
        )
        selections = pd.concat(
            [pd.read_csv(path) for path in selection_files], ignore_index=True
        )
        if predictions["battery_id"].nunique() != 40:
            raise ValueError(
                f"Expected 40 outer batteries, found {predictions['battery_id'].nunique()}"
            )
        if predictions.duplicated(
            ["battery_id", "cutoff", "forecast_horizon", "ablation"]
        ).any():
            raise ValueError("Duplicate outer predictions across batches")
        overall, policy, horizon = metric_tables(predictions)
        importance = aggregate_importance(raw_importance)
        runtime_seconds = 0.0
        for path in metrics_files:
            with path.open("r", encoding="utf-8") as handle:
                runtime_seconds += float(json.load(handle)["runtime_seconds"])
        predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
        overall.to_csv(OUTPUT_DIR / "ablation.csv", index=False)
        policy.to_csv(OUTPUT_DIR / "policy_metrics.csv", index=False)
        horizon.to_csv(OUTPUT_DIR / "horizon_metrics.csv", index=False)
        importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
        raw_importance.to_csv(
            OUTPUT_DIR / "feature_importance_outer_folds.csv", index=False
        )
        selections.to_csv(OUTPUT_DIR / "nested_selections.csv", index=False)
        with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                build_metrics_payload(
                    predictions, overall, policy, horizon, selections, runtime_seconds
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(overall.sort_values(["cutoff", "overall_RMSE"]).to_string(index=False))
        print(f"combined_runtime_seconds={runtime_seconds:.3f}")
        return

    summary, cycles = load_data()
    train_ids, _ = split_battery_ids(summary)
    table = build_learning_table(cycles, summary, train_ids)
    features = feature_sets(table)
    predictions, raw_importance, selections = run_nested_backtest(
        table,
        features,
        train_ids,
        outer_start=args.outer_start,
        outer_limit=args.outer_limit,
    )
    overall, policy, horizon = metric_tables(predictions)
    importance = aggregate_importance(raw_importance)
    runtime_seconds = float(time.perf_counter() - started)

    suffix = f"_{args.batch_tag}" if args.batch_tag else ""
    predictions.to_csv(OUTPUT_DIR / f"predictions{suffix}.csv", index=False)
    overall.to_csv(OUTPUT_DIR / f"ablation{suffix}.csv", index=False)
    policy.to_csv(OUTPUT_DIR / f"policy_metrics{suffix}.csv", index=False)
    horizon.to_csv(OUTPUT_DIR / f"horizon_metrics{suffix}.csv", index=False)
    importance.to_csv(OUTPUT_DIR / f"feature_importance{suffix}.csv", index=False)
    raw_importance.to_csv(
        OUTPUT_DIR / f"feature_importance_outer_folds{suffix}.csv", index=False
    )
    selections.to_csv(OUTPUT_DIR / f"nested_selections{suffix}.csv", index=False)
    with (OUTPUT_DIR / f"metrics{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            build_metrics_payload(
                predictions, overall, policy, horizon, selections, runtime_seconds
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(overall.sort_values(["cutoff", "overall_RMSE"]).to_string(index=False))
    print(f"runtime_seconds={runtime_seconds:.3f}")


if __name__ == "__main__":
    main()
