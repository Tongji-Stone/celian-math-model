from __future__ import annotations

import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from problem3.common.data import cutoff_view, future_view, load_data, split_battery_ids
from problem3.common.metrics import HORIZON_BINS, HORIZON_LABELS, mae, rmse, summarize_predictions
from problem3.common.validation import EARLY_CUTOFFS, FORECAST_HORIZON, SEED, leave_one_battery_out

from .model import (
    MAX_EOL_CYCLE,
    MEAN_KINDS,
    GPConfig,
    block_bootstrap_eol,
    candidate_configs,
    fit_mean,
    fit_predict_gp,
)


OUTPUT_DIR = Path(__file__).resolve().parent
PSEUDO_THRESHOLDS = (0.997, 0.995, 0.990, 0.980, 0.970)
MIN_THRESHOLD_CROSSINGS = 10


def _conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if not len(values):
        return np.nan
    rank = min(len(values) - 1, int(np.ceil((len(values) + 1) * coverage)) - 1)
    return float(values[max(rank, 0)])


def _build_forecast_cache(
    cycles: pd.DataFrame,
    train_ids: list[int],
    configs: list[GPConfig],
) -> dict[tuple[int, int, str], dict[str, object]]:
    cache: dict[tuple[int, int, str], dict[str, object]] = {}
    for cutoff in EARLY_CUTOFFS:
        future_cycles = np.arange(cutoff + 1, cutoff + FORECAST_HORIZON + 1, dtype=float)
        for battery_id in train_ids:
            history = cutoff_view(cycles, battery_id, cutoff)
            future = future_view(cycles, battery_id, cutoff, FORECAST_HORIZON)
            x = history["cycle"].to_numpy(dtype=float)
            y = history["SOH"].to_numpy(dtype=float)
            truth = future["SOH"].to_numpy(dtype=float)
            for config in configs:
                forecast = fit_predict_gp(x, y, future_cycles, config)
                cache[(cutoff, battery_id, config.name)] = {
                    "prediction": forecast.mean,
                    "std": forecast.std,
                    "truth": truth,
                    "signal_std": forecast.signal_std,
                    "noise_std": forecast.noise_std,
                }
    return cache


def run_nested_backtest(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    train_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    configs = candidate_configs()
    cache = _build_forecast_cache(cycles, train_ids, configs)
    rows: list[dict[str, object]] = []
    selection_counter: Counter[str] = Counter()
    inner_scores_by_fold: list[dict[str, object]] = []
    candidate_performance: list[dict[str, object]] = []

    for cutoff in EARLY_CUTOFFS:
        for config in configs:
            per_battery_rmse: list[float] = []
            all_errors: list[np.ndarray] = []
            for battery_id in train_ids:
                result = cache[(cutoff, battery_id, config.name)]
                errors = np.asarray(result["prediction"]) - np.asarray(result["truth"])
                all_errors.append(errors)
                per_battery_rmse.append(float(np.sqrt(np.mean(errors**2))))
            concatenated = np.concatenate(all_errors)
            candidate_performance.append(
                {
                    "cutoff": cutoff,
                    "config": config.name,
                    "mean_kind": config.mean_kind,
                    "kernel_kind": config.kernel_kind,
                    "length_scale": config.length_scale,
                    "overall_RMSE": float(np.sqrt(np.mean(concatenated**2))),
                    "MAE": float(np.mean(np.abs(concatenated))),
                    "mean_battery_RMSE": float(np.mean(per_battery_rmse)),
                    "worst_battery_RMSE": float(np.max(per_battery_rmse)),
                }
            )
        for inner_ids, target_id in leave_one_battery_out(train_ids):
            scores: dict[str, float] = {}
            for config in configs:
                errors = np.concatenate(
                    [
                        np.asarray(cache[(cutoff, battery_id, config.name)]["prediction"])
                        - np.asarray(cache[(cutoff, battery_id, config.name)]["truth"])
                        for battery_id in inner_ids
                    ]
                )
                scores[config.name] = float(np.sqrt(np.mean(errors**2)))
            chosen = min(configs, key=lambda config: (scores[config.name], config.name))
            selection_counter[chosen.name] += 1

            inner_errors = np.concatenate(
                [
                    np.abs(
                        np.asarray(cache[(cutoff, battery_id, chosen.name)]["prediction"])
                        - np.asarray(cache[(cutoff, battery_id, chosen.name)]["truth"])
                    )
                    for battery_id in inner_ids
                ]
            )
            inner_std = np.concatenate(
                [
                    np.asarray(cache[(cutoff, battery_id, chosen.name)]["std"])
                    for battery_id in inner_ids
                ]
            )
            standardized = inner_errors / np.maximum(inner_std, 1e-9)
            q90 = _conformal_quantile(standardized, 0.90)
            q95 = _conformal_quantile(standardized, 0.95)
            inner_scores_by_fold.append(
                {
                    "cutoff": cutoff,
                    "outer_battery_id": target_id,
                    "selected": chosen.name,
                    "selected_inner_RMSE": scores[chosen.name],
                    "q90": q90,
                    "q95": q95,
                }
            )

            target = cache[(cutoff, target_id, chosen.name)]
            prediction = np.asarray(target["prediction"])
            std = np.asarray(target["std"])
            truth = np.asarray(target["truth"])
            policy = summary.loc[summary["battery_id"].eq(target_id), "policy"].iloc[0]
            for index in range(FORECAST_HORIZON):
                rows.append(
                    {
                        "battery_id": target_id,
                        "policy": policy,
                        "cutoff": cutoff,
                        "horizon": index + 1,
                        "cycle": cutoff + index + 1,
                        "y_true": float(truth[index]),
                        "SOH_pred": float(prediction[index]),
                        "SOH_lower_90": float(prediction[index] - q90 * std[index]),
                        "SOH_upper_90": float(prediction[index] + q90 * std[index]),
                        "SOH_lower_95": float(prediction[index] - q95 * std[index]),
                        "SOH_upper_95": float(prediction[index] + q95 * std[index]),
                        "mean": float(prediction[index]),
                        "lower": float(prediction[index] - q95 * std[index]),
                        "upper": float(prediction[index] + q95 * std[index]),
                        "raw_GP_std": float(std[index]),
                        "interval_q90": q90,
                        "interval_q95": q95,
                        "model": chosen.name,
                        "signal_std": float(target["signal_std"]),
                        "noise_std": float(target["noise_std"]),
                    }
                )
    metadata = {
        "candidate_count": len(configs),
        "candidates": [config.name for config in configs],
        "selection_counts": dict(sorted(selection_counter.items())),
        "inner_folds": inner_scores_by_fold,
        "candidate_performance": candidate_performance,
    }
    return pd.DataFrame(rows), metadata


def overall_metric_rows(predictions: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cutoff, group in predictions.groupby("cutoff"):
        row = summarize_predictions(group, "SOH_pred", "Nested mean+GP residual")
        row["cutoff"] = int(cutoff)
        rows.append(row)
    return rows


def policy_metric_rows(predictions: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (cutoff, policy), group in predictions.groupby(["cutoff", "policy"]):
        rows.append(
            {
                "cutoff": int(cutoff),
                "policy": str(policy),
                "n_batteries": int(group["battery_id"].nunique()),
                "RMSE": rmse(group["y_true"].to_numpy(), group["SOH_pred"].to_numpy()),
                "MAE": mae(group["y_true"].to_numpy(), group["SOH_pred"].to_numpy()),
            }
        )
    return rows


def horizon_metric_rows(predictions: pd.DataFrame) -> list[dict[str, object]]:
    work = predictions.assign(
        horizon_bin=pd.cut(
            predictions["horizon"],
            HORIZON_BINS,
            labels=HORIZON_LABELS,
            include_lowest=True,
        )
    )
    rows: list[dict[str, object]] = []
    for (cutoff, horizon_bin), group in work.groupby(["cutoff", "horizon_bin"], observed=True):
        rows.append(
            {
                "cutoff": int(cutoff),
                "horizon_bin": str(horizon_bin),
                "n_points": int(len(group)),
                "RMSE": rmse(group["y_true"].to_numpy(), group["SOH_pred"].to_numpy()),
                "MAE": mae(group["y_true"].to_numpy(), group["SOH_pred"].to_numpy()),
            }
        )
    return rows


def uncertainty_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(cutoff: int, group_type: str, group_value: str, group: pd.DataFrame) -> None:
        rows.append(
            {
                "cutoff": int(cutoff),
                "group_type": group_type,
                "group_value": group_value,
                "n_points": int(len(group)),
                "coverage_90": float(
                    ((group["y_true"] >= group["SOH_lower_90"]) & (group["y_true"] <= group["SOH_upper_90"])).mean()
                ),
                "mean_width_90": float((group["SOH_upper_90"] - group["SOH_lower_90"]).mean()),
                "coverage_95": float(
                    ((group["y_true"] >= group["SOH_lower_95"]) & (group["y_true"] <= group["SOH_upper_95"])).mean()
                ),
                "mean_width_95": float((group["SOH_upper_95"] - group["SOH_lower_95"]).mean()),
            }
        )

    for cutoff, cutoff_df in predictions.groupby("cutoff"):
        add_row(int(cutoff), "overall", "all", cutoff_df)
        for policy, group in cutoff_df.groupby("policy"):
            add_row(int(cutoff), "policy", str(policy), group)
        binned = cutoff_df.assign(
            horizon_bin=pd.cut(
                cutoff_df["horizon"],
                HORIZON_BINS,
                labels=HORIZON_LABELS,
                include_lowest=True,
            )
        )
        for horizon_bin, group in binned.groupby("horizon_bin", observed=True):
            add_row(int(cutoff), "horizon", str(horizon_bin), group)
    return pd.DataFrame(rows)


def _first_sustained_crossing(group: pd.DataFrame, threshold: float, run_length: int = 3) -> float:
    values = group.sort_values("cycle")["SOH_smooth"].to_numpy(dtype=float)
    cycles = group.sort_values("cycle")["cycle"].to_numpy(dtype=int)
    below = values <= threshold
    if len(below) < run_length:
        return np.nan
    sustained = np.convolve(below.astype(int), np.ones(run_length, dtype=int), mode="valid") == run_length
    indices = np.flatnonzero(sustained)
    return float(cycles[indices[0]]) if len(indices) else np.nan


def pseudo_threshold_validation(
    cycles: pd.DataFrame,
    train_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    crossing_lookup: dict[tuple[int, float], float] = {}
    counts: dict[str, int] = {}
    for threshold in PSEUDO_THRESHOLDS:
        for battery_id in train_ids:
            group = cycles.loc[cycles["battery_id"].eq(battery_id)]
            crossing_lookup[(battery_id, threshold)] = _first_sustained_crossing(group, threshold)
        counts[f"{threshold:.3f}"] = int(
            sum(np.isfinite(crossing_lookup[(battery_id, threshold)]) for battery_id in train_ids)
        )
    selected = [
        threshold
        for threshold in PSEUDO_THRESHOLDS
        if counts[f"{threshold:.3f}"] >= MIN_THRESHOLD_CROSSINGS
    ]

    rows: list[dict[str, object]] = []
    for threshold in selected:
        for cutoff in EARLY_CUTOFFS:
            for mean_kind in MEAN_KINDS:
                errors: list[float] = []
                predictions: list[float] = []
                truths: list[float] = []
                for battery_id in train_ids:
                    true_crossing = crossing_lookup[(battery_id, threshold)]
                    if not np.isfinite(true_crossing) or true_crossing <= cutoff:
                        continue
                    history = cutoff_view(cycles, battery_id, cutoff)
                    model = fit_mean(
                        history["cycle"].to_numpy(dtype=float),
                        history["SOH"].to_numpy(dtype=float),
                        mean_kind,
                    )
                    predicted = model.crossing_cycle(threshold, cutoff)
                    if np.isfinite(predicted):
                        predictions.append(float(predicted))
                        truths.append(float(true_crossing))
                        errors.append(float(abs(predicted - true_crossing)))
                eligible = sum(
                    np.isfinite(crossing_lookup[(battery_id, threshold)])
                    and crossing_lookup[(battery_id, threshold)] > cutoff
                    for battery_id in train_ids
                )
                rows.append(
                    {
                        "threshold": threshold,
                        "cutoff": cutoff,
                        "mean_kind": mean_kind,
                        "eligible_batteries": int(eligible),
                        "finite_predictions": len(predictions),
                        "finite_rate": float(len(predictions) / eligible) if eligible else np.nan,
                        "crossing_MAE": float(np.mean(errors)) if errors else np.nan,
                        "crossing_median_AE": float(np.median(errors)) if errors else np.nan,
                        "crossing_RMSE": rmse(np.asarray(truths), np.asarray(predictions)) if errors else np.nan,
                    }
                )
    return pd.DataFrame(rows), counts


def eol_sensitivity_table(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, summary_row in summary.sort_values("battery_id").iterrows():
        battery_id = int(summary_row["battery_id"])
        history = cutoff_view(cycles, battery_id, 150)
        x = history["cycle"].to_numpy(dtype=float)
        y = history["SOH"].to_numpy(dtype=float)
        for mean_index, mean_kind in enumerate(MEAN_KINDS):
            result = block_bootstrap_eol(
                x,
                y,
                mean_kind,
                threshold=0.8,
                n_samples=100,
                seed=SEED + 1000 * battery_id + mean_index,
            )
            rows.append(
                {
                    "battery_id": battery_id,
                    "prediction_test": int(summary_row["prediction_test"]),
                    "policy": summary_row["policy"],
                    "cutoff": 150,
                    "mean_kind": mean_kind,
                    "EOL_point": result["point"],
                    "EOL_median": result["median"],
                    "EOL_lower_90": result["lower_90"],
                    "EOL_upper_90": result["upper_90"],
                    "EOL_lower_95": result["lower_95"],
                    "EOL_upper_95": result["upper_95"],
                    "bootstrap_crossing_probability": result["crossing_probability"],
                    "bootstrap_samples": int(result["n_samples"]),
                    "max_reported_cycle": MAX_EOL_CYCLE,
                }
            )
    table = pd.DataFrame(rows)
    disagreement = (
        table.groupby("battery_id")["EOL_point"]
        .agg(lambda values: float(np.nanmax(values) - np.nanmin(values)) if np.isfinite(values).any() else np.nan)
        .rename("model_disagreement")
    )
    return table.merge(disagreement, on="battery_id", how="left")


def _baseline_comparison(overall_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline_path = Path(__file__).resolve().parents[2] / "common" / "baseline_metrics.csv"
    baseline = pd.read_csv(baseline_path)
    gp = pd.DataFrame(overall_rows)
    rows: list[dict[str, object]] = []
    for cutoff in EARLY_CUTOFFS:
        group = baseline.loc[
            baseline["cutoff"].eq(cutoff)
            & baseline["model"].isin(["Linear-10", "Linear-20", "Linear-30", "Linear-50"])
        ].sort_values(["overall_RMSE", "model"])
        best = group.iloc[0]
        gp_rmse = float(gp.loc[gp["cutoff"].eq(cutoff), "overall_RMSE"].iloc[0])
        baseline_rmse = float(best["overall_RMSE"])
        nested = baseline.loc[
            baseline["cutoff"].eq(cutoff) & baseline["model"].eq("Linear-Nested")
        ].iloc[0]
        nested_rmse = float(nested["overall_RMSE"])
        rows.append(
            {
                "cutoff": cutoff,
                "best_recent_linear": str(best["model"]),
                "baseline_RMSE": baseline_rmse,
                "gp_RMSE": gp_rmse,
                "relative_RMSE_change": (gp_rmse - baseline_rmse) / baseline_rmse,
                "nested_linear_RMSE": nested_rmse,
                "relative_change_vs_nested_linear": (gp_rmse - nested_rmse) / nested_rmse,
            }
        )
    return rows


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fmt(value: float, digits: int = 6) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    metrics: dict[str, object],
    uncertainty: pd.DataFrame,
    pseudo: pd.DataFrame,
    eol: pd.DataFrame,
) -> None:
    overall = pd.DataFrame(metrics["overall_metrics"])
    comparison = pd.DataFrame(metrics["baseline_comparison"])
    cutoff150 = overall.loc[overall["cutoff"].eq(150)].iloc[0]
    compare150 = comparison.loc[comparison["cutoff"].eq(150)].iloc[0]
    interval150 = uncertainty.loc[
        uncertainty["cutoff"].eq(150)
        & uncertainty["group_type"].eq("overall")
    ].iloc[0]
    candidate_metrics = pd.DataFrame(metrics["selection"]["candidate_performance"])
    candidate150 = candidate_metrics.loc[candidate_metrics["cutoff"].eq(150)]
    best_by_mean = (
        candidate150.sort_values(["overall_RMSE", "config"])
        .groupby("mean_kind", as_index=False)
        .first()
        .sort_values("overall_RMSE")
    )
    best_by_kernel = (
        candidate150.sort_values(["overall_RMSE", "config"])
        .groupby("kernel_kind", as_index=False)
        .first()
        .sort_values("overall_RMSE")
    )
    horizon150 = pd.DataFrame(metrics["horizon_metrics"])
    horizon150 = horizon150.loc[horizon150["cutoff"].eq(150)]
    rmse50 = float(overall.loc[overall["cutoff"].eq(50), "overall_RMSE"].iloc[0])
    rmse150 = float(cutoff150["overall_RMSE"])
    best_cutoff_row = overall.sort_values(["overall_RMSE", "cutoff"]).iloc[0]
    selected_pseudo = pseudo.loc[
        pseudo["eligible_batteries"].ge(5) & pseudo["finite_predictions"].ge(5)
    ].sort_values(["threshold", "crossing_MAE"])
    pseudo_lines = []
    for threshold, group in selected_pseudo.groupby("threshold"):
        best = group.sort_values("crossing_MAE").iloc[0]
        pseudo_lines.append(
            f"- 阈值 {threshold:.3f}：在 cutoff={int(best['cutoff'])} 的可评估子集上，"
            f"{best['mean_kind']} 的 crossing MAE={_fmt(float(best['crossing_MAE']), 2)} cycles "
            f"(n={int(best['finite_predictions'])})。"
        )
    if not pseudo_lines:
        pseudo_lines.append("- 没有形成至少 5 块电池的可评估 cutoff 子集；不报告伪阈值误差。")

    test_eol = eol.loc[eol["prediction_test"].eq(1)]
    disagreement = test_eol.groupby("battery_id")["EOL_point"].agg(lambda values: np.nanmax(values) - np.nanmin(values))
    finite_disagreement = disagreement.replace([np.inf, -np.inf], np.nan).dropna()
    median_disagreement = float(finite_disagreement.median()) if len(finite_disagreement) else np.nan

    lines = [
        "# Agent 04：显式均值函数 + Gaussian Process 残差",
        "",
        "## 结论摘要",
        "",
        f"在严格的外层 Leave-One-Battery-Out、内层其余电池选型下，cutoff=150 的 GP 总体 "
        f"RMSE={_fmt(float(cutoff150['overall_RMSE']))}，MAE={_fmt(float(cutoff150['MAE']))}。"
        f"同一 cutoff 最佳固定 recent-linear 基线是 {compare150['best_recent_linear']}，"
        f"RMSE={_fmt(float(compare150['baseline_RMSE']))}；GP 相对变化 "
        f"{100.0 * float(compare150['relative_RMSE_change']):+.1f}%。严格内层选窗的 Linear-Nested "
        f"RMSE={_fmt(float(compare150['nested_linear_RMSE']))}，GP 相对变化 "
        f"{100.0 * float(compare150['relative_change_vs_nested_linear']):+.1f}%。",
        "",
        "这一路线的主要价值是给出经过跨电池校准的预测区间，而不是保证点预测一定胜过局部线性。"
        f"cutoff=150 的 90%/95% 区间经验覆盖率分别为 "
        f"{100.0 * float(interval150['coverage_90']):.1f}% / {100.0 * float(interval150['coverage_95']):.1f}%，"
        f"平均宽度为 {_fmt(float(interval150['mean_width_90']))} / {_fmt(float(interval150['mean_width_95']))}。",
        "",
        "## 方法与泄漏控制",
        "",
        "对每块目标电池只使用 `cycle <= cutoff` 的原始 SOH。模型写作 "
        "$SOH(n)=m(n)+r(n)$；$m(n)$ 比较局部线性、幂律和加速指数形式，"
        "$r(n)$ 使用 RBF、Matérn 3/2 或 Matérn 5/2 核。核长度尺度候选为 10、30、60 cycles。"
        "共 27 个候选组合。所有候选在外层目标电池以外的 39 块电池上比较 50-step RMSE，"
        "目标电池未来值不参与均值函数、核或长度尺度选择。",
        "",
        "原始 `SOH_smooth` 未用于拟合，因为题目未给出其平滑算法，无法排除 cutoff 边界处的双向平滑。"
        "非线性均值用稳健 soft-L1 拟合；GP 残差的信号/噪声尺度只从当前电池 cutoff 内残差估计。"
        "90% 和 95% 区间用内层电池的标准化绝对误差分位数校准，再应用于被留出的外层电池。",
        "",
        "## Short-horizon 回测",
        "",
        "| cutoff | GP RMSE | GP mean-battery RMSE | GP worst-battery RMSE | best fixed linear | fixed RMSE | nested linear RMSE | GP vs nested |",
        "|---:|---:|---:|---:|:---|---:|---:|---:|",
    ]
    for _, row in overall.sort_values("cutoff").iterrows():
        comp = comparison.loc[comparison["cutoff"].eq(row["cutoff"])].iloc[0]
        lines.append(
            f"| {int(row['cutoff'])} | {_fmt(float(row['overall_RMSE']))} | "
            f"{_fmt(float(row['mean_battery_RMSE']))} | {_fmt(float(row['worst_battery_RMSE']))} | "
            f"{comp['best_recent_linear']} | {_fmt(float(comp['baseline_RMSE']))} | "
            f"{_fmt(float(comp['nested_linear_RMSE']))} | "
            f"{100.0 * float(comp['relative_change_vs_nested_linear']):+.1f}% |"
        )
    lines.extend(
        [
            "",
            "### Forecast horizon",
            "",
            "| horizon after cutoff=150 | RMSE | MAE |",
            "|:---|---:|---:|",
            *[
                f"| {row['horizon_bin']} | {_fmt(float(row['RMSE']))} | {_fmt(float(row['MAE']))} |"
                for _, row in horizon150.iterrows()
            ],
            "",
            "误差总体随外推距离扩大；41–50 step 的 RMSE 明显高于 1–10 step，说明局部残差修正的作用会随距离衰减，"
            "长期部分更依赖均值函数。",
            "",
            "### Early-data length",
            "",
            f"从 cutoff=50 增加到 cutoff=150，GP RMSE 由 {_fmt(rmse50)} 降至 {_fmt(rmse150)}，"
            f"相对下降 {100.0 * (rmse50 - rmse150) / rmse50:.1f}%。但性能并不单调："
            f"本实验最佳 cutoff={int(best_cutoff_row['cutoff'])}，RMSE={_fmt(float(best_cutoff_row['overall_RMSE']))}。"
            "因此不能把更多早期循环机械地等同于更低误差；当前均值族与最近 50-cycle 线性基线的局部适配仍会影响结果。",
            "",
            "### 均值函数与核的独立比较",
            "",
            "下面给出 cutoff=150 时每类均值函数跨核/长度尺度的最佳候选；它们是候选诊断，"
            "最终逐电池预测仍使用严格内层选型。",
            "",
            "| mean function | best kernel | length scale | RMSE |",
            "|:---|:---|---:|---:|",
            *[
                f"| {row['mean_kind']} | {row['kernel_kind']} | {float(row['length_scale']):.0f} | "
                f"{_fmt(float(row['overall_RMSE']))} |"
                for _, row in best_by_mean.iterrows()
            ],
            "",
            "按核函数聚合后的 cutoff=150 最佳候选为：",
            "",
            "| kernel | best mean | length scale | RMSE |",
            "|:---|:---|---:|---:|",
            *[
                f"| {row['kernel_kind']} | {row['mean_kind']} | {float(row['length_scale']):.0f} | "
                f"{_fmt(float(row['overall_RMSE']))} |"
                for _, row in best_by_kernel.iterrows()
            ],
            "",
            "完整逐点预测、分 policy 指标和 horizon 分段区间指标分别见 `predictions.csv`、"
            "`metrics.json`、`horizon_metrics.csv` 与 `uncertainty_metrics.csv`；"
            "全部 27 个候选的统一指标另见 `candidate_metrics.csv`。"
            "原始 GP 标准差不直接当置信区间；"
            "报告区间均使用严格内层电池误差校准。",
            "",
            "## EOL：外推而非监督预测",
            "",
            "40 块训练电池的 200-cycle 数据和 9 块测试电池的 150-cycle 数据均没有 SOH<=0.8。"
            "因此没有可报告的真实 EOL RMSE。`eol_sensitivity.csv` 分别给出线性、幂律、指数均值函数的 "
            "0.8 crossing；区间来自 100 条移动块残差 bootstrap 轨迹后重新拟合均值参数。"
            "GP 残差在训练域外回归到 0，不能凭空提供远至 0.8 的物理信息，所以这些区间是给定函数族的条件敏感性区间，"
            "不是已校准寿命置信区间。",
            "",
            f"9 块测试电池三种函数形式的有限 point EOL 跨模型差异中位数为 "
            f"{_fmt(median_disagreement, 1)} cycles。若某一模型在 {MAX_EOL_CYCLE} cycles 内不 crossing，"
            "CSV 保留 NA 并报告 bootstrap crossing probability，不以截断上限伪装成寿命。",
            "",
            "### 伪阈值验证",
            "",
            "伪阈值只保留至少 10 块训练电池发生连续 3-cycle crossing 的阈值；"
            "当前入选阈值及可评估子集结果如下：",
            "",
            *pseudo_lines,
            "",
            "伪阈值离 0.8 很远，只能检验近域 crossing 排序和数值稳定性，不能证明真正 EOL 的绝对准确度。",
            "",
            "## 不确定性解释",
            "",
            "短期区间的覆盖率是 40 块训练电池的外层 LOBO 经验覆盖率；policy 级结果样本仅 2–7 块电池，"
            "不可把单个 policy 的覆盖率解释为严格概率保证。EOL bootstrap 同时暴露拟合残差与参数不确定性，"
            "但没有包含未知老化机理改变、knee onset 或训练域外分布漂移，因此必须与三种均值函数的模型分歧联合报告。",
            "",
            "## 复杂度与复现",
            "",
            f"- Seed: {metrics['run']['seed']}。",
            f"- 运行时间: {float(metrics['run']['runtime_seconds']):.1f} s。",
            f"- Python {metrics['versions']['python']}；NumPy {metrics['versions']['numpy']}；"
            f"Pandas {metrics['versions']['pandas']}；SciPy {metrics['versions']['scipy']}。",
            "- 每块电池的精确 GP 保存 cutoff 个对偶系数；连续量包括 2 个（linear）或 3 个（power/exponential）"
            "均值参数以及 2 个从残差估计的信号/噪声尺度，因此 cutoff=150 时预测状态约为 154–155 个数。"
            "核种类和长度尺度是内层 battery-level validation 选择的离散超参数。",
            "- GP 为每块电池独立的一维精确 GP；没有引入不稳定的多输出 GP 或策略核。"
            "由于本 Agent 的任务是检验单电池趋势+残差 GP，本结果不声称已量化 charging-policy 的增益。",
            "",
            "## 文献依据",
            "",
            "1. Richardson, R. R., Osborne, M. A., & Howey, D. A. (2017). "
            "*Gaussian process regression for forecasting battery state of health*. Journal of Power Sources, 357, 209–219. "
            "DOI: [10.1016/j.jpowsour.2017.05.004](https://doi.org/10.1016/j.jpowsour.2017.05.004). "
            "该文明确讨论显式均值函数、核选择、SOH/RUL 预测与 GP 不确定性；本实现采用其“显式退化均值 + GP 残差”思想，"
            "但没有读取其任何电池未来数据。",
            "2. Severson, K. A., Attia, P. M., Jin, N., et al. (2019). "
            "*Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 4, 383–391. "
            "DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8). "
            "该文支持按电池而非按循环行验证早期寿命信息；本实现仅借鉴方法原则。",
            "",
            "## 建议",
            "",
            "若 GP 的 cutoff=150 点预测 RMSE 未优于 recent-linear，则不建议把 GP 单独作为 Stage I 主模型；"
            "可将其校准区间或均值+残差预测作为候选 expert。Stage II 必须继续采用跨函数族敏感性报告，"
            "不能把任一 GP/均值函数无限外推得到的单个 0.8 crossing 当作精确寿命。",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    summary, cycles = load_data()
    train_ids, test_ids = split_battery_ids(summary)
    predictions, selection_metadata = run_nested_backtest(summary, cycles, train_ids)
    overall = overall_metric_rows(predictions)
    policy = policy_metric_rows(predictions)
    horizon = horizon_metric_rows(predictions)
    uncertainty = uncertainty_metric_table(predictions)
    pseudo, threshold_counts = pseudo_threshold_validation(cycles, train_ids)
    eol = eol_sensitivity_table(summary, cycles)
    baseline_comparison = _baseline_comparison(overall)
    runtime = time.perf_counter() - started

    metrics: dict[str, object] = {
        "agent": "agent_04_gp",
        "method": "explicit degradation mean plus exact single-battery GP residual",
        "run": {
            "seed": SEED,
            "runtime_seconds": runtime,
            "train_batteries": len(train_ids),
            "held_out_test_batteries_not_used_for_validation": test_ids,
            "cutoffs": EARLY_CUTOFFS,
            "horizon": FORECAST_HORIZON,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "platform": sys.platform,
        },
        "hyperparameters": {
            "mean_functions": list(MEAN_KINDS),
            "kernels": ["rbf", "matern32", "matern52"],
            "length_scales_cycles": [10.0, 30.0, 60.0],
            "linear_mean_window": 50,
            "mean_fit_loss": "soft_l1",
            "interval_calibration": "inner-battery standardized absolute-error conformal quantile",
            "eol_bootstrap_samples": 100,
            "eol_bootstrap_block_length": 10,
            "max_eol_cycle": MAX_EOL_CYCLE,
        },
        "complexity": {
            "gp_dual_coefficients_per_battery": "equal to cutoff: 50, 75, 100, 125, or 150",
            "mean_parameters": {"linear": 2, "power": 3, "exponential": 3},
            "residual_scale_parameters": 2,
            "discrete_selected_hyperparameters": ["mean family", "kernel family", "length scale"],
            "trainable_multioutput_parameters": 0,
        },
        "validation": {
            "outer": "Leave-One-Battery-Out over 40 non-test batteries",
            "inner": "select mean/kernel/length scale using only the other 39 batteries",
            "target": "raw SOH; provided SOH_smooth excluded from fitting to avoid unknown smoothing-boundary leakage",
        },
        "overall_metrics": overall,
        "policy_metrics": policy,
        "horizon_metrics": horizon,
        "baseline_comparison": baseline_comparison,
        "selection": selection_metadata,
        "pseudo_threshold_crossing_counts": threshold_counts,
        "pseudo_threshold_metrics": pseudo.to_dict(orient="records"),
    }
    metrics = _json_safe(metrics)

    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    pd.DataFrame(selection_metadata["candidate_performance"]).to_csv(
        OUTPUT_DIR / "candidate_metrics.csv", index=False
    )
    pd.DataFrame(horizon).to_csv(OUTPUT_DIR / "horizon_metrics.csv", index=False)
    uncertainty.to_csv(OUTPUT_DIR / "uncertainty_metrics.csv", index=False)
    pseudo.to_csv(OUTPUT_DIR / "pseudo_threshold_validation.csv", index=False)
    eol.to_csv(OUTPUT_DIR / "eol_sensitivity.csv", index=False)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics, uncertainty, pseudo, eol)
    print(pd.DataFrame(overall).sort_values("cutoff").to_string(index=False))
    print("\nUncertainty (overall):")
    print(uncertainty.loc[uncertainty["group_type"].eq("overall")].to_string(index=False))
    print(f"\nRuntime: {runtime:.1f} s")


if __name__ == "__main__":
    main()
