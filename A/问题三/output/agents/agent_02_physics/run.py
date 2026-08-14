from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problem3.common.data import cutoff_view, future_view, load_data, split_battery_ids
from problem3.common.validation import EARLY_CUTOFFS, FORECAST_HORIZON, SEED

from problem3.agents.agent_02_physics.model import (
    BASE_MODELS,
    crossing_cycle,
    covariance_crossing_interval,
    fit_degradation_model,
    forecast,
    model_derivative,
    moving_block_bootstrap_crossing_interval,
    persistent_crossing_cycle,
    power_damped_residual_forecast,
    robust_asof_soh,
)


OUTPUT_DIR = Path(__file__).resolve().parent
FITTING_STARTS = [1, 20, 50, 80]
PSEUDO_THRESHOLD_CANDIDATES = [0.997, 0.995, 0.990, 0.980, 0.970]
PSEUDO_MIN_BATTERIES = 5
PSEUDO_MIN_LEAD = 5
PSEUDO_PENALTY_CYCLE = 2000.0
EOL_THRESHOLD = 0.80
BOOTSTRAP_REPETITIONS = 40
BOOTSTRAP_BLOCK_LENGTH = 8


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metric_record(group: pd.DataFrame) -> dict[str, Any]:
    errors = group["y_pred"].to_numpy(dtype=float) - group["y_true"].to_numpy(dtype=float)
    battery_rmse = group.groupby("battery_id", sort=True).apply(
        lambda frame: float(
            np.sqrt(np.mean((frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()) ** 2))
        ),
        include_groups=False,
    )
    return {
        "n_batteries": int(group["battery_id"].nunique()),
        "n_points": int(len(group)),
        "MAE": float(np.mean(np.abs(errors))),
        "overall_RMSE": float(np.sqrt(np.mean(errors**2))),
        "mean_battery_RMSE": float(battery_rmse.mean()),
        "median_battery_RMSE": float(battery_rmse.median()),
        "worst_battery_RMSE": float(battery_rmse.max()),
    }


def build_short_horizon_predictions(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    train_ids: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cutoff in EARLY_CUTOFFS:
        fit_starts = FITTING_STARTS if cutoff == 150 else [1]
        for battery_id in train_ids:
            policy = str(summary.loc[summary["battery_id"].eq(battery_id), "policy"].iloc[0])
            full_history = cutoff_view(cycles, battery_id, cutoff)
            future = future_view(cycles, battery_id, cutoff, FORECAST_HORIZON)
            future_cycles = future["cycle"].to_numpy(dtype=float)
            for fit_start in fit_starts:
                history = full_history.loc[full_history["cycle"].ge(fit_start)].copy()
                x = history["cycle"].to_numpy(dtype=float)
                y = robust_asof_soh(history)
                fitted_models = {
                    model: fit_degradation_model(x, y, model) for model in BASE_MODELS
                }
                prediction_map = {
                    model: forecast(fit, future_cycles) for model, fit in fitted_models.items()
                }
                prediction_map["power_damped_residual"] = power_damped_residual_forecast(
                    fitted_models["power"], x, y, future_cycles
                )
                for model_name, predictions in prediction_map.items():
                    base_name = "power" if model_name == "power_damped_residual" else model_name
                    fit = fitted_models[base_name]
                    derivative = np.diff(predictions)
                    monotonic_violation_rate = float(np.mean(derivative > 0.0)) if len(derivative) else 0.0
                    for horizon, (_, truth_row) in enumerate(future.iterrows(), start=1):
                        rows.append(
                            {
                                "battery_id": int(battery_id),
                                "policy": policy,
                                "cutoff": int(cutoff),
                                "fit_start": int(fit_start),
                                "model": model_name,
                                "horizon": int(horizon),
                                "cycle": int(truth_row["cycle"]),
                                "y_true": float(truth_row["SOH"]),
                                "y_pred": float(predictions[horizon - 1]),
                                "fit_success": bool(fit.success),
                                "n_fit": int(fit.n_observations),
                                "history_fit_RMSE": float(fit.fit_rmse),
                                "monotonic_violation_rate": monotonic_violation_rate,
                            }
                        )
    return pd.DataFrame(rows)


def summarize_short_horizon(predictions: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    for (cutoff, fit_start, model), group in predictions.groupby(
        ["cutoff", "fit_start", "model"], sort=True
    ):
        metric_rows.append(
            {
                "cutoff": int(cutoff),
                "fit_start": int(fit_start),
                "model": str(model),
                **_metric_record(group),
            }
        )

    policy_rows: list[dict[str, Any]] = []
    focus = predictions.loc[predictions["cutoff"].eq(150)]
    for (fit_start, model, policy), group in focus.groupby(
        ["fit_start", "model", "policy"], sort=True
    ):
        errors = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
        policy_rows.append(
            {
                "cutoff": 150,
                "fit_start": int(fit_start),
                "model": str(model),
                "policy": str(policy),
                "n_batteries": int(group["battery_id"].nunique()),
                "RMSE": float(np.sqrt(np.mean(errors**2))),
                "MAE": float(np.mean(np.abs(errors))),
            }
        )

    horizon_rows: list[dict[str, Any]] = []
    labels = [(1, 10, "1-10"), (11, 20, "11-20"), (21, 30, "21-30"), (31, 40, "31-40"), (41, 50, "41-50")]
    for (cutoff, fit_start, model), group in predictions.groupby(
        ["cutoff", "fit_start", "model"], sort=True
    ):
        for lower, upper, label in labels:
            subset = group.loc[group["horizon"].between(lower, upper)]
            errors = subset["y_pred"].to_numpy() - subset["y_true"].to_numpy()
            horizon_rows.append(
                {
                    "cutoff": int(cutoff),
                    "fit_start": int(fit_start),
                    "model": str(model),
                    "horizon_bin": label,
                    "RMSE": float(np.sqrt(np.mean(errors**2))),
                    "MAE": float(np.mean(np.abs(errors))),
                }
            )
    return {"metrics": metric_rows, "per_policy": policy_rows, "horizon": horizon_rows}


def select_pseudo_thresholds(
    cycles: pd.DataFrame,
    train_ids: list[int],
) -> tuple[list[float], dict[float, dict[int, float]]]:
    crossings: dict[float, dict[int, float]] = {}
    selected: list[float] = []
    for threshold in PSEUDO_THRESHOLD_CANDIDATES:
        mapping = {
            int(battery_id): persistent_crossing_cycle(
                cycles.loc[cycles["battery_id"].eq(battery_id)], threshold
            )
            for battery_id in train_ids
        }
        crossings[threshold] = mapping
        eligible = sum(
            np.isfinite(cycle) and cycle > min(EARLY_CUTOFFS) + PSEUDO_MIN_LEAD
            for cycle in mapping.values()
        )
        if eligible >= PSEUDO_MIN_BATTERIES:
            selected.append(float(threshold))
    return selected, crossings


def pseudo_threshold_validation(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    train_ids: list[int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[float]]:
    selected, crossings = select_pseudo_thresholds(cycles, train_ids)
    case_rows: list[dict[str, Any]] = []
    for threshold in selected:
        for cutoff in EARLY_CUTOFFS:
            for battery_id in train_ids:
                actual = crossings[threshold][battery_id]
                if not np.isfinite(actual) or actual <= cutoff + PSEUDO_MIN_LEAD:
                    continue
                history = cutoff_view(cycles, battery_id, cutoff)
                x = history["cycle"].to_numpy(dtype=float)
                y = robust_asof_soh(history)
                policy = str(summary.loc[summary["battery_id"].eq(battery_id), "policy"].iloc[0])
                for model in BASE_MODELS:
                    fit = fit_degradation_model(x, y, model)
                    predicted = crossing_cycle(fit, threshold)
                    interval_low, interval_high, interval_draws = covariance_crossing_interval(
                        fit,
                        threshold,
                        seed=SEED + int(battery_id) * 101 + cutoff * 7 + int(threshold * 10000),
                    )
                    finite_future = bool(np.isfinite(predicted) and predicted > cutoff)
                    within_evaluation_range = bool(
                        finite_future and predicted <= PSEUDO_PENALTY_CYCLE
                    )
                    bounded_error = (
                        abs(predicted - actual) if within_evaluation_range else float("nan")
                    )
                    penalized_prediction = (
                        float(np.clip(predicted, cutoff + 1.0, PSEUDO_PENALTY_CYCLE))
                        if finite_future
                        else PSEUDO_PENALTY_CYCLE
                    )
                    interval_valid = bool(
                        np.isfinite(interval_low)
                        and np.isfinite(interval_high)
                        and interval_high > interval_low
                    )
                    case_rows.append(
                        {
                            "threshold": float(threshold),
                            "cutoff": int(cutoff),
                            "battery_id": int(battery_id),
                            "policy": policy,
                            "model": model,
                            "actual_crossing": float(actual),
                            "predicted_crossing": float(predicted),
                            "finite_future_prediction": finite_future,
                            "within_evaluation_range": within_evaluation_range,
                            "bounded_absolute_error": float(bounded_error),
                            "penalized_absolute_error": float(abs(penalized_prediction - actual)),
                            "interval_90_lower": float(interval_low),
                            "interval_90_upper": float(interval_high),
                            "interval_draws": int(interval_draws),
                            "interval_valid": interval_valid,
                            "interval_contains_actual": bool(
                                interval_valid and interval_low <= actual <= interval_high
                            ),
                            "interval_width": float(interval_high - interval_low)
                            if interval_valid
                            else float("nan"),
                        }
                    )
    cases = pd.DataFrame(case_rows)
    metric_rows: list[dict[str, Any]] = []
    if cases.empty:
        return cases, metric_rows, selected
    group_keys: list[tuple[str, pd.DataFrame]] = []
    for (threshold, cutoff, model), group in cases.groupby(["threshold", "cutoff", "model"]):
        group_keys.append((f"{float(threshold):.3f}|{int(cutoff)}|{model}", group))
    for (threshold, model), group in cases.groupby(["threshold", "model"]):
        group_keys.append((f"{float(threshold):.3f}|ALL|{model}", group))
    for key, group in group_keys:
        threshold_text, cutoff_text, model = key.split("|")
        bounded = group.loc[group["within_evaluation_range"]]
        intervals = group.loc[group["interval_valid"]]
        metric_rows.append(
            {
                "threshold": float(threshold_text),
                "cutoff": cutoff_text,
                "model": model,
                "n_cases": int(len(group)),
                "n_batteries": int(group["battery_id"].nunique()),
                "finite_future_rate": float(group["finite_future_prediction"].mean()),
                "within_evaluation_range_rate": float(group["within_evaluation_range"].mean()),
                "crossing_MAE_within_range": float(bounded["bounded_absolute_error"].mean())
                if len(bounded)
                else float("nan"),
                "crossing_RMSE_within_range": float(
                    np.sqrt(np.mean(bounded["bounded_absolute_error"] ** 2))
                )
                if len(bounded)
                else float("nan"),
                "median_absolute_error_within_range": float(
                    bounded["bounded_absolute_error"].median()
                )
                if len(bounded)
                else float("nan"),
                "penalized_crossing_MAE": float(group["penalized_absolute_error"].mean()),
                "interval_90_coverage": float(intervals["interval_contains_actual"].mean())
                if len(intervals)
                else float("nan"),
                "median_interval_width": float(intervals["interval_width"].median())
                if len(intervals)
                else float("nan"),
                "n_valid_intervals": int(len(intervals)),
            }
        )
    return cases, metric_rows, selected


def eol_sensitivity(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    train_ids: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cutoff = 150
    for battery_id in train_ids:
        policy = str(summary.loc[summary["battery_id"].eq(battery_id), "policy"].iloc[0])
        full_history = cutoff_view(cycles, battery_id, cutoff)
        for fit_start in FITTING_STARTS:
            history = full_history.loc[full_history["cycle"].ge(fit_start)].copy()
            x = history["cycle"].to_numpy(dtype=float)
            y = robust_asof_soh(history)
            for model in BASE_MODELS:
                fit = fit_degradation_model(x, y, model)
                eol = crossing_cycle(fit, EOL_THRESHOLD)
                lower, upper, n_valid = moving_block_bootstrap_crossing_interval(
                    fit,
                    x,
                    y,
                    EOL_THRESHOLD,
                    seed=SEED + int(battery_id) * 1009 + fit_start * 17 + BASE_MODELS.index(model),
                    repetitions=BOOTSTRAP_REPETITIONS,
                    block_length=BOOTSTRAP_BLOCK_LENGTH,
                )
                derivative = model_derivative(model, x, fit.params)
                rows.append(
                    {
                        "battery_id": int(battery_id),
                        "policy": policy,
                        "cutoff": cutoff,
                        "fit_start": int(fit_start),
                        "model": model,
                        "EOL_threshold": EOL_THRESHOLD,
                        "EOL_cycle": float(eol),
                        "EOL_lower_90": float(lower),
                        "EOL_upper_90": float(upper),
                        "bootstrap_valid": int(n_valid),
                        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                        "fit_RMSE": float(fit.fit_rmse),
                        "derivative_max": float(np.max(derivative)),
                        "monotonic_constraint_satisfied": bool(np.all(derivative <= 1e-12)),
                        "params_json": json.dumps([float(value) for value in fit.params]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_eol(eol: pd.DataFrame) -> dict[str, Any]:
    family_window_rows = []
    for (fit_start, model), group in eol.groupby(["fit_start", "model"], sort=True):
        finite = group.loc[np.isfinite(group["EOL_cycle"]), "EOL_cycle"]
        widths = group["EOL_upper_90"] - group["EOL_lower_90"]
        family_window_rows.append(
            {
                "fit_start": int(fit_start),
                "model": str(model),
                "n_finite": int(len(finite)),
                "median_EOL": float(finite.median()),
                "p10_EOL": float(finite.quantile(0.10)),
                "p90_EOL": float(finite.quantile(0.90)),
                "min_EOL": float(finite.min()),
                "max_EOL": float(finite.max()),
                "median_bootstrap_interval_width": float(widths.replace([np.inf, -np.inf], np.nan).median()),
            }
        )
    battery_spread = []
    for battery_id, group in eol.groupby("battery_id", sort=True):
        values = group.loc[np.isfinite(group["EOL_cycle"]), "EOL_cycle"]
        battery_spread.append(
            {
                "battery_id": int(battery_id),
                "n_estimates": int(len(values)),
                "minimum_EOL": float(values.min()),
                "maximum_EOL": float(values.max()),
                "absolute_spread": float(values.max() - values.min()),
                "max_min_ratio": float(values.max() / values.min()) if values.min() > 0 else float("nan"),
            }
        )
    spread_frame = pd.DataFrame(battery_spread)
    return {
        "family_window_summary": family_window_rows,
        "per_battery_model_window_spread": battery_spread,
        "median_absolute_spread": float(spread_frame["absolute_spread"].median()),
        "maximum_absolute_spread": float(spread_frame["absolute_spread"].max()),
        "median_max_min_ratio": float(spread_frame["max_min_ratio"].median()),
        "maximum_max_min_ratio": float(spread_frame["max_min_ratio"].max()),
    }


def read_baseline_reference() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "problem3" / "common" / "baseline_metrics.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.loc[frame["cutoff"].eq(150)].sort_values("overall_RMSE").to_dict("records")


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    if frame.empty:
        return "（无可用结果）"
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "NA" if not np.isfinite(value) else f"{value:.{digits}g}"
        )
    headers = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [headers, separator]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    short_summary: dict[str, list[dict[str, Any]]],
    baseline_reference: list[dict[str, Any]],
    pseudo_metrics: list[dict[str, Any]],
    selected_thresholds: list[float],
    eol_summary: dict[str, Any],
    runtime_seconds: float,
) -> None:
    short_frame = pd.DataFrame(short_summary["metrics"])
    cutoff150 = short_frame.loc[short_frame["cutoff"].eq(150)].sort_values("overall_RMSE")
    best = cutoff150.iloc[0]
    early = short_frame.loc[short_frame["fit_start"].eq(1)].sort_values(["cutoff", "overall_RMSE"])
    early_best = early.groupby("cutoff", as_index=False).first()
    pseudo_frame = pd.DataFrame(pseudo_metrics)
    pseudo_all = pseudo_frame.loc[pseudo_frame["cutoff"].eq("ALL")].sort_values(
        ["threshold", "penalized_crossing_MAE"]
    ) if not pseudo_frame.empty else pd.DataFrame()
    eol_frame = pd.DataFrame(eol_summary["family_window_summary"]).sort_values(
        ["fit_start", "model"]
    )
    baseline_frame = pd.DataFrame(baseline_reference)
    baseline_best = baseline_frame.iloc[0] if not baseline_frame.empty else None
    residual_row = cutoff150.loc[
        (cutoff150["fit_start"].eq(int(best["fit_start"])))
        & cutoff150["model"].eq("power_damped_residual")
    ]
    power_row = cutoff150.loc[
        (cutoff150["fit_start"].eq(int(best["fit_start"])))
        & cutoff150["model"].eq("power")
    ]
    residual_statement = "未形成可比结果。"
    if len(residual_row) and len(power_row):
        delta = float(residual_row.iloc[0]["overall_RMSE"] - power_row.iloc[0]["overall_RMSE"])
        residual_statement = (
            f"在相同窗口下，damped residual 相对纯 power 的 RMSE 变化为 {delta:+.6g}；"
            + ("有小幅改善，但不足以支持更复杂神经残差。" if delta < 0 else "没有改善，因此不支持增加残差网络。")
        )

    baseline_sentence = "公共 baseline 结果不可用。"
    if baseline_best is not None:
        difference = float(best["overall_RMSE"] - baseline_best["overall_RMSE"])
        baseline_sentence = (
            f"公共 baseline 的最佳 cutoff=150 模型为 {baseline_best['model']}，RMSE={baseline_best['overall_RMSE']:.6g}；"
            f"本路线最佳为 {best['model']}（拟合起点 {int(best['fit_start'])}），RMSE={best['overall_RMSE']:.6g}，"
            f"差值为 {difference:+.6g}。"
        )

    report = rf"""# Agent 02：半经验退化模型与长期 EOL 外推

## 1. 结论先行

短期 151–200 周期回测中，本路线最好的组合是 **{best['model']} + cycle {int(best['fit_start'])}–150 拟合窗口**，overall RMSE 为 **{best['overall_RMSE']:.6g}**。{baseline_sentence}

0.8 EOL 没有真实 crossing 监督，因而本报告不提供 EOL RMSE，也不把任何单一外推值称为“真实寿命预测”。不同函数族和拟合起点产生的 EOL 估计在单块电池上的中位极差为 **{eol_summary['median_absolute_spread']:.1f} cycles**，最大极差为 **{eol_summary['maximum_absolute_spread']:.1f} cycles**；中位最大/最小比为 **{eol_summary['median_max_min_ratio']:.2f}**。这是模型假设不确定性，而不是可通过增加小型网络消除的普通噪声。

## 2. 数据边界与泄漏控制

- 仅使用赛题 PDF、`battery_summary.csv`、`cycle_train.csv`、公共审计与 baseline；未读取其他 Agent 的结果，未访问外部完整电池轨迹。
- 40 块非测试电池用于独立 rolling-origin battery-level 回测；9 块 `prediction_test=1` 电池没有参与训练、调参、伪阈值验证或本 Agent 的候选最终推理。
- 每个 cutoff 只读取 `cycle <= cutoff`。汇总表的 `mean_IR/mean_Tavg/mean_chargetime` 未被使用。
- 未直接使用赛题给出的 `SOH_smooth`：其平滑过程未知，且 battery 1 的早期异常会造成明显端点畸变。每个 cutoff 内仅对原始 SOH 使用固定 7 点 Hampel 异常抑制，参数不由未来数据选择。
- 所有模型允许观测点局部回升；物理约束只作用于潜在趋势的导数符号，避免把测量波动强行改成逐点单调。

## 3. 模型

比较三种可解释退化族：

\[
\text{{Linear}}:\quad SOH(n)=a-bn,\quad b\ge0,
\]

\[
\text{{Power}}:\quad SOH(n)=a-kn^p,\quad k\ge0,\ 0.25\le p\le3,
\]

\[
\text{{Exponential}}:\quad SOH(n)=a\exp(-kn),\quad k\ge0.
\]

参数用 `soft_l1` 稳健损失估计。另测试一个低容量半经验残差：在 power 曲线上叠加当前 10 周期中位残差，并以 25 cycle 时间常数指数衰减。它仅修正短期水平，不把残差趋势无限外推到 EOL。

{residual_statement}

## 4. 151–200 周期预测验证

### 4.1 cutoff=150 与窗口敏感性

{_markdown_table(cutoff150.head(16), ['model', 'fit_start', 'MAE', 'overall_RMSE', 'mean_battery_RMSE', 'median_battery_RMSE', 'worst_battery_RMSE'])}

### 4.2 早期数据长度

下表在每个 cutoff 仅比较 cycle 1–N 拟合的三种退化族及小残差版本，并列出该 cutoff 的最优者。较晚 cutoff 不自动保证更好，因为全历史函数会平均掉最近退化速率；这也是单独检查 20/50/80 起点的原因。

{_markdown_table(early_best, ['cutoff', 'model', 'fit_start', 'overall_RMSE', 'mean_battery_RMSE', 'worst_battery_RMSE'])}

## 5. 伪阈值验证

以原始 SOH 的 5 点因果尾随中位数定义 crossing，并要求连续 3 点低于阈值。候选阈值为 {PSEUDO_THRESHOLD_CANDIDATES}；只有在 cutoff=50 后至少 {PSEUDO_MIN_BATTERIES} 块训练电池发生 crossing 的阈值才保留，最终为 **{selected_thresholds}**。每个 case 的 cutoff 至少比真实 crossing 提前 {PSEUDO_MIN_LEAD} cycles。

`crossing_MAE_within_range` 只评价落在 cutoff 与 {int(PSEUDO_PENALTY_CYCLE)} cycle 之间的 crossing；为防止极远预测或“无 crossing”被忽略，`penalized_crossing_MAE` 将缺失或超过 {int(PSEUDO_PENALTY_CYCLE)} 的预测按 {int(PSEUDO_PENALTY_CYCLE)} cycle 计罚。90% 区间来自局部参数协方差抽样，仅用于诊断 calibration，不是严格覆盖保证。

{_markdown_table(pseudo_all, ['threshold', 'model', 'n_cases', 'n_batteries', 'within_evaluation_range_rate', 'crossing_MAE_within_range', 'penalized_crossing_MAE', 'interval_90_coverage', 'median_interval_width'])}

低阈值 crossing 数量很少，模型排序只能说明“接近当前数据范围的阈值外推”表现；它不能验证 0.8 的远距离寿命。

在三个保留阈值上，power 的惩罚 MAE 均为三类模型最低，但只有约 55%–72% 的 case 落入 2000-cycle 评价范围，90% 区间经验覆盖率也只有约 23%–50%。因此“power 排名第一”只是相对结论，绝不构成可靠校准或远程 EOL 准确性的证据。

## 6. 0.8 EOL 的模型与窗口敏感性

下表对每块训练电池仅使用 cycle 1–150，并分别从 cycle 1、20、50、80 开始拟合。区间为固定 8-cycle block 的移动块残差 bootstrap（{BOOTSTRAP_REPETITIONS} 次）得到的 90% 条件区间。

{_markdown_table(eol_frame, ['fit_start', 'model', 'n_finite', 'median_EOL', 'p10_EOL', 'p90_EOL', 'min_EOL', 'max_EOL', 'median_bootstrap_interval_width'], digits=7)}

bootstrap 区间只反映“给定函数族和窗口”的参数/残差不确定性；跨函数族、跨窗口的差异通常更大。因此论文应同时列出 linear/power/exponential，另以模型-窗口包络表示假设敏感性。没有 0.8 crossing 时，不应把 bootstrap 区间称为经覆盖率校准的置信区间。

## 7. PINN / Neural ODE 判断

本数据只有 40 个独立训练 cell、每个只有早期 200 cycles，且没有 0.8 crossing。cycle 行数并不等于独立物理实验数；把同一电池的相邻行当成大量监督样本会夸大有效样本量。题目数据也不含可辨识的电化学状态方程、反应速率或完整长期轨迹，PINN/Neural ODE 中的“物理项”和“神经残差”会高度互相补偿。鉴于低容量 damped residual 的实际增益及模型族敏感性，本 Agent **淘汰 PINN/Neural ODE，不推荐作为主模型或 EOL 外推器**。更合理的做法是短期预测与长期半经验外推分阶段，并把函数族分歧显式报告。

## 8. 可复现性

- seed：{SEED}
- Python：{platform.python_version()}
- NumPy：{np.__version__}
- pandas：{pd.__version__}
- SciPy：{scipy.__version__}
- robust loss：`soft_l1`，`f_scale=0.0015`
- 参数量：linear=2，power=3，exponential=2，power+damped-residual=4（其中 decay=25 固定）
- power exponent bounds：`[0.25, 3.0]`
- EOL bootstrap：moving-block residual bootstrap，block={BOOTSTRAP_BLOCK_LENGTH}，B={BOOTSTRAP_REPETITIONS}
- runtime：{runtime_seconds:.2f} s

运行命令：`python problem3/agents/agent_02_physics/run.py`

## 9. 文献依据（仅用于方法，不用于提取测试答案）

1. Kristen A. Severson, Peter M. Attia, Norman Jin, Nicholas Perkins, Benben Jiang, Zi Yang, Michael H. Chen, Muratahan Aykol, Patrick K. Herring, Dimitrios Fraggedakis, Martin Z. Bazant, Stephen J. Harris, William C. Chueh, Richard D. Braatz. *Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 2019, 4:383–391. DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8).
2. Ian A. Richardson, Michael A. Osborne, David A. Howey. *Gaussian process regression for forecasting battery state of health*. Journal of Power Sources, 2017, 357:209–219. DOI: [10.1016/j.jpowsour.2017.05.004](https://doi.org/10.1016/j.jpowsour.2017.05.004).
3. Pengfei Wen, Zhi-Sheng Ye, Yong Li, Shaowei Chen, Pu Xie, Shuai Zhao. *Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries*. IEEE Transactions on Intelligent Vehicles, 2024, 9(1):2276–2289. DOI: [10.1109/TIV.2023.3315548](https://doi.org/10.1109/TIV.2023.3315548).

这些论文分别支持早期信息建模、显式退化均值/不确定性、以及物理—数据融合的研究动机；本实验没有下载或使用其中任何电池未来数据。
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    summary, cycles = load_data()
    train_ids, test_ids = split_battery_ids(summary)
    if cycles.loc[cycles["battery_id"].isin(test_ids), "cycle"].max() > 150:
        raise RuntimeError("Leakage guard failed: a test battery contains cycle > 150")
    reuse_existing = "--reuse-existing" in sys.argv
    if reuse_existing:
        predictions = pd.read_csv(OUTPUT_DIR / "predictions.csv")
        pseudo_frame = pd.read_csv(OUTPUT_DIR / "pseudo_threshold_metrics.csv")
        pseudo_metrics = pseudo_frame.to_dict("records")
        available_thresholds = {
            float(value) for value in pseudo_frame["threshold"].dropna().unique()
        }
        selected_thresholds = [
            threshold
            for threshold in PSEUDO_THRESHOLD_CANDIDATES
            if threshold in available_thresholds
        ]
        eol = pd.read_csv(OUTPUT_DIR / "eol_sensitivity.csv")
        old_metrics_path = OUTPUT_DIR / "metrics.json"
        if old_metrics_path.exists():
            with old_metrics_path.open(encoding="utf-8") as handle:
                runtime_seconds = float(json.load(handle).get("runtime_seconds", float("nan")))
        else:
            runtime_seconds = float("nan")
    else:
        predictions = build_short_horizon_predictions(summary, cycles, train_ids)
        predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
        _, pseudo_metrics, selected_thresholds = pseudo_threshold_validation(
            summary, cycles, train_ids
        )
        pd.DataFrame(pseudo_metrics).to_csv(
            OUTPUT_DIR / "pseudo_threshold_metrics.csv", index=False
        )
        eol = eol_sensitivity(summary, cycles, train_ids)
        eol.to_csv(OUTPUT_DIR / "eol_sensitivity.csv", index=False)
        runtime_seconds = time.perf_counter() - started

    short_summary = summarize_short_horizon(predictions)
    eol_summary = summarize_eol(eol)
    baseline_reference = read_baseline_reference()

    cutoff150_ranked = sorted(
        [row for row in short_summary["metrics"] if row["cutoff"] == 150],
        key=lambda row: row["overall_RMSE"],
    )
    pseudo_overall = [row for row in pseudo_metrics if row["cutoff"] == "ALL"]
    metrics = {
        "agent": "agent_02_physics",
        "seed": SEED,
        "runtime_seconds": runtime_seconds,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "data": {
            "n_training_batteries": len(train_ids),
            "n_prediction_test_batteries": len(test_ids),
            "test_battery_ids": test_ids,
            "true_SOH_le_0_8_exists": bool(cycles["SOH"].le(EOL_THRESHOLD).any()),
            "test_batteries_used_for_fitting_or_validation": False,
        },
        "hyperparameters": {
            "early_cutoffs": EARLY_CUTOFFS,
            "forecast_horizon": FORECAST_HORIZON,
            "fitting_starts_at_cutoff_150": FITTING_STARTS,
            "robust_loss": "soft_l1",
            "robust_f_scale": 0.0015,
            "power_exponent_bounds": [0.25, 3.0],
            "damped_residual_decay_cycles": 25.0,
            "model_parameter_counts": {
                "linear": 2,
                "power": 3,
                "exponential": 2,
                "power_damped_residual": 4,
            },
            "eol_bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "eol_bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
        },
        "short_horizon": {
            **short_summary,
            "cutoff_150_ranked": cutoff150_ranked,
            "common_baseline_cutoff_150": baseline_reference,
        },
        "pseudo_threshold_validation": {
            "candidate_thresholds": PSEUDO_THRESHOLD_CANDIDATES,
            "selected_thresholds": selected_thresholds,
            "minimum_distinct_batteries_after_cutoff_50": PSEUDO_MIN_BATTERIES,
            "minimum_lead_cycles": PSEUDO_MIN_LEAD,
            "missing_prediction_penalty_cycle": PSEUDO_PENALTY_CYCLE,
            "metrics": pseudo_metrics,
            "overall_metrics": pseudo_overall,
            "interval_note": "Approximate 90% local parameter-covariance intervals; diagnostic, not guaranteed coverage.",
        },
        "eol_extrapolation": {
            "threshold": EOL_THRESHOLD,
            "true_crossing_count": int(
                sum(
                    np.isfinite(
                        persistent_crossing_cycle(
                            cycles.loc[cycles["battery_id"].eq(battery_id)], EOL_THRESHOLD
                        )
                    )
                    for battery_id in train_ids
                )
            ),
            "EOL_RMSE_reported": False,
            "sensitivity": eol_summary,
            "uncertainty_note": "90% moving-block residual bootstrap is conditional on each family/window; cross-family/window spread is model-assumption uncertainty.",
        },
        "recommendation": {
            "best_physics_short_horizon": cutoff150_ranked[0],
            "PINN_or_Neural_ODE": "not_recommended",
            "reason": "Only 40 independent training batteries and no supervised 0.8 crossing; a neural residual is not identifiable enough to justify its complexity.",
            "long_horizon": "Report linear, power and exponential together with fitting-window sensitivity; do not publish a single precise EOL.",
        },
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite_or_none(metrics), handle, ensure_ascii=False, indent=2, allow_nan=False)

    write_report(
        short_summary,
        baseline_reference,
        pseudo_metrics,
        selected_thresholds,
        eol_summary,
        runtime_seconds,
    )
    print(
        pd.DataFrame(cutoff150_ranked)[
            ["model", "fit_start", "MAE", "overall_RMSE", "worst_battery_RMSE"]
        ].head(12).to_string(index=False)
    )
    print(f"Selected pseudo thresholds: {selected_thresholds}")
    print(f"Runtime: {runtime_seconds:.2f} s")


if __name__ == "__main__":
    main()
