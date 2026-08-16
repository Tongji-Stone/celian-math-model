from __future__ import annotations

import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problem3.agents.agent_05_intercell.model import (  # noqa: E402
    CONFIG_GRIDS,
    build_cutoff_data,
    predict_cohort,
    select_config_nested,
)
from problem3.common.data import load_data, split_battery_ids  # noqa: E402
from problem3.common.metrics import detailed_metric_tables, mae, rmse  # noqa: E402
from problem3.common.validation import EARLY_CUTOFFS, FORECAST_HORIZON, SEED  # noqa: E402


MODEL_COLUMNS = {
    "Public Linear-Nested": "pred_linear_nested",
    "Exact-policy mean delta": "pred_exact_policy",
    "Trajectory similarity": "pred_trajectory",
    "Slope-adapted cohort": "pred_slope",
    "Multivariate similarity": "pred_multivariate",
    "Soft-policy neighbors": "pred_soft_policy",
}

METHOD_TO_COLUMN = {
    "trajectory": "pred_trajectory",
    "slope": "pred_slope",
    "multivariate": "pred_multivariate",
    "soft_policy": "pred_soft_policy",
}


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _run_nested_backtest(summary: pd.DataFrame, cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids, test_ids = split_battery_ids(summary)
    if set(train_ids).intersection(test_ids):
        raise AssertionError("Train/test battery sets overlap")
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for cutoff in EARLY_CUTOFFS:
        data = build_cutoff_data(cycles, summary, train_ids, cutoff, FORECAST_HORIZON)
        all_indices = np.arange(data.n_batteries, dtype=int)
        for outer_target in all_indices:
            reference_indices = all_indices[all_indices != outer_target]
            exact_prediction, exact_diagnostics = predict_cohort(
                data, int(outer_target), reference_indices, "exact_policy"
            )
            predictions = {"pred_exact_policy": exact_prediction}
            config_text: dict[str, str] = {}
            diagnostic_by_method: dict[str, dict[str, float | int]] = {
                "exact_policy": exact_diagnostics
            }

            for method, configs in CONFIG_GRIDS.items():
                selected, inner = select_config_nested(
                    data, int(outer_target), method, configs
                )
                prediction, diagnostics = predict_cohort(
                    data, int(outer_target), reference_indices, method, selected
                )
                predictions[METHOD_TO_COLUMN[method]] = prediction
                config_text[method] = _compact_json(selected)
                diagnostic_by_method[method] = diagnostics
                selected_rows.append(
                    {
                        "cutoff": int(cutoff),
                        "battery_id": int(data.battery_ids[outer_target]),
                        "policy": str(data.policies[outer_target]),
                        "method": method,
                        "selected_config": config_text[method],
                        "inner_RMSE": float(inner["inner_RMSE"]),
                        "candidate_count": int(inner["candidate_count"]),
                        **diagnostics,
                    }
                )

            for horizon in range(1, FORECAST_HORIZON + 1):
                row: dict[str, Any] = {
                    "battery_id": int(data.battery_ids[outer_target]),
                    "policy": str(data.policies[outer_target]),
                    "cutoff": int(cutoff),
                    "horizon": int(horizon),
                    "cycle": int(cutoff + horizon),
                    "y_true": float(data.future_soh[outer_target, horizon - 1]),
                    "exact_peer_count": int(exact_diagnostics["exact_peer_count"]),
                    "reference_count": int(exact_diagnostics["reference_count"]),
                }
                for column, prediction in predictions.items():
                    row[column] = float(prediction[horizon - 1])
                for method in CONFIG_GRIDS:
                    row[f"{method}_config"] = config_text[method]
                    row[f"{method}_effective_peer_count"] = float(
                        diagnostic_by_method[method]["effective_peer_count"]
                    )
                rows.append(row)

    predictions = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    return predictions, selected


def _attach_public_baseline(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline_path = PROJECT_ROOT / "problem3" / "common" / "baseline_predictions.csv"
    baseline = pd.read_csv(baseline_path)
    keys = ["battery_id", "cutoff", "horizon", "cycle"]
    keep = keys + ["y_true", "pred_linear_nested", "pred_same_policy", "nested_best_window"]
    merged = predictions.merge(
        baseline[keep], on=keys, how="left", suffixes=("", "_baseline"), validate="one_to_one"
    )
    if merged["pred_linear_nested"].isna().any():
        raise AssertionError("Public baseline merge is incomplete")
    truth_difference = float(np.max(np.abs(merged["y_true"] - merged["y_true_baseline"])))
    cohort_difference = float(
        np.max(np.abs(merged["pred_exact_policy"] - merged["pred_same_policy"]))
    )
    if truth_difference > 1e-12 or cohort_difference > 1e-12:
        raise AssertionError(
            f"Public baseline reconciliation failed: truth={truth_difference}, cohort={cohort_difference}"
        )
    return merged.drop(columns=["y_true_baseline", "pred_same_policy"])


def _cohort_size_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cutoff, peer_count), group in predictions.groupby(["cutoff", "exact_peer_count"]):
        for model, column in MODEL_COLUMNS.items():
            per_battery = group.groupby("battery_id").apply(
                lambda frame: rmse(frame["y_true"].to_numpy(), frame[column].to_numpy()),
                include_groups=False,
            )
            rows.append(
                {
                    "cutoff": int(cutoff),
                    "model": model,
                    "exact_peer_count": int(peer_count),
                    "n_batteries": int(group["battery_id"].nunique()),
                    "n_points": int(len(group)),
                    "MAE": mae(group["y_true"].to_numpy(), group[column].to_numpy()),
                    "RMSE": rmse(group["y_true"].to_numpy(), group[column].to_numpy()),
                    "mean_battery_RMSE": float(per_battery.mean()),
                }
            )
    return pd.DataFrame(rows)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _configuration_frequencies(selected: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (cutoff, method), group in selected.groupby(["cutoff", "method"]):
        for config, count in Counter(group["selected_config"]).most_common():
            rows.append(
                {
                    "cutoff": int(cutoff),
                    "method": str(method),
                    "selected_config": str(config),
                    "outer_fold_count": int(count),
                }
            )
    return rows


def _markdown_table(frame: pd.DataFrame, digits: int = 7) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else value
        )
    return display.to_markdown(index=False)


def _write_report(
    overall: pd.DataFrame,
    policy: pd.DataFrame,
    horizon: pd.DataFrame,
    cohort_size: pd.DataFrame,
    selected: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    cutoff_150 = overall.loc[overall["cutoff"].eq(150)].sort_values("overall_RMSE")
    cohort_150 = cutoff_150.loc[~cutoff_150["model"].eq("Public Linear-Nested")]
    best_row = cohort_150.iloc[0]
    best_model = str(best_row["model"])
    linear_row = cutoff_150.loc[cutoff_150["model"].eq("Public Linear-Nested")].iloc[0]
    exact_row = cutoff_150.loc[cutoff_150["model"].eq("Exact-policy mean delta")].iloc[0]
    relative_vs_linear = 100.0 * (
        float(linear_row["overall_RMSE"]) - float(best_row["overall_RMSE"])
    ) / float(linear_row["overall_RMSE"])

    best_by_cutoff = (
        overall.loc[~overall["model"].eq("Public Linear-Nested")]
        .sort_values(["cutoff", "overall_RMSE"])
        .groupby("cutoff", as_index=False)
        .first()
    )
    cutoff_comparison = overall.loc[
        overall["model"].isin([best_model, "Public Linear-Nested", "Exact-policy mean delta"]),
        ["cutoff", "model", "MAE", "overall_RMSE", "mean_battery_RMSE", "worst_battery_RMSE"],
    ].sort_values(["cutoff", "overall_RMSE"])
    best_policy = policy.loc[
        policy["cutoff"].eq(150) & policy["model"].eq(best_model),
        ["policy", "n_batteries", "mean_exact_peer_count", "RMSE", "MAE"],
    ].sort_values("RMSE")
    best_horizon = horizon.loc[
        horizon["cutoff"].eq(150) & horizon["model"].eq(best_model),
        ["horizon_bin", "RMSE", "MAE"],
    ]
    best_cohort = cohort_size.loc[
        cohort_size["cutoff"].eq(150) & cohort_size["model"].eq(best_model),
        ["exact_peer_count", "n_batteries", "RMSE", "mean_battery_RMSE"],
    ].sort_values("exact_peer_count")

    selected_summary_rows = []
    for method, group in selected.loc[selected["cutoff"].eq(150)].groupby("method"):
        config, count = Counter(group["selected_config"]).most_common(1)[0]
        selected_summary_rows.append(
            {"method": method, "most_common_config": config, "outer_fold_count": count}
        )
    selected_summary = pd.DataFrame(selected_summary_rows)

    trajectory_row = cutoff_150.loc[cutoff_150["model"].eq("Trajectory similarity")].iloc[0]
    slope_row = cutoff_150.loc[cutoff_150["model"].eq("Slope-adapted cohort")].iloc[0]
    multi_row = cutoff_150.loc[cutoff_150["model"].eq("Multivariate similarity")].iloc[0]
    soft_row = cutoff_150.loc[cutoff_150["model"].eq("Soft-policy neighbors")].iloc[0]
    soft_vs_exact = 100.0 * (
        float(exact_row["overall_RMSE"]) - float(soft_row["overall_RMSE"])
    ) / float(exact_row["overall_RMSE"])

    recommendation = (
        "可作为候选主模型进一步进入统一 Judge。"
        if relative_vs_linear > 0
        else "不推荐单独作为主模型；公共局部线性基线更准确，cohort 输出更适合作为集成专家。"
    )

    report = rf"""# Agent 05：Inter-cell / Same-policy Cohort Transfer

## 摘要

本 Agent 将目标电池在 cutoff 之后的变化写成其他电池未来变化的加权迁移，并在 40 块非测试电池上完成 outer Leave-One-Battery-Out、inner Leave-One-Battery-Out 的严格嵌套回测。所有目标电池未来值只用于最外层评分，从未进入 reference pool 或超参数选择。

在核心 `150 -> 151--200` 回测中，cohort 系列最低 overall RMSE 的方案为 **{best_model}**，RMSE={float(best_row['overall_RMSE']):.7f}，MAE={float(best_row['MAE']):.7f}，最差单电池 RMSE={float(best_row['worst_battery_RMSE']):.7f}。相对公共 nested recent-linear 的 RMSE 变化为 {relative_vs_linear:+.2f}%（正值表示降低误差，负值表示更差）。结论：**{recommendation}**

## 1 模型

对 cutoff 为 $N$ 的目标电池 $i$ 与参考电池 $j$，定义

$$D_j(h)=SOH_j(N+h)-SOH_j(N),\qquad h=1,\ldots,50,$$

并预测

$$\widehat{{SOH}}_i(N+h)=SOH_i(N)+\sum_j w_{{ij}}D_j(h),\qquad w_{{ij}}\ge 0,\ \sum_jw_{{ij}}=1.$$

比较的 cohort 路线为：

1. **Exact-policy mean delta**：仅使用同策略参考电池并等权平均；公共 baseline 与本实现逐预测点一致。
2. **Trajectory similarity**：对最后 $L$ 个 `SOH_smooth` 相对轨迹计算 RMSE，并用高斯核加权。
3. **Slope-adapted cohort**：在轨迹核权重上，用平滑 SOH 的近期退化速率比缩放参考增量，并将比例截断；速率下限仅用于避免近零斜率除法不稳定。
4. **Multivariate similarity**：SOH 相对轨迹与 IR、Tavg、chargetime 的标准化距离共同决定权重；缩放参数只由当前 reference pool 的 `cycle <= N` 数据计算。
5. **Soft-policy neighbors**：允许所有训练策略参与，权重同时惩罚 SOH 轨迹距离与标准化 $(C_1,Q_1,C_2)$ 策略距离；结构性缺失的单阶段策略使用 $C_1^{{effective}}=C_2$，没有总体均值填补。

## 2 无泄漏验证协议

- 9 块 `prediction_test == 1` 电池完全未进入本 Agent 的训练、选参和回测。
- 外层对 40 块训练电池逐块留一；其余 39 块是唯一可用 reference pool，目标自身的 future 永不进入 pool。
- 对每个 outer fold、每个 cutoff、每个改进方法，在 39 块 outer-training 电池上再做 inner LOO；$L$、核尺度、斜率 clip、多变量距离权重和策略核尺度都由 inner RMSE 决定。
- inner fold 如果因 outer 留一导致某策略没有同策略同伴，使用当时仍可用的其他 inner reference，且仍不读取 inner target future 作为 reference。
- cutoff 分别为 50、75、100、125、150；每次只从 `cycle <= cutoff` 构造相似度与标准化量，预测后 50 个 cycle。
- 预测行与公共 baseline 的真实值、same-policy mean delta 均逐点对账，最大差异不超过 $10^{{-12}}$。

## 3 核心回测结果

### 3.1 cutoff=150 全模型

{_markdown_table(cutoff_150[['model','MAE','overall_RMSE','mean_battery_RMSE','median_battery_RMSE','worst_battery_RMSE']])}

### 3.2 early cutoff 比较

{_markdown_table(cutoff_comparison)}

各 cutoff 的 cohort 系列最优候选如下；这是跨 outer folds 的回测总结，不用于对某块 outer battery 看未来后择优：

{_markdown_table(best_by_cutoff[['cutoff','model','overall_RMSE','worst_battery_RMSE']])}

### 3.3 cutoff=150 的 horizon 分段

{_markdown_table(best_horizon)}

### 3.4 cutoff=150 的分策略表现

{_markdown_table(best_policy)}

### 3.5 同伴数与误差

{_markdown_table(best_cohort)}

同伴数与误差并非随机分配：peer count 由 policy 样本量决定，因此该表只能描述鲁棒性，不能解释为同伴数量的因果效应。

## 4 消融与解释

- Exact-policy RMSE={float(exact_row['overall_RMSE']):.7f}；SOH 轨迹相似度 RMSE={float(trajectory_row['overall_RMSE']):.7f}；斜率自适应 RMSE={float(slope_row['overall_RMSE']):.7f}。
- 加入 IR、温度与充电时间后的 multivariate RMSE={float(multi_row['overall_RMSE']):.7f}。这只是预测增益消融，不应被解释为这些量对寿命的因果效应。
- soft-policy RMSE={float(soft_row['overall_RMSE']):.7f}，相对 exact-policy 变化 {soft_vs_exact:+.2f}%（正值表示误差降低）。若该值为负，说明在当前样本中跨策略借用引入的偏差大于扩充同伴的收益。
- cutoff=150 最常见的内层选参结果如下；每个 outer fold 都独立重选，因此不存在用 outer target 的 151--200 误差选择 $L$/核尺度/clip 的行为。

{_markdown_table(selected_summary, digits=4)}

## 5 适用性与局限

1. 小策略仅有 2 块训练电池，outer fold 中 exact-policy 实际只有 1 个同伴，相似度权重无法在同策略内部发挥作用；这会导致分策略误差高度离散。
2. cohort 方法迁移的是 50-cycle 真实变化模板，只适合本题的短期预测；它没有提供可信的 $SOH=0.8$ 长期外推机制，不报告 EOL。
3. 未给出形式化预测区间。不同同伴的分歧可作为后续集成的不确定性信号，但当前样本量不足以把它直接称为校准置信区间。
4. 策略参数是离散实验组合，soft-policy 的参数距离不等价于物理退化距离，且 `policy` 与 $(C_1,Q_1,C_2)$ 共线。
5. 本模型是非参数加权器，训练参数量为 0；主要复杂度来自 nested LOO。运行时间为 {runtime_seconds:.2f} 秒，随机种子记录为 {SEED}（算法本身确定性）。

## 6 文献依据

- Han Zhang, Yuqi Li, Shun Zheng, Ziheng Lu, Xiaofan Gui, Wei Xu, Jiang Bian. *Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning*. **Nature Machine Intelligence**, 7, 270--277, 2025. DOI: [10.1038/s42256-024-00972-x](https://doi.org/10.1038/s42256-024-00972-x). 该文把未知寿命电池视为 target、完整电池视为 reference，并学习 cell 间差异；本 Agent 采用不需要神经网络的核加权短期差分迁移作为小样本对应物。
- Kristen A. Severson, Peter M. Attia, Norman Jin, Nicholas Perkins, Benben Jiang, Zi Yang, Michael H. Chen, Muratahan Aykol, Patrick K. Herring, Dimitrios Fraggedakis, Martin Z. Bazant, Stephen J. Harris, William C. Chueh, Richard D. Braatz. *Data-driven prediction of battery cycle life before capacity degradation*. **Nature Energy**, 4, 383--391, 2019. DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8). 该文支持从早期循环提取退化差异信息；本报告仅借鉴方法，不使用其公开数据补全本题测试电池未来。

## 7 可复现产物

- `model.py`：cohort 模型、距离、权重与 nested selection。
- `run.py`：全量回测、公共 baseline 对账、指标与报告生成。
- `predictions.csv`：10,000 个 outer-CV 预测点及逐折配置。
- `ablation.csv`、`policy_metrics.csv`、`horizon_metrics.csv`、`cohort_size_metrics.csv`：总体、策略、horizon 与同伴数指标。
- `selected_hyperparameters.csv`、`metrics.json`：逐折选参与完整机器可读摘要。
"""
    (AGENT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    np.random.seed(SEED)
    summary, cycles = load_data()
    train_ids, test_ids = split_battery_ids(summary)

    test_cycles = cycles.loc[cycles["battery_id"].isin(test_ids), "cycle"]
    if int(test_cycles.max()) != 150:
        raise AssertionError("Test data boundary changed; refusing to continue")
    if any(int(cycles.loc[cycles["battery_id"].eq(battery_id), "cycle"].max()) < 200 for battery_id in train_ids):
        raise AssertionError("A training battery lacks the required 200 cycles")

    predictions, selected = _run_nested_backtest(summary, cycles)
    predictions = _attach_public_baseline(predictions)

    overall, policy, horizon = detailed_metric_tables(predictions, MODEL_COLUMNS)
    peer_counts = (
        predictions.groupby(["cutoff", "policy"], as_index=False)["exact_peer_count"]
        .mean()
        .rename(columns={"exact_peer_count": "mean_exact_peer_count"})
    )
    policy = policy.merge(peer_counts, on=["cutoff", "policy"], how="left", validate="many_to_one")
    cohort_size = _cohort_size_metrics(predictions)
    runtime_seconds = float(time.perf_counter() - started)

    predictions.to_csv(AGENT_DIR / "predictions.csv", index=False)
    overall.to_csv(AGENT_DIR / "ablation.csv", index=False)
    policy.to_csv(AGENT_DIR / "policy_metrics.csv", index=False)
    horizon.to_csv(AGENT_DIR / "horizon_metrics.csv", index=False)
    cohort_size.to_csv(AGENT_DIR / "cohort_size_metrics.csv", index=False)
    selected.to_csv(AGENT_DIR / "selected_hyperparameters.csv", index=False)

    cutoff_150 = overall.loc[overall["cutoff"].eq(150)].sort_values("overall_RMSE")
    best_overall = cutoff_150.iloc[0]
    best = cutoff_150.loc[~cutoff_150["model"].eq("Public Linear-Nested")].iloc[0]
    linear = cutoff_150.loc[cutoff_150["model"].eq("Public Linear-Nested")].iloc[0]
    payload = {
        "agent": "agent_05_intercell",
        "seed": SEED,
        "runtime_seconds": runtime_seconds,
        "parameter_count": 0,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "data": {
            "training_batteries": len(train_ids),
            "excluded_prediction_test_batteries": len(test_ids),
            "cutoffs": EARLY_CUTOFFS,
            "forecast_horizon": FORECAST_HORIZON,
            "prediction_rows": int(len(predictions)),
        },
        "validation": {
            "outer": "Leave-One-Battery-Out over 40 non-test batteries",
            "inner": "Leave-One-Battery-Out over each outer training pool",
            "selection_metric": "point-weighted inner RMSE over horizons 1-50",
            "cutoff_features": "cycle <= cutoff only",
            "target_future_in_reference_pool": False,
            "prediction_test_used": False,
            "public_same_policy_reconciliation_max_abs_diff": 0.0,
        },
        "hyperparameter_grids": CONFIG_GRIDS,
        "overall_metrics": _json_records(overall),
        "best_cutoff_150": {
            **json.loads(best.to_json()),
            "RMSE_reduction_vs_public_linear_percent": 100.0
            * (float(linear["overall_RMSE"]) - float(best["overall_RMSE"]))
            / float(linear["overall_RMSE"]),
        },
        "best_overall_including_public_baseline_cutoff_150": json.loads(best_overall.to_json()),
        "configuration_frequencies": _configuration_frequencies(selected),
        "artifacts": {
            "predictions": "predictions.csv",
            "ablation": "ablation.csv",
            "policy_metrics": "policy_metrics.csv",
            "horizon_metrics": "horizon_metrics.csv",
            "cohort_size_metrics": "cohort_size_metrics.csv",
            "selected_hyperparameters": "selected_hyperparameters.csv",
            "report": "report.md",
        },
        "literature": [
            {
                "title": "Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning",
                "authors": ["Han Zhang", "Yuqi Li", "Shun Zheng", "Ziheng Lu", "Xiaofan Gui", "Wei Xu", "Jiang Bian"],
                "journal": "Nature Machine Intelligence",
                "year": 2025,
                "doi": "10.1038/s42256-024-00972-x",
            },
            {
                "title": "Data-driven prediction of battery cycle life before capacity degradation",
                "authors": ["Kristen A. Severson", "Peter M. Attia", "Norman Jin", "Nicholas Perkins", "Benben Jiang", "Zi Yang", "Michael H. Chen", "Muratahan Aykol", "Patrick K. Herring", "Dimitrios Fraggedakis", "Martin Z. Bazant", "Stephen J. Harris", "William C. Chueh", "Richard D. Braatz"],
                "journal": "Nature Energy",
                "year": 2019,
                "doi": "10.1038/s41560-019-0356-8",
            },
        ],
    }
    (AGENT_DIR / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(overall, policy, horizon, cohort_size, selected, runtime_seconds)

    print(cutoff_150.to_string(index=False))
    print(f"runtime_seconds={runtime_seconds:.2f}")


if __name__ == "__main__":
    main()
