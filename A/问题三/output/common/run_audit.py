from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_data, split_battery_ids


def table(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_markdown(index=index, floatfmt=".6f")


def generate_audit() -> str:
    summary, cycles = load_data()
    train_ids, test_ids = split_battery_ids(summary)
    joined = cycles.merge(summary[["battery_id", "prediction_test"]], on="battery_id")

    policy = (
        summary.groupby("policy", as_index=False)
        .agg(
            battery_count=("battery_id", "size"),
            train_count=("prediction_test", lambda x: int((x == 0).sum())),
            test_count=("prediction_test", "sum"),
            C1=("C1", "first"),
            Q1=("Q1", "first"),
            C2=("C2", "first"),
        )
        .sort_values("policy")
    )
    test_table = summary.loc[
        summary["battery_id"].isin(test_ids), ["battery_id", "policy", "C1", "Q1", "C2"]
    ].copy()
    train_summary = summary.loc[summary["battery_id"].isin(train_ids)]
    test_table["same_policy_training_peers"] = test_table["policy"].map(
        train_summary["policy"].value_counts()
    )

    cycle_coverage = (
        cycles.groupby("battery_id", as_index=False)
        .agg(min_cycle=("cycle", "min"), max_cycle=("cycle", "max"), row_count=("cycle", "size"))
        .merge(summary[["battery_id", "policy", "prediction_test"]], on="battery_id")
        .sort_values("battery_id")
    )

    missing_rows = []
    for source, frame in (("battery_summary.csv", summary), ("cycle_train.csv", cycles)):
        for column, count in frame.isna().sum().items():
            if count:
                missing_rows.append(
                    {
                        "source": source,
                        "column": column,
                        "missing_count": int(count),
                        "missing_rate": float(count / len(frame)),
                    }
                )
    missing = pd.DataFrame(missing_rows)

    range_columns = ["capacity", "SOH", "SOH_smooth", "IR", "Tavg", "chargetime"]
    ranges = cycles[range_columns].agg(["min", "max", "mean", "median"]).T.reset_index()
    ranges = ranges.rename(columns={"index": "variable"})

    threshold_rows = []
    for threshold in (0.997, 0.995, 0.990, 0.980, 0.970, 0.800):
        for target in ("SOH", "SOH_smooth"):
            crossed = joined.loc[
                joined[target].le(threshold) & joined["prediction_test"].eq(0), "battery_id"
            ].nunique()
            threshold_rows.append(
                {"target": target, "threshold": threshold, "training_batteries_crossed": int(crossed)}
            )
    thresholds = pd.DataFrame(threshold_rows)

    c1_missing = summary.loc[summary["C1"].isna(), ["battery_id", "policy", "Q1", "C2", "prediction_test"]]
    ir_zero = cycles.loc[cycles["IR"].le(0), ["battery_id", "cycle", "IR"]]
    high_soh = cycles.loc[
        cycles["SOH"].gt(1.05),
        ["battery_id", "cycle", "capacity", "SOH", "SOH_smooth", "IR", "Tavg", "chargetime"],
    ]

    summary_difference = []
    for summary_col, cycle_col in (
        ("mean_IR", "IR"),
        ("mean_Tavg", "Tavg"),
        ("mean_chargetime", "chargetime"),
    ):
        cutoff_means = cycles.loc[cycles["cycle"].le(150)].groupby("battery_id")[cycle_col].mean()
        declared = summary.set_index("battery_id")[summary_col]
        delta = (cutoff_means - declared.loc[cutoff_means.index]).abs()
        summary_difference.append(
            {
                "summary_field": summary_col,
                "mean_abs_difference_vs_cycles_1_150": float(delta.mean()),
                "max_abs_difference_vs_cycles_1_150": float(delta.max()),
            }
        )

    lines = [
        "# 问题三数据审计",
        "",
        "## 1. 数据来源、范围与审计结论",
        "",
        "本审计仅使用赛题 PDF、`battery_summary.csv` 与 `cycle_train.csv`。目录中未发现已完成的问题一、问题二代码或结果。赛题说明非测试电池提供至第 200 次循环，9 块测试电池仅提供至第 150 次循环；本地文件与该说明一致。",
        "",
        f"- 电池总数：{len(summary)}；训练电池：{len(train_ids)}；`prediction_test == 1` 测试电池：{len(test_ids)}。",
        f"- 策略数：{summary['policy'].nunique()}；循环行数：{len(cycles)}。",
        "- `battery_id` 在汇总表唯一；`(battery_id, cycle)` 在循环表唯一；无重复行、无循环缺口、两表策略标签一致。",
        "- 所有 40 块训练电池均有 cycle 1–200；所有 9 块测试电池均且仅有 cycle 1–150。",
        "- 数据中不存在真实 `SOH <= 0.8` 或 `SOH_smooth <= 0.8`。因此 EOL 不能做普通监督回归，必须单独进行长期外推、伪阈值验证、函数形式敏感性和不确定性分析。",
        "- 三个汇总动态均值与 cycle 1–150 重算值不一致，且赛题未说明聚合窗口；所有模拟 cutoff 验证禁止使用 `mean_IR`、`mean_Tavg`、`mean_chargetime`。",
        "",
        "## 2. 充电策略与电池数量",
        "",
        table(policy),
        "",
        "`80PER_3_6C` 的 C1 缺失是同一策略全部 3 块电池共享的结构性缺失，而非随机漏填。后续将其解释为 0–80% SOC 的单阶段 3.6C 策略：构造 `C1_effective = C2 = 3.6`，同时保留 `C1_missing_indicator = 1`，不使用总体均值填补。",
        "",
        "## 3. 测试电池与同策略训练同伴",
        "",
        table(test_table),
        "",
        "每块测试电池均存在同策略训练电池，可开展严格的 same-policy cohort transfer；同伴数从 2 到 7 不等，必须同时报告同伴数与分策略误差。",
        "",
        "## 4. 每块电池循环覆盖",
        "",
        table(cycle_coverage),
        "",
        "## 5. 缺失值、重复与完整性",
        "",
        table(missing) if not missing.empty else "两个文件均无缺失值。",
        "",
        f"除上表 C1 的 3 个结构性缺失外，循环动态字段无缺失。汇总表主键重复数为 {int(summary['battery_id'].duplicated().sum())}；循环复合主键重复数为 {int(cycles.duplicated(['battery_id', 'cycle']).sum())}。",
        "",
        "## 6. 数值范围与数据特点",
        "",
        table(ranges),
        "",
        "- `SOH = capacity / initial_capacity`，早期测量波动允许轻微高于 1。绝大多数 SOH 位于约 0.95–1.01，但 battery 1 cycle 12 有单点容量 1.539054 Ah、SOH 1.437443；其平滑值仍被抬高至 1.090374。该点属于高影响异常，建模时使用平滑序列、稳健特征或在训练窗口内做稳健清洗，并保留原始数据不改写。",
        "- IR 有 2 个零值（battery 2 与 3 的 cycle 12），物理上不合理。动态特征提取时仅在各自 cutoff 内将非正 IR 置为空并按电池时间插值，防止全序列插值造成未来泄漏。",
        "- C1、Q1、C2 由 policy 离散组合决定，并非独立随机设计；策略效应与参数主效应存在共线/混杂，解释时以消融和留一电池验证为主，不把简单相关当因果。",
        "- Q1=80 时第二阶段宽度为 0，`E2=C2(80-Q1)=0`；策略暴露特征必须按这一物理边界构造。",
        "",
        "异常记录：",
        "",
        table(high_soh),
        "",
        "非正 IR 记录：",
        "",
        table(ir_zero),
        "",
        "结构性 C1 缺失记录：",
        "",
        table(c1_missing),
        "",
        "## 7. SOH 阈值可监督性",
        "",
        table(thresholds),
        "",
        "伪阈值不能预先固定。正式 EOL 验证将从实际 crossing 数筛选阈值，并要求足够训练电池发生 crossing；0.8 没有任何监督 crossing。较低阈值（如 0.97/0.98）样本仍很少，因此须报告每个阈值的有效电池数，避免伪造所谓 EOL RMSE。",
        "",
        "## 8. 汇总动态字段的 cutoff 泄漏风险",
        "",
        table(pd.DataFrame(summary_difference)),
        "",
        "这些差异证明汇总均值不能当作 cutoff 时刻已知特征。所有验证和最终推理只直接使用静态 `C1/Q1/C2/policy/initial_capacity`；IR、温度、充电时间的统计量均从 `cycle <= cutoff` 现场重算，清洗、标准化、特征选择和超参数选择也只在对应训练折内完成。",
        "",
        "## 9. 泄漏控制清单",
        "",
        "1. 9 块测试电池在冻结模型前完全不进入训练、验证、超参数选择或模型权重学习。",
        "2. 所有外层验证按 `battery_id` 留一电池；禁止随机拆分 cycle 行。",
        "3. rolling cutoff 样本先按 battery 分折，再在折内生成。",
        "4. cutoff=N 的动态特征、清洗和标准化只读取 `cycle <= N`。",
        "5. same-policy transfer 的目标电池未来轨迹永远不进入 reference pool。",
        "6. 最终测试电池只读取 cycle 1–150；不使用 `global_id` 查询外部完整轨迹。",
        "",
        "## 10. 可用性判断",
        "",
        "该数据可用于 50-cycle 短期 SOH 的严格 battery-level 回测，也可用于比较 early cutoff=50/75/100/125/150。它不包含 0.8 EOL 监督，EOL 结论只能是带模型假设、敏感性和区间说明的长程外推。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data_audit.md"
    output.write_text(generate_audit(), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
