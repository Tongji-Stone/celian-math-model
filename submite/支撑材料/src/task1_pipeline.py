"""问题一：数据清洗、策略提取、200 次循环汇总与绘图。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from SOH_plot import configure_chinese_font, plot_all_battery_soh, plot_battery_soh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_COLUMNS = ["capacity", "SOH", "SOH_smooth", "chargetime", "IR", "Tavg"]


def normalize_policy_name(policy: str) -> str:
    """按第三次迭代规则合并策略并删除 NEWSTRUCTURE 后缀。"""
    normalized = str(policy).strip()
    direct_mapping = {
        "80PER_3_6C": "3_6C-80PER_3_6C",
        "3_6C-80PER_3_6C": "3_6C-80PER_3_6C",
        "4_8C_80PER_4_8C_NEWSTRUCTURE": "4_8C_80PER_4_8C",
        "4_8C_80PER_4_8C": "4_8C_80PER_4_8C",
    }
    if normalized in direct_mapping:
        return direct_mapping[normalized]
    upper = normalized.upper()
    if upper.endswith("_NEWSTRUCTURE"):
        normalized = normalized[: -len("_NEWSTRUCTURE")]
    return normalized


def normalize_policies(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """在处理副本中统一两个数据表的策略名称，原始数据保持不变。"""
    normalized_summary = summary.copy()
    normalized_cycles = cycles.copy()
    original_names = sorted(str(value) for value in summary["policy"].unique())
    mapping = {name: normalize_policy_name(name) for name in original_names}
    normalized_summary["policy"] = normalized_summary["policy"].map(mapping)
    normalized_cycles["policy"] = normalized_cycles["policy"].map(mapping)
    if normalized_summary["policy"].isna().any() or normalized_cycles["policy"].isna().any():
        raise ValueError("策略统一命名过程中产生空值")
    return normalized_summary, normalized_cycles, mapping


def validate_inputs(summary: pd.DataFrame, cycles: pd.DataFrame) -> None:
    summary_required = {"battery_id", "policy", "initial_capacity", "prediction_test"}
    cycle_required = {
        "battery_id",
        "cycle",
        "capacity",
        "SOH",
        "SOH_smooth",
        "chargetime",
        "IR",
        "Tavg",
        "policy",
    }
    summary_missing = summary_required.difference(summary.columns)
    cycle_missing = cycle_required.difference(cycles.columns)
    if summary_missing or cycle_missing:
        raise ValueError(
            f"输入字段不完整；battery_summary 缺少 {sorted(summary_missing)}，"
            f"cycle_train 缺少 {sorted(cycle_missing)}"
        )
    if summary["battery_id"].duplicated().any():
        raise ValueError("battery_summary.csv 中 battery_id 不唯一")
    if cycles.duplicated(["battery_id", "cycle"]).any():
        raise ValueError("cycle_train.csv 中存在重复的 battery_id + cycle")
    if set(summary["battery_id"]) != set(cycles["battery_id"]):
        raise ValueError("两个输入文件的 battery_id 集合不一致")

    policy_check = cycles.merge(
        summary[["battery_id", "policy"]],
        on="battery_id",
        suffixes=("_cycle", "_summary"),
        validate="many_to_one",
    )
    if (policy_check["policy_cycle"] != policy_check["policy_summary"]).any():
        raise ValueError("cycle_train.csv 与 battery_summary.csv 的策略字段不一致")


def iqr_clean_by_battery(
    cycles: pd.DataFrame,
    columns: list[str],
    factor: float = 1.5,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]]]:
    """逐电池、逐字段进行 IQR 检测，并用最近前后正常值的均值替换。"""
    cleaned = cycles.sort_values(["battery_id", "cycle"]).copy()
    audit: list[dict[str, float | int | str]] = []

    for battery_id, group in cycles.sort_values(["battery_id", "cycle"]).groupby("battery_id"):
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue

            q1, q3 = np.quantile(finite_values, [0.25, 0.75])
            iqr = q3 - q1
            if np.isclose(iqr, 0.0):
                continue
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outlier_mask = (~np.isfinite(values)) | (values < lower) | (values > upper)
            normal_positions = np.flatnonzero(~outlier_mask)

            for position in np.flatnonzero(outlier_mask):
                left_candidates = normal_positions[normal_positions < position]
                right_candidates = normal_positions[normal_positions > position]
                neighbors: list[float] = []
                if left_candidates.size:
                    neighbors.append(float(values[left_candidates[-1]]))
                if right_candidates.size:
                    neighbors.append(float(values[right_candidates[0]]))
                if not neighbors:
                    continue

                replacement = float(np.mean(neighbors))
                row_index = group.index[position]
                cleaned.at[row_index, column] = replacement
                audit.append(
                    {
                        "battery_id": int(battery_id),
                        "cycle": int(group.iloc[position]["cycle"]),
                        "column": column,
                        "original": float(values[position]),
                        "replacement": replacement,
                        "lower_bound": float(lower),
                        "upper_bound": float(upper),
                    }
                )

    return cleaned.sort_values(["battery_id", "cycle"]), audit


def build_policy_table(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary[["battery_id", "policy"]]
        .sort_values("battery_id")
        .reset_index(drop=True)
    )


def build_cycle_200_summary(
    cleaned_cycles: pd.DataFrame,
    policy_table: pd.DataFrame,
) -> pd.DataFrame:
    max_cycle = cleaned_cycles.groupby("battery_id")["cycle"].max()
    eligible_ids = max_cycle[max_cycle == 200].index
    eligible = cleaned_cycles[cleaned_cycles["battery_id"].isin(eligible_ids)]

    cycle_200 = eligible.loc[eligible["cycle"] == 200, ["battery_id", "SOH"]]
    if cycle_200["battery_id"].duplicated().any() or len(cycle_200) != len(eligible_ids):
        raise ValueError("达到 200 次循环的电池没有且仅有一条 cycle=200 记录")

    charge_mean = (
        eligible.groupby("battery_id", as_index=False)["chargetime"]
        .mean()
        .rename(columns={"chargetime": "charge_time_mean"})
    )
    result = (
        cycle_200.merge(policy_table, on="battery_id", validate="one_to_one")
        .merge(charge_mean, on="battery_id", validate="one_to_one")
        [["battery_id", "policy", "SOH", "charge_time_mean"]]
        .sort_values("battery_id")
        .reset_index(drop=True)
    )
    return result


def build_policy_summary(
    data: pd.DataFrame,
    policy_order: list[str],
    critical_soh_low: float,
    critical_soh_high: float,
    summary: pd.DataFrame,
    cleaned_cycles: pd.DataFrame,
) -> pd.DataFrame:
    """汇总统一策略的最终 SOH、充电时间和其他实验参数。"""
    classified = data.assign(
        lifetime_group=np.select(
            [data["SOH"] >= critical_soh_high, data["SOH"] <= critical_soh_low],
            ["长寿命", "短寿命"],
            default="中等寿命",
        )
    )
    grouped = classified.groupby("policy", sort=False)
    result = grouped.agg(
        battery_count=("battery_id", "size"),
        SOH_min=("SOH", "min"),
        SOH_max=("SOH", "max"),
        SOH_mean=("SOH", "mean"),
        SOH_median=("SOH", "median"),
        charge_time_mean=("charge_time_mean", "mean"),
        charge_time_min=("charge_time_mean", "min"),
        charge_time_max=("charge_time_mean", "max"),
    ).reindex(policy_order)
    result["battery_ids"] = grouped["battery_id"].apply(
        lambda values: ",".join(str(int(value)) for value in sorted(values))
    ).reindex(policy_order)

    eligible_ids = set(int(value) for value in data["battery_id"])
    battery_cycle_metrics = (
        cleaned_cycles.loc[cleaned_cycles["battery_id"].isin(eligible_ids)]
        .groupby(["battery_id", "policy"], as_index=False)
        .agg(IR_mean=("IR", "mean"), Tavg_mean=("Tavg", "mean"))
    )
    cycle_metrics = battery_cycle_metrics.groupby("policy").agg(
        IR_mean=("IR_mean", "mean"),
        Tavg_mean=("Tavg_mean", "mean"),
    ).reindex(policy_order)
    result = result.join(cycle_metrics)

    eligible_summary = summary.loc[summary["battery_id"].isin(eligible_ids)]
    capacity_metrics = eligible_summary.groupby("policy").agg(
        initial_capacity_mean=("initial_capacity", "mean"),
        initial_capacity_min=("initial_capacity", "min"),
        initial_capacity_max=("initial_capacity", "max"),
    ).reindex(policy_order)
    result = result.join(capacity_metrics)

    def parameter_values(series: pd.Series) -> str:
        values = [f"{float(value):g}" for value in sorted(series.dropna().unique())]
        if series.isna().any():
            values.append("缺失")
        return "/".join(values)

    for column in ["C1", "Q1", "C2"]:
        if column in eligible_summary.columns:
            result[f"{column}_values"] = (
                eligible_summary.groupby("policy")[column].apply(parameter_values).reindex(policy_order)
            )
    group_counts = (
        classified.groupby(["policy", "lifetime_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(policy_order, fill_value=0)
    )
    for group_name, column_name in [
        ("长寿命", "long_battery_count"),
        ("中等寿命", "middle_battery_count"),
        ("短寿命", "short_battery_count"),
    ]:
        result[column_name] = group_counts.get(group_name, pd.Series(0, index=result.index)).astype(int)

    result["lifetime_label"] = np.select(
        [result["SOH_median"] >= critical_soh_high, result["SOH_median"] <= critical_soh_low],
        ["长寿命策略", "短寿命策略"],
        default="中等寿命策略",
    )
    max_median = result["SOH_median"].max()
    min_median = result["SOH_median"].min()
    result["longest_strategy"] = np.where(np.isclose(result["SOH_median"], max_median), "是", "否")
    result["shortest_strategy"] = np.where(np.isclose(result["SOH_median"], min_median), "是", "否")
    result["SOH_range"] = result.apply(lambda row: f"{row['SOH_min']:.6f}–{row['SOH_max']:.6f}", axis=1)
    result["charge_time_range"] = result.apply(
        lambda row: f"{row['charge_time_min']:.6f}–{row['charge_time_max']:.6f}", axis=1
    )
    return result.reset_index()


def save_soh_histogram(
    data: pd.DataFrame,
    output_path: Path,
    critical_soh_low: float,
    critical_soh_high: float,
) -> None:
    values = data["SOH"].to_numpy(dtype=float)
    bin_edges = np.histogram_bin_edges(values, bins="fd")
    if len(bin_edges) < 7:
        bin_edges = np.linspace(values.min(), values.max(), 8)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        values,
        bins=bin_edges,
        color="#4C78A8",
        edgecolor="white",
        linewidth=1.0,
        label="电池数量",
    )
    ax.axvline(
        critical_soh_high,
        color="#D62728",
        linestyle="--",
        linewidth=2,
        label=f"上四分位线（Q3）={critical_soh_high:.4f}",
    )
    ax.axvline(
        critical_soh_low,
        color="#1F77B4",
        linestyle="--",
        linewidth=2,
        label=f"下四分位线（Q1）={critical_soh_low:.4f}",
    )
    margin = max((values.max() - values.min()) * 0.08, 0.003)
    ax.set_xlim(values.min() - margin, values.max() + margin)
    ax.set_title("第 200 次循环 SOH 分布")
    ax.set_xlabel("第 200 次循环的 SOH")
    ax.set_ylabel("电池数量")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_soh_charge_scatter(
    data: pd.DataFrame,
    output_path: Path,
    policy_order: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    color_map = matplotlib.colormaps.get_cmap("tab10")
    colors = color_map(np.linspace(0, 1, max(len(policy_order), 2)))
    for color, policy in zip(colors, policy_order):
        group = data.loc[data["policy"] == policy]
        if group.empty:
            continue
        ax.scatter(
            group["SOH"],
            group["charge_time_mean"],
            s=64,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.88,
            label=policy,
        )
    ax.set_title("第 200 次循环 SOH 与平均充电时间")
    ax.set_xlabel("第 200 次循环的 SOH")
    ax.set_ylabel("平均充电时间（min）")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92, title="充电策略", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_policy_boxplot(
    data: pd.DataFrame,
    output_path: Path,
    critical_soh_low: float,
    critical_soh_high: float,
    policy_order: list[str],
) -> None:
    available_order = [policy for policy in policy_order if policy in set(data["policy"])]
    groups = [data.loc[data["policy"] == policy, "SOH"].to_numpy() for policy in available_order]

    fig, ax = plt.subplots(figsize=(16, 8))
    box = ax.boxplot(
        groups,
        tick_labels=available_order,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "#1F1F1F", "linewidth": 1.5},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#4C78A8",
            "markersize": 5,
        },
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#9ECAE1")
        patch.set_edgecolor("#4C78A8")

    ax.axhline(
        critical_soh_high,
        color="#D62728",
        linestyle="--",
        linewidth=2,
    )
    ax.axhline(
        critical_soh_low,
        color="#1F77B4",
        linestyle="--",
        linewidth=2,
    )
    all_values = data["SOH"].to_numpy(dtype=float)
    margin = max((all_values.max() - all_values.min()) * 0.08, 0.003)
    ax.set_ylim(all_values.min() - margin, all_values.max() + margin)
    ax.set_title("不同充电策略下的第 200 次循环 SOH 分布")
    ax.set_xlabel("充电策略")
    ax.set_ylabel("使用寿命指标：第 200 次循环的 SOH")
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    legend_handles = [
        Patch(facecolor="#9ECAE1", edgecolor="#4C78A8", label="策略内 SOH 分布"),
        plt.Line2D(
            [0], [0], color="#D62728", linestyle="--", linewidth=2,
            label=f"上四分位线（Q3）={critical_soh_high:.4f}",
        ),
        plt.Line2D(
            [0], [0], color="#1F77B4", linestyle="--", linewidth=2,
            label=f"下四分位线（Q1）={critical_soh_low:.4f}",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_typical_policy_doc(
    doc_path: Path,
    policy_summary: pd.DataFrame,
    critical_soh_low: float,
    critical_soh_high: float,
) -> None:
    long_policies = policy_summary.loc[policy_summary["lifetime_label"] == "长寿命策略"]
    short_policies = policy_summary.loc[policy_summary["lifetime_label"] == "短寿命策略"]
    longest = policy_summary.loc[policy_summary["longest_strategy"] == "是"].iloc[0]
    shortest = policy_summary.loc[policy_summary["shortest_strategy"] == "是"].iloc[0]

    lines = [
        "# 典型长寿命与短寿命充电策略",
        "",
        "本分析使用第 200 次循环的 SOH 作为早期使用寿命代理指标。",
        f"长寿命阈值为样本 SOH 的 75% 分位数（{critical_soh_high:.6f}），",
        f"短寿命阈值为 25% 分位数（{critical_soh_low:.6f}）。策略类型按策略内 SOH 中位数判定。",
        "",
        "## 关键结论",
        "",
        f"- **寿命最长策略：`{longest['policy']}`**，SOH 中位数 {longest['SOH_median']:.6f}，范围 {longest['SOH_range']}。",
        f"- **寿命最短策略：`{shortest['policy']}`**，SOH 中位数 {shortest['SOH_median']:.6f}，范围 {shortest['SOH_range']}。",
        "",
        "## 典型长寿命策略",
        "",
        "| 策略 | 电池数 | SOH中位数 | SOH范围 | 平均充电时间范围/min | 标记 |",
        "|---|---:|---:|---|---|---|",
    ]
    for _, row in long_policies.sort_values("SOH_median", ascending=False).iterrows():
        mark = "寿命最长" if row["longest_strategy"] == "是" else "典型长寿命"
        lines.append(
            f"| `{row['policy']}` | {int(row['battery_count'])} | {row['SOH_median']:.6f} | "
            f"{row['SOH_range']} | {row['charge_time_range']} | **{mark}** |"
        )
    lines.extend(
        [
            "",
            "## 典型短寿命策略",
            "",
            "| 策略 | 电池数 | SOH中位数 | SOH范围 | 平均充电时间范围/min | 标记 |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for _, row in short_policies.sort_values("SOH_median").iterrows():
        mark = "寿命最短" if row["shortest_strategy"] == "是" else "典型短寿命"
        lines.append(
            f"| `{row['policy']}` | {int(row['battery_count'])} | {row['SOH_median']:.6f} | "
            f"{row['SOH_range']} | {row['charge_time_range']} | **{mark}** |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 各策略的完整 200 循环样本数较少，结论用于本数据集内的典型策略比较，不宜直接外推为因果结论。",
            "- `4_8C_80PER_4_8C_NEWSTRUCTURE` 的中位数较高但包含一个明显较低的 SOH 个体，需结合离散程度解读。",
            "- 数据未覆盖 SOH<80% 的真实寿命终止循环，因此这里比较的是第 200 次循环的健康保持程度。",
        ]
    )
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_policy_report(
    report_path: Path,
    policy_summary: pd.DataFrame,
    critical_soh_low: float,
    critical_soh_high: float,
) -> None:
    """输出第三次迭代要求的统一策略统计报告。"""
    lines = [
        "# 统一充电策略统计报告",
        "",
        "## 统计口径",
        "",
        "- 策略名称已按第三次迭代规则合并，并移除 `_NEWSTRUCTURE` 后缀。",
        "- 最终 SOH 为完整记录至第 200 次循环的电池在 `cycle=200` 时的清洗后 SOH。",
        "- 充电时间、内阻和温度先按电池对 1–200 次循环取均值，再按策略汇总。",
        f"- 下四分位线 Q1={critical_soh_low:.6f}；上四分位线 Q3={critical_soh_high:.6f}。",
        "",
        "## 最终 SOH 与充电时间",
        "",
        "| 统一策略 | 电池记号 | 电池数 | 最终SOH平均 | 最小 | 最大 | 充电时间平均/min | 最小/min | 最大/min |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in policy_summary.iterrows():
        lines.append(
            f"| `{row['policy']}` | {row['battery_ids']} | {int(row['battery_count'])} | "
            f"{row['SOH_mean']:.6f} | {row['SOH_min']:.6f} | {row['SOH_max']:.6f} | "
            f"{row['charge_time_mean']:.6f} | {row['charge_time_min']:.6f} | {row['charge_time_max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 其他参数",
            "",
            "| 统一策略 | C1取值 | Q1取值 | C2取值 | 初始容量平均/Ah | 最小/Ah | 最大/Ah | 平均内阻 | 平均温度/°C |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in policy_summary.iterrows():
        lines.append(
            f"| `{row['policy']}` | {row.get('C1_values', '')} | {row.get('Q1_values', '')} | "
            f"{row.get('C2_values', '')} | {row['initial_capacity_mean']:.6f} | "
            f"{row['initial_capacity_min']:.6f} | {row['initial_capacity_max']:.6f} | "
            f"{row['IR_mean']:.6f} | {row['Tavg_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 9块仅有150次循环数据的测试电池未纳入最终SOH策略统计。",
            "- 统一策略是按题目指定规则进行的分析分组；报告中的相关性不代表充电策略对寿命的因果效应。",
            "- 本数据片段尚未覆盖 SOH<80% 的真实寿命终止点，最终SOH用于比较第200次循环时的健康保持程度。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_acceptance_results(
    original_summary: pd.DataFrame,
    normalized_summary: pd.DataFrame,
    normalized_cycles: pd.DataFrame,
    policy_table: pd.DataFrame,
    cycle_200: pd.DataFrame,
    policy_summary: pd.DataFrame,
    soh_dir: Path,
    table_write_paths: dict[str, Path],
) -> list[tuple[str, bool, str]]:
    """执行第三次迭代的显式验收检查。"""
    original_count = int(original_summary["policy"].nunique())
    normalized_count = int(normalized_summary["policy"].nunique())
    expected_names = {normalize_policy_name(value) for value in original_summary["policy"].unique()}
    actual_names = set(normalized_summary["policy"].unique())
    forbidden = {"80PER_3_6C", "4_8C_80PER_4_8C_NEWSTRUCTURE"}
    plot_files = list(soh_dir.glob("SOH_*.png"))
    individual_files = [path for path in plot_files if path.stem != "SOH_ALL"]
    results = [
        (
            "策略数量恰好减少2种",
            original_count - normalized_count == 2,
            f"原始{original_count}种，统一后{normalized_count}种，减少{original_count - normalized_count}种",
        ),
        (
            "统一策略集合正确",
            actual_names == expected_names,
            f"实际策略：{sorted(actual_names)}",
        ),
        (
            "旧策略名称已消除",
            not (actual_names & forbidden),
            f"禁止名称残留：{sorted(actual_names & forbidden)}",
        ),
        (
            "NEWSTRUCTURE后缀已全部移除",
            not any("NEWSTRUCTURE" in name.upper() for name in actual_names),
            "统一后的策略名不含NEWSTRUCTURE",
        ),
        (
            "全部处理表命名一致",
            set(normalized_cycles["policy"].unique()) == actual_names
            and set(policy_table["policy"].unique()) == actual_names
            and set(cycle_200["policy"].unique()).issubset(actual_names),
            "清洗循环表、charge_policy和battery_SOH_charge均采用统一名称",
        ),
        (
            "charge_policy覆盖49块电池",
            len(policy_table) == 49
            and policy_table["battery_id"].nunique() == 49,
            f"记录数={len(policy_table)}，唯一电池数={policy_table['battery_id'].nunique()}",
        ),
        (
            "核心CSV均写入标准文件名",
            all(path.name == expected for expected, path in table_write_paths.items()),
            ", ".join(f"{expected}->{path.name}" for expected, path in table_write_paths.items()),
        ),
        (
            "策略报告覆盖全部统一策略",
            set(policy_summary["policy"]) == actual_names and len(policy_summary) == normalized_count,
            f"报告策略数={len(policy_summary)}",
        ),
        (
            "49张单体曲线与SOH_ALL齐全",
            len(individual_files) == 49 and (soh_dir / "SOH_ALL.png").is_file(),
            f"单体曲线={len(individual_files)}，SOH_ALL={int((soh_dir / 'SOH_ALL.png').is_file())}",
        ),
    ]
    return results


def write_acceptance_report(
    path: Path,
    results: list[tuple[str, bool, str]],
    mapping: dict[str, str],
) -> None:
    lines = [
        "# 第三次迭代验收确认",
        "",
        "| 验收项 | 结果 | 证据 |",
        "|---|---|---|",
    ]
    for name, passed, evidence in results:
        lines.append(f"| {name} | {'通过' if passed else '失败'} | {evidence} |")
    lines.extend(["", "## 策略名称映射", "", "| 原始名称 | 统一名称 |", "|---|---|"])
    for source, target in mapping.items():
        lines.append(f"| `{source}` | `{target}` |")
    lines.extend(["", f"**总体结论：{'全部通过' if all(item[1] for item in results) else '存在失败项'}。**"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_if_changed(data: pd.DataFrame, path: Path) -> Path:
    """内容不变时避免重写，兼容 CSV 正被表格软件打开的情形。"""
    if path.exists():
        try:
            existing = pd.read_csv(path)
            pd.testing.assert_frame_equal(existing, data, check_dtype=False)
            return path
        except (AssertionError, OSError, ValueError):
            pass
    pending_path = path.with_name(f"{path.stem}_v3_pending{path.suffix}")
    try:
        data.to_csv(path, index=False, encoding="utf-8-sig")
        if pending_path.exists():
            pending_path.unlink()
        return path
    except PermissionError:
        data.to_csv(pending_path, index=False, encoding="utf-8-sig")
        return pending_path


def write_log(
    log_path: Path,
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    audit: list[dict[str, float | int | str]],
    cycle_200: pd.DataFrame,
    policy_summary: pd.DataFrame,
    critical_soh_low: float,
    critical_soh_high: float,
    iqr_factor: float,
    font_name: str,
    output_paths: list[Path],
    prompt_text: str,
    policy_mapping: dict[str, str],
    acceptance_results: list[tuple[str, bool, str]],
) -> None:
    counts = Counter(str(item["column"]) for item in audit)
    max_cycle_counts = cycles.groupby("battery_id")["cycle"].max().value_counts().sort_index()
    missing_summary = int(summary.isna().sum().sum())
    missing_cycles = int(cycles.isna().sum().sum())
    long_count = int((cycle_200["SOH"] >= critical_soh_high).sum())
    short_count = int((cycle_200["SOH"] <= critical_soh_low).sum())
    middle_count = len(cycle_200) - long_count - short_count
    longest = policy_summary.loc[policy_summary["longest_strategy"] == "是", "policy"].iloc[0]
    shortest = policy_summary.loc[policy_summary["shortest_strategy"] == "是", "policy"].iloc[0]
    normalized_policy_count = len(set(policy_mapping.values()))

    lines = [
        "# 问题一第三次迭代运行日志（log_3）",
        "",
        f"- 运行时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- IQR 系数：{iqr_factor:.6g}",
        f"- 上四分位线 Q3（75%分位数）：{critical_soh_high:.6f}",
        f"- 下四分位线 Q1（25%分位数）：{critical_soh_low:.6f}",
        f"- 绘图字体：{font_name}",
        "",
        "## 输入质量检查",
        "",
        f"- battery_summary.csv：{len(summary)} 行，{summary.shape[1]} 列，{summary['battery_id'].nunique()} 块电池。",
        f"- cycle_train.csv：{len(cycles)} 行，{cycles.shape[1]} 列。",
        f"- battery_id + cycle 重复记录：{int(cycles.duplicated(['battery_id', 'cycle']).sum())}。",
        f"- 汇总表缺失单元格：{missing_summary}（其中 C1 的结构性缺失保留，不参与循环数据清洗）。",
        f"- 循环表缺失单元格：{missing_cycles}。",
        f"- 最大循环数分布：{dict((int(k), int(v)) for k, v in max_cycle_counts.items())}。",
        "",
        "## 策略统一命名",
        "",
        f"- 原始策略数：{len(policy_mapping)}；统一后策略数：{normalized_policy_count}；恰好减少 {len(policy_mapping) - normalized_policy_count} 种。",
        "- `80PER_3_6C` 统一为 `3_6C-80PER_3_6C`。",
        "- `4_8C_80PER_4_8C_NEWSTRUCTURE` 统一为 `4_8C_80PER_4_8C`。",
        "- 其余策略仅移除 `_NEWSTRUCTURE` 后缀。",
        "",
        "## IQR 清洗",
        "",
        "- 方法：按 battery_id 分组，对 capacity、SOH、SOH_smooth、chargetime、IR、Tavg 分别计算 Q1、Q3 和 IQR。",
        "- 判定：小于 Q1 - 1.5×IQR 或大于 Q3 + 1.5×IQR 的值标记为异常。",
        "- 填补：使用同一电池、同一字段中距离最近的前后正常值均值；边界点仅有一侧正常值时使用该侧值。",
        f"- 共替换 {len(audit)} 个单元格；分字段数量：{dict(counts)}。",
        "- 原始 data 文件未改写；清洗结果另存为 output/A/cycle_train_cleaned.csv。",
        "",
        "## 200 次循环样本",
        "",
        f"- 纳入：最大 cycle 恰为 200 的 {len(cycle_200)} 块电池。",
        f"- 排除：仅记录至 150 次循环的 {summary['prediction_test'].sum()} 块测试电池。",
        f"- SOH 范围：{cycle_200['SOH'].min():.6f}–{cycle_200['SOH'].max():.6f}，均值 {cycle_200['SOH'].mean():.6f}。",
        f"- 按两个分位数分类：长寿命 {long_count} 块，中等寿命 {middle_count} 块，短寿命 {short_count} 块。",
        f"- 寿命最长策略（按策略 SOH 中位数）：{longest}。",
        f"- 寿命最短策略（按策略 SOH 中位数）：{shortest}。",
        "",
        "## SOH 曲线",
        "",
        "- 使用清洗后的 `SOH_smooth` 绘制49张单体曲线和1张全部电池曲线。",
        "- 单体曲线命名为 `SOH_N.png`；总图命名为 `SOH_ALL.png`，颜色按电池编号由蓝到红渐变。",
        "",
        "## 验收确认",
        "",
    ]
    for name, passed, evidence in acceptance_results:
        lines.append(f"- [{'x' if passed else ' '}] {name}：{evidence}。")
    lines.extend(["", "## 输出文件", ""])
    lines.extend(f"- `{path.relative_to(PROJECT_ROOT).as_posix()}`" for path in output_paths)
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 箱型图纵轴采用“第 200 次循环 SOH”作为使用寿命代理指标；本题数据尚未覆盖 SOH<80% 的真实寿命终止循环。",
            "- IQR 是统计异常筛查，不等同于确认传感器故障；清洗结果用于完成本题指定分析。",
            "- 图中的两条参考线统一命名为上四分位线（Q3）和下四分位线（Q1）。",
            "- 策略报告使用统一后的7种策略，最终SOH统计只纳入完整200循环的40块电池。",
            "",
            "## 原始提示词",
            "",
            "以下完整保留本次执行时 `A/问题一/AGENTS.md` 的内容：",
            "",
            "```text",
            prompt_text.rstrip(),
            "```",
        ]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行问题一完整分析流程")
    parser.add_argument("--iqr-factor", type=float, default=1.5, help="IQR 异常判定系数，默认 1.5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iqr_factor <= 0:
        raise ValueError("--iqr-factor 必须大于 0")

    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "output" / "A"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_summary = pd.read_csv(data_dir / "battery_summary.csv")
    original_cycles = pd.read_csv(data_dir / "cycle_train.csv")
    validate_inputs(original_summary, original_cycles)
    summary, cycles, policy_mapping = normalize_policies(original_summary, original_cycles)
    validate_inputs(summary, cycles)

    cleaned, audit = iqr_clean_by_battery(cycles, MEASUREMENT_COLUMNS, args.iqr_factor)
    policy_table = build_policy_table(summary)
    cycle_200 = build_cycle_200_summary(cleaned, policy_table)
    critical_soh_low = float(cycle_200["SOH"].quantile(0.25))
    critical_soh_high = float(cycle_200["SOH"].quantile(0.75))
    policy_order = list(dict.fromkeys(summary.sort_values("battery_id")["policy"]))
    policy_summary = build_policy_summary(
        cycle_200,
        policy_order,
        critical_soh_low,
        critical_soh_high,
        summary,
        cleaned,
    )

    cleaned_path = output_dir / "cycle_train_cleaned.csv"
    policy_path = output_dir / "charge_policy.csv"
    summary_path = output_dir / "battery_SOH_charge.csv"
    soh_example_path = output_dir / "SOH_1_example.png"
    histogram_path = output_dir / "SOH_summary.png"
    scatter_path = output_dir / "SOH_chargetime.png"
    boxplot_path = output_dir / "charge_policy.png"
    policy_summary_path = output_dir / "policy.csv"
    report_path = PROJECT_ROOT / "output" / "doc" / "1.md"
    acceptance_path = output_dir / "acceptance_3.md"
    soh_dir = PROJECT_ROOT / "A" / "问题一" / "SOH"

    cleaned_write_path = write_csv_if_changed(cleaned, cleaned_path)
    policy_write_path = write_csv_if_changed(policy_table, policy_path)
    summary_write_path = write_csv_if_changed(cycle_200, summary_path)
    policy_summary_write_path = write_csv_if_changed(policy_summary, policy_summary_path)

    font_name = configure_chinese_font()
    plot_battery_soh(cleaned, 1, soh_example_path)
    save_soh_histogram(cycle_200, histogram_path, critical_soh_low, critical_soh_high)
    save_soh_charge_scatter(cycle_200, scatter_path, policy_order)
    save_policy_boxplot(
        cycle_200,
        boxplot_path,
        critical_soh_low,
        critical_soh_high,
        policy_order,
    )
    write_policy_report(
        report_path,
        policy_summary,
        critical_soh_low,
        critical_soh_high,
    )

    soh_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in soh_dir.glob("SOH_*.png"):
        stale_plot.unlink()
    for battery_id in sorted(int(value) for value in cleaned["battery_id"].unique()):
        plot_battery_soh(cleaned, battery_id, soh_dir / f"SOH_{battery_id}.png")
    plot_all_battery_soh(cleaned, soh_dir / "SOH_ALL.png")

    acceptance_results = build_acceptance_results(
        original_summary,
        summary,
        cleaned,
        policy_table,
        cycle_200,
        policy_summary,
        soh_dir,
        {
            cleaned_path.name: cleaned_write_path,
            policy_path.name: policy_write_path,
            summary_path.name: summary_write_path,
            policy_summary_path.name: policy_summary_write_path,
        },
    )
    write_acceptance_report(acceptance_path, acceptance_results, policy_mapping)

    output_paths = [
        cleaned_write_path,
        policy_write_path,
        summary_write_path,
        soh_example_path,
        histogram_path,
        scatter_path,
        boxplot_path,
        policy_summary_write_path,
        report_path,
        acceptance_path,
        soh_dir,
    ]
    write_log(
        PROJECT_ROOT / "A" / "问题一" / "log_3.md",
        summary,
        cycles,
        audit,
        cycle_200,
        policy_summary,
        critical_soh_low,
        critical_soh_high,
        args.iqr_factor,
        font_name,
        output_paths,
        (PROJECT_ROOT / "A" / "问题一" / "AGENTS.md").read_text(encoding="utf-8"),
        policy_mapping,
        acceptance_results,
    )

    print(f"IQR 替换单元格数: {len(audit)}")
    print(f"200 次循环样本数: {len(cycle_200)}")
    print(f"critical_SOH_low: {critical_soh_low:.6f}")
    print(f"critical_SOH_high: {critical_soh_high:.6f}")
    print(f"策略数量: {len(policy_mapping)} -> {len(set(policy_mapping.values()))}")
    print(f"SOH曲线数量: {len(list(soh_dir.glob('SOH_*.png')))}")
    failed = [name for name, passed, _ in acceptance_results if not passed]
    print(f"验收失败项: {failed}")
    for path in output_paths:
        print(f"已生成: {path}")


if __name__ == "__main__":
    main()
