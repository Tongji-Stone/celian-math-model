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

from SOH_plot import configure_chinese_font, plot_battery_soh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_COLUMNS = ["capacity", "SOH", "SOH_smooth", "chargetime", "IR", "Tavg"]


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
) -> pd.DataFrame:
    """汇总各充电策略，并用策略 SOH 中位数划分寿命类型。"""
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
        charge_time_min=("charge_time_mean", "min"),
        charge_time_max=("charge_time_mean", "max"),
    ).reindex(policy_order)
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
        label=f"长寿命分界线（75%分位）={critical_soh_high:.4f}",
    )
    ax.axvline(
        critical_soh_low,
        color="#1F77B4",
        linestyle="--",
        linewidth=2,
        label=f"短寿命分界线（25%分位）={critical_soh_low:.4f}",
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
            label=f"长寿命分界线={critical_soh_high:.4f}",
        ),
        plt.Line2D(
            [0], [0], color="#1F77B4", linestyle="--", linewidth=2,
            label=f"短寿命分界线={critical_soh_low:.4f}",
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


def write_csv_if_changed(data: pd.DataFrame, path: Path) -> None:
    """内容不变时避免重写，兼容 CSV 正被表格软件打开的情形。"""
    if path.exists():
        try:
            existing = pd.read_csv(path)
            pd.testing.assert_frame_equal(existing, data, check_dtype=False)
            return
        except (AssertionError, OSError, ValueError):
            pass
    data.to_csv(path, index=False, encoding="utf-8-sig")


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

    lines = [
        "# 问题一第二次迭代运行日志（log_2）",
        "",
        f"- 运行时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- IQR 系数：{iqr_factor:.6g}",
        f"- critical_SOH_high（75%分位数）：{critical_soh_high:.6f}",
        f"- critical_SOH_low（25%分位数）：{critical_soh_low:.6f}",
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
        "## 第一轮归档",
        "",
        "- 第一轮输出已整理到 `output/A/1/`，共 7 个交付文件。",
        "",
        "## 输出文件",
        "",
    ]
    lines.extend(f"- `{path.relative_to(PROJECT_ROOT).as_posix()}`" for path in output_paths)
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 箱型图纵轴采用“第 200 次循环 SOH”作为使用寿命代理指标；本题数据尚未覆盖 SOH<80% 的真实寿命终止循环。",
            "- IQR 是统计异常筛查，不等同于确认传感器故障；清洗结果用于完成本题指定分析。",
            "- 策略寿命标签按策略内 SOH 中位数与总体 25%/75% 分位阈值比较得到。",
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
    summary = pd.read_csv(data_dir / "battery_summary.csv")
    cycles = pd.read_csv(data_dir / "cycle_train.csv")
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
    )

    cleaned_path = output_dir / "cycle_train_cleaned.csv"
    policy_path = output_dir / "charge_policy.csv"
    summary_path = output_dir / "battery_SOH_charge.csv"
    soh_example_path = output_dir / "SOH_1_example.png"
    histogram_path = output_dir / "SOH_summary.png"
    scatter_path = output_dir / "SOH_chargetime.png"
    boxplot_path = output_dir / "charge_policy.png"
    policy_summary_path = output_dir / "policy.csv"
    typical_doc_path = output_dir / "doc" / "1.md"

    cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    write_csv_if_changed(policy_table, policy_path)
    cycle_200.to_csv(summary_path, index=False, encoding="utf-8-sig")
    policy_summary.to_csv(policy_summary_path, index=False, encoding="utf-8-sig")

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
    write_typical_policy_doc(
        typical_doc_path,
        policy_summary,
        critical_soh_low,
        critical_soh_high,
    )

    output_paths = [
        cleaned_path,
        policy_path,
        summary_path,
        soh_example_path,
        histogram_path,
        scatter_path,
        boxplot_path,
        policy_summary_path,
        typical_doc_path,
    ]
    write_log(
        PROJECT_ROOT / "A" / "问题一" / "log_2.md",
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
    )

    print(f"IQR 替换单元格数: {len(audit)}")
    print(f"200 次循环样本数: {len(cycle_200)}")
    print(f"critical_SOH_low: {critical_soh_low:.6f}")
    print(f"critical_SOH_high: {critical_soh_high:.6f}")
    for path in output_paths:
        print(f"已生成: {path}")


if __name__ == "__main__":
    main()
