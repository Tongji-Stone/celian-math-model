"""问题四：用问题二策略层模型做充电时间--衰减权衡。

主衰减模型采用 Tongji-Stone 两窗口 PySR（与策略加权 WLS 同结构）。
原始参数 OLS/岭系数与自由 PySR 只作对照，不作为推荐依据。
问题三逐电池曲线与公开 N_0.88 不进入优化。

运行（项目根目录）：
    python src/A/task4_optimize.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from SOH_plot import configure_chinese_font  # noqa: E402
from task2_exposure_model import (  # noqa: E402
    POLICY_ORDER,
    Q_STAR_MAIN,
    add_exposure,
    extract_primary,
    two_window_exposure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
P1_CHARGE = PROJECT_ROOT / "A" / "问题一" / "output" / "A" / "battery_SOH_charge.csv"
OUT_DIR = PROJECT_ROOT / "A" / "问题四" / "output"
FIG_DIR = OUT_DIR / "figures"
PAPER_FIG_DIR = PROJECT_ROOT / "output" / "A" / "question4" / "figures"

# Tongji-Stone 两窗口 PySR，复杂度 7（Pareto 拐点）；单位已换回 Fade_200
PYSR_A = 1.0 / 1000.0
PYSR_B = 1.3419437769134037 / 1000.0
PYSR_C = -430.7336192485949 / 1000.0

# 同结构 WLS（复杂度 9 的 PySR 与此几乎相同）
WLS_A = 0.0010170244725916937
WLS_B = 0.00135618754807407
WLS_C = -0.4369763403393811

# 原始参数 OLS（问题二对照回归，C1 符号为负）
OLS_B0 = 0.1132844102263081
OLS_C1 = -0.016923603100034136
OLS_Q1 = -0.00028079760706489564
OLS_C2 = -0.001554110998602276

# 自由 PySR 全样本拐点（不含 C2）
FREE_K0 = 354.96955557538615
FREE_K1 = 54.574553623224716

SOC_END = 80.0
C_HIGH_CAP = 4.80  # 不超过恒流 4.8C，远低于短寿组 5.9C
T_LO, T_HI = 9.90, 10.25  # 10 分钟快充邻域；两阶段实验点几乎都在此
C1_MIN, C1_MAX = 3.6, 5.6
Q1_MIN, Q1_MAX = 19.0, 80.0
C2_MIN, C2_MAX = 4.0, 4.8  # 连续搜索邻域：长寿组 C2 范围
TYPICAL_LONG = {
    "5C_67PER_4C",
    "5_3C_54PER_4C",
    "5_6C_36PER_4_3C",
    "5_6C_19PER_4_6C",
}
SHORT = "3_7C_31PER_5_9C"
ATYPICAL = "4_8C_80PER_4_8C"


def charge_time_cc_min(c1: float, q1: float, c2: float) -> float:
    """两阶段恒流充至 80% SOC 的时间（分钟）。"""
    return 60.0 * ((q1 / 100.0) / c1 + (SOC_END / 100.0 - q1 / 100.0) / c2)


def fade_pysr(e_l: float, e_h: float) -> float:
    return PYSR_A * e_l + PYSR_B * e_h + PYSR_C


def fade_wls(e_l: float, e_h: float) -> float:
    return WLS_A * e_l + WLS_B * e_h + WLS_C


def fade_ols(c1: float, q1: float, c2: float) -> float:
    return OLS_B0 + OLS_C1 * c1 + OLS_Q1 * q1 + OLS_C2 * c2


def fade_free_pysr(c1: float, q1: float, c2: float) -> float:
    del c2
    return ((FREE_K0 - q1) / c1 - FREE_K1) / 1000.0


def protocol_row(c1: float, q1: float, c2: float, time_a: float, time_b: float) -> dict:
    e_l, e_h = two_window_exposure(c1, q1, c2, Q_STAR_MAIN)
    c_high = e_h / (SOC_END - Q_STAR_MAIN)
    t_cc = charge_time_cc_min(c1, q1, c2)
    return {
        "C1": c1,
        "Q1": q1,
        "C2": c2,
        "E_L": e_l,
        "E_H": e_h,
        "C_high": c_high,
        "T_cc_min": t_cc,
        "T_hat_min": time_a * t_cc + time_b,
        "fade_pysr": fade_pysr(e_l, e_h),
        "fade_wls": fade_wls(e_l, e_h),
        "fade_ols": fade_ols(c1, q1, c2),
        "fade_free_pysr": fade_free_pysr(c1, q1, c2),
        "c1_ge_c2": int(c1 + 1e-12 >= c2),
    }


def attach_p1_charge_time(primary: pd.DataFrame) -> pd.DataFrame:
    """用问题一清洗后的逐圈 chargetime 均值，不用 summary.mean_chargetime。"""
    charge = pd.read_csv(P1_CHARGE)[["battery_id", "charge_time_mean"]].rename(
        columns={"charge_time_mean": "charge_time_p1"}
    )
    out = primary.merge(charge, on="battery_id", how="left", validate="one_to_one")
    missing = out.loc[out["charge_time_p1"].isna(), "battery_id"]
    if len(missing):
        raise ValueError(f"问题一充电时间缺失 battery_id={missing.tolist()}")
    return out


def fit_time_calibration(primary: pd.DataFrame) -> tuple[float, float, float]:
    """T_CC 在两阶段策略上几乎没有变异，不能拟合斜率。只估恒定偏移。"""
    typical = primary.loc[primary["policy"].isin(TYPICAL_LONG)]
    t_cc = np.array(
        [charge_time_cc_min(r.C1, r.Q1, r.C2) for r in typical.itertuples()],
        dtype=float,
    )
    t_obs = typical["charge_time_p1"].to_numpy(dtype=float)
    beta = float(np.mean(t_obs - t_cc))
    pred = t_cc + beta
    rmse = float(np.sqrt(np.mean((t_obs - pred) ** 2)))
    return 1.0, beta, rmse


def pareto_min2(frame: pd.DataFrame, t_col: str, y_col: str) -> pd.DataFrame:
    ordered = frame.sort_values([t_col, y_col], kind="mergesort")
    best_y = np.inf
    keep: list[int] = []
    for idx, y in zip(ordered.index, ordered[y_col].to_numpy()):
        if y < best_y - 1e-15:
            keep.append(int(idx))
            best_y = float(y)
    return ordered.loc[keep].reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, floatfmt: str = ".4g") -> str:
    data = frame if columns is None else frame[columns]
    cols = list(data.columns)

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(cell(row[c]) for c in cols) + " |" for _, row in data.iterrows()]
    return "\n".join([header, sep, *body])


def build_grid(time_a: float, time_b: float) -> pd.DataFrame:
    c1_vals = np.round(np.arange(C1_MIN, C1_MAX + 1e-9, 0.1), 1)
    q1_vals = np.arange(int(Q1_MIN), int(Q1_MAX) + 1)
    c2_vals = np.round(np.arange(C2_MIN, C2_MAX + 1e-9, 0.1), 1)
    rows = []
    for c1 in c1_vals:
        for q1 in q1_vals:
            for c2 in c2_vals:
                rec = protocol_row(float(c1), float(q1), float(c2), time_a, time_b)
                rec["feasible"] = int(
                    rec["c1_ge_c2"] == 1
                    and rec["C_high"] <= C_HIGH_CAP + 1e-12
                    and T_LO <= rec["T_hat_min"] <= T_HI
                )
                rows.append(rec)
    return pd.DataFrame(rows)


def make_figures(
    observed: pd.DataFrame,
    grid: pd.DataFrame,
    feasible: pd.DataFrame,
    front: pd.DataFrame,
    rec: pd.Series,
    contrast: pd.DataFrame,
) -> None:
    configure_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for _, row in observed.iterrows():
        color = "#C44E52" if row["policy"] == SHORT else ("#B07AA1" if row["policy"] == ATYPICAL else "#4C78A8")
        ax.scatter(row["T_obs_min"], row["fade_200"], s=70, color=color, zorder=3)
        ax.annotate(row["policy"].replace("_", "\n"), (row["T_obs_min"], row["fade_200"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("实测平均充电时间（min）")
    ax.set_ylabel(r"实测 $\mathrm{Fade}_{200}$")
    ax.set_title("已有策略：充电时间与 200 圈衰减")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "discrete_time_fade.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    sample = feasible.sample(n=min(4000, len(feasible)), random_state=20260816)
    ax.scatter(sample["T_hat_min"], sample["C_high"], s=6, alpha=0.18, color="#9ecae1", label="可行网格")
    ax.plot(front["T_hat_min"], front["C_high"], color="#4C78A8", lw=1.8, label="Pareto 前沿")
    ax.scatter(observed["T_hat_min"], observed["C_high"], s=55, color="#E45756", zorder=4, label="实验策略")
    ax.scatter([rec["T_hat_min"]], [rec["C_high"]], s=110, marker="*", color="#F2CF5B", edgecolor="black", zorder=5, label="邻域候选（min C_high）")
    ax.set_xlabel("校准后充电时间（min）")
    ax.set_ylabel("C_high")
    ax.set_title("邻域网格上的时间--高SOC倍率 Pareto（C1≥C2，C_high≤4.8）")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pareto_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(observed["C_high"], observed["fade_200"], s=70, color="#4C78A8", label="实验策略实测")
    ax.axvline(C_HIGH_CAP, color="#E45756", ls="--", lw=1.2, label=f"C_high={C_HIGH_CAP} 约束")
    ax.axvline(5.9, color="#B07AA1", ls=":", lw=1.2, label="短寿组 5.9C")
    for _, row in observed.iterrows():
        ax.annotate(row["policy"], (row["C_high"], row["fade_200"]), fontsize=7)
    ax.set_xlabel("高 SOC 等效倍率 C_high")
    ax.set_ylabel(r"实测 $\mathrm{Fade}_{200}$")
    ax.set_title("问题四约束落在高 SOC 倍率，而不是充电时间")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chigh_constraint.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.scatter(grid["fade_ols"], grid["C_high"], s=4, alpha=0.08, color="#bbbbbb")
    ax.scatter(feasible["fade_pysr"], feasible["C_high"], s=6, alpha=0.2, color="#9ecae1", label="可行域（两窗口）")
    colors = {"ols_box": "#E45756", "free_pysr_box": "#F58518", "neighborhood_candidate": "#54A24B"}
    labels = {
        "ols_box": "OLS 盒约束最优（危险）",
        "free_pysr_box": "自由 PySR 盒约束最优（危险）",
        "neighborhood_candidate": "两窗口邻域候选",
    }
    for _, row in contrast.iterrows():
        ax.scatter(row["fade_used"], row["C_high"], s=90, color=colors[row["name"]], zorder=4, label=labels[row["name"]])
    ax.set_xlabel("该模型给出的预测衰减")
    ax.set_ylabel("C_high")
    ax.set_title("对照：原始参数模型会把优化器推向高 SOC 大电流")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wrong_model_optima.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    observed: pd.DataFrame,
    time_a: float,
    time_b: float,
    time_rmse: float,
    rec: pd.Series,
    nearest: pd.Series,
    front: pd.DataFrame,
    contrast: pd.DataFrame,
    n_grid: int,
    n_feasible: int,
) -> None:
    rec_policy = f"{rec['C1']:.1f}C({rec['Q1']:.0f}%)-{rec['C2']:.1f}C"
    lines = [
        "# 问题四优化报告",
        "",
        "衰减主模型：Tongji-Stone 两窗口 PySR（复杂度 7）。充电时间与问题一相同：清洗后逐圈 chargetime 均值。",
        "连续搜索限制在 C1>=C2、C_high<=4.8、时间约 10 min 的实验邻域。",
        "正式目标是压低 C_high，而不是把两窗口预测 Fade 的小数差当成可分辨寿命。",
        "",
        "## 时间校准",
        "",
        f"- T = T_CC + beta，beta={time_b:.4f} min（长寿四策略、问题一口径充电时间上的平均残差）。",
        f"- 34 块主队列上校准 RMSE={time_rmse:.4f} min。两阶段实验策略的恒流时间几乎都是 10 min。",
        "",
        "## 已有 6 个策略",
        "",
        markdown_table(
            observed,
            [
                "policy",
                "n",
                "C1",
                "Q1",
                "C2",
                "C_high",
                "T_obs_min",
                "fade_200",
                "fade_pysr",
            ],
        ),
        "",
        "## 推荐",
        "",
        f"- 正式推荐（已实验）：**{nearest['policy']}**，实测 Fade={nearest['fade_200']:.5f}，C_high={nearest['C_high']:.3f}，T≈{nearest['T_obs_min']:.3f} min。",
        f"- 邻域候选（未实验）：**{rec_policy}**，T≈{rec['T_hat_min']:.3f} min，C_high={rec['C_high']:.3f}。只说明可把高 SOC 段保持在 4.0C；不把 PySR Fade={rec['fade_pysr']:.5f} 写成预期寿命。",
        "",
        "## 对照模型会推荐什么",
        "",
        markdown_table(contrast),
        "",
        f"网格规模 {n_grid}，可行 {n_feasible}。Pareto 前沿 {len(front)} 个点见 `pareto_front.csv`。",
        "",
        "## 边界",
        "",
        "- 响应仍是 200 圈衰减，不是 SOH=80% 寿命。",
        "- 两窗口高 R^2 依赖短寿 2 块；约束写成 C_high 上限，而不是精细曲面寻优。",
        "- 未使用问题三 hat N 与公开 N_0.88。",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(DATA_DIR / "battery_summary.csv")
    cycles = pd.read_csv(DATA_DIR / "cycle_train.csv")
    primary = extract_primary(summary, cycles)
    primary = add_exposure(primary, Q_STAR_MAIN)
    primary = attach_p1_charge_time(primary)
    time_a, time_b, time_rmse = fit_time_calibration(primary)

    grouped = (
        primary.groupby("policy", as_index=False)
        .agg(
            n=("battery_id", "size"),
            C1=("C1", "mean"),
            Q1=("Q1", "mean"),
            C2=("C2", "mean"),
            fade_200=("fade_200", "mean"),
            T_obs_min=("charge_time_p1", "mean"),
            E_L=("E_L", "mean"),
            E_H=("E_H", "mean"),
            C_high=("C_high", "mean"),
        )
    )
    extra = pd.DataFrame([protocol_row(r.C1, r.Q1, r.C2, time_a, time_b) for r in grouped.itertuples()])
    observed = grouped.merge(extra[["C1", "Q1", "C2", "T_cc_min", "T_hat_min", "fade_pysr", "fade_wls", "fade_ols", "fade_free_pysr"]], on=["C1", "Q1", "C2"])
    observed["policy"] = pd.Categorical(observed["policy"], POLICY_ORDER, ordered=True)
    observed = observed.sort_values("policy").reset_index(drop=True)
    observed.to_csv(OUT_DIR / "observed_policies.csv", index=False)

    typical = observed.loc[observed["policy"].isin(TYPICAL_LONG)].copy()
    discrete_best = typical.sort_values(["fade_200", "C_high", "T_obs_min"]).iloc[0]
    nearest = discrete_best

    print("校准时间模型并扫描网格…", flush=True)
    grid = build_grid(time_a, time_b)
    feasible = grid.loc[grid["feasible"] == 1].copy()
    if feasible.empty:
        raise RuntimeError("可行域为空，检查约束")
    front = pareto_min2(feasible, "T_hat_min", "C_high")

    # 邻域候选：在长寿两策略附近压低 C2，Q1 放到高 SOC 门槛，C1 升高以保持约 10 min。
    # 不在全盒内最小化 PySR Fade：该曲面在 C_high<4.8 时过平，小数差不可信。
    neighborhood = feasible.loc[
        feasible["C1"].between(5.0, 5.6)
        & feasible["Q1"].between(50, 67)
        & feasible["C2"].between(4.0, 4.3)
    ].copy()
    neighborhood["dist_to_official"] = (
        (neighborhood["C1"] - 5.3) ** 2
        + ((neighborhood["Q1"] - 54.0) / 10.0) ** 2
        + (neighborhood["C2"] - 4.0) ** 2
    )
    if neighborhood.empty:
        rec = pd.Series(protocol_row(5.3, 54.0, 4.0, time_a, time_b))
    else:
        rec = neighborhood.sort_values(["C_high", "dist_to_official", "T_hat_min"]).iloc[0]

    # 盒约束上对照模型的“最优”：OLS/自由 PySR 的梯度都指向最大 C1、Q1
    ols_best = protocol_row(C1_MAX, Q1_MAX, 5.9, time_a, time_b)
    free_best = protocol_row(C1_MAX, Q1_MAX, 4.0, time_a, time_b)
    contrast = pd.DataFrame(
        [
            {
                "name": "ols_box",
                "C1": ols_best["C1"],
                "Q1": ols_best["Q1"],
                "C2": ols_best["C2"],
                "C_high": ols_best["C_high"],
                "T_hat_min": ols_best["T_hat_min"],
                "fade_used": ols_best["fade_ols"],
                "fade_pysr_at_point": ols_best["fade_pysr"],
            },
            {
                "name": "free_pysr_box",
                "C1": free_best["C1"],
                "Q1": free_best["Q1"],
                "C2": free_best["C2"],
                "C_high": free_best["C_high"],
                "T_hat_min": free_best["T_hat_min"],
                "fade_used": free_best["fade_free_pysr"],
                "fade_pysr_at_point": free_best["fade_pysr"],
            },
            {
                "name": "neighborhood_candidate",
                "C1": rec["C1"],
                "Q1": rec["Q1"],
                "C2": rec["C2"],
                "C_high": rec["C_high"],
                "T_hat_min": rec["T_hat_min"],
                "fade_used": rec["fade_pysr"],
                "fade_pysr_at_point": rec["fade_pysr"],
            },
        ]
    )

    make_figures(observed, grid, feasible, front, rec, contrast)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for png in FIG_DIR.glob("*.png"):
        shutil.copy2(png, PAPER_FIG_DIR / png.name)
    write_report(observed, time_a, time_b, time_rmse, rec, nearest, front, contrast, len(grid), len(feasible))

    rec.to_frame("value").T.assign(source="neighborhood_candidate").to_csv(OUT_DIR / "recommendation.csv", index=False)
    nearest.to_frame("value").T.assign(source="nearest_experiment").to_csv(OUT_DIR / "nearest_experiment.csv", index=False)
    discrete_best.to_frame("value").T.assign(source="discrete_typical_min_fade").to_csv(
        OUT_DIR / "discrete_best.csv", index=False
    )
    front.to_csv(OUT_DIR / "pareto_front.csv", index=False)
    contrast.to_csv(OUT_DIR / "contrast_optima.csv", index=False)
    feasible.sample(n=min(2000, len(feasible)), random_state=1).to_csv(OUT_DIR / "feasible_sample.csv", index=False)

    summary_json = {
        "time_alpha": time_a,
        "time_beta": time_b,
        "time_rmse_min": time_rmse,
        "n_grid": int(len(grid)),
        "n_feasible": int(len(feasible)),
        "n_pareto": int(len(front)),
        "recommend_C1": float(rec["C1"]),
        "recommend_Q1": float(rec["Q1"]),
        "recommend_C2": float(rec["C2"]),
        "recommend_T": float(rec["T_hat_min"]),
        "recommend_fade_pysr": float(rec["fade_pysr"]),
        "recommend_C_high": float(rec["C_high"]),
        "nearest_policy": str(nearest["policy"]),
        "discrete_best_policy": str(discrete_best["policy"]),
        "discrete_best_fade_obs": float(discrete_best["fade_200"]),
        "discrete_best_C_high": float(discrete_best["C_high"]),
        "official_policy": str(discrete_best["policy"]),
        "objective": "discrete_observed_fade_plus_Chigh_constraint",
        "charge_time_source": "problem1_cleaned_cycle_chargetime_mean",
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(json.dumps(summary_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
