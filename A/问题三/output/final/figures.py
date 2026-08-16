from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from problem3.agents.agent_02_physics.model import robust_asof_soh


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
BLUE = "#2367A3"
ORANGE = "#D97706"
GOLD = "#C8A951"
PINK = "#B45572"
OLIVE = "#718355"
INK = "#233142"
GREY = "#9AA5B1"
LIGHT = "#E8EEF3"


def configure_style() -> str:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    selected = "DejaVu Sans"
    for candidate in candidates:
        try:
            fm.findfont(candidate, fallback_to_default=False)
            selected = candidate
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#7C8793",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": "#4E5D6C",
            "ytick.color": "#4E5D6C",
            "grid.color": "#DDE3E8",
            "grid.linewidth": 0.7,
            "axes.titleweight": "semibold",
        }
    )
    return selected


def finish(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_01(summary: pd.DataFrame, cycles: pd.DataFrame) -> None:
    policies = sorted(summary["policy"].unique())
    train_ids = set(summary.loc[summary["prediction_test"].eq(0), "battery_id"])
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey=True)
    for ax, policy in zip(axes.flat, policies):
        ids = summary.loc[
            summary["policy"].eq(policy) & summary["battery_id"].isin(train_ids), "battery_id"
        ].astype(int)
        trajectories = []
        for battery_id in ids:
            frame = cycles.loc[cycles["battery_id"].eq(battery_id)].sort_values("cycle")
            cleaned = robust_asof_soh(frame)
            trajectories.append(cleaned)
            ax.plot(frame["cycle"], cleaned, color=BLUE, alpha=0.28, linewidth=1.0)
        if trajectories:
            mean = np.mean(np.vstack(trajectories), axis=0)
            ax.plot(np.arange(1, len(mean) + 1), mean, color=INK, linewidth=2.0, label="策略均值")
        ax.set_title(f"{policy}\n训练电池 n={len(ids)}", fontsize=10)
        ax.grid(True, alpha=0.75)
        ax.set_ylim(0.944, 1.009)
    for ax in axes[-1, :]:
        ax.set_xlabel("循环次数")
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH")
    fig.suptitle("Figure 1  40 块训练电池的 SOH 轨迹（按充电策略分面）", fontsize=16, y=1.01)
    fig.text(0.5, 0.005, "显示用 cutoff 内稳健处理仅抑制 battery 1 cycle 12 单点异常；原始数据未改写。", ha="center", fontsize=9, color="#5F6B76")
    fig.tight_layout()
    finish(fig, "figure_01_train_trajectories.png")


def figure_02(summary: pd.DataFrame, cycles: pd.DataFrame, predictions: pd.DataFrame) -> None:
    test_ids = summary.loc[summary["prediction_test"].eq(1), "battery_id"].astype(int).tolist()
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey=True)
    for ax, battery_id in zip(axes.flat, test_ids):
        history = cycles.loc[cycles["battery_id"].eq(battery_id)].sort_values("cycle")
        future = predictions.loc[predictions["battery_id"].eq(battery_id)].sort_values("cycle")
        ax.plot(history["cycle"], history["SOH"], color=BLUE, linewidth=1.5, label="已观测")
        ax.plot(future["cycle"], future["SOH_pred"], color=ORANGE, linewidth=2.0, label="预测")
        ax.fill_between(
            future["cycle"].to_numpy(),
            future["SOH_lower"].to_numpy(),
            future["SOH_upper"].to_numpy(),
            color=GOLD,
            alpha=0.22,
            label="经验 90% 区间",
        )
        policy = summary.loc[summary["battery_id"].eq(battery_id), "policy"].iloc[0]
        ax.set_title(f"Battery {battery_id} | {policy}", fontsize=9.5)
        ax.axvline(150, color="#6B7280", linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.75)
    for ax in axes[-1, :]:
        ax.set_xlabel("循环次数")
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Figure 2  9 块测试电池：cycle 1–150 观测与 151–200 预测", fontsize=16, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    finish(fig, "figure_02_test_predictions.png")


def figure_03(cv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    scatter = ax.scatter(
        cv["y_true"],
        cv["SOH_pred"],
        c=cv["horizon"],
        cmap="cividis",
        s=17,
        alpha=0.62,
        linewidths=0,
    )
    lower = min(cv["y_true"].min(), cv["SOH_pred"].min())
    upper = max(cv["y_true"].max(), cv["SOH_pred"].max())
    ax.plot([lower, upper], [lower, upper], color=INK, linestyle="--", linewidth=1.4, label="理想线")
    rmse = np.sqrt(np.mean((cv["SOH_pred"] - cv["y_true"]) ** 2))
    ax.text(0.03, 0.96, f"LOBO RMSE = {rmse:.6f}\nn = {cv['battery_id'].nunique()} batteries", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlabel("真实 SOH")
    ax.set_ylabel("预测 SOH")
    ax.set_title("Figure 3  150→151–200 严格电池级回测：真实值 vs 预测值")
    ax.grid(True, alpha=0.7)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("预测 horizon")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    finish(fig, "figure_03_cv_true_vs_predicted.png")


def figure_04(cv: pd.DataFrame) -> None:
    ensemble = pd.read_csv(ROOT / "problem3" / "agents" / "agent_06_ensemble" / "predictions.csv")
    ensemble = ensemble.loc[ensemble["cutoff"].eq(150), ["battery_id", "horizon", "pred_adaptive"]]
    frame = cv.merge(ensemble, on=["battery_id", "horizon"], validate="one_to_one")
    series = {
        "Final Hybrid": "SOH_pred",
        "Power(50–150)": "pred_power",
        "Linear-50": "pred_linear_50",
        "Adaptive Ensemble": "pred_adaptive",
    }
    colors = [INK, ORANGE, BLUE, PINK]
    styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    for (label, column), color, style in zip(series.items(), colors, styles):
        values = frame.groupby("horizon").apply(
            lambda group: np.sqrt(np.mean((group[column] - group["y_true"]) ** 2)),
            include_groups=False,
        )
        ax.plot(values.index, values.values, label=label, color=color, linestyle=style, linewidth=2.0)
    ax.set_xlabel("预测 horizon（cycles）")
    ax.set_ylabel("RMSE")
    ax.set_title("Figure 4  RMSE 随预测 horizon 的变化（cutoff=150）")
    ax.grid(True, alpha=0.75)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    finish(fig, "figure_04_rmse_vs_horizon.png")


def figure_05() -> None:
    route = pd.read_csv(ROOT / "problem3" / "judge" / "route_metrics.csv")
    baseline = pd.read_csv(ROOT / "problem3" / "common" / "baseline_metrics.csv")
    baseline = baseline.loc[baseline["model"].eq("Linear-Nested"), ["cutoff", "overall_RMSE"]]
    chosen = {
        "Nested Linear": baseline.set_index("cutoff")["overall_RMSE"],
        "SOH-only Tree ML": route.loc[route["agent"].eq("Agent 03 Tree ML")].set_index("cutoff")["overall_RMSE"],
        "Adaptive Ensemble": route.loc[route["agent"].eq("Agent 06 Ensemble")].set_index("cutoff")["overall_RMSE"],
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for (label, values), color, marker in zip(chosen.items(), [BLUE, OLIVE, PINK], ["o", "s", "^"]):
        ax.plot(values.index, values.values, marker=marker, linewidth=2.0, color=color, label=label)
    ax.scatter([150], [0.000495239144964], s=90, color=ORANGE, edgecolor=INK, zorder=5, label="Final Hybrid (N=150)")
    ax.set_xticks([50, 75, 100, 125, 150])
    ax.set_xlabel("可用早期循环数 N")
    ax.set_ylabel("未来 50 cycles RMSE")
    ax.set_title("Figure 5  早期数据长度与 50-cycle 预测误差")
    ax.grid(True, alpha=0.75)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    finish(fig, "figure_05_rmse_vs_cutoff.png")


def figure_06() -> None:
    leaderboard = pd.read_csv(ROOT / "problem3" / "judge" / "leaderboard.csv").sort_values("judge_score")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    bars = ax.barh(leaderboard["agent"], leaderboard["judge_score"], color=BLUE, edgecolor=INK, linewidth=0.5)
    for bar, value in zip(bars, leaderboard["judge_score"]):
        ax.text(value + 0.7, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Judge 综合得分")
    ax.set_title("Figure 6  六 Agent 统一 leaderboard")
    ax.grid(axis="x", alpha=0.75)
    fig.tight_layout()
    finish(fig, "figure_06_agent_leaderboard.png")


def figure_07() -> None:
    importance = pd.read_csv(ROOT / "problem3" / "agents" / "agent_03_ml" / "feature_importance.csv")
    top = importance.sort_values("mean_RMSE_increase", ascending=False).head(12).sort_values("mean_RMSE_increase")
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.barh(top["feature"], top["mean_RMSE_increase"], color=OLIVE, edgecolor=INK, linewidth=0.5)
    ax.set_xlabel("置换后 RMSE 增量（越大越重要）")
    ax.set_title("Figure 7  外层安全 permutation importance（Top 12）")
    ax.grid(axis="x", alpha=0.75)
    fig.tight_layout()
    finish(fig, "figure_07_feature_importance.png")


def figure_08(cycles: pd.DataFrame, cv: pd.DataFrame) -> None:
    battery_rmse = (
        cv.assign(se=(cv["SOH_pred"] - cv["y_true"]) ** 2)
        .groupby("battery_id")["se"]
        .mean()
        .pow(0.5)
        .sort_values()
    )
    battery_id = int(battery_rmse.index[len(battery_rmse) // 2])
    history = cycles.loc[cycles["battery_id"].eq(battery_id) & cycles["cycle"].le(150)].sort_values("cycle")
    future = cv.loc[cv["battery_id"].eq(battery_id)].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    ax.plot(history["cycle"], history["SOH"], color=BLUE, linewidth=1.5, label="已观测 1–150")
    ax.plot(future["cycle"], future["y_true"], color=INK, linewidth=1.6, label="真实 151–200")
    ax.plot(future["cycle"], future["SOH_pred"], color=ORANGE, linewidth=2.2, label="预测")
    ax.fill_between(future["cycle"], future["SOH_lower"], future["SOH_upper"], color=GOLD, alpha=0.24, label="经验 90% 区间")
    ax.axvline(150, color="#6B7280", linestyle="--", linewidth=1.0)
    ax.set_xlabel("循环次数")
    ax.set_ylabel("SOH")
    ax.set_title(f"Figure 8  代表性 Battery {battery_id} 的预测区间（battery RMSE={battery_rmse.loc[battery_id]:.6f}）")
    ax.grid(True, alpha=0.75)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    finish(fig, "figure_08_representative_interval.png")


def figure_09(eol: pd.DataFrame) -> None:
    ordered = eol.sort_values("battery_id")
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for column, label, color, marker in (
        ("EOL_linear", "Linear", BLUE, "o"),
        ("EOL_power", "Power", ORANGE, "s"),
        ("EOL_alternative", "Exponential", PINK, "^"),
    ):
        ax.scatter(x, ordered[column], label=label, color=color, marker=marker, s=55)
        ax.plot(x, ordered[column], color=color, alpha=0.42, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(value) for value in ordered["battery_id"]])
    ax.set_xlabel("测试 battery_id")
    ax.set_ylabel("预测 EOL cycle（对数尺度）")
    ax.set_title("Figure 9  SOH=0.8 的函数族敏感性（无真实 EOL 监督）")
    ax.grid(True, which="both", alpha=0.7)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    finish(fig, "figure_09_eol_model_sensitivity.png")


def figure_10(summary: pd.DataFrame, cycles: pd.DataFrame) -> None:
    test = summary.loc[summary["prediction_test"].eq(1)].copy()
    train = summary.loc[summary["prediction_test"].eq(0)].copy()
    peer_counts = test["policy"].map(train["policy"].value_counts())
    target_id = int(test.loc[peer_counts.idxmax(), "battery_id"])
    policy = str(test.loc[test["battery_id"].eq(target_id), "policy"].iloc[0])
    peer_ids = train.loc[train["policy"].eq(policy), "battery_id"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    peer_values = []
    for peer_id in peer_ids:
        frame = cycles.loc[cycles["battery_id"].eq(peer_id)].sort_values("cycle")
        anchor = float(frame.loc[frame["cycle"].eq(150), "SOH"].iloc[0])
        relative = frame["SOH"].to_numpy() - anchor
        peer_values.append(relative)
        ax.plot(frame["cycle"], relative, color=GREY, alpha=0.55, linewidth=1.0)
    mean_peer = np.mean(np.vstack(peer_values), axis=0)
    ax.plot(np.arange(1, 201), mean_peer, color=ORANGE, linewidth=2.2, label=f"同策略训练同伴均值 (n={len(peer_ids)})")
    target = cycles.loc[cycles["battery_id"].eq(target_id)].sort_values("cycle")
    target_anchor = float(target.loc[target["cycle"].eq(150), "SOH"].iloc[0])
    ax.plot(target["cycle"], target["SOH"] - target_anchor, color=BLUE, linewidth=2.2, label=f"测试 Battery {target_id}（仅至150）")
    ax.axvline(150, color=INK, linestyle="--", linewidth=1.0)
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_xlabel("循环次数")
    ax.set_ylabel("相对 cycle 150 的 ΔSOH")
    ax.set_title(f"Figure 10  同策略 cohort 轨迹比较：{policy}")
    ax.grid(True, alpha=0.75)
    ax.legend(frameon=False)
    fig.tight_layout()
    finish(fig, "figure_10_same_policy_cohort.png")


def write_chart_map(font_name: str) -> None:
    rows = [
        (1, "训练轨迹", "Trend / 3x3 small multiples", "40 training batteries; policy facets"),
        (2, "测试预测", "Uncertainty / 3x3 small multiples", "9 test batteries; observed, prediction, interval"),
        (3, "CV一致性", "Relationship / scatter", "2,000 cutoff-150 OOF points"),
        (4, "horizon误差", "Trend / multi-series line", "RMSE by horizons 1-50"),
        (5, "早期长度", "Trend / line with markers", "cutoff 50/75/100/125/150"),
        (6, "Agent评审", "Comparison / ranked horizontal bar", "six routes"),
        (7, "特征重要性", "Ranking / horizontal bar", "outer-safe permutation importance"),
        (8, "区间示例", "Uncertainty / line and band", "median-RMSE training battery"),
        (9, "EOL敏感性", "Uncertainty / log-scale point-line", "three degradation families"),
        (10, "同策略迁移", "Cohort / highlighted line", "one test battery and same-policy peers"),
    ]
    frame = pd.DataFrame(rows, columns=["figure", "question", "chart", "data_scope"])
    text = "# Figure chart map\n\n" + frame.to_markdown(index=False) + f"\n\n- 字体：{font_name}\n- 输出：240 dpi PNG，白底，静态论文图。\n- 颜色不是唯一编码；关键系列同时使用线型、标记或直接标签。\n"
    (Path(__file__).resolve().parent / "chart_map.md").write_text(text, encoding="utf-8")


def generate_all_figures(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    cv: pd.DataFrame,
    eol: pd.DataFrame,
    eol_detail: pd.DataFrame,
) -> None:
    del eol_detail
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_style()
    figure_01(summary, cycles)
    figure_02(summary, cycles, predictions)
    figure_03(cv)
    figure_04(cv)
    figure_05()
    figure_06()
    figure_07()
    figure_08(cycles, cv)
    figure_09(eol)
    figure_10(summary, cycles)
    write_chart_map(font_name)
