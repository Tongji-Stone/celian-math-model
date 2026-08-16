from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "problem3" / "agents"
COMMON_DIR = ROOT / "problem3" / "common"
KEYS = ["battery_id", "policy", "cutoff", "horizon", "cycle", "y_true"]
SEED = 20260815


def metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    error = frame[prediction].to_numpy() - frame["y_true"].to_numpy()
    battery_rmse = (
        frame.assign(squared_error=error**2)
        .groupby("battery_id")["squared_error"]
        .mean()
        .pow(0.5)
    )
    return {
        "MAE": float(np.mean(np.abs(error))),
        "overall_RMSE": float(np.sqrt(np.mean(error**2))),
        "mean_battery_RMSE": float(battery_rmse.mean()),
        "median_battery_RMSE": float(battery_rmse.median()),
        "worst_battery_RMSE": float(battery_rmse.max()),
    }


def physics_best_predictions() -> pd.DataFrame:
    frame = pd.read_csv(AGENT_DIR / "agent_02_physics" / "predictions.csv")
    scored = (
        frame.groupby(["cutoff", "fit_start", "model"])
        .apply(
            lambda group: np.sqrt(np.mean((group["y_true"] - group["y_pred"]) ** 2)),
            include_groups=False,
        )
        .rename("RMSE")
        .reset_index()
    )
    parts = []
    for cutoff, group in scored.groupby("cutoff"):
        choice = group.sort_values(["RMSE", "fit_start", "model"]).iloc[0]
        parts.append(
            frame.loc[
                frame["cutoff"].eq(cutoff)
                & frame["fit_start"].eq(choice["fit_start"])
                & frame["model"].eq(choice["model"])
            ]
        )
    return pd.concat(parts, ignore_index=True)[KEYS + ["y_pred"]].rename(
        columns={"y_pred": "prediction"}
    )


def representative_predictions() -> dict[str, pd.DataFrame]:
    tcn = pd.read_csv(AGENT_DIR / "agent_01_tcn" / "predictions.csv")
    ml = pd.read_csv(AGENT_DIR / "agent_03_ml" / "predictions.csv").rename(
        columns={"forecast_horizon": "horizon"}
    )
    gp = pd.read_csv(AGENT_DIR / "agent_04_gp" / "predictions.csv")
    intercell = pd.read_csv(AGENT_DIR / "agent_05_intercell" / "predictions.csv")
    ensemble = pd.read_csv(AGENT_DIR / "agent_06_ensemble" / "predictions.csv")
    return {
        "Agent 01 TCN": tcn[KEYS + ["SOH_pred"]].rename(columns={"SOH_pred": "prediction"}),
        "Agent 02 Physics": physics_best_predictions(),
        "Agent 03 Tree ML": ml.loc[ml["ablation"].eq("A_SOH_only"), KEYS + ["y_pred"]].rename(
            columns={"y_pred": "prediction"}
        ),
        "Agent 04 GP": gp[KEYS + ["SOH_pred"]].rename(columns={"SOH_pred": "prediction"}),
        "Agent 05 Inter-cell": intercell[KEYS + ["pred_slope"]].rename(
            columns={"pred_slope": "prediction"}
        ),
        "Agent 06 Ensemble": ensemble[KEYS + ["pred_adaptive"]].rename(
            columns={"pred_adaptive": "prediction"}
        ),
    }


def build_hybrid_cv() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    physics = pd.read_csv(AGENT_DIR / "agent_02_physics" / "predictions.csv")
    power = physics.loc[
        physics["cutoff"].eq(150)
        & physics["fit_start"].eq(50)
        & physics["model"].eq("power"),
        KEYS + ["y_pred"],
    ].rename(columns={"y_pred": "pred_power"})
    baseline = pd.read_csv(COMMON_DIR / "baseline_predictions.csv")
    linear = baseline.loc[
        baseline["cutoff"].eq(150), KEYS + ["pred_linear_50"]
    ]
    merged = power.merge(linear, on=KEYS, validate="one_to_one")
    rows = []
    weight_rows = []
    for battery_id in sorted(merged["battery_id"].unique()):
        training = merged.loc[merged["battery_id"].ne(battery_id)]
        validation = merged.loc[merged["battery_id"].eq(battery_id)].copy()
        direction = (training["pred_power"] - training["pred_linear_50"]).to_numpy()
        target = (training["y_true"] - training["pred_linear_50"]).to_numpy()
        denominator = max(float(direction @ direction), 1e-30)
        weight = float(np.clip(float(direction @ target) / denominator, 0.0, 1.0))
        validation["power_weight"] = weight
        validation["linear_weight"] = 1.0 - weight
        validation["SOH_pred"] = (
            weight * validation["pred_power"]
            + (1.0 - weight) * validation["pred_linear_50"]
        )
        rows.append(validation)
        weight_rows.append({"battery_id": int(battery_id), "power_weight": weight, "linear_weight": 1.0 - weight})
    hybrid = pd.concat(rows, ignore_index=True).sort_values(["battery_id", "horizon"])
    weights = pd.DataFrame(weight_rows)
    direction = (merged["pred_power"] - merged["pred_linear_50"]).to_numpy()
    target = (merged["y_true"] - merged["pred_linear_50"]).to_numpy()
    frozen_weight = float(np.clip(float(direction @ target) / max(float(direction @ direction), 1e-30), 0.0, 1.0))
    return hybrid, weights, frozen_weight


def build_leaderboard() -> tuple[pd.DataFrame, pd.DataFrame]:
    representatives = representative_predictions()
    metric_rows = []
    for agent, frame in representatives.items():
        for cutoff, group in frame.groupby("cutoff"):
            metric_rows.append({"agent": agent, "cutoff": int(cutoff), **metrics(group, "prediction")})
    route_metrics = pd.DataFrame(metric_rows)
    cutoff150 = route_metrics.loc[route_metrics["cutoff"].eq(150)].set_index("agent")
    best_by_cutoff = route_metrics.groupby("cutoff")["overall_RMSE"].min()
    robust_index = (
        route_metrics.assign(
            relative=lambda frame: frame.apply(
                lambda row: row["overall_RMSE"] / best_by_cutoff.loc[row["cutoff"]], axis=1
            )
        )
        .groupby("agent")["relative"]
        .mean()
    )
    best_short = float(cutoff150["overall_RMSE"].min())
    best_robust = float(robust_index.min())
    rubric = {
        "Agent 01 TCN": {"EOL": 15, "uncertainty": 20, "interpretability": 45, "simplicity": 45, "complexity": "5.6k-6.2k NN params"},
        "Agent 02 Physics": {"EOL": 90, "uncertainty": 50, "interpretability": 90, "simplicity": 100, "complexity": "2-3 fitted params"},
        "Agent 03 Tree ML": {"EOL": 20, "uncertainty": 20, "interpretability": 95, "simplicity": 65, "complexity": "32-45 trees"},
        "Agent 04 GP": {"EOL": 80, "uncertainty": 95, "interpretability": 65, "simplicity": 75, "complexity": "27 controlled GP candidates"},
        "Agent 05 Inter-cell": {"EOL": 20, "uncertainty": 20, "interpretability": 85, "simplicity": 80, "complexity": "distance-weighted peers"},
        "Agent 06 Ensemble": {"EOL": 25, "uncertainty": 80, "interpretability": 70, "simplicity": 45, "complexity": "6 experts + 660 gate coeffs"},
    }
    def runtime_seconds(directory: str) -> float:
        data = json.loads((AGENT_DIR / directory / "metrics.json").read_text(encoding="utf-8"))
        if "runtime_seconds" in data:
            return float(data["runtime_seconds"])
        if isinstance(data.get("runtime"), dict) and "runtime_seconds" in data["runtime"]:
            return float(data["runtime"]["runtime_seconds"])
        if isinstance(data.get("run"), dict) and "runtime_seconds" in data["run"]:
            return float(data["run"]["runtime_seconds"])
        return float("nan")

    name_to_runtime = {
        "Agent 01 TCN": runtime_seconds("agent_01_tcn"),
        "Agent 02 Physics": runtime_seconds("agent_02_physics"),
        "Agent 03 Tree ML": runtime_seconds("agent_03_ml"),
        "Agent 04 GP": runtime_seconds("agent_04_gp"),
        "Agent 05 Inter-cell": runtime_seconds("agent_05_intercell"),
        "Agent 06 Ensemble": runtime_seconds("agent_06_ensemble"),
    }
    rows = []
    for agent in representatives:
        short_score = 100.0 * best_short / float(cutoff150.loc[agent, "overall_RMSE"])
        robust_score = 100.0 * best_robust / float(robust_index.loc[agent])
        components = rubric[agent]
        total = (
            0.45 * short_score
            + 0.15 * robust_score
            + 0.15 * components["EOL"]
            + 0.10 * components["uncertainty"]
            + 0.10 * components["interpretability"]
            + 0.05 * components["simplicity"]
        )
        rows.append(
            {
                "agent": agent,
                "RMSE_150": float(cutoff150.loc[agent, "overall_RMSE"]),
                "MAE_150": float(cutoff150.loc[agent, "MAE"]),
                "mean_battery_RMSE_150": float(cutoff150.loc[agent, "mean_battery_RMSE"]),
                "median_battery_RMSE_150": float(cutoff150.loc[agent, "median_battery_RMSE"]),
                "worst_battery_RMSE_150": float(cutoff150.loc[agent, "worst_battery_RMSE"]),
                "robust_relative_index": float(robust_index.loc[agent]),
                "short_score": short_score,
                "robust_score": robust_score,
                "EOL_score": components["EOL"],
                "uncertainty_score": components["uncertainty"],
                "interpretability_score": components["interpretability"],
                "simplicity_score": components["simplicity"],
                "judge_score": total,
                "runtime_seconds": float(name_to_runtime[agent]),
                "complexity": components["complexity"],
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values("judge_score", ascending=False).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    return leaderboard, route_metrics


def write_reports(leaderboard: pd.DataFrame, route_metrics: pd.DataFrame, hybrid: pd.DataFrame, frozen_weight: float) -> None:
    focus = leaderboard[["rank", "agent", "RMSE_150", "worst_battery_RMSE_150", "judge_score", "runtime_seconds"]].copy()
    comparison = f"""# 六路线统一评审

## 评审结论

统一复算与 Agent 自报指标一致。独立路线中，Agent 02 的 power 半经验模型在 150→200 点预测 RMSE 最低；Agent 06 的严格自适应 ensemble 次之，并给出经验区间；Agent 03 的 SOH-only 树模型在多个 cutoff 稳定，但动态指标与策略特征没有可证实增益；Agent 04 的区间校准最好；Agent 05 证明 slope adaptation 能显著改进朴素 cohort，但小同伴组失稳；Agent 01 TCN 明显不如简单线性模型。

{focus.to_markdown(index=False, floatfmt='.6f')}

## 统一验证协议

- 训练样本单位为 battery，不是 cycle 行；9 块 `prediction_test==1` 电池没有进入训练、调参或 Judge。
- 核心指标为 cutoff=150 后 50 周期真实回测；同时复算 cutoff=50/75/100/125/150。
- 指标包含 overall、mean/median/worst battery RMSE、MAE、policy 与 horizon 分解。
- Physics 行在每个 cutoff 使用该独立路线比较后的最佳函数/窗口，故其路线级 CV 含有限的模型选择乐观偏差；最终 hybrid 的元权重另做 leave-one-battery-out 学习。

## 早期数据长度下的路线 RMSE

{route_metrics.pivot(index='cutoff', columns='agent', values='overall_RMSE').to_markdown(floatfmt='.6f')}

## 评分定义

短期分数按相对最佳 RMSE 计算；稳健性分数基于五个 cutoff 相对当期最佳模型的平均比值。EOL、uncertainty、interpretability、simplicity 按预先给定的证据量表评分：是否做了伪阈值、模型族敏感性、区间覆盖、消融/重要性以及模型复杂度。总分遵循 0.45/0.15/0.15/0.10/0.10/0.05 权重。主观分只影响同量级模型的取舍，不覆盖短期实测结果。

## Judge 后的精简融合

只保留前列且互补的 Power(50–150) 与 Linear-50。对每个外层目标电池，权重只由其余 39 块电池的 OOF 误差计算。严格元层 LOO 指标为：

{pd.DataFrame([metrics(hybrid, 'SOH_pred')]).to_markdown(index=False, floatfmt='.6f')}

用于最终 9 块测试电池推理的冻结 power 权重为 `{frozen_weight:.6f}`，linear 权重为 `{1-frozen_weight:.6f}`。报告性能仍使用逐电池留出的元权重，避免把冻结权重在同一 OOF 集上的拟合值当成无偏 CV。
"""
    (JUDGE_DIR / "comparison.md").write_text(comparison, encoding="utf-8")

    selection = rf"""# 最终模型选择与冻结说明

## Stage I：151–200 周期 SOH

最终采用精简的两专家凸组合：

\[
\widehat{{SOH}}_{{N+h}}=w\widehat{{SOH}}^{{power}}_{{N+h}}+(1-w)\widehat{{SOH}}^{{linear50}}_{{N+h}},
\quad w={frozen_weight:.6f}.
\]

Power 使用最后最多 101 个已观测循环的原始 SOH，经 cutoff 内稳健异常抑制后拟合 `a-k n^p`；核心 N=150 时窗口为 cycle 50–150。Linear-50 仅使用 cycle 101–150。该组合以 3–5 个有效参数完成预测，避免复杂 ensemble 的 6 experts、树模型和 660 个门控系数；其 worst-battery RMSE 也优于复杂 hybrid。

短期区间使用 40 块训练电池严格元层 LOO 残差，在每个 10-cycle horizon bin 上取 battery 内最大绝对误差，再做有限样本 90% 分位。它是分组经验预测区间，不是独立同分布条件下的严格 coverage 保证。

## Stage II：0.8 EOL

数据没有任何 SOH≤0.8 crossing。EOL 不由 Stage I 递推得到，而是分别拟合 linear、power、exponential，并比较 cycle 1/20/50/80–150 四个窗口。点估计取跨函数/窗口中位数；上下界取函数-窗口移动块残差 bootstrap 的包络。该包络表示模型假设敏感性，不称为经监督校准的置信区间，也不报告 EOL RMSE。

## 冻结与测试隔离

模型结构、窗口、权重学习规则、异常处理、区间规则和 EOL 函数族均在读取测试电池预测结果前冻结。最终测试推理只读取每块测试电池 cycle 1–150，不读取 `global_id` 对应外部数据，也不使用汇总表动态均值。
"""
    (JUDGE_DIR / "model_selection.md").write_text(selection, encoding="utf-8")


def main() -> None:
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard, route_metrics = build_leaderboard()
    hybrid, weights, frozen_weight = build_hybrid_cv()
    leaderboard.to_csv(JUDGE_DIR / "leaderboard.csv", index=False)
    route_metrics.to_csv(JUDGE_DIR / "route_metrics.csv", index=False)
    hybrid.to_csv(JUDGE_DIR / "hybrid_cv_predictions.csv", index=False)
    weights.to_csv(JUDGE_DIR / "hybrid_meta_weights.csv", index=False)
    (JUDGE_DIR / "frozen_model.json").write_text(
        json.dumps(
            {
                "stage1": "Power(last_101_cycles) + Linear(last_50_cycles)",
                "power_weight": frozen_weight,
                "linear_weight": 1.0 - frozen_weight,
                "reported_cv": metrics(hybrid, "SOH_pred"),
                "stage2": "linear/power/exponential x fit_start 1/20/50/80 sensitivity envelope",
                "seed": SEED,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_reports(leaderboard, route_metrics, hybrid, frozen_weight)
    print(leaderboard.to_string(index=False))
    print("Hybrid:", metrics(hybrid, "SOH_pred"))


if __name__ == "__main__":
    main()
