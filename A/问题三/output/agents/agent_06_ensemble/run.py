from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problem3.agents.agent_06_ensemble.model import (  # noqa: E402
    EXPERTS,
    GATE_PRIOR_BLEND,
    GATE_RIDGE_ALPHA,
    GATE_TEMPERATURE,
    HORIZON_BINS,
    INNER_SPLITS,
    META_SPLITS,
    ML_MAX_DEPTH,
    ML_MIN_SAMPLES_LEAF,
    ML_TREES,
    SCALE_FLOOR,
    run_nested_backtest,
)
from problem3.common.data import load_data, split_battery_ids  # noqa: E402
from problem3.common.metrics import detailed_metric_tables  # noqa: E402
from problem3.common.validation import EARLY_CUTOFFS, SEED  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent
PREDICTION_COLUMNS = {
    "Expert Linear-20": "pred_linear20",
    "Expert Linear-30": "pred_linear30",
    "Expert Linear-50": "pred_linear50",
    "Expert Same-policy": "pred_same_policy",
    "Expert Feature-ML": "pred_feature_ml",
    "Expert GP": "pred_gp",
    "Ensemble Global": "pred_global",
    "Ensemble Horizon": "pred_horizon",
    "Ensemble Adaptive": "pred_adaptive",
}


def uncertainty_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff, cutoff_frame in predictions.groupby("cutoff"):
        for variant in ("global", "horizon", "adaptive"):
            for level in (90, 95):
                lower = f"lower_{level}_{variant}"
                upper = f"upper_{level}_{variant}"
                inside = cutoff_frame["y_true"].between(
                    cutoff_frame[lower], cutoff_frame[upper], inclusive="both"
                )
                battery_path = inside.groupby(cutoff_frame["battery_id"]).all()
                rows.append(
                    {
                        "cutoff": int(cutoff),
                        "ensemble": variant,
                        "nominal_level": level / 100,
                        "point_coverage": float(inside.mean()),
                        "full_50_cycle_path_coverage": float(battery_path.mean()),
                        "mean_interval_width": float(
                            (cutoff_frame[upper] - cutoff_frame[lower]).mean()
                        ),
                        "median_interval_width": float(
                            (cutoff_frame[upper] - cutoff_frame[lower]).median()
                        ),
                        "calibration_batteries_per_outer_fold": 39,
                        "calibration_note": (
                            "battery-grouped meta cross-fit; max normalized residual calibrated "
                            "separately in five 10-cycle horizon bins"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def format_metric_table(overall: pd.DataFrame, cutoff: int) -> str:
    columns = [
        "model",
        "MAE",
        "overall_RMSE",
        "mean_battery_RMSE",
        "median_battery_RMSE",
        "worst_battery_RMSE",
    ]
    table = overall.loc[overall["cutoff"].eq(cutoff), columns].sort_values("overall_RMSE")
    return table.to_markdown(index=False, floatfmt=".7f")


def report_text(
    overall: pd.DataFrame,
    policy: pd.DataFrame,
    horizon: pd.DataFrame,
    uncertainty: pd.DataFrame,
    weights: pd.DataFrame,
    baseline: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    at_150 = overall.loc[overall["cutoff"].eq(150)].sort_values("overall_RMSE")
    best = at_150.iloc[0]
    ensemble_150 = at_150.loc[at_150["model"].str.startswith("Ensemble")]
    best_ensemble = ensemble_150.iloc[0]
    baseline_150 = baseline.loc[baseline["cutoff"].eq(150)].sort_values("overall_RMSE")
    best_baseline = baseline_150.iloc[0]
    adaptive_uncertainty = uncertainty.loc[
        uncertainty["cutoff"].eq(150) & uncertainty["ensemble"].eq("adaptive")
    ]
    mean_weights = (
        weights.loc[weights["cutoff"].eq(150)]
        .groupby(["ensemble", "horizon_bin", "expert"], as_index=False)["weight"]
        .mean()
    )
    horizon_150 = horizon.loc[
        horizon["cutoff"].eq(150)
        & horizon["model"].isin(["Ensemble Global", "Ensemble Horizon", "Ensemble Adaptive"])
    ]
    policy_150 = policy.loc[
        policy["cutoff"].eq(150) & policy["model"].eq(str(best_ensemble["model"]))
    ].sort_values("RMSE")
    early_model = (
        overall.loc[overall["model"].eq(str(best_ensemble["model"])), ["cutoff", "MAE", "overall_RMSE"]]
        .sort_values("cutoff")
    )
    early_baseline = (
        baseline.sort_values("overall_RMSE")
        .groupby("cutoff", as_index=False)
        .first()[["cutoff", "model", "overall_RMSE"]]
        .rename(
            columns={
                "model": "best_public_baseline",
                "overall_RMSE": "baseline_RMSE",
            }
        )
    )
    early = early_model.merge(early_baseline, on="cutoff")
    early["RMSE_reduction_vs_baseline_pct"] = 100 * (
        early["baseline_RMSE"] - early["overall_RMSE"]
    ) / early["baseline_RMSE"]
    baseline_gain = 100 * (
        float(best_baseline["overall_RMSE"]) - float(best_ensemble["overall_RMSE"])
    ) / float(best_baseline["overall_RMSE"])
    return f"""# Agent 06：混合专家与自适应集成

## 1. 结论摘要

在严格的外层 Leave-One-Battery-Out（LOBO）验证下，cycle 150→151–200 的最佳单项为 **{best['model']}**，overall RMSE 为 **{float(best['overall_RMSE']):.7f}**。三个集成中最佳的是 **{best_ensemble['model']}**（RMSE **{float(best_ensemble['overall_RMSE']):.7f}**）；公共基线最佳为 **{best_baseline['model']}**（RMSE **{float(best_baseline['overall_RMSE']):.7f}**），相对变化为 **{baseline_gain:+.2f}%**（正值表示集成降低 RMSE，负值表示未超过基线）。因此不以“集成”名称作为推荐依据；若集成没有稳定超过简单局部趋势，Judge 应保留更简单模型。

## 2. 数据边界与泄漏控制

- 数据源仅为赛题 PDF、`battery_summary.csv`、`cycle_train.csv`、公共审计和公共基线；未读取其他研究 Agent 的目录或结果，也未检索测试电池的外部未来数据。
- 9 块 `prediction_test==1` 电池完全未进入本报告的训练、调参或验证。40 块训练电池均按 battery_id 做外层 LOBO。
- 每个外层训练集内再进行 3 折 battery-grouped cross-fitting，生成权重学习所需的专家 OOF 预测。feature-ML 与 same-policy expert 在每个内层折均排除该折电池及外层目标电池的未来。
- 区间校准又对内层 OOF 预测做 3 折 battery-grouped 元层 cross-fitting，避免用拟合权重的同一条电池误差直接校准自身区间。
- cutoff=N 的全部动态特征、清洗、GP 拟合与相似度只读取 `cycle<=N`；没有使用汇总表中的动态均值。

## 3. 六个独立 Expert

1. recent 20-cycle linear；
2. recent 30-cycle linear；
3. recent 50-cycle linear；
4. exact same-policy mean future delta；若内层参考池中没有同策略电池才退化为全体参考均值；
5. Extra Trees 多步回归，输入 cutoff 内 SOH、IR、温度、充电时间和策略物理特征，并以 horizon 为输入预测相对末次观测的 SOH 增量；
6. linear mean + Matérn-3/2 GP residual。GP 使用固定的小规模核参数，避免在 40 块电池上做无意义的大搜索。

## 4. 集成方法

设六个专家输出为 $\\hat y_k$。global 与 10-cycle horizon-bin 权重通过带约束最小二乘求解：

$$
\\min_{{w_k}} \\sum( y-\\sum_k w_k\\hat y_k)^2,
\\qquad w_k\\ge 0,\\quad \\sum_k w_k=1.
$$

自适应门控以 recent slope、curvature、局部残差波动、IR/温度/充电时间趋势、策略参数和同伴相似度预测各专家在五个 horizon bin 的 log-RMSE，再转为 softmax 权重，并与内层 horizon 权重各 50% 混合以降低小样本门控的不稳定性。整个门控训练仅用外层训练电池的 OOF 结果。

## 5. cycle 150 的统一指标

{format_metric_table(overall, 150)}

公共基线（相同 SOH 目标和 40 块训练电池）：

{baseline_150[['model','MAE','overall_RMSE','mean_battery_RMSE','median_battery_RMSE','worst_battery_RMSE']].to_markdown(index=False, floatfmt='.7f')}

最佳集成的分策略误差：

{policy_150[['policy','n_batteries','RMSE','MAE']].to_markdown(index=False, floatfmt='.7f')}

三个集成的 horizon-bin 误差：

{horizon_150[['model','horizon_bin','RMSE','MAE']].to_markdown(index=False, floatfmt='.7f')}

## 6. 早期数据长度

以下表格固定使用 cycle 150 时表现最好的集成类型 `{best_ensemble['model']}`，避免按每个 cutoff 重新挑选赢家：

{early.to_markdown(index=False, floatfmt='.7f')}

这组结果只表示当前 50-cycle 预测任务在不同 cutoff 的回测表现；不应把 cutoff 变长与训练电池老化阶段变化混为严格因果效应。Adaptive 在 cutoff 75 和 100 未超过当时最佳公共线性基线，说明其跨 cutoff 优势并不稳定；本 Agent 仅推荐把它保留为 cycle 150 的候选，而不是无条件替代局部线性模型。

## 7. 权重与自适应性

cycle 150 的外层折平均权重：

{mean_weights.to_markdown(index=False, floatfmt='.5f')}

`weights.csv` 保留每个外层目标、集成类型、horizon bin 与 expert 的完整权重，可检查非负性及和为 1。adaptive 权重在预测目标时只由截止时点特征决定，不会先查看该电池的 151–200 误差再选专家。

## 8. 不确定性

区间基础尺度由加权 expert disagreement、GP posterior standard deviation 和数值稳定下限共同构成。随后在元层 battery-grouped OOF 残差上，对每个 10-cycle horizon bin 的“单电池最大标准化残差”做有限样本 90%/95% conformal quantile 校准。

cycle 150 的 adaptive 区间表现：

{adaptive_uncertainty.to_markdown(index=False, floatfmt='.5f')}

这里的 coverage 是 40 块训练电池的外层回测经验覆盖率，不是对 9 块测试电池的严格概率保证。策略组最小仅 2 块训练电池，且各 horizon 内同一电池误差相关；因此样本不足以声称有限样本条件在测试分布上完全成立。`full_50_cycle_path_coverage` 只是描述性指标，校准实际按五个 10-cycle bin 分别进行。

## 9. 复现配置

- seed：{SEED}
- inner/meta battery folds：{INNER_SPLITS}/{META_SPLITS}
- Extra Trees：n_estimators={ML_TREES}, max_depth={ML_MAX_DEPTH}, min_samples_leaf={ML_MIN_SAMPLES_LEAF}, max_features=0.80
- adaptive gate：Ridge alpha={GATE_RIDGE_ALPHA}, temperature={GATE_TEMPERATURE}, horizon-prior blend={GATE_PRIOR_BLEND}
- interval scale floor：{SCALE_FLOOR}
- 复杂度：global 6 个权重（5 自由度）；horizon 30 个权重（25 自由度）；最终 adaptive gate 为 21 输入×30 输出加 30 个截距，共 660 个 Ridge 系数；Extra Trees 固定 40 棵、深度上限 7，节点数随外层折变化，不伪报固定“参数量”；GP 不优化核超参数，每次最多拟合 50 个残差观测。
- 总运行时间：{runtime_seconds:.1f} s

## 10. 局限性与建议

- 这是一套 short-horizon 模型比较，不输出 0.8 EOL。当前 CSV 没有任何真实 SOH≤0.8；把本模型无限递推到 0.8 会混淆短期预测与长期外推。
- same-policy 组仅 2–7 块训练电池，分策略 RMSE 波动大；不能把组均值差异解释为充电策略的因果效应。
- feature-ML 和 adaptive gate 的有效独立样本单位是 battery，而不是 50 个 horizon rows；模型复杂度已据此限制。
- Judge 应依据完整 LOBO、worst-battery RMSE 与跨 cutoff 稳定性决定是否保留 ensemble；若相近，应优先 recent-linear。

## 11. 已核验方法文献

1. Severson, K. A., Attia, P. M., Jin, N., et al. *Data-driven prediction of battery cycle life before capacity degradation*. **Nature Energy**, 4, 383–391 (2019). DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)。用于早期循环特征与小样本寿命预测的设计依据。
2. Richardson, R. R., Osborne, M. A., & Howey, D. A. *Gaussian process regression for forecasting battery state of health*. **Journal of Power Sources**, 357, 209–219 (2017). DOI: [10.1016/j.jpowsour.2017.05.004](https://doi.org/10.1016/j.jpowsour.2017.05.004)。用于显式均值函数与 GP 残差不确定性的设计依据。
3. Zhang, H., Li, Y., Zheng, S., et al. *Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning*. **Nature Machine Intelligence**, 7, 270–277 (2025). DOI: [10.1038/s42256-024-00972-x](https://doi.org/10.1038/s42256-024-00972-x)。用于 inter-cell / reference-cell 思路；本研究没有下载或使用其数据与测试真值。
"""


def main() -> None:
    started = time.perf_counter()
    summary, cycles = load_data()
    train_ids, _ = split_battery_ids(summary)
    predictions, weights = run_nested_backtest(
        summary, cycles, train_ids, list(EARLY_CUTOFFS)
    )
    overall, policy, horizon = detailed_metric_tables(predictions, PREDICTION_COLUMNS)
    uncertainty = uncertainty_metrics(predictions)
    runtime_seconds = time.perf_counter() - started

    baseline = pd.read_csv(PROJECT_ROOT / "problem3" / "common" / "baseline_metrics.csv")
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    weights.to_csv(OUTPUT_DIR / "weights.csv", index=False)
    uncertainty.to_csv(OUTPUT_DIR / "uncertainty_metrics.csv", index=False)

    package_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    payload = {
        "agent": "agent_06_ensemble",
        "seed": SEED,
        "runtime_seconds": runtime_seconds,
        "package_versions": package_versions,
        "model_complexity": {
            "global_weights": {"stored_coefficients": 6, "degrees_of_freedom": 5},
            "horizon_weights": {"stored_coefficients": 30, "degrees_of_freedom": 25},
            "adaptive_gate": {
                "maximum_input_features": 21,
                "outputs": 30,
                "ridge_coefficients_including_intercepts": 660,
            },
            "extra_trees": {
                "trees": ML_TREES,
                "max_depth": ML_MAX_DEPTH,
                "fixed_parameter_count": None,
                "note": "fitted node count varies by outer fold",
            },
            "gp": {
                "optimized_kernel_hyperparameters": 0,
                "maximum_training_observations_per_forecast": 50,
            },
        },
        "validation": {
            "outer": "Leave-One-Battery-Out over 40 training batteries",
            "inner": f"{INNER_SPLITS}-fold battery-grouped cross-fitting within each outer training set",
            "meta_calibration": f"{META_SPLITS}-fold battery-grouped cross-fitting over inner OOF expert predictions",
            "cutoffs": list(EARLY_CUTOFFS),
            "forecast_horizon": 50,
            "test_batteries_used": False,
        },
        "hyperparameters": {
            "experts": list(EXPERTS),
            "extra_trees": {
                "n_estimators": ML_TREES,
                "max_depth": ML_MAX_DEPTH,
                "min_samples_leaf": ML_MIN_SAMPLES_LEAF,
                "max_features": 0.80,
            },
            "gp": {
                "mean": "recent-50 linear",
                "kernel": "2e-7 * Matern(length_scale=12, nu=1.5) + WhiteKernel(2e-8)",
                "optimizer": None,
            },
            "adaptive_gate": {
                "model": "standardized multi-output Ridge on log expert RMSE",
                "alpha": GATE_RIDGE_ALPHA,
                "temperature": GATE_TEMPERATURE,
                "horizon_prior_blend": GATE_PRIOR_BLEND,
            },
            "interval": {
                "scale_floor": SCALE_FLOOR,
                "levels": [0.90, 0.95],
                "score": "battery maximum normalized residual within each 10-cycle horizon bin",
            },
        },
        "overall_metrics": json_records(overall),
        "policy_metrics": json_records(policy),
        "horizon_metrics": json_records(horizon),
        "uncertainty_metrics": json_records(uncertainty),
        "public_baseline_metrics": json_records(baseline),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "report.md").write_text(
        report_text(overall, policy, horizon, uncertainty, weights, baseline, runtime_seconds),
        encoding="utf-8",
    )
    print(
        overall.loc[overall["cutoff"].eq(150)]
        .sort_values("overall_RMSE")
        .to_string(index=False)
    )
    print(f"runtime_seconds={runtime_seconds:.1f}")


if __name__ == "__main__":
    main()
