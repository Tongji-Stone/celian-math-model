"""问题二：两阶段快充参数与 200 圈健康保持程度的统计分析。

运行：python src/A/task2_statistical_analysis.py

说明：赛题 CSV 未观测 SOH=0.8 的 EOL，因此本脚本仅分析 SOH_200、
200 圈容量保持率和早期衰减斜率；不将任何外推结果称为真实循环寿命。
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from SOH_plot import configure_chinese_font


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "A" / "question2"
FIG_DIR = OUTPUT_DIR / "figures"

POLICY_ORDER = [
    "4_8C_80PER_4_8C",
    "5C_67PER_4C",
    "5_3C_54PER_4C",
    "5_6C_19PER_4_6C",
    "5_6C_36PER_4_3C",
    "3_7C_31PER_5_9C",
]


def canonical_policy(value: str) -> str:
    """统一问题一已采用的策略命名规则。"""
    value = str(value).replace("_NEWSTRUCTURE", "")
    if value == "80PER_3_6C":
        return "3_6C-80PER_3_6C"
    return value


def holm_pairwise(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policies = [x for x in POLICY_ORDER if x in set(data["policy"])]
    for left, right in combinations(policies, 2):
        x = data.loc[data["policy"] == left, "SOH_200"]
        y = data.loc[data["policy"] == right, "SOH_200"]
        result = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
        rows.append(
            {
                "policy_left": left,
                "policy_right": right,
                "n_left": len(x),
                "n_right": len(y),
                "median_difference_left_minus_right": float(x.median() - y.median()),
                "U": float(result.statistic),
                "p_raw": float(result.pvalue),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = multipletests(result["p_raw"], method="holm")[1]
    return result


def policy_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("policy", observed=True)
        .agg(
            battery_count=("battery_id", "size"),
            SOH_200_mean=("SOH_200", "mean"),
            SOH_200_median=("SOH_200", "median"),
            SOH_200_std=("SOH_200", "std"),
            SOH_200_min=("SOH_200", "min"),
            SOH_200_max=("SOH_200", "max"),
            fade_200_mean=("fade_200", "mean"),
            fade_slope_mean=("fade_slope", "mean"),
            charge_time_mean=("charge_time_mean", "mean"),
        )
        .reindex([p for p in POLICY_ORDER if p in set(data["policy"])])
        .reset_index()
    )


def extract_cell_metrics(summary: pd.DataFrame, cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.copy()
    cycles = cycles.copy()
    summary["policy"] = summary["policy"].map(canonical_policy)
    cycles["policy"] = cycles["policy"].map(canonical_policy)

    max_cycle = cycles.groupby("battery_id")["cycle"].max().rename("max_cycle")
    cell = summary.merge(max_cycle, on="battery_id", validate="one_to_one")
    cell = cell.query("prediction_test == 0 and max_cycle == 200").copy()

    cycle_200 = cycles.loc[cycles["cycle"] == 200, ["battery_id", "SOH", "SOH_smooth"]].copy()
    cycle_200 = cycle_200.rename(columns={"SOH": "SOH_200_raw", "SOH_smooth": "SOH_200"})
    cycle_1 = cycles.loc[cycles["cycle"] == 1, ["battery_id", "SOH_smooth"]].rename(
        columns={"SOH_smooth": "SOH_1"}
    )
    cell = cell.merge(cycle_1, on="battery_id", validate="one_to_one").merge(
        cycle_200, on="battery_id", validate="one_to_one"
    )
    cell["fade_200"] = cell["SOH_1"] - cell["SOH_200"]
    cell["fade_slope"] = cell["fade_200"] / 199.0
    cell["charge_time_mean"] = cell["mean_chargetime"]

    # 主分析：同一实验结构（dataset 3）；全体完整样本仅作敏感性对照。
    primary = cell.query("dataset_id == 3").copy()
    longitudinal = cycles.merge(
        primary[["battery_id", "policy"]], on=["battery_id", "policy"], how="inner", validate="many_to_one"
    ).copy()
    longitudinal["cycle_scaled"] = longitudinal["cycle"] / 100.0
    return cell, primary, longitudinal


def save_figures(primary: pd.DataFrame, longitudinal: pd.DataFrame, model) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    configure_chinese_font()

    order = [p for p in POLICY_ORDER if p in set(primary["policy"])]
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=primary, x="policy", y="SOH_200", order=order, ax=ax, color="#9ecae1", fliersize=0)
    sns.stripplot(data=primary, x="policy", y="SOH_200", order=order, ax=ax, color="#1f1f1f", size=5, jitter=0.15)
    ax.set(title="同一实验结构下各策略的第 200 圈 SOH", xlabel="两阶段快充策略", ylabel="SOH at cycle 200")
    ax.tick_params(axis="x", rotation=22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "policy_soh200_boxplot.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for policy in order:
        group = longitudinal.loc[longitudinal["policy"] == policy]
        agg = group.groupby("cycle", as_index=False)["SOH_smooth"].agg(["mean", "sem"]).reset_index()
        x = agg["cycle"].to_numpy()
        y = agg["mean"].to_numpy()
        sem = agg["sem"].fillna(0).to_numpy()
        ax.plot(x, y, linewidth=2, label=policy)
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, alpha=0.12)
    ax.set(title="同一实验结构下的 SOH 轨迹（均值 ± 95% 近似区间）", xlabel="Cycle", ylabel="Smoothed SOH")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "policy_soh_trajectories.png", dpi=300)
    plt.close(fig)

    residuals = model.resid
    fitted = model.fittedvalues
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(fitted, residuals, color="#3182bd", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="参数回归残差图", xlabel="Fitted fade at cycle 200", ylabel="Residual")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("参数回归 Q-Q 图")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "parameter_regression_diagnostics.png", dpi=300)
    plt.close(fig)


def nested_loo_models(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare collinearity-resistant models with battery-level nested LOOCV."""
    features = ["C1", "Q1", "C2"]
    X = primary[features].to_numpy(dtype=float)
    y = primary["fade_200"].to_numpy(dtype=float)
    outer = LeaveOneOut()
    predictions = {name: np.zeros(len(primary)) for name in ("OLS", "Ridge", "PLS_1component", "PLS_2components")}
    alphas = np.logspace(-6, 3, 80)
    for train_idx, test_idx in outer.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        # Scaling and penalty selection occur inside every training fold.
        models = {
            "OLS": make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-12], cv=LeaveOneOut(), scoring="neg_mean_squared_error")),
            "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=LeaveOneOut(), scoring="neg_mean_squared_error")),
            "PLS_1component": make_pipeline(StandardScaler(), PLSRegression(n_components=1)),
            "PLS_2components": make_pipeline(StandardScaler(), PLSRegression(n_components=2)),
        }
        for name, fitted_model in models.items():
            fitted_model.fit(X_train, y_train)
            predictions[name][test_idx[0]] = float(fitted_model.predict(X_test).ravel()[0])
    comparison = pd.DataFrame([
        {"model": name, "LOOCV_MAE": mean_absolute_error(y, pred), "LOOCV_RMSE": mean_squared_error(y, pred) ** 0.5, "LOOCV_R2": r2_score(y, pred)}
        for name, pred in predictions.items()
    ]).sort_values("LOOCV_RMSE").reset_index(drop=True)
    prediction_df = primary[["battery_id", "policy", "fade_200"]].copy()
    for name, pred in predictions.items():
        prediction_df[f"pred_{name}"] = pred
    return comparison, prediction_df


def write_report(
    cell: pd.DataFrame,
    primary: pd.DataFrame,
    summary: pd.DataFrame,
    kruskal: stats.KruskalResult,
    pairwise: pd.DataFrame,
    correlations: pd.DataFrame,
    regression,
    vif: pd.DataFrame,
    model_comparison: pd.DataFrame,
    mixed_note: str,
) -> None:
    significant = pairwise.loc[pairwise["p_holm"] < 0.05] if not pairwise.empty else pairwise
    lines = [
        "# 问题二统计分析报告",
        "",
        "## 结论口径",
        "",
        "- 赛题数据未观测 SOH=0.8 的 EOL；本报告的响应变量是第 200 次循环 SOH（SOH_200）及其对应衰减量，不是实际循环寿命。",
        "- 主分析仅保留 dataset_id=3、prediction_test=0 且完整记录至 200 圈的电池，以降低跨批次和实验结构混杂；全体 40 个完整样本只作敏感性参考。",
        f"- 全体可用完整样本：{len(cell)}；主分析样本：{len(primary)}。",
        "",
        "## 策略差异检验",
        "",
        f"- Kruskal--Wallis: H={kruskal.statistic:.4f}, p={kruskal.pvalue:.4g}。该检验比较各策略 SOH_200 分布是否一致。",
        f"- Holm 校正后显著的两两比较数：{len(significant)}。",
        "- 分组样本量有限且参数非正交，因此 p 值用于描述样本内证据，不作严格因果结论。",
        "",
        "## 参数回归与相关",
        "",
        "- 回归响应：fade_200 = SOH_1 - SOH_200；解释变量：C1、Q1、C2；标准误使用 HC3 稳健协方差。",
        f"- OLS adjusted R²={regression.rsquared_adj:.4f}, overall F-test p={regression.f_pvalue:.4g}。",
        "- 回归和相关性均为探索性结果，因为同一策略参数组合存在重复测量，而可区分策略组合数有限。",
        "",
        "## 纵向模型",
        "",
        "- 使用 1--200 圈 SOH_smooth 拟合“cycle × policy”的线性混合效应模型，电池设为随机截距和随机循环斜率。",
        f"- {mixed_note}",
        "",
        "## 结果文件",
        "",
        "- cell_metrics.csv：逐电池的 SOH_200、200 圈衰减量与策略参数。",
        "- policy_summary.csv：策略汇总。",
        "- group_test.csv / pairwise_mannwhitney_holm.csv：组间检验。",
        "- correlations.csv / regression_coefficients.csv / vif.csv：参数关联、回归与共线性诊断。",
        "- figures/：箱线图、SOH 轨迹与回归诊断图。",
        "",
        "## 策略汇总",
        "",
        summary.round(6).to_markdown(index=False),
        "",
        "## Spearman 相关",
        "",
        correlations.round(6).to_markdown(index=False),
        "",
        "## OLS 系数（HC3）",
        "",
        pd.DataFrame({"term": regression.params.index, "coefficient": regression.params.values, "p_value_hc3": regression.pvalues.values}).round(6).to_markdown(index=False),
        "",
        "## VIF",
        "",
        vif.round(6).to_markdown(index=False),
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_raw = pd.read_csv(DATA_DIR / "battery_summary.csv")
    cycles_raw = pd.read_csv(DATA_DIR / "cycle_train.csv")
    cell, primary, longitudinal = extract_cell_metrics(summary_raw, cycles_raw)
    cell.to_csv(OUTPUT_DIR / "cell_metrics.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(OUTPUT_DIR / "primary_cohort_cell_metrics.csv", index=False, encoding="utf-8-sig")

    summary = policy_summary(primary)
    summary.to_csv(OUTPUT_DIR / "policy_summary.csv", index=False, encoding="utf-8-sig")
    groups = [primary.loc[primary["policy"] == p, "SOH_200"] for p in summary["policy"]]
    kruskal = stats.kruskal(*groups)
    epsilon_squared = max(0.0, (kruskal.statistic - len(groups) + 1) / (len(primary) - len(groups)))
    pd.DataFrame([{"test": "Kruskal-Wallis", "H": kruskal.statistic, "p_value": kruskal.pvalue, "epsilon_squared": epsilon_squared, "n": len(primary)}]).to_csv(
        OUTPUT_DIR / "group_test.csv", index=False, encoding="utf-8-sig"
    )
    pairwise = holm_pairwise(primary)
    pairwise.to_csv(OUTPUT_DIR / "pairwise_mannwhitney_holm.csv", index=False, encoding="utf-8-sig")

    corr_rows = []
    for parameter in ["C1", "Q1", "C2"]:
        rho, p_value = stats.spearmanr(primary[parameter], primary["fade_200"])
        corr_rows.append({"parameter": parameter, "spearman_rho_with_fade_200": rho, "p_value": p_value})
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(OUTPUT_DIR / "correlations.csv", index=False, encoding="utf-8-sig")

    regression = smf.ols("fade_200 ~ C1 + Q1 + C2", data=primary).fit(cov_type="HC3")
    regression.summary().as_text()
    (OUTPUT_DIR / "parameter_regression_summary.txt").write_text(regression.summary().as_text(), encoding="utf-8")
    pd.DataFrame({"term": regression.params.index, "coefficient": regression.params.values, "std_error_hc3": regression.bse.values, "p_value_hc3": regression.pvalues.values}).to_csv(
        OUTPUT_DIR / "regression_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    X = primary[["C1", "Q1", "C2"]].astype(float)
    vif = pd.DataFrame({"variable": X.columns, "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
    vif.to_csv(OUTPUT_DIR / "vif.csv", index=False, encoding="utf-8-sig")
    model_comparison, loo_predictions = nested_loo_models(primary)
    model_comparison.to_csv(OUTPUT_DIR / "alternative_model_loocv.csv", index=False, encoding="utf-8-sig")
    loo_predictions.to_csv(OUTPUT_DIR / "alternative_model_loo_predictions.csv", index=False, encoding="utf-8-sig")

    try:
        mixed = smf.mixedlm("SOH_smooth ~ cycle_scaled * C(policy)", longitudinal, groups=longitudinal["battery_id"], re_formula="~cycle_scaled").fit(method="lbfgs", reml=False)
        (OUTPUT_DIR / "mixed_effects_summary.txt").write_text(mixed.summary().as_text(), encoding="utf-8")
        mixed_note = f"模型收敛：{bool(mixed.converged)}；AIC={mixed.aic:.3f}。完整估计见 mixed_effects_summary.txt。"
    except Exception as exc:  # 保证其余统计结果仍可复现。
        mixed_note = f"混合效应模型未稳定收敛：{type(exc).__name__}: {exc}。不据此输出系数结论。"
        (OUTPUT_DIR / "mixed_effects_summary.txt").write_text(mixed_note + "\n", encoding="utf-8")

    save_figures(primary, longitudinal, regression)
    write_report(cell, primary, summary, kruskal, pairwise, correlations, regression, vif, model_comparison, mixed_note)
    print(f"主分析样本数: {len(primary)}")
    print(f"Kruskal-Wallis H={kruskal.statistic:.4f}, p={kruskal.pvalue:.6g}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
