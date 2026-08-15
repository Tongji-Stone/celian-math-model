"""问题二：以公开同源 EOL 为响应的策略寿命分析。

数据边界：只使用 prediction_test=0 的公开 MATR 标签；不读取、训练、比较或输出
9 块问题三测试电池的 EOL。本脚本分析的是六个既有策略组合的关联，不能产生
单个 C1、Q1、C2 参数的严格因果效应。
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data" / "battery_summary.csv"
AUDIT = ROOT / "references" / "matr_source" / "external_eol_audit" / "contest_to_matr_eol_audit.csv"
OUT = ROOT / "A" / "问题二" / "output" / "eol_lifetime"
FIG = OUT / "figures"
Q_STAR = 50.0
SOC_END = 80.0


def canonical_policy(value: str) -> str:
    value = str(value).replace("_NEWSTRUCTURE", "")
    return "3_6C-80PER_3_6C" if value == "80PER_3_6C" else value


def exposure(c1: float, q1: float, c2: float) -> tuple[float, float, float]:
    """固定 50% SOC 分界下的低/高 SOC 电流暴露。"""
    if q1 >= Q_STAR:
        low = c1 * Q_STAR
        high = c1 * (q1 - Q_STAR) + c2 * (SOC_END - q1)
    else:
        low = c1 * q1 + c2 * (Q_STAR - q1)
        high = c2 * (SOC_END - Q_STAR)
    return float(low), float(high), float(high / (SOC_END - Q_STAR))


def holm_pairs(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policies = list(data.policy.drop_duplicates())
    for left, right in combinations(policies, 2):
        x = data.loc[data.policy == left, "external_eol_cycles"]
        y = data.loc[data.policy == right, "external_eol_cycles"]
        test = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        rows.append({"policy_a": left, "policy_b": right, "u": test.statistic, "p_raw": test.pvalue})
    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(out)
    adjusted = []
    running = 0.0
    for index, p in enumerate(out.p_raw):
        running = max(running, min(1.0, (m - index) * p))
        adjusted.append(running)
    out["p_holm"] = adjusted
    return out


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """不依赖可选 tabulate 包的简易 Markdown 表格。"""
    def display(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)

    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(display(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def weighted_ols(data: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    """策略均值上的 WLS；推断单位为策略配方而非重复循环记录。"""
    x = np.column_stack([np.ones(len(data)), data[columns].to_numpy(float)])
    y = data.log_eol_mean.to_numpy(float)
    weights = data.n.to_numpy(float)
    root_w = np.sqrt(weights)
    beta = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]
    fitted = x @ beta
    resid = y - fitted
    n, k = x.shape
    ssr = float(np.sum(weights * resid**2))
    sst = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
    r2 = 1.0 - ssr / sst
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    # df=3，只把区间作为模型内描述，不把它解释为强假设检验。
    sigma2 = ssr / max(n - k, 1)
    covariance = sigma2 * np.linalg.inv((x * weights[:, None]).T @ x)
    se = np.sqrt(np.diag(covariance))
    t = beta / np.maximum(se, 1e-12)
    p = 2 * stats.t.sf(abs(t), df=max(n - k, 1))
    return {
        "beta": beta,
        "se": se,
        "p": p,
        "fitted": fitted,
        "resid": resid,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
    }


def fit_rows(policy: pd.DataFrame, columns: list[str], label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = weighted_ols(policy, columns)
    terms = ["Intercept", *columns]
    coef = pd.DataFrame({
        "model": label,
        "term": terms,
        "coefficient_log_eol": fit["beta"],
        "std_error": fit["se"],
        "p_classical_descriptive": fit["p"],
    })
    stats_row = pd.DataFrame([{
        "model": label, "n_policies": fit["n"], "parameters": fit["k"],
        "r_squared": fit["r2"], "adj_r_squared": fit["adj_r2"],
    }])
    return coef, stats_row


def leave_one_policy(policy: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for held in policy.policy:
        train = policy.loc[policy.policy != held]
        test = policy.loc[policy.policy == held].iloc[0]
        fit = weighted_ols(train, columns)
        x = np.array([1.0, *test[columns].to_numpy(float)])
        pred = float(np.exp(x @ fit["beta"]))
        rows.append({
            "held_out_policy": held,
            "observed_mean_eol": float(test.eol_mean),
            "predicted_eol": pred,
            "absolute_percentage_error": abs(pred - test.eol_mean) / test.eol_mean * 100,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    # 先在标签表中删除测试记录，使其 EOL 不会进入后续 merge 的任何列。
    audit = pd.read_csv(AUDIT)
    audit = audit.loc[(audit.prediction_test == 0) & (audit.eol_status == "available"),
                      ["battery_id", "matr_source_key", "external_eol_cycles"]]
    data = summary.merge(audit, on="battery_id", how="inner", validate="one_to_one")
    data = data.loc[data.dataset_id == 3].copy()
    data["policy"] = data.policy.map(canonical_policy)
    # batch-2 的接续不是新电池；主队列虽都是 batch-3，仍保留这条保护。
    data = data.sort_values("global_id").drop_duplicates("matr_source_key", keep="first")
    data = data.dropna(subset=["C1", "Q1", "C2", "external_eol_cycles"])
    exposures = [exposure(r.C1, r.Q1, r.C2) for r in data.itertuples()]
    data[["E_low", "E_high", "C_high"]] = exposures
    data["log_eol"] = np.log(data.external_eol_cycles)

    policy = data.groupby("policy", as_index=False).agg(
        n=("battery_id", "size"), eol_mean=("external_eol_cycles", "mean"),
        eol_median=("external_eol_cycles", "median"), eol_sd=("external_eol_cycles", "std"),
        log_eol_mean=("log_eol", "mean"), C1=("C1", "mean"), Q1=("Q1", "mean"),
        C2=("C2", "mean"), E_low=("E_low", "mean"), E_high=("E_high", "mean"),
        C_high=("C_high", "mean"), battery_ids=("battery_id", lambda x: ",".join(map(str, sorted(x)))),
    ).sort_values("eol_mean", ascending=False)
    kw = stats.kruskal(*[part.external_eol_cycles for _, part in data.groupby("policy")])
    kw_table = pd.DataFrame([{
        "n_batteries": len(data), "n_policies": data.policy.nunique(), "H": kw.statistic, "p_value": kw.pvalue,
        "effect_epsilon_squared": max(0.0, (kw.statistic - data.policy.nunique() + 1) / (len(data) - data.policy.nunique())),
    }])
    pairs = holm_pairs(data)
    coef, fit = fit_rows(policy, ["E_low", "E_high"], "two_window_log_eol_wls")
    raw_coef, raw_fit = fit_rows(policy, ["C1", "Q1", "C2"], "raw_parameter_log_eol_wls")
    loo = leave_one_policy(policy, ["E_low", "E_high"])
    no_short = policy.loc[policy.policy != "3_7C_31PER_5_9C"]
    _, sensitivity = fit_rows(no_short, ["E_low", "E_high"], "drop_3_7C")
    # 原始参数的相关性提示共线性；只有 6 个设计点，不能拆成独立边际效应。
    corr = policy[["C1", "Q1", "C2", "E_low", "E_high"]].corr(method="spearman")

    data.to_csv(OUT / "eol_cell_level.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(OUT / "eol_policy_summary.csv", index=False, encoding="utf-8-sig")
    kw_table.to_csv(OUT / "kruskal_eol.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT / "pairwise_eol_holm.csv", index=False, encoding="utf-8-sig")
    pd.concat([coef, raw_coef], ignore_index=True).to_csv(OUT / "wls_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.concat([fit, raw_fit, sensitivity], ignore_index=True).to_csv(OUT / "wls_fit_summary.csv", index=False, encoding="utf-8-sig")
    loo.to_csv(OUT / "leave_one_policy_eol.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT / "spearman_design_correlation.csv", encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    order = list(policy.policy)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    values = [data.loc[data.policy == name, "external_eol_cycles"] for name in order]
    ax.boxplot(values, tick_labels=order, showmeans=True)
    for pos, values_i in enumerate(values, start=1):
        ax.scatter(np.full(len(values_i), pos), values_i, color="#2574a9", zorder=3)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.set_ylabel("公开 EOL / 圈")
    ax.set_title("同一 batch-3 结构下的策略寿命分布（非测试电池）")
    fig.tight_layout()
    fig.savefig(FIG / "policy_eol_boxplot.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(policy.E_high, policy.eol_mean, s=50 + policy.n * 16, color="#d95f02")
    for row in policy.itertuples():
        ax.annotate(row.policy, (row.E_high, row.eol_mean), fontsize=7, xytext=(3, 4), textcoords="offset points")
    ax.set_xlabel(r"高 SOC 电流暴露 $E^H$")
    ax.set_ylabel("策略平均 EOL / 圈")
    ax.set_title("高 SOC 暴露不能单独决定寿命")
    fig.tight_layout()
    fig.savefig(FIG / "eol_vs_high_soc_exposure.png", dpi=300)
    plt.close(fig)

    report = [
        "# 问题二：以 EOL 为响应的寿命分析（脚本生成）", "",
        "## 数据边界", "",
        "- 仅纳入 `dataset_id=3`、`prediction_test=0` 且公开 EOL 可用的去重原电池。",
        f"- 样本为 {len(data)} 块电池、{data.policy.nunique()} 个既有策略点。测试电池 EOL 标签在合并前已排除。",
        "- 这是非正交、小样本的策略关联分析，不是 C1、Q1、C2 的独立因果实验。", "",
        "## 策略汇总", "", markdown_table(policy), "",
        "## 总体差异", "", markdown_table(kw_table, digits=5), "",
        "Kruskal--Wallis 仅检验策略分布总体是否不同；两两比较需看 Holm 校正表。", "",
        "## 两窗口寿命模型", "", markdown_table(fit, digits=5), "", markdown_table(coef, digits=5), "",
        "高 SOC 系数只能作为在六个策略点上的描述性关联。删除 3.7C--31%--5.9C 策略后的拟合见 `wls_fit_summary.csv`，用于检查结论是否依赖短寿策略。", "",
        "## 留一策略外推", "", markdown_table(loo), "",
        "留一策略误差说明该模型不应用于凭空设计远离已有配方的新策略。",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"n={len(data)}, policies={data.policy.nunique()}, KW p={kw.pvalue:.6g}")
    print(policy[["policy", "n", "eol_mean", "eol_median"]].to_string(index=False))
    print(fit.to_string(index=False))
    print(loo.to_string(index=False))


if __name__ == "__main__":
    main()
