"""问题二：两窗口电流暴露模型。

运行（项目根目录）：
    python src/A/task2_exposure_model.py

响应为 fade_200 = SOH_1 - SOH_200，是寿命的反向代理，不是 N_EOL。
本脚本只用 numpy / pandas / scipy / matplotlib，避免 statsmodels 环境冲突。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from SOH_plot import configure_chinese_font  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "A" / "问题二" / "output"
FIG_DIR = OUT_DIR / "figures"

POLICY_ORDER = [
    "4_8C_80PER_4_8C",
    "5C_67PER_4C",
    "5_3C_54PER_4C",
    "5_6C_19PER_4_6C",
    "5_6C_36PER_4_3C",
    "3_7C_31PER_5_9C",
]

Q_STAR_MAIN = 50.0
Q_STAR_LIST = (40.0, 50.0, 60.0)
POWER_GRID = (1.0, 1.5, 2.0)
SOC_HIGH_END = 80.0


def canonical_policy(value: str) -> str:
    value = str(value).replace("_NEWSTRUCTURE", "")
    if value == "80PER_3_6C":
        return "3_6C-80PER_3_6C"
    return value


def extract_primary(summary: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    cycles = cycles.copy()
    summary["policy"] = summary["policy"].map(canonical_policy)
    cycles["policy"] = cycles["policy"].map(canonical_policy)
    max_cycle = cycles.groupby("battery_id")["cycle"].max().rename("max_cycle")
    cell = summary.merge(max_cycle, on="battery_id", validate="one_to_one")
    cell = cell.query("prediction_test == 0 and max_cycle == 200").copy()
    cycle_200 = cycles.loc[cycles["cycle"] == 200, ["battery_id", "SOH_smooth"]].rename(columns={"SOH_smooth": "SOH_200"})
    cycle_1 = cycles.loc[cycles["cycle"] == 1, ["battery_id", "SOH_smooth"]].rename(columns={"SOH_smooth": "SOH_1"})
    cell = cell.merge(cycle_1, on="battery_id", validate="one_to_one").merge(cycle_200, on="battery_id", validate="one_to_one")
    cell["fade_200"] = cell["SOH_1"] - cell["SOH_200"]
    primary = cell.query("dataset_id == 3").dropna(subset=["C1", "Q1", "C2", "fade_200"]).copy()
    return primary


def two_window_exposure(c1: float, q1: float, c2: float, q_star: float, p: float = 1.0, q: float = 1.0) -> tuple[float, float]:
    c1_p = float(c1) ** p
    c2_q = float(c2) ** q
    q1 = float(q1)
    if q1 >= q_star:
        e_low = c1_p * q_star
        e_high = c1_p * (q1 - q_star) + c2_q * (SOC_HIGH_END - q1)
    else:
        e_low = c1_p * q1 + c2_q * (q_star - q1)
        e_high = c2_q * (SOC_HIGH_END - q_star)
    return float(e_low), float(e_high)


def add_exposure(frame: pd.DataFrame, q_star: float, p: float = 1.0, q: float = 1.0) -> pd.DataFrame:
    out = frame.copy()
    pairs = [two_window_exposure(r.C1, r.Q1, r.C2, q_star, p, q) for r in out.itertuples()]
    out["E_L"] = [x[0] for x in pairs]
    out["E_H"] = [x[1] for x in pairs]
    out["C_high"] = out["E_H"] / (SOC_HIGH_END - q_star)
    out["q_star"] = q_star
    out["power_p"] = p
    out["power_q"] = q
    return out


@dataclass
class OLSResult:
    name: str
    terms: list[str]
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    fittedvalues: np.ndarray
    resid: np.ndarray
    rsquared: float
    rsquared_adj: float
    aic: float
    bic: float
    n: int
    k: int

    def coef_frame(self) -> pd.DataFrame:
        z = stats.norm.ppf(0.975)
        return pd.DataFrame(
            {
                "model": self.name,
                "term": self.terms,
                "coef": self.params.values,
                "std_error": self.bse.values,
                "p_value": self.pvalues.values,
                "ci_low": self.params.values - z * self.bse.values,
                "ci_high": self.params.values + z * self.bse.values,
            }
        )


def _design(data: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    x = np.column_stack([np.ones(len(data)), data[columns].to_numpy(dtype=float)])
    return x, ["Intercept", *columns]


def fit_ols(
    data: pd.DataFrame,
    columns: list[str],
    y_col: str,
    name: str,
    se: str = "hc3",
    cluster: pd.Series | None = None,
    weights: np.ndarray | None = None,
) -> OLSResult:
    y = data[y_col].to_numpy(dtype=float)
    x, terms = _design(data, columns)
    n, k = x.shape
    if weights is None:
        xtx = x.T @ x
        xty = x.T @ y
    else:
        w = np.asarray(weights, dtype=float)
        sw = np.sqrt(w)
        xw = x * sw[:, None]
        yw = y * sw
        xtx = xw.T @ xw
        xty = xw.T @ yw
    beta = np.linalg.solve(xtx, xty)
    fitted = x @ beta
    resid = y - fitted
    xtx_inv = np.linalg.inv(xtx)

    if se == "cluster":
        if cluster is None:
            raise ValueError("cluster series required")
        meat = np.zeros((k, k))
        for _, idx in pd.Series(np.arange(n), index=data.index).groupby(cluster.to_numpy()):
            xi = x[idx]
            ei = resid[idx]
            score = xi.T @ ei
            meat += np.outer(score, score)
        g = pd.Series(cluster).nunique()
        # 少簇校正
        scale = (g / (g - 1)) * ((n - 1) / (n - k))
        cov = scale * xtx_inv @ meat @ xtx_inv
    elif se == "hc3":
        h = np.sum((x @ xtx_inv) * x, axis=1)
        adj = resid / np.clip(1.0 - h, 1e-8, None)
        xe = x * adj[:, None]
        meat = xe.T @ xe
        cov = xtx_inv @ meat @ xtx_inv
    else:
        sigma2 = float(resid @ resid) / (n - k)
        cov = sigma2 * xtx_inv

    se_vec = np.sqrt(np.clip(np.diag(cov), 0, None))
    tstat = beta / np.clip(se_vec, 1e-18, None)
    pval = 2 * stats.t.sf(np.abs(tstat), df=max(n - k, 1))

    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float((resid ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else np.nan
    aic = n * np.log(ssr / n) + 2 * k
    bic = n * np.log(ssr / n) + k * np.log(n)
    return OLSResult(
        name=name,
        terms=terms,
        params=pd.Series(beta, index=terms),
        bse=pd.Series(se_vec, index=terms),
        pvalues=pd.Series(pval, index=terms),
        fittedvalues=fitted,
        resid=resid,
        rsquared=float(r2),
        rsquared_adj=float(r2_adj),
        aic=float(aic),
        bic=float(bic),
        n=n,
        k=k,
    )


def vif_two_columns(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for i, col in enumerate(cols):
        others = [c for j, c in enumerate(cols) if j != i]
        model = fit_ols(data, others, col, name=f"vif_{col}", se="classical")
        vif = 1.0 / (1.0 - model.rsquared) if model.rsquared < 1 else np.inf
        rows.append({"variable": col, "VIF": float(vif)})
    return pd.DataFrame(rows)


def policy_means(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("policy", as_index=False).agg(
        fade_200=("fade_200", "mean"),
        E_L=("E_L", "mean"),
        E_H=("E_H", "mean"),
        C_high=("C_high", "mean"),
        C1=("C1", "mean"),
        Q1=("Q1", "mean"),
        C2=("C2", "mean"),
        n=("battery_id", "size"),
        SOH_200=("SOH_200", "mean"),
        battery_ids=("battery_id", lambda s: ",".join(map(str, sorted(s.astype(int))))),
    )


def loo_policy(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for held in data["policy"].drop_duplicates():
        train = data.loc[data["policy"] != held]
        test = data.loc[data["policy"] == held]
        model = fit_ols(train, columns, "fade_200", name="loo", se="classical")
        x_test, terms = _design(test, columns)
        pred = x_test @ model.params.reindex(terms).to_numpy()
        resid = test["fade_200"].to_numpy() - pred
        rows.append(
            {
                "held_out_policy": held,
                "n_test": int(len(test)),
                "observed_mean": float(test["fade_200"].mean()),
                "predicted_mean": float(pred.mean()),
                "mae": float(np.mean(np.abs(resid))),
                "rmse": float(np.sqrt(np.mean(resid**2))),
                "mean_residual": float(np.mean(resid)),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.6g}") -> str:
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda x: floatfmt.format(x) if pd.notna(x) else "")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join([header, sep, body])


def save_figures(primary: pd.DataFrame, main_model: OLSResult) -> None:
    configure_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = [p for p in POLICY_ORDER if p in set(primary["policy"])]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(order)))
    color_map = dict(zip(order, colors))

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for policy in order:
        part = primary.loc[primary["policy"] == policy]
        ax.scatter(part["E_H"], part["fade_200"], s=42, color=color_map[policy], label=policy, alpha=0.9, edgecolors="white", linewidths=0.4)
    xline = np.linspace(primary["E_H"].min() * 0.98, primary["E_H"].max() * 1.02, 50)
    e_l_bar = float(primary["E_L"].mean())
    intercept = float(main_model.params.get("Intercept", 0.0))
    b_l = float(main_model.params.get("E_L", 0.0))
    b_h = float(main_model.params.get("E_H", 0.0))
    ax.plot(xline, intercept + b_l * e_l_bar + b_h * xline, color="black", linewidth=1.2, linestyle="--", label="主模型切片（E_L 取均值）")
    ax.set_xlabel(r"高 SOC 暴露 $E^{H}$（$Q^*=50\%$）")
    ax.set_ylabel(r"寿命反向代理 $\mathrm{Fade}_{200}$")
    ax.set_title("高 SOC 电流暴露与 200 圈衰减")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fade_vs_E_high.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for policy in order:
        part = primary.loc[primary["policy"] == policy]
        ax.scatter(part["C_high"], part["fade_200"], s=42, color=color_map[policy], label=policy, alpha=0.9, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(r"高 SOC 等效倍率 $C^{\mathrm{high}}$")
    ax.set_ylabel(r"寿命反向代理 $\mathrm{Fade}_{200}$")
    ax.set_title("高 SOC 等效倍率与 200 圈衰减")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fade_vs_C_high.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(main_model.fittedvalues, main_model.resid, c="#3182bd", alpha=0.85)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="主模型残差图", xlabel="拟合的 Fade_200", ylabel="残差")
    stats.probplot(main_model.resid, dist="norm", plot=axes[1])
    axes[1].set_title("主模型 Q-Q 图")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "main_residuals.png", dpi=300)
    plt.close(fig)


def kruskal_on(frame: pd.DataFrame, column: str, subset: str) -> dict:
    policies = [p for p in POLICY_ORDER if p in set(frame["policy"])]
    groups = [frame.loc[frame["policy"] == p, column] for p in policies]
    result = stats.kruskal(*groups)
    k = len(groups)
    n = len(frame)
    epsilon_squared = max(0.0, (result.statistic - k + 1) / (n - k))
    return {"subset": subset, "response": column, "n": n, "k": k, "H": float(result.statistic), "p_value": float(result.pvalue), "epsilon_squared": float(epsilon_squared)}


def fit_row(model: OLSResult) -> dict:
    return {
        "model": model.name,
        "n": model.n,
        "k": model.k,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
    }


def write_generated_report(payload: dict) -> None:
    primary = payload["primary"]
    policy = payload["policy"]
    lines = [
        "# 问题二暴露模型数值报告（脚本生成）",
        "",
        "本文件由 `src/A/task2_exposure_model.py` 写入，供 `模型.md` / `结论.md` 引用。",
        "响应 Y = Fade_200 为寿命反向代理：越大表示 200 圈内掉得越多。",
        "",
        f"- 主分析样本：{len(primary)} 块，策略 {primary['policy'].nunique()} 种。",
        f"- 主模型门槛 Q* = {Q_STAR_MAIN:.0f}%。",
        f"- 起始健康异常（SOH_1 < 0.98）：电池 {payload['start_outliers']}。",
        "",
        "## 策略暴露量（主设定）",
        "",
        markdown_table(policy[["policy", "n", "C1", "Q1", "C2", "E_L", "E_H", "C_high", "fade_200"]].rename(columns={"fade_200": "fade_200_mean", "n": "battery_count"})),
        "",
        "## Spearman（与 Fade_200）",
        "",
        markdown_table(payload["spearman"]),
        "",
        "## 拟合优度",
        "",
        markdown_table(payload["fits"]),
        "",
        "## 系数",
        "",
        markdown_table(payload["coefs"]),
        "",
        "## 暴露量 VIF",
        "",
        markdown_table(payload["vif"]),
        "",
        "## Kruskal–Wallis（Fade_200）",
        "",
        markdown_table(pd.DataFrame(payload["kw"])),
        "",
        "## 稳健性",
        "",
        markdown_table(payload["robust"]),
        "",
        "## 留一策略预测",
        "",
        markdown_table(payload["loo"]),
        "",
        "## 幂次网格（策略加权 WLS，按 AIC 升序）",
        "",
        markdown_table(payload["grid"].sort_values("aic").head(8)),
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_raw = pd.read_csv(DATA_DIR / "battery_summary.csv")
    cycles_raw = pd.read_csv(DATA_DIR / "cycle_train.csv")
    primary = add_exposure(extract_primary(summary_raw, cycles_raw), Q_STAR_MAIN)
    start_outliers = primary.loc[primary["SOH_1"] < 0.98, "battery_id"].astype(int).tolist()
    policy = policy_means(primary).sort_values("E_H", ascending=False)

    models = [
        fit_ols(primary, ["E_L", "E_H"], "fade_200", "main_EH_EL_cluster", se="cluster", cluster=primary["policy"]),
        fit_ols(primary, ["E_L", "E_H"], "fade_200", "main_EH_EL_hc3", se="hc3"),
        fit_ols(primary, ["E_H"], "fade_200", "nested_EH_cluster", se="cluster", cluster=primary["policy"]),
        fit_ols(primary, ["E_H"], "fade_200", "nested_EH_hc3", se="hc3"),
        fit_ols(primary, ["C_high"], "fade_200", "simple_Chigh_cluster", se="cluster", cluster=primary["policy"]),
        fit_ols(primary, ["C_high"], "fade_200", "simple_Chigh_hc3", se="hc3"),
        fit_ols(primary, ["C1", "Q1", "C2"], "fade_200", "baseline_raw_cluster", se="cluster", cluster=primary["policy"]),
        fit_ols(primary, ["C1", "Q1", "C2"], "fade_200", "baseline_raw_hc3", se="hc3"),
    ]
    grouped = policy_means(primary)
    wls_models = [
        fit_ols(grouped, ["E_L", "E_H"], "fade_200", "main_EH_EL_policy_wls", se="classical", weights=grouped["n"].to_numpy()),
        fit_ols(grouped, ["E_H"], "fade_200", "nested_EH_policy_wls", se="classical", weights=grouped["n"].to_numpy()),
        fit_ols(grouped, ["C_high"], "fade_200", "simple_Chigh_policy_wls", se="classical", weights=grouped["n"].to_numpy()),
        fit_ols(grouped, ["C1", "Q1", "C2"], "fade_200", "baseline_raw_policy_wls", se="classical", weights=grouped["n"].to_numpy()),
    ]
    all_models = models + wls_models
    main_cluster = models[0]
    wls_main = wls_models[0]
    coefs = pd.concat([m.coef_frame() for m in all_models], ignore_index=True)
    fits = pd.DataFrame([fit_row(m) for m in all_models])
    vif = vif_two_columns(primary, ["E_L", "E_H"])

    spearman_rows = []
    for col in ["E_L", "E_H", "C_high", "C1", "Q1", "C2"]:
        rho, p_value = stats.spearmanr(primary[col], primary["fade_200"])
        spearman_rows.append({"variable": col, "spearman_rho": float(rho), "p_value": float(p_value)})
    spearman = pd.DataFrame(spearman_rows)

    no_short = primary.loc[primary["policy"] != "3_7C_31PER_5_9C"].copy()
    kw = [
        kruskal_on(primary, "fade_200", "primary_34"),
        kruskal_on(no_short, "fade_200", "drop_3_7C"),
    ]

    robust_rows = []
    raw = primary.drop(columns=["E_L", "E_H", "C_high", "q_star", "power_p", "power_q"])
    for q_star in Q_STAR_LIST:
        tmp = add_exposure(raw, q_star)
        g = policy_means(tmp)
        model = fit_ols(g, ["E_L", "E_H"], "fade_200", f"qstar{int(q_star)}", se="classical", weights=g["n"].to_numpy())
        robust_rows.append(
            {
                "check": f"q_star={q_star:.0f}",
                "n": len(tmp),
                "beta_L": float(model.params.get("E_L", np.nan)),
                "beta_H": float(model.params.get("E_H", np.nan)),
                "p_beta_H": float(model.pvalues.get("E_H", np.nan)),
                "adj_r_squared": model.rsquared_adj,
            }
        )
    g_drop = policy_means(no_short)
    m_drop = fit_ols(g_drop, ["E_L", "E_H"], "fade_200", "drop_3_7C", se="classical", weights=g_drop["n"].to_numpy())
    robust_rows.append(
        {
            "check": "drop_3_7C_policy_wls",
            "n": len(no_short),
            "beta_L": float(m_drop.params.get("E_L", np.nan)),
            "beta_H": float(m_drop.params.get("E_H", np.nan)),
            "p_beta_H": float(m_drop.pvalues.get("E_H", np.nan)),
            "adj_r_squared": m_drop.rsquared_adj,
        }
    )
    no_start = primary.loc[~primary["battery_id"].isin(start_outliers)].copy()
    g_start = policy_means(no_start)
    m_start = fit_ols(g_start, ["E_L", "E_H"], "fade_200", "drop_low_SOH1", se="classical", weights=g_start["n"].to_numpy())
    robust_rows.append(
        {
            "check": "drop_SOH1_lt_0.98",
            "n": len(no_start),
            "beta_L": float(m_start.params.get("E_L", np.nan)),
            "beta_H": float(m_start.params.get("E_H", np.nan)),
            "p_beta_H": float(m_start.pvalues.get("E_H", np.nan)),
            "adj_r_squared": m_start.rsquared_adj,
        }
    )
    robust = pd.DataFrame(robust_rows)
    loo = loo_policy(primary, ["E_L", "E_H"])

    grid_rows = []
    for p, q in product(POWER_GRID, POWER_GRID):
        tmp = add_exposure(raw, Q_STAR_MAIN, p=p, q=q)
        g = policy_means(tmp)
        model = fit_ols(g, ["E_L", "E_H"], "fade_200", f"p{p}_q{q}", se="classical", weights=g["n"].to_numpy())
        grid_rows.append(
            {
                "p": p,
                "q": q,
                "beta_L": float(model.params.get("E_L", np.nan)),
                "beta_H": float(model.params.get("E_H", np.nan)),
                "p_beta_H": float(model.pvalues.get("E_H", np.nan)),
                "adj_r_squared": model.rsquared_adj,
                "aic": model.aic,
            }
        )
    grid = pd.DataFrame(grid_rows)

    primary.to_csv(OUT_DIR / "cell_exposure.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(OUT_DIR / "policy_exposure.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv(OUT_DIR / "coefficients.csv", index=False, encoding="utf-8-sig")
    fits.to_csv(OUT_DIR / "fit_stats.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(OUT_DIR / "vif_exposure.csv", index=False, encoding="utf-8-sig")
    spearman.to_csv(OUT_DIR / "spearman_exposure.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(kw).to_csv(OUT_DIR / "kruskal_fade.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(OUT_DIR / "robustness.csv", index=False, encoding="utf-8-sig")
    loo.to_csv(OUT_DIR / "loo_policy.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(OUT_DIR / "power_grid.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "run_summary.json").write_text(
        json.dumps({"n_primary": int(len(primary)), "n_policy": int(primary["policy"].nunique()), "q_star_main": Q_STAR_MAIN, "start_outliers": start_outliers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_figures(primary, main_cluster)
    write_generated_report(
        {
            "primary": primary,
            "policy": policy,
            "coefs": coefs,
            "fits": fits,
            "loo": loo,
            "robust": robust,
            "grid": grid,
            "vif": vif,
            "kw": kw,
            "spearman": spearman,
            "start_outliers": start_outliers,
        }
    )
    print(f"n={len(primary)} policies={primary['policy'].nunique()}")
    print(f"main cluster beta_H={main_cluster.params['E_H']:.6g} p={main_cluster.pvalues['E_H']:.4g}")
    print(f"WLS beta_H={wls_main.params['E_H']:.6g} p={wls_main.pvalues['E_H']:.4g}")
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()
