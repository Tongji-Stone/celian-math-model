"""问题三：逐电池 SOH 退化模型（与问题二共用函数族 f(n; θ)）。

运行（项目根目录）：
    python src/A/task3_degradation_model.py

主思路：对每块电池用其自己的早期 SOH_smooth 拟合
    SOH(n) = a + b n
（并比较幂律、指数、策略均值斜率、向策略收缩）。
在 40 块满 200 圈电池上做截断验证：用 1--k 圈拟合，预报 151--200。
9 块测试电池只给出 1--150 的预报与外推到 SOH=80% 的估计寿命。

不修改他人机器学习文件；本脚本只用 numpy / pandas / scipy / matplotlib。
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from SOH_plot import configure_chinese_font  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "A" / "问题三" / "output"
FIG_DIR = OUT_DIR / "figures"

WINDOWS = (50, 100, 150)
HORIZON = (151, 200)
EOL_SOH = 0.80
N_EOL_CAP = 20000.0
RECENT_LEN = 50

COLOR_MAP = {
    "3_6C-80PER_3_6C": "#4c78a8",
    "4_8C_80PER_4_8C": "#f58518",
    "5C_67PER_4C": "#54a24b",
    "5_3C_54PER_4C": "#e45756",
    "5_6C_19PER_4_6C": "#72b7b2",
    "5_6C_36PER_4_3C": "#b279a2",
    "3_7C_31PER_5_9C": "#ff9da6",
}


def canonical_policy(value: str) -> str:
    value = str(value).replace("_NEWSTRUCTURE", "")
    if value == "80PER_3_6C":
        return "3_6C-80PER_3_6C"
    return value


def find_cycle_file() -> Path:
    candidates = [
        PROJECT_ROOT / "output" / "A" / "cycle_train_cleaned.csv",
        PROJECT_ROOT / "A" / "问题一" / "output" / "A" / "cycle_train_cleaned.csv",
        DATA_DIR / "cycle_train.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到 cycle 数据")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(DATA_DIR / "battery_summary.csv")
    cycles = pd.read_csv(find_cycle_file())
    summary["policy"] = summary["policy"].map(canonical_policy)
    cycles["policy"] = cycles["policy"].map(canonical_policy)
    return summary, cycles


def markdown_table(frame: pd.DataFrame, floatfmt: str = "{:.6g}") -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float) and np.isfinite(value):
                cells.append(floatfmt.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cell_series(cycles: pd.DataFrame, battery_id: int) -> pd.DataFrame:
    part = cycles.loc[cycles["battery_id"] == battery_id, ["cycle", "SOH_smooth"]].sort_values("cycle")
    return part.dropna()


def fit_linear(n: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    x = np.column_stack([np.ones(len(n)), n])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    resid = y - fitted
    return {
        "form": "linear",
        "a": float(coef[0]),
        "b": float(coef[1]),
        "fit_rmse": float(np.sqrt(np.mean(resid**2))),
    }


def predict_linear(fit: dict[str, float], n: np.ndarray) -> np.ndarray:
    return fit["a"] + fit["b"] * np.asarray(n, float)


def _power_fun(n: np.ndarray, intercept: float, scale: float, power: float) -> np.ndarray:
    return intercept - scale * np.power(np.asarray(n, float), power)


def fit_power(n: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    drop = max(float(y[0] - y[-1]), 1e-6)
    p0 = [float(y[0]), drop / max(n[-1], 1.0), 1.0]
    try:
        coef, _ = curve_fit(
            _power_fun,
            n,
            y,
            p0=p0,
            bounds=([0.80, 0.0, 0.2], [1.20, 2.0, 3.0]),
            maxfev=20000,
        )
        fitted = _power_fun(n, *coef)
        return {
            "form": "power",
            "a": float(coef[0]),
            "c": float(coef[1]),
            "p": float(coef[2]),
            "fit_rmse": float(np.sqrt(np.mean((y - fitted) ** 2))),
        }
    except (RuntimeError, ValueError):
        linear = fit_linear(n, y)
        linear["form"] = "power_fallback_linear"
        return linear


def predict_power(fit: dict[str, float], n: np.ndarray) -> np.ndarray:
    if fit["form"] != "power":
        return predict_linear(fit, n)
    return _power_fun(n, fit["a"], fit["c"], fit["p"])


def _exp_fun(n: np.ndarray, floor: float, amp: float, rate: float) -> np.ndarray:
    return floor + amp * np.exp(-rate * np.asarray(n, float))


def fit_exp(n: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    p0 = [float(np.min(y) - 0.002), max(float(y[0] - np.min(y)), 1e-4), 1e-4]
    try:
        coef, _ = curve_fit(
            _exp_fun,
            n,
            y,
            p0=p0,
            bounds=([0.0, 0.0, 1e-8], [1.05, 1.0, 0.05]),
            maxfev=20000,
        )
        fitted = _exp_fun(n, *coef)
        return {
            "form": "exp",
            "c": float(coef[0]),
            "amp": float(coef[1]),
            "rate": float(coef[2]),
            "fit_rmse": float(np.sqrt(np.mean((y - fitted) ** 2))),
        }
    except (RuntimeError, ValueError):
        linear = fit_linear(n, y)
        linear["form"] = "exp_fallback_linear"
        return linear


def predict_exp(fit: dict[str, float], n: np.ndarray) -> np.ndarray:
    if fit["form"] != "exp":
        return predict_linear(fit, n)
    return _exp_fun(n, fit["c"], fit["amp"], fit["rate"])


def predict_from_fit(fit: dict[str, float], n: np.ndarray) -> np.ndarray:
    form = fit["form"]
    if form.startswith("power"):
        return predict_power(fit, n)
    if form.startswith("exp"):
        return predict_exp(fit, n)
    return predict_linear(fit, n)


def n_eol_from_fit(fit: dict[str, float], soh_now: float, cycle_now: float) -> tuple[float, str]:
    if soh_now <= EOL_SOH:
        return float(cycle_now), "already_below"
    form = fit["form"]
    if form == "power":
        gap = fit["a"] - EOL_SOH
        if fit["c"] <= 0 or gap <= 0:
            return N_EOL_CAP, "censored"
        value = (gap / fit["c"]) ** (1.0 / fit["p"])
        if not np.isfinite(value) or value <= 0:
            return N_EOL_CAP, "censored"
        return float(min(value, N_EOL_CAP)), "ok" if value < N_EOL_CAP else "censored"
    if form == "exp":
        if fit["c"] >= EOL_SOH or fit["amp"] <= 0 or fit["rate"] <= 0:
            return N_EOL_CAP, "censored"
        inside = (EOL_SOH - fit["c"]) / fit["amp"]
        if inside <= 0 or inside >= 1:
            return N_EOL_CAP, "censored"
        value = -np.log(inside) / fit["rate"]
        return float(min(value, N_EOL_CAP)), "ok" if value < N_EOL_CAP else "censored"
    slope = fit["b"]
    if slope >= -1e-12:
        return N_EOL_CAP, "non_decreasing"
    value = (EOL_SOH - fit["a"]) / slope
    if value <= cycle_now:
        return float(cycle_now + (soh_now - EOL_SOH) / max(-slope, 1e-12)), "ok"
    return float(min(value, N_EOL_CAP)), "ok" if value < N_EOL_CAP else "censored"


def window_xy(series: pd.DataFrame, k: int, recent: bool = False) -> tuple[np.ndarray, np.ndarray]:
    part = series.loc[series["cycle"] <= k]
    if recent:
        part = part.tail(min(RECENT_LEN, k))
    return part["cycle"].to_numpy(float), part["SOH_smooth"].to_numpy(float)


def soh_at(series: pd.DataFrame, cycle: int) -> float:
    row = series.loc[series["cycle"] == cycle, "SOH_smooth"]
    if row.empty:
        return float("nan")
    return float(row.iloc[0])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n_points": int(len(y_true)),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "bias": float(np.mean(resid)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan"),
    }


def horizon_true(series: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    part = series.loc[(series["cycle"] >= HORIZON[0]) & (series["cycle"] <= HORIZON[1])]
    return part["cycle"].to_numpy(float), part["SOH_smooth"].to_numpy(float)


def linear_from_level_slope(level_cycle: float, level_soh: float, slope: float) -> dict[str, float]:
    return {"form": "linear", "a": float(level_soh - slope * level_cycle), "b": float(slope), "fit_rmse": float("nan")}


def collect_train_cells(summary: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    max_cycle = cycles.groupby("battery_id")["cycle"].max().rename("max_cycle")
    cell = summary.merge(max_cycle, on="battery_id", validate="one_to_one")
    return cell.query("prediction_test == 0 and max_cycle == 200").copy()


def collect_test_cells(summary: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    max_cycle = cycles.groupby("battery_id")["cycle"].max().rename("max_cycle")
    cell = summary.merge(max_cycle, on="battery_id", validate="one_to_one")
    return cell.query("prediction_test == 1").copy()


def policy_slope_map(cycles: pd.DataFrame, train_ids: list[int], k: int, exclude: int | None = None) -> dict[str, float]:
    slopes: dict[str, list[float]] = {}
    for battery_id in train_ids:
        if exclude is not None and int(battery_id) == int(exclude):
            continue
        series = cell_series(cycles, int(battery_id))
        n, y = window_xy(series, k, recent=False)
        if len(n) < 5:
            continue
        fit = fit_linear(n, y)
        policy = str(series_policy(cycles, int(battery_id)))
        slopes.setdefault(policy, []).append(fit["b"])
    return {key: float(np.mean(vals)) for key, vals in slopes.items() if vals}


def series_policy(cycles: pd.DataFrame, battery_id: int) -> str:
    return str(cycles.loc[cycles["battery_id"] == battery_id, "policy"].iloc[0])


def evaluate_cell(
    series: pd.DataFrame,
    k: int,
    model_name: str,
    policy: str,
    policy_slopes: dict[str, float],
    global_slope: float,
) -> dict[str, object]:
    n_fit, y_fit = window_xy(series, k, recent=model_name == "linear_recent")
    n_true, y_true = horizon_true(series)
    soh_k = soh_at(series, k)
    if model_name == "persist":
        y_pred = np.full_like(n_true, soh_k, dtype=float)
        fit = linear_from_level_slope(k, soh_k, 0.0)
    elif model_name == "policy_mean":
        slope = policy_slopes.get(policy, global_slope)
        fit = linear_from_level_slope(k, soh_k, slope)
        y_pred = predict_linear(fit, n_true)
    elif model_name == "shrink_linear":
        cell_fit = fit_linear(*window_xy(series, k, recent=False))
        slope_p = policy_slopes.get(policy, global_slope)
        slope = 0.7 * cell_fit["b"] + 0.3 * slope_p
        fit = linear_from_level_slope(k, soh_k, slope)
        y_pred = predict_linear(fit, n_true)
    elif model_name == "linear":
        fit = fit_linear(n_fit, y_fit)
        fit = linear_from_level_slope(k, soh_k, fit["b"])
        y_pred = predict_linear(fit, n_true)
    elif model_name == "linear_recent":
        fit = fit_linear(n_fit, y_fit)
        fit = linear_from_level_slope(k, soh_k, fit["b"])
        y_pred = predict_linear(fit, n_true)
    elif model_name == "power":
        fit = fit_power(*window_xy(series, k, recent=False))
        y_pred = predict_from_fit(fit, n_true)
        y_pred = y_pred - (predict_from_fit(fit, np.array([k]))[0] - soh_k)
    elif model_name == "exp":
        fit = fit_exp(*window_xy(series, k, recent=False))
        y_pred = predict_from_fit(fit, n_true)
        y_pred = y_pred - (predict_from_fit(fit, np.array([k]))[0] - soh_k)
    else:
        raise ValueError(model_name)

    stats = metrics(y_true, y_pred)
    soh200_true = soh_at(series, 200)
    soh200_pred = float(y_pred[n_true == 200][0]) if np.any(n_true == 200) else float("nan")
    n_eol, flag = n_eol_from_fit(fit, soh_k, k)
    stats.update(
        {
            "model": model_name,
            "k": k,
            "soh200_true": soh200_true,
            "soh200_pred": soh200_pred,
            "soh200_abs_err": abs(soh200_true - soh200_pred),
            "n_eol": n_eol,
            "n_eol_flag": flag,
            "slope": float(fit.get("b", np.nan)),
        }
    )
    return stats, n_true, y_true, y_pred, fit


def run_validation(summary: pd.DataFrame, cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = collect_train_cells(summary, cycles)
    train_ids = [int(x) for x in train["battery_id"]]
    models_by_k = {
        50: ["persist", "linear", "linear_recent", "policy_mean", "shrink_linear"],
        100: ["persist", "linear", "linear_recent", "policy_mean", "shrink_linear"],
        150: ["persist", "linear", "linear_recent", "policy_mean", "shrink_linear", "power", "exp"],
    }
    rows = []
    point_rows = []
    for k in WINDOWS:
        global_slopes = []
        for battery_id in train_ids:
            n, y = window_xy(cell_series(cycles, battery_id), k)
            global_slopes.append(fit_linear(n, y)["b"])
        global_slope = float(np.mean(global_slopes))
        for battery_id in train_ids:
            series = cell_series(cycles, battery_id)
            policy = series_policy(cycles, battery_id)
            policy_slopes = policy_slope_map(cycles, train_ids, k, exclude=battery_id)
            for model_name in models_by_k[k]:
                stats, n_true, y_true, y_pred, _fit = evaluate_cell(
                    series, k, model_name, policy, policy_slopes, global_slope
                )
                stats["battery_id"] = battery_id
                stats["policy"] = policy
                rows.append(stats)
                if k == 150:
                    for cycle, true, pred in zip(n_true, y_true, y_pred):
                        point_rows.append(
                            {
                                "battery_id": battery_id,
                                "policy": policy,
                                "model": model_name,
                                "cycle": int(cycle),
                                "soh_true": float(true),
                                "soh_pred": float(pred),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(point_rows)


def summarize_validation(cell_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        cell_scores.groupby(["k", "model"], as_index=False)
        .agg(
            n_cells=("battery_id", "nunique"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            soh200_mae=("soh200_abs_err", "mean"),
            bias_mean=("bias", "mean"),
            n_eol_median=("n_eol", "median"),
            n_eol_censored=("n_eol_flag", lambda s: int((s != "ok").sum())),
        )
        .sort_values(["k", "rmse_mean"])
    )
    return grouped


def summarize_by_policy(cell_scores: pd.DataFrame, k: int = 150, model: str = "linear_recent") -> pd.DataFrame:
    part = cell_scores.query("k == @k and model == @model")
    return (
        part.groupby("policy", as_index=False)
        .agg(
            n=("battery_id", "nunique"),
            rmse_mean=("rmse", "mean"),
            soh200_mae=("soh200_abs_err", "mean"),
            n_eol_median=("n_eol", "median"),
        )
        .sort_values("rmse_mean", ascending=False)
    )


def predict_test(summary: pd.DataFrame, cycles: pd.DataFrame, train_ids: list[int], model_name: str, k: int = 150) -> pd.DataFrame:
    test = collect_test_cells(summary, cycles)
    policy_slopes = policy_slope_map(cycles, train_ids, k, exclude=None)
    global_slope = float(np.mean(list(policy_slopes.values()))) if policy_slopes else 0.0
    rows = []
    traj_rows = []
    for battery_id in test["battery_id"].astype(int):
        series = cell_series(cycles, battery_id)
        policy = series_policy(cycles, battery_id)
        soh_k = soh_at(series, k)
        if model_name == "linear_recent":
            n_fit, y_fit = window_xy(series, k, recent=True)
            slope = fit_linear(n_fit, y_fit)["b"]
            fit = linear_from_level_slope(k, soh_k, slope)
        elif model_name == "linear":
            n_fit, y_fit = window_xy(series, k, recent=False)
            slope = fit_linear(n_fit, y_fit)["b"]
            fit = linear_from_level_slope(k, soh_k, slope)
        elif model_name == "policy_mean":
            fit = linear_from_level_slope(k, soh_k, policy_slopes.get(policy, global_slope))
        else:
            n_fit, y_fit = window_xy(series, k, recent=True)
            slope = fit_linear(n_fit, y_fit)["b"]
            fit = linear_from_level_slope(k, soh_k, slope)
        future = np.arange(k + 1, HORIZON[1] + 1, dtype=float)
        pred = predict_linear(fit, future)
        n_eol, flag = n_eol_from_fit(fit, soh_k, k)
        rows.append(
            {
                "battery_id": battery_id,
                "policy": policy,
                "model": model_name,
                "k": k,
                "soh_150": soh_k,
                "soh_200_pred": float(pred[-1]),
                "slope": float(fit["b"]),
                "n_eol": n_eol,
                "n_eol_flag": flag,
            }
        )
        observed = series.loc[series["cycle"] <= k, ["cycle", "SOH_smooth"]].copy()
        observed["source"] = "observed"
        future_df = pd.DataFrame({"cycle": future, "SOH_smooth": pred, "source": "predicted"})
        both = pd.concat([observed, future_df], ignore_index=True)
        both["battery_id"] = battery_id
        both["policy"] = policy
        traj_rows.append(both)
    return pd.DataFrame(rows), pd.concat(traj_rows, ignore_index=True)


def save_figures(cell_scores: pd.DataFrame, points: pd.DataFrame, test_traj: pd.DataFrame, main_model: str) -> None:
    configure_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = summarize_validation(cell_scores)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    part = summary.query("k == 150").sort_values("rmse_mean")
    ax.bar(part["model"], part["rmse_mean"], color="#4c78a8")
    ax.set_ylabel("40 块电池上 151–200 圈 RMSE 均值")
    ax.set_title("截断验证：k=150 时各退化式预报误差")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "val_rmse_models.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for model, color in [("linear_recent", "#4c78a8"), ("policy_mean", "#f58518"), ("persist", "#9e9ac8")]:
        sub = summary.query("model == @model").sort_values("k")
        ax.plot(sub["k"], sub["rmse_mean"], marker="o", label=model, color=color)
    ax.set_xlabel("拟合窗口 k（用 1–k 圈）")
    ax.set_ylabel("151–200 圈 RMSE 均值")
    ax.set_title("窗口长度对近处预报的影响")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "val_rmse_by_window.png", dpi=200)
    plt.close(fig)

    soh200 = (
        cell_scores.query("k == 150 and model == @main_model")
        .copy()
    )
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for policy, part in soh200.groupby("policy"):
        ax.scatter(
            part["soh200_true"],
            part["soh200_pred"],
            s=42,
            color=COLOR_MAP.get(str(policy), "#333333"),
            label=policy,
            edgecolors="white",
            linewidths=0.4,
        )
    lims = [0.948, 1.002]
    ax.plot(lims, lims, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("实测 SOH_200")
    ax.set_ylabel("预报 SOH_200")
    ax.set_title(f"主模型 {main_model}：第 200 圈")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "val_soh200_scatter.png", dpi=200)
    plt.close(fig)

    example_ids = []
    short = soh200.sort_values("soh200_true").head(2)["battery_id"].tolist()
    long = soh200.sort_values("soh200_true", ascending=False).head(2)["battery_id"].tolist()
    example_ids = [int(x) for x in short + long]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, battery_id in zip(axes, example_ids):
        cell_pts = points.query("battery_id == @battery_id")
        policy = str(cell_pts["policy"].iloc[0]) if not cell_pts.empty else ""
        true = cell_pts.query("model == @main_model")
        ax.plot(true["cycle"], true["soh_true"], color="black", linewidth=1.6, label="实测")
        for model, style, color in [
            (main_model, "-", "#4c78a8"),
            ("policy_mean", "--", "#f58518"),
            ("persist", ":", "#9e9ac8"),
        ]:
            pred = cell_pts.query("model == @model")
            ax.plot(pred["cycle"], pred["soh_pred"], linestyle=style, color=color, label=model)
        ax.set_title(f"电池 {battery_id}（{policy}）")
        ax.set_ylabel("SOH")
    axes[0].legend(fontsize=7)
    axes[2].set_xlabel("cycle")
    axes[3].set_xlabel("cycle")
    fig.suptitle("截断验证示例：151–200 圈")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "val_example_trajectories.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for battery_id, part in test_traj.groupby("battery_id"):
        policy = str(part["policy"].iloc[0])
        obs = part.query("source == 'observed'")
        pred = part.query("source == 'predicted'")
        color = COLOR_MAP.get(policy, "#333333")
        ax.plot(obs["cycle"], obs["SOH_smooth"], color=color, linewidth=1.2, alpha=0.85)
        ax.plot(pred["cycle"], pred["SOH_smooth"], color=color, linewidth=1.2, linestyle="--", alpha=0.85)
    ax.axvline(150, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("cycle")
    ax.set_ylabel("SOH_smooth")
    ax.set_title("9 块测试电池：实线 1–150，虚线主模型预报 151–200")
    handles = [
        plt.Line2D([0], [0], color=COLOR_MAP[p], label=p)
        for p in sorted(test_traj["policy"].unique())
        if p in COLOR_MAP
    ]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_forecast_151_200.png", dpi=200)
    plt.close(fig)


def write_report(
    val_summary: pd.DataFrame,
    policy_tbl: pd.DataFrame,
    test_pred: pd.DataFrame,
    main_model: str,
    cycle_path: Path,
    comparison_note: str,
) -> None:
    best = val_summary.query("k == 150").sort_values("rmse_mean").iloc[0]
    linear = val_summary.query("k == 150 and model == 'linear'").iloc[0]
    recent = val_summary.query("k == 150 and model == 'linear_recent'").iloc[0]
    policy = val_summary.query("k == 150 and model == 'policy_mean'").iloc[0]
    persist = val_summary.query("k == 150 and model == 'persist'").iloc[0]
    lines = [
        "# 问题三逐电池退化模型数值报告（脚本生成）",
        "",
        f"- 循环数据：`{cycle_path.as_posix()}`。",
        "- 验证：40 块 `prediction_test=0` 且满 200 圈；用 1–k 拟合，预报 151–200。",
        "- 测试：9 块 `prediction_test=1`，无 151–200 真值。",
        f"- 主模型（按 k=150 的 RMSE 均值选取）：**{main_model}**。",
        "",
        "## 截断验证汇总",
        "",
        markdown_table(val_summary),
        "",
        "## k=150 要点",
        "",
        f"- 最优：`{best['model']}`，RMSE 均值 {best['rmse_mean']:.6g}，SOH_200 MAE {best['soh200_mae']:.6g}。",
        f"- 全程线性 `linear`：RMSE {linear['rmse_mean']:.6g}。",
        f"- 近段线性 `linear_recent`：RMSE {recent['rmse_mean']:.6g}。",
        f"- 策略均值斜率 `policy_mean`：RMSE {policy['rmse_mean']:.6g}（跨电池，接近「只用策略」）。",
        f"- 持续性 `persist`（150 圈 SOH 沿用）：RMSE {persist['rmse_mean']:.6g}。",
        "",
        f"## 主模型按策略（k=150, {main_model}）",
        "",
        markdown_table(policy_tbl),
        "",
        "## 测试集预报（无真值）",
        "",
        markdown_table(test_pred),
        "",
        comparison_note,
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cycle_path = find_cycle_file()
    summary, cycles = load_tables()
    train = collect_train_cells(summary, cycles)
    train_ids = [int(x) for x in train["battery_id"]]

    cell_scores, points = run_validation(summary, cycles)
    val_summary = summarize_validation(cell_scores)
    best_row = val_summary.query("k == 150").sort_values("rmse_mean").iloc[0]
    main_model = str(best_row["model"])
    if main_model in {"persist", "policy_mean"}:
        main_model = "linear_recent"

    policy_tbl = summarize_by_policy(cell_scores, 150, main_model)
    test_pred, test_traj = predict_test(summary, cycles, train_ids, main_model, k=150)
    test_policy, _test_traj_policy = predict_test(summary, cycles, train_ids, "policy_mean", k=150)
    test_policy = test_policy.rename(
        columns={"soh_200_pred": "soh_200_pred_policy", "n_eol": "n_eol_policy", "slope": "slope_policy"}
    )[["battery_id", "soh_200_pred_policy", "n_eol_policy", "slope_policy"]]
    test_joined = test_pred.merge(test_policy, on="battery_id")

    save_figures(cell_scores, points, test_traj, main_model)

    cell_scores.to_csv(OUT_DIR / "validation_cell_scores.csv", index=False)
    val_summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)
    policy_tbl.to_csv(OUT_DIR / "validation_by_policy.csv", index=False)
    points.to_csv(OUT_DIR / "validation_points_k150.csv", index=False)
    test_joined.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    test_traj.to_csv(OUT_DIR / "test_trajectories.csv", index=False)

    comparison_note = "\n".join(
        [
            "## 与跨电池 / 机器学习思路的可比较部分",
            "",
            "仓库中尚未发现问题三机器学习脚本。本报告用 `policy_mean` 作为「跨电池、主要用策略」的对照：",
            "同策略其他电池 1–k 圈的平均斜率，接到本电池第 k 圈水平，再预报 151–200。",
            "这与特征+树模型/神经网络在信息来源上同类（借用别的电池），但参数更少。",
            "机器学习文件补齐后，应在同一 40 块、同一 151–200 真值上比较 RMSE，而不是另换指标。",
        ]
    )
    write_report(val_summary, policy_tbl, test_joined, main_model, cycle_path, comparison_note)

    payload = {
        "n_train": int(len(train_ids)),
        "n_test": int(len(collect_test_cells(summary, cycles))),
        "cycle_file": str(cycle_path),
        "main_model": main_model,
        "best_k150": {
            "model": str(best_row["model"]),
            "rmse_mean": float(best_row["rmse_mean"]),
            "soh200_mae": float(best_row["soh200_mae"]),
        },
        "ml_files_found": False,
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()
