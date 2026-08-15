"""问题三：检验 SOH 外的可见过程特征是否改善 150->200 SOH 预测。

只使用第 150 圈及以前的 capacity、chargetime、IR、Tavg、初始容量和充电策略；
不读取 EOL 标签，也不使用 summary 中可能汇总到后续循环的均值字段。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from task3_mlp_capacity_scan import CapacityMLP, INIT_SEEDS, ROOT, count_parameters, predict
from task3_window_ablation import HORIZON, enforce_monotone, make_target


OUT = ROOT / "A" / "问题三" / "output" / "mlp_feature_ablation"
END = 150
FOLD_SEED = 20260815
ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "linear": (),
    "hidden_16_8": (16, 8),
    "hidden_32_16": (32, 16),
}


def slopes_and_levels(values: pd.Series) -> list[float]:
    """每个过程量只压缩为可解释的水平、变化和前后斜率，避免直接塞入 150 个点。"""
    x = values.index.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    split = max(2, END // 2)
    early = (x <= split)
    late = (x >= split)
    return [
        float(y[0]), float(y[-1]), float(y[-1] - y[0]),
        float(np.polyfit(x[early], y[early], 1)[0]),
        float(np.polyfit(x[late], y[late], 1)[0]),
        float(np.std(y)),
    ]


def make_feature_groups(frame: pd.DataFrame, meta: pd.Series) -> dict[str, np.ndarray]:
    obs = frame.loc[frame.cycle <= END].set_index("cycle").sort_index()
    soh = obs.SOH_smooth
    first, split = float(soh.loc[1]), END // 2
    marks = np.linspace(1, END, 7).round().astype(int)
    soh_only = [
        *[float(soh.loc[m] - first) for m in marks],
        float((soh.loc[END] - soh.loc[split]) / (END - split)),
        float((soh.loc[split] - first) / (split - 1)),
    ]
    # 与已有窗口 MLP 一致的两个过程特征，作为公平对照。
    existing = [
        *soh_only,
        float(obs.IR.loc[split:END].mean() - obs.IR.loc[1:split].mean()),
        float(obs.Tavg.loc[split:END].mean() - obs.Tavg.loc[1:split].mean()),
    ]
    initial = float(meta.initial_capacity)
    capacity_relative = obs.capacity / initial
    process = [
        *existing,
        *slopes_and_levels(capacity_relative),
        *slopes_and_levels(obs.chargetime),
        *slopes_and_levels(obs.IR),
        *slopes_and_levels(obs.Tavg),
    ]
    c1 = 0.0 if pd.isna(meta.C1) else float(meta.C1)
    policy = [c1, float(meta.Q1), float(meta.C2), initial]
    return {
        "soh_only": np.asarray(soh_only, dtype=np.float32),
        "soh_ir_temp": np.asarray(existing, dtype=np.float32),
        "soh_process": np.asarray(process, dtype=np.float32),
        "soh_process_policy": np.asarray([*process, *policy], dtype=np.float32),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    table = frame.copy()
    for col in table.select_dtypes(include="number"):
        table[col] = table[col].map(lambda value: f"{value:.6g}")
    headers = list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "data" / "battery_summary.csv").set_index("battery_id")
    cycles = pd.read_csv(ROOT / "data" / "cycle_train.csv")
    full_ids = summary.index[(summary.prediction_test.eq(0)) & (cycles.groupby("battery_id").cycle.max().eq(200))].to_numpy()
    groups = [make_feature_groups(cycles.loc[cycles.battery_id == bid], summary.loc[bid]) for bid in full_ids]
    y = np.vstack([make_target(cycles.loc[cycles.battery_id == bid], END) for bid in full_ids])
    folds = list(KFold(5, shuffle=True, random_state=FOLD_SEED).split(full_ids))
    rows: list[dict[str, object]] = []
    for feature_set in groups[0]:
        x = np.vstack([features[feature_set] for features in groups])
        for name, hidden in ARCHITECTURES.items():
            params = count_parameters(x.shape[1], hidden)
            for seed in INIT_SEEDS:
                for fold, (fit, valid) in enumerate(folds, 1):
                    delta = enforce_monotone(predict(x[fit], y[fit], x[valid], hidden, seed))
                    for pos, bid in enumerate(full_ids[valid]):
                        frame = cycles.loc[cycles.battery_id == bid]
                        base = float(frame.loc[frame.cycle.eq(END), "SOH_smooth"].iloc[0])
                        actual, estimated = base + y[valid][pos], base + delta[pos]
                        rows.append({
                            "feature_set": feature_set, "architecture": name,
                            "hidden_layers": "-".join(map(str, hidden)) or "none",
                            "n_input": x.shape[1], "parameters": params, "seed": seed, "fold": fold,
                            "battery_id": int(bid),
                            "rmse": float(np.sqrt(mean_squared_error(actual, estimated))),
                            "mae": float(mean_absolute_error(actual, estimated)),
                        })
    detail = pd.DataFrame(rows)
    summary_table = detail.groupby(["feature_set", "architecture", "hidden_layers", "n_input", "parameters"], as_index=False).agg(
        batteries_times_seeds=("battery_id", "size"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
    ).sort_values("rmse_mean")
    detail.to_csv(OUT / "feature_cv_detail.csv", index=False, encoding="utf-8-sig")
    summary_table.to_csv(OUT / "feature_cv_summary.csv", index=False, encoding="utf-8-sig")

    labels = {
        "soh_only": "SOH 摘要", "soh_ir_temp": "SOH+IR+温度",
        "soh_process": "再加容量/充电时间/过程趋势", "soh_process_policy": "再加策略/初始容量",
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    for feature_set, subset in summary_table.groupby("feature_set"):
        ax.scatter(subset.parameters, subset.rmse_mean, s=55, label=labels[feature_set])
        for row in subset.itertuples():
            ax.annotate(row.architecture.replace("hidden_", ""), (row.parameters, row.rmse_mean), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("可训练参数量（对数刻度）")
    ax.set_ylabel("5 折 × 3 初始化平均 RMSE")
    ax.set_title("问题三：输入特征与 MLP 容量联合消融")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "feature_vs_rmse.png", dpi=250)
    plt.close(fig)

    lines = [
        "# 问题三：SOH 之外的输入特征消融", "",
        "## 严格的信息边界", "",
        "所有特征均由第 1--150 圈记录构成。明确不使用 `battery_summary.csv` 中的 `mean_chargetime`、`mean_IR`、`mean_Tavg`，因为对完整训练电池这些量可能汇总了第 151--200 圈，属于未来信息。仅使用初始容量、策略参数和每圈可见的容量、充电时间、IR、Tavg。外部 EOL 标签没有被读取。", "",
        "## 特征组", "",
        "- `soh_only`：7 个 SOH 检查点变化 + 早/晚期斜率（9 维）。",
        "- `soh_ir_temp`：上项加 IR、温度的前后均值差（11 维），即原窗口 MLP 的动态特征组。",
        "- `soh_process`：再加相对容量、充电时间、IR、温度的起止水平、变化、早/晚期斜率与波动（35 维）。",
        "- `soh_process_policy`：再加 `C1,Q1,C2` 与初始容量（39 维）。", "",
        "## 结果", "", markdown_table(summary_table), "",
        "## 判定原则", "",
        "每一行都是相同 40 块完整电池、5 折按电池验证、3 个初始化的 120 个误差汇总。只有特征组在多个网络容量上稳定降低验证 RMSE，才认为过程信息真正有用；否则不因训练拟合变好而加入最终模型。仍须与最近 50 圈线性尾段基线 RMSE=0.000308 比较。",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
