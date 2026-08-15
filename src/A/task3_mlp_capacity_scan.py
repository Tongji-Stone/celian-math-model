"""问题三：扫描不同容量 MLP 在 150->200 SOH 任务上的泛化表现。

所有模型复用 task3_window_ablation 的特征、目标、单调后处理和按电池 5 折划分；
只改变 MLP 隐藏层宽度。外部 EOL 标签完全不读取。
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from task3_window_ablation import HORIZON, ROOT, enforce_monotone, make_features, make_target


OUT = ROOT / "A" / "问题三" / "output" / "mlp_capacity_scan"
END = 150
FOLD_SEED = 20260815
INIT_SEEDS = (20260815, 20260816, 20260817)
ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "linear": (),
    "bottleneck_4": (4,),
    "hidden_8": (8,),
    "hidden_16": (16,),
    "hidden_16_8": (16, 8),
    "hidden_32_16": (32, 16),
    "hidden_64_32": (64, 32),
    "hidden_128_64": (128, 64),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class CapacityMLP(nn.Module):
    def __init__(self, n_input: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        widths = (n_input, *hidden, HORIZON)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(n_input: int, hidden: tuple[int, ...]) -> int:
    return sum((left + 1) * right for left, right in zip((n_input, *hidden), (*hidden, HORIZON)))


def predict(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray, hidden: tuple[int, ...], seed: int) -> np.ndarray:
    """训练设置保持和原 MLP 一致，避免容量之外的因素混入比较。"""
    set_seed(seed)
    sx, sy = StandardScaler(), StandardScaler()
    xf = torch.tensor(sx.fit_transform(x_fit), dtype=torch.float32)
    yf = torch.tensor(sy.fit_transform(y_fit), dtype=torch.float32)
    xe = torch.tensor(sx.transform(x_eval), dtype=torch.float32)
    model = CapacityMLP(x_fit.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.03)
    for _ in range(700):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(xf), yf)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return sy.inverse_transform(model(xe).numpy())


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
    fold_indices = list(KFold(5, shuffle=True, random_state=FOLD_SEED).split(full_ids))
    rows: list[dict[str, object]] = []
    for use_policy in (False, True):
        x = np.vstack([make_features(cycles[cycles.battery_id == bid], summary.loc[bid], END, use_policy) for bid in full_ids])
        y = np.vstack([make_target(cycles[cycles.battery_id == bid], END) for bid in full_ids])
        for name, hidden in ARCHITECTURES.items():
            params = count_parameters(x.shape[1], hidden)
            for init_seed in INIT_SEEDS:
                for fold, (fit, valid) in enumerate(fold_indices, 1):
                    delta = enforce_monotone(predict(x[fit], y[fit], x[valid], hidden, init_seed))
                    for position, bid in enumerate(full_ids[valid]):
                        frame = cycles[cycles.battery_id == bid]
                        base = float(frame.loc[frame.cycle.eq(END), "SOH_smooth"].iloc[0])
                        actual = base + y[valid][position]
                        estimated = base + delta[position]
                        rows.append({
                            "use_policy": use_policy, "architecture": name,
                            "hidden_layers": "-".join(map(str, hidden)) or "none",
                            "parameters": params, "n_input": x.shape[1], "seed": init_seed,
                            "fold": fold, "battery_id": int(bid),
                            "rmse": float(np.sqrt(mean_squared_error(actual, estimated))),
                            "mae": float(mean_absolute_error(actual, estimated)),
                        })
    detail = pd.DataFrame(rows)
    summary_table = detail.groupby(["use_policy", "architecture", "hidden_layers", "parameters", "n_input"], as_index=False).agg(
        batteries_times_seeds=("battery_id", "size"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
    ).sort_values(["use_policy", "rmse_mean"])
    detail.to_csv(OUT / "capacity_cv_detail.csv", index=False, encoding="utf-8-sig")
    summary_table.to_csv(OUT / "capacity_cv_summary.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for use_policy, subset in summary_table.groupby("use_policy"):
        label = "含 C1,Q1,C2" if use_policy else "仅动态特征"
        ax.errorbar(subset.parameters, subset.rmse_mean, yerr=subset.rmse_sd, marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("可训练参数量（对数刻度）")
    ax.set_ylabel("5 折 × 3 初始化的平均 RMSE")
    ax.set_title("问题三：MLP 容量扫描（150 圈预测后 50 圈）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "capacity_vs_rmse.png", dpi=250)
    plt.close(fig)

    lines = [
        "# 问题三：MLP 参数量扫描", "",
        "## 目的", "",
        "检验原始 `16-8` 小 MLP 的容量是否任意。所有模型只改变隐藏层宽度；使用相同的前 150 圈特征、后 50 圈目标、5 折按电池划分、AdamW（700 轮）和单调后处理。",
        "每个结构使用 3 个固定随机初始化，因此每行汇总 40 块电池 × 3 次初始化 = 120 个验证误差。`use_policy=True` 额外加入 C1、Q1、C2；外部 EOL 标签没有被读取。", "",
        "## 结果", "", markdown_table(summary_table), "",
        "## 如何解读", "",
        "- 参数量是包含权重和偏置的总可训练参数数。例如 11 维输入的 `16-8` 为 778 个，14 维输入的 `16-8` 为 826 个。",
        "- 该扫描的对象是 MLP 间的相对比较；仍应与线性尾段基线 RMSE=0.000308 对照。若所有 MLP 都更差，则不因某个网络相对最优而取代线性模型。",
        "- 只有 40 块独立电池，参数量上升后的误差变化要结合标准差和留一电池泛化理解，不能只看单次训练误差。",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
