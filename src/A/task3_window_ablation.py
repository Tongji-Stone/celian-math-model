"""问题三小问 1 与 3：不同早期窗口、策略特征的 SOH 预测比较。

训练及验证只使用赛题 CSV。每次以整块电池为单位留出，避免同一电池的循环点
同时出现在训练与验证中。外部 EOL 标签不被本脚本读取。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "A" / "问题三" / "output" / "window_ablation"
SEED = 20260815
HORIZON = 50
WINDOWS = (50, 100, 150)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def make_features(frame: pd.DataFrame, meta: pd.Series, end: int, use_policy: bool) -> np.ndarray:
    """提取到 end 圈为止真正可观测的低维特征。"""
    obs = frame[frame.cycle <= end].set_index("cycle").sort_index()
    soh, ir, temp = obs.SOH_smooth, obs.IR, obs.Tavg
    marks = np.linspace(1, end, 7).round().astype(int)
    first, split = soh.loc[1], max(2, end // 2)
    dynamic = [
        *[float(soh.loc[m] - first) for m in marks],
        float((soh.loc[end] - soh.loc[split]) / (end - split)),
        float((soh.loc[split] - first) / (split - 1)),
        float(ir.loc[split:end].mean() - ir.loc[1:split].mean()),
        float(temp.loc[split:end].mean() - temp.loc[1:split].mean()),
    ]
    if not use_policy:
        return np.asarray(dynamic, dtype=np.float32)
    c1 = 0.0 if pd.isna(meta.C1) else float(meta.C1)
    return np.asarray([c1, float(meta.Q1), float(meta.C2), *dynamic], dtype=np.float32)


def make_target(frame: pd.DataFrame, end: int) -> np.ndarray:
    soh = frame.set_index("cycle").SOH_smooth
    return (soh.loc[np.arange(end + 1, end + HORIZON + 1)].to_numpy() - soh.loc[end]).astype(np.float32)


class SmallMLP(nn.Module):
    def __init__(self, n_input: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_input, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, HORIZON))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mlp_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """在每个训练折内部独立标准化；固定训练轮数以免用验证折早停。"""
    sx, sy = StandardScaler(), StandardScaler()
    xf = torch.tensor(sx.fit_transform(x_fit), dtype=torch.float32)
    yf = torch.tensor(sy.fit_transform(y_fit), dtype=torch.float32)
    xe = torch.tensor(sx.transform(x_eval), dtype=torch.float32)
    model = SmallMLP(x_fit.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.03)
    for _ in range(700):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xf), yf)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return sy.inverse_transform(model(xe).numpy())


def enforce_monotone(delta: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.minimum(delta, 0.0), axis=1)


def linear_predict(frame: pd.DataFrame, end: int) -> np.ndarray:
    tail = frame[(frame.cycle > end - min(50, end - 1)) & (frame.cycle <= end)]
    return np.polyval(np.polyfit(tail.cycle, tail.SOH_smooth, 1), np.arange(end + 1, end + HORIZON + 1))


def main() -> None:
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "data" / "battery_summary.csv").set_index("battery_id")
    cycles = pd.read_csv(ROOT / "data" / "cycle_train.csv")
    full_ids = summary.index[(summary.prediction_test.eq(0)) & (cycles.groupby("battery_id").cycle.max().eq(200))].to_numpy()
    rows: list[dict[str, object]] = []
    for end in WINDOWS:
        for use_policy in (False, True):
            x = np.vstack([make_features(cycles[cycles.battery_id == bid], summary.loc[bid], end, use_policy) for bid in full_ids])
            y = np.vstack([make_target(cycles[cycles.battery_id == bid], end) for bid in full_ids])
            for fold, (fit, valid) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(full_ids), 1):
                set_seed()
                delta = enforce_monotone(mlp_predict(x[fit], y[fit], x[valid]))
                for pos, bid in enumerate(full_ids[valid]):
                    frame = cycles[cycles.battery_id == bid]
                    base = frame.loc[frame.cycle.eq(end), "SOH_smooth"].iloc[0]
                    actual = base + y[valid][pos]
                    mlp = base + delta[pos]
                    linear = linear_predict(frame, end)
                    for method, prediction in (("torch_mlp", mlp), ("linear_tail", linear)):
                        rows.append({
                            "window_end": end, "use_policy": use_policy, "fold": fold, "battery_id": int(bid), "method": method,
                            "rmse": float(np.sqrt(mean_squared_error(actual, prediction))), "mae": float(mean_absolute_error(actual, prediction)),
                            "r2": float(r2_score(actual, prediction)),
                        })
    detail = pd.DataFrame(rows).sort_values(["window_end", "use_policy", "method", "battery_id"])
    detail.to_csv(OUT / "cv_detail.csv", index=False, encoding="utf-8-sig")
    report = detail.groupby(["window_end", "use_policy", "method"], as_index=False).agg(
        batteries=("battery_id", "count"), rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), r2_mean=("r2", "mean")
    )
    report.to_csv(OUT / "cv_summary.csv", index=False, encoding="utf-8-sig")
    # 对题设 9 块测试电池，采用在 end=150 上验证最优的线性尾段外推。
    test_ids = summary.index[summary.prediction_test.eq(1)].to_numpy()
    prediction_rows: list[dict[str, object]] = []
    for bid in test_ids:
        forecast = linear_predict(cycles[cycles.battery_id == bid], 150)
        prediction_rows.extend(
            {"battery_id": int(bid), "cycle": int(cycle), "SOH_prediction": float(value), "method": "linear_tail_50_selected"}
            for cycle, value in zip(np.arange(151, 201), forecast)
        )
    final_prediction = pd.DataFrame(prediction_rows)
    final_prediction.to_csv(OUT / "selected_test_soh_151_200.csv", index=False, encoding="utf-8-sig")
    final_prediction.groupby("battery_id", as_index=False).agg(soh_151=("SOH_prediction", "first"), soh_200=("SOH_prediction", "last")).merge(
        summary[["policy"]], left_on="battery_id", right_index=True
    ).to_csv(OUT / "selected_test_soh_overview.csv", index=False, encoding="utf-8-sig")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for bid in test_ids:
        observed = cycles[cycles.battery_id.eq(bid)]
        predicted = final_prediction[final_prediction.battery_id.eq(bid)]
        color = ax.plot(observed.cycle, observed.SOH_smooth, linewidth=1.15, label=f"{bid}：观测")[0].get_color()
        ax.plot(predicted.cycle, predicted.SOH_prediction, "--", color=color, linewidth=1.7)
    ax.axvline(150, color="black", linestyle=":", linewidth=1, label="第150圈：预测起点")
    ax.set(xlabel="循环次数", ylabel="平滑 SOH", title="问题三：测试电池第151--200圈 SOH 预测")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "selected_test_soh_forecast.png", dpi=220)
    plt.close(fig)
    (OUT / "README.md").write_text(
        "# 问题三：窗口与策略特征消融\n\n"
        "每一行验证结果都由一块完整电池的连续 50 圈构成；整块电池不会出现在对应训练折。"
        "`use_policy=True` 表示加入 C1、Q1、C2；外部 EOL 标签没有被读取。\n",
        encoding="utf-8",
    )
    print(report.to_string(index=False))
    print(json.dumps({"full_batteries": int(len(full_ids)), "windows": list(WINDOWS), "horizon": HORIZON}, ensure_ascii=False))


if __name__ == "__main__":
    main()
