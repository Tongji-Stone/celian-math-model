"""问题三：PyTorch MLP，用第 1--150 圈预测第 151--200 圈 SOH。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "A" / "问题三" / "output" / "torch_mlp"
OBSERVED_CYCLES = 150
FUTURE_CYCLES = np.arange(151, 201)
SEED = 20260814


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def features(frame: pd.DataFrame, meta: pd.Series) -> np.ndarray:
    """把前 150 圈浓缩为 14 个低维、可解释的输入特征。"""
    observed = frame[frame.cycle <= OBSERVED_CYCLES].set_index("cycle").sort_index()
    soh, ir, temp = observed.SOH_smooth, observed.IR, observed.Tavg
    marks = [1, 25, 50, 75, 100, 125, 150]
    soh_change = [float(soh.loc[c] - soh.loc[1]) for c in marks]
    c1 = 0.0 if pd.isna(meta.C1) else float(meta.C1)
    return np.array(
        [
            c1, float(meta.Q1), float(meta.C2),
            *soh_change,
            float((soh.loc[100] - soh.loc[50]) / 50),
            float((soh.loc[150] - soh.loc[125]) / 25),
            float(ir.loc[126:150].mean() - ir.loc[1:25].mean()),
            float(temp.loc[126:150].mean() - temp.loc[1:25].mean()),
        ], dtype=np.float32,
    )


def target(frame: pd.DataFrame) -> np.ndarray:
    soh = frame.set_index("cycle").SOH_smooth
    return (soh.loc[FUTURE_CYCLES].to_numpy() - soh.loc[OBSERVED_CYCLES]).astype(np.float32)


class ForecastMLP(nn.Module):
    """小网络：14 个输入，50 个未来循环输出。"""

    def __init__(self, n_features: int, n_outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_predict: np.ndarray) -> np.ndarray:
    """仅用拟合折统计标准化参数，防止验证电池的信息泄漏。"""
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_fit = x_scaler.fit_transform(x_fit).astype(np.float32)
    y_fit = y_scaler.fit_transform(y_fit).astype(np.float32)
    x_predict = x_scaler.transform(x_predict).astype(np.float32)
    model = ForecastMLP(x_fit.shape[1], y_fit.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.03)
    criterion = nn.MSELoss()
    x_tensor, y_tensor = torch.from_numpy(x_fit), torch.from_numpy(y_fit)
    model.train()
    for _ in range(1400):
        optimizer.zero_grad()
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x_predict)).numpy()
    return y_scaler.inverse_transform(pred_scaled)


def physical_postprocess(delta: np.ndarray) -> np.ndarray:
    """短期 SOH 不应超过第150圈，且不应在后续循环中持续回升。"""
    return np.minimum.accumulate(np.minimum(delta, 0.0), axis=1)


def linear_baseline(frame: pd.DataFrame) -> np.ndarray:
    tail = frame[(frame.cycle >= 101) & (frame.cycle <= 150)]
    return np.polyval(np.polyfit(tail.cycle, tail.SOH_smooth, 1), FUTURE_CYCLES)


def main() -> None:
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(DATA_DIR / "battery_summary.csv").set_index("battery_id")
    cycles = pd.read_csv(DATA_DIR / "cycle_train.csv")
    last_cycle = cycles.groupby("battery_id").cycle.max()
    train_ids = summary.index[(summary.prediction_test == 0) & (last_cycle == 200)].to_numpy()
    test_ids = summary.index[summary.prediction_test == 1].to_numpy()
    x = np.vstack([features(cycles[cycles.battery_id == bid], summary.loc[bid]) for bid in train_ids])
    y = np.vstack([target(cycles[cycles.battery_id == bid]) for bid in train_ids])

    records = []
    cv_predictions = []
    for fold, (fit, valid) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(train_ids), 1):
        seed_everything()
        predicted = physical_postprocess(fit_predict(x[fit], y[fit], x[valid]))
        for idx, bid in enumerate(train_ids[valid]):
            observed = cycles[cycles.battery_id == bid]
            base = observed.loc[observed.cycle == 150, "SOH_smooth"].iloc[0]
            actual_soh, pred_soh = base + y[valid][idx], base + predicted[idx]
            linear = linear_baseline(observed)
            records.append({
                "fold": fold, "battery_id": int(bid), "policy": summary.loc[bid, "policy"],
                "torch_mlp_rmse": float(np.sqrt(mean_squared_error(actual_soh, pred_soh))),
                "torch_mlp_mae": float(mean_absolute_error(actual_soh, pred_soh)),
                "linear_rmse": float(np.sqrt(mean_squared_error(actual_soh, linear))),
                "linear_mae": float(mean_absolute_error(actual_soh, linear)),
            })
            cv_predictions.extend({"battery_id": int(bid), "cycle": int(cycle), "actual_SOH": float(actual), "prediction_SOH": float(pred)} for cycle, actual, pred in zip(FUTURE_CYCLES, actual_soh, pred_soh))
    cv = pd.DataFrame(records).sort_values("battery_id")
    cv.to_csv(OUT_DIR / "cv_by_battery.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv_predictions).to_csv(OUT_DIR / "cv_predictions.csv", index=False, encoding="utf-8-sig")

    seed_everything()
    test_delta = physical_postprocess(fit_predict(x, y, np.vstack([features(cycles[cycles.battery_id == bid], summary.loc[bid]) for bid in test_ids])))
    test_rows = []
    for idx, bid in enumerate(test_ids):
        base = cycles.loc[(cycles.battery_id == bid) & (cycles.cycle == 150), "SOH_smooth"].iloc[0]
        test_rows.extend({"battery_id": int(bid), "cycle": int(cycle), "SOH_prediction": float(base + delta), "method": "PyTorch_MLP_150_to_200"} for cycle, delta in zip(FUTURE_CYCLES, test_delta[idx]))
    test_prediction = pd.DataFrame(test_rows)
    test_prediction.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    overview = test_prediction.groupby("battery_id").agg(soh_151=("SOH_prediction", "first"), soh_200=("SOH_prediction", "last")).reset_index().merge(summary[["policy"]], left_on="battery_id", right_index=True)
    overview.to_csv(OUT_DIR / "test_prediction_overview.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for bid in test_ids:
        observed, forecast = cycles[cycles.battery_id == bid], test_prediction[test_prediction.battery_id == bid]
        color = ax.plot(observed.cycle, observed.SOH_smooth, linewidth=1.2, label=f"{bid}：已观测")[0].get_color()
        ax.plot(forecast.cycle, forecast.SOH_prediction, "--", color=color, linewidth=1.6)
    ax.axvline(150, color="black", linestyle=":", linewidth=1, label="第150圈：预测起点")
    ax.set(xlabel="循环次数", ylabel="SOH（平滑值）", title="问题三：PyTorch MLP 对测试电池的 SOH 外推")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "test_forecasts.png", dpi=220)
    plt.close(fig)

    metrics = {
        "implementation": "PyTorch 2.13.0+cpu",
        "training_batteries": int(len(train_ids)), "test_batteries": int(len(test_ids)), "forecast_horizon": 50,
        "cv_torch_mlp_rmse_mean": float(cv.torch_mlp_rmse.mean()), "cv_torch_mlp_mae_mean": float(cv.torch_mlp_mae.mean()),
        "cv_linear_rmse_mean": float(cv.linear_rmse.mean()), "cv_linear_mae_mean": float(cv.linear_mae.mean()),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
