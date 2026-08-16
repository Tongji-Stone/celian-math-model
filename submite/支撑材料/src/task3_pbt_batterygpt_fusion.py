"""问题三方案 B：PBT 寿命头 × BatteryGPT 轨迹头的双头融合（另存，不覆盖原模型）。

运行（项目根目录）：
    python src/A/task3_pbt_batterygpt_fusion.py

设计（与论文权重无关，仅用赛题附件训练）：
    共享 Transformer 编码器（PBT 风格的协议门控双专家 MoE）
    + 轨迹头：在近段线性基线上生成 151–200 圈 SOH 残差（BatteryGPT 风格）
    + 寿命头：预测 log N_EOL
    + 一致性损失：由轨迹外推的寿命与寿命头对齐。

验证口径与原脚本一致：40 块满 200 圈电池、只给 1–150、预报 151–200。
输出目录：A/问题三/fusion_pbt_batterygpt/ （不写回 A/问题三/output/）
"""

from __future__ import annotations

import json
import math
import random
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from SOH_plot import configure_chinese_font  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "A" / "问题三" / "fusion_pbt_batterygpt"
FIG_DIR = OUT_DIR / "figures"
WEIGHT_DIR = OUT_DIR / "weights"

K = 150
HORIZON = (151, 200)
HORIZON_LEN = HORIZON[1] - K
N_EOL_CAP = 20000.0
RECENT_LEN = 50
SEQ_FEAT = ["soh", "dsoh", "ir", "tavg", "chargetime", "cycle_norm"]
STATIC_KEYS = [
    "C1",
    "Q1",
    "C2",
    "initial_capacity",
    "mean_chargetime",
    "mean_IR",
    "mean_Tavg",
    "soh_k",
    "recent_slope",
    "policy_slope",
]
SEED = 20260815


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    return {"a": float(coef[0]), "b": float(coef[1])}


def linear_from_level_slope(level_cycle: float, level_soh: float, slope: float) -> dict[str, float]:
    return {"a": float(level_soh - slope * level_cycle), "b": float(slope)}


def n_eol_from_fit(fit: dict[str, float], soh_now: float, cycle_now: float) -> tuple[float, str]:
    slope = fit["b"]
    if slope >= -1e-12:
        return N_EOL_CAP, "non_decreasing"
    value = (0.80 - fit["a"]) / slope
    if value <= cycle_now:
        return float(cycle_now + (soh_now - 0.80) / max(-slope, 1e-12)), "ok"
    return float(min(value, N_EOL_CAP)), "ok" if value < N_EOL_CAP else "censored"


def n_eol_from_level_slope(soh_k: float, cycle_k: float, slope: float) -> float:
    fit = linear_from_level_slope(cycle_k, soh_k, slope)
    value, _flag = n_eol_from_fit(fit, soh_k, cycle_k)
    return float(value)


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


def collect_train_cells(summary: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    max_cycle = cycles.groupby("battery_id")["cycle"].max().rename("max_cycle")
    cell = summary.merge(max_cycle, on="battery_id", validate="one_to_one")
    return cell.query("prediction_test == 0 and max_cycle == 200").copy()


def collect_test_cells(summary: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    max_cycle = cycles.groupby("battery_id")["cycle"].max().rename("max_cycle")
    cell = summary.merge(max_cycle, on="battery_id", validate="one_to_one")
    return cell.query("prediction_test == 1").copy()


def series_policy(cycles: pd.DataFrame, battery_id: int) -> str:
    return str(cycles.loc[cycles["battery_id"] == battery_id, "policy"].iloc[0])


def policy_slope_map(cycles: pd.DataFrame, train_ids: list[int], k: int, exclude: int | None = None) -> dict[str, float]:
    slopes: dict[str, list[float]] = {}
    for battery_id in train_ids:
        if exclude is not None and int(battery_id) == int(exclude):
            continue
        series = cell_series(cycles, battery_id)
        n, y = window_xy(series, k, recent=False)
        if len(n) < 5:
            continue
        fit = fit_linear(n, y)
        policy = str(series_policy(cycles, int(battery_id)))
        slopes.setdefault(policy, []).append(fit["b"])
    return {key: float(np.mean(vals)) for key, vals in slopes.items() if vals}


def pack_one_cell(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    battery_id: int,
    k: int,
    policy_slopes: dict[str, float],
    global_slope: float,
    with_future: bool,
) -> dict[str, np.ndarray | float | int | str]:
    series = cell_series(cycles, battery_id)
    full = cycles.loc[cycles["battery_id"] == battery_id].sort_values("cycle")
    part = full.loc[full["cycle"] <= k].copy()
    if len(part) < k:
        raise ValueError(f"battery {battery_id} has only {len(part)} cycles before {k}")
    part = part.tail(k)
    soh = part["SOH_smooth"].to_numpy(float)
    ir = part["IR"].to_numpy(float)
    ir = np.where(ir <= 0, np.nan, ir)
    if np.all(np.isnan(ir)):
        ir = np.zeros_like(soh)
    else:
        med = float(np.nanmedian(ir))
        ir = np.where(np.isnan(ir), med, ir)
    tavg = part["Tavg"].to_numpy(float)
    chg = part["chargetime"].to_numpy(float)
    dsoh = np.diff(soh, prepend=soh[0])
    cycle_norm = part["cycle"].to_numpy(float) / 200.0
    seq = np.stack([soh, dsoh, ir, tavg, chg, cycle_norm], axis=1)

    meta = summary.loc[summary["battery_id"] == battery_id].iloc[0]
    c1 = float(meta["C1"]) if pd.notna(meta["C1"]) else float(meta["C2"])
    c2 = float(meta["C2"])
    q1 = float(meta["Q1"])
    soh_k = soh_at(series, k)
    n_recent, y_recent = window_xy(series, k, recent=True)
    recent_slope = fit_linear(n_recent, y_recent)["b"]
    policy = series_policy(cycles, battery_id)
    policy_slope = float(policy_slopes.get(policy, global_slope))
    static = np.array(
        [
            c1,
            q1,
            c2,
            float(meta["initial_capacity"]),
            float(meta["mean_chargetime"]),
            float(meta["mean_IR"]),
            float(meta["mean_Tavg"]),
            soh_k,
            recent_slope,
            policy_slope,
        ],
        dtype=float,
    )

    baseline = soh_k + recent_slope * (np.arange(1, HORIZON_LEN + 1, dtype=float))
    payload: dict[str, np.ndarray | float | int | str] = {
        "battery_id": int(battery_id),
        "policy": policy,
        "seq": seq.astype(np.float32),
        "static": static.astype(np.float32),
        "soh_k": float(soh_k),
        "recent_slope": float(recent_slope),
        "policy_slope": float(policy_slope),
        "baseline": baseline.astype(np.float32),
        "n_eol_lin": n_eol_from_level_slope(soh_k, k, recent_slope),
    }
    if with_future:
        n_true = np.arange(k + 1, HORIZON[1] + 1, dtype=float)
        y_true = np.array([soh_at(series, int(n)) for n in n_true], dtype=float)
        payload["y_true"] = y_true.astype(np.float32)
        payload["residual"] = (y_true - baseline).astype(np.float32)
        true_slope = float((y_true[-1] - soh_k) / HORIZON_LEN)
        payload["n_eol_trueish"] = n_eol_from_level_slope(soh_k, k, true_slope)
        payload["soh200_true"] = float(y_true[-1])
    return payload


def stack_field(packs: list[dict], key: str) -> np.ndarray:
    return np.stack([np.asarray(p[key]) for p in packs], axis=0)


class Standardizer:
    def __init__(self) -> None:
        self.seq_mean = None
        self.seq_std = None
        self.static_mean = None
        self.static_std = None

    def fit(self, packs: list[dict]) -> "Standardizer":
        seq = stack_field(packs, "seq")
        static = stack_field(packs, "static")
        self.seq_mean = seq.mean(axis=(0, 1))
        self.seq_std = seq.std(axis=(0, 1)) + 1e-6
        self.static_mean = static.mean(axis=0)
        self.static_std = static.std(axis=0) + 1e-6
        return self

    def transform_seq(self, seq: np.ndarray) -> np.ndarray:
        return (seq - self.seq_mean) / self.seq_std

    def transform_static(self, static: np.ndarray) -> np.ndarray:
        return (static - self.static_mean) / self.static_std

    def state_dict(self) -> dict:
        return {
            "seq_mean": self.seq_mean.tolist(),
            "seq_std": self.seq_std.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.seq_mean = np.array(state["seq_mean"], dtype=np.float32)
        self.seq_std = np.array(state["seq_std"], dtype=np.float32)
        self.static_mean = np.array(state["static_mean"], dtype=np.float32)
        self.static_std = np.array(state["static_std"], dtype=np.float32)


class MixtureOfExperts(nn.Module):
    """PBT 风格：协议特征门控的两个专家。"""

    def __init__(self, dim: int, n_experts: int = 2) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(h), dim=-1)
        stacked = torch.stack([expert(h) for expert in self.experts], dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)


class DualHeadFusion(nn.Module):
    """共享编码 + 轨迹头（BatteryGPT）+ 寿命头（PBT）。"""

    def __init__(
        self,
        seq_dim: int = 6,
        static_dim: int = 10,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        horizon: int = HORIZON_LEN,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.seq_in = nn.Linear(seq_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.moe = MixtureOfExperts(d_model, n_experts=2)
        self.recon_head = nn.Linear(d_model, 1)
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self.life_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.traj_head[-1].weight)
        nn.init.zeros_(self.traj_head[-1].bias)
        nn.init.zeros_(self.life_head[-1].weight)
        nn.init.zeros_(self.life_head[-1].bias)

    def encode(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        x = self.seq_in(seq)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        fused = pooled + self.static_mlp(static)
        return self.moe(fused)

    def reconstruct(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        x = self.seq_in(seq)
        h = self.encoder(x)
        return self.recon_head(h).squeeze(-1)

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(seq, static)
        raw = self.traj_head(h)
        # 有界修正：最多大约 ±5e-5 /圈 的斜率调整，避免 50 维残差爆炸。
        slope_delta = 5e-5 * torch.tanh(raw[:, 0])
        curve = 2e-8 * torch.tanh(raw[:, 1])
        steps = torch.arange(1, self.horizon + 1, device=seq.device, dtype=seq.dtype)
        residual = slope_delta.unsqueeze(1) * steps + curve.unsqueeze(1) * (steps**2)
        log_n_adj = 0.25 * torch.tanh(self.life_head(h).squeeze(-1))
        return residual, slope_delta, log_n_adj


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def batches(n: int, batch_size: int) -> list[np.ndarray]:
    idx = np.arange(n)
    np.random.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def pretrain_encoder(
    model: DualHeadFusion,
    scaler: Standardizer,
    packs: list[dict],
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    seq_all = scaler.transform_seq(stack_field(packs, "seq"))
    static_all = scaler.transform_static(stack_field(packs, "static"))
    soh_true = stack_field(packs, "seq")[:, :, 0]
    soh_mean = scaler.seq_mean[0]
    soh_std = scaler.seq_std[0]
    soh_z = (soh_true - soh_mean) / soh_std
    model.train()
    for _epoch in range(epochs):
        for idx in batches(len(packs), batch_size):
            seq = to_tensor(seq_all[idx], device)
            static = to_tensor(static_all[idx], device)
            target = to_tensor(soh_z[idx], device)
            mask = torch.rand(seq.shape[:2], device=device) < 0.2
            seq_masked = seq.clone()
            seq_masked[:, :, 0] = torch.where(mask, torch.zeros_like(seq_masked[:, :, 0]), seq_masked[:, :, 0])
            pred = model.reconstruct(seq_masked, static)
            loss = F.mse_loss(pred[mask], target[mask]) if mask.any() else F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def finetune_dual_head(
    model: DualHeadFusion,
    scaler: Standardizer,
    packs: list[dict],
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 8e-4,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    seq_all = scaler.transform_seq(stack_field(packs, "seq"))
    static_all = scaler.transform_static(stack_field(packs, "static"))
    residual = stack_field(packs, "residual")
    baseline = stack_field(packs, "baseline")
    y_true = stack_field(packs, "y_true")
    log_n = np.log(np.clip(np.array([p["n_eol_trueish"] for p in packs], dtype=np.float32), 200.0, N_EOL_CAP))
    model.train()
    for _epoch in range(epochs):
        for idx in batches(len(packs), batch_size):
            seq = to_tensor(seq_all[idx], device)
            static = to_tensor(static_all[idx], device)
            res_t = to_tensor(residual[idx], device)
            base_t = to_tensor(baseline[idx], device)
            y_t = to_tensor(y_true[idx], device)
            log_t = to_tensor(log_n[idx], device)
            soh_k = to_tensor(np.array([packs[i]["soh_k"] for i in idx], dtype=np.float32), device)
            pred_res, _slope_delta, log_n_adj = model(seq, static)
            y_hat = base_t + pred_res
            log_lin = torch.log(
                to_tensor(np.array([packs[i]["n_eol_lin"] for i in idx], dtype=np.float32), device).clamp(min=200.0)
            )
            pred_logn = log_lin + log_n_adj
            loss_traj = F.mse_loss(y_hat, y_t)
            loss_res = F.mse_loss(pred_res, res_t)
            loss_life = F.mse_loss(pred_logn, log_t)
            last_slope = (y_hat[:, -1] - soh_k) / float(HORIZON_LEN)
            n_from_traj = torch.clamp(
                K + (soh_k - 0.80) / (-last_slope).clamp(min=1e-6),
                min=200.0,
                max=float(N_EOL_CAP),
            )
            loss_cons = F.mse_loss(torch.log(n_from_traj), pred_logn)
            loss = loss_traj + 0.3 * loss_res + 0.2 * loss_life + 0.1 * loss_cons
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


@torch.no_grad()
def predict_pack(
    model: DualHeadFusion,
    scaler: Standardizer,
    pack: dict,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    model.eval()
    seq = to_tensor(scaler.transform_seq(pack["seq"][None, ...]), device)
    static = to_tensor(scaler.transform_static(pack["static"][None, ...]), device)
    residual, _slope_delta, log_n_adj = model(seq, static)
    residual = residual.cpu().numpy()[0]
    y_hat = np.asarray(pack["baseline"], dtype=float) + residual
    log_lin = math.log(max(float(pack["n_eol_lin"]), 200.0))
    n_eol_life = float(np.clip(math.exp(log_lin + float(log_n_adj.cpu().numpy()[0])), 200.0, N_EOL_CAP))
    slope_hat = float((y_hat[-1] - pack["soh_k"]) / HORIZON_LEN)
    n_eol_traj = n_eol_from_level_slope(float(pack["soh_k"]), K, slope_hat)
    n_eol_blend = float(np.exp(0.5 * np.log(n_eol_life) + 0.5 * np.log(max(n_eol_traj, 200.0))))
    return {
        "y_pred": y_hat.astype(float),
        "residual": residual.astype(float),
        "n_eol_life": n_eol_life,
        "n_eol_traj": n_eol_traj,
        "n_eol": float(np.clip(n_eol_blend, 200.0, N_EOL_CAP)),
        "slope_hat": slope_hat,
    }


def cell_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "bias": float(np.mean(resid)),
        "soh200_abs_err": float(abs(y_true[-1] - y_pred[-1])),
    }


def build_policy_slopes(cycles: pd.DataFrame, train_ids: list[int], exclude: int | None) -> tuple[dict[str, float], float]:
    slopes = policy_slope_map(cycles, train_ids, K, exclude=exclude)
    global_slope = float(np.mean(list(slopes.values()))) if slopes else 0.0
    return slopes, global_slope


def run_loocv(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    train_ids: list[int],
    device: torch.device,
    pretrain_epochs: int,
    finetune_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    point_rows = []
    # 自监督只看 1–150，可在全部验证电池上预训练一次，不泄漏 151–200。
    slopes_all, g_all = build_policy_slopes(cycles, train_ids, exclude=None)
    pre_packs = [
        pack_one_cell(summary, cycles, i, K, slopes_all, g_all, with_future=False) for i in train_ids
    ]
    pre_scaler = Standardizer().fit(pre_packs)
    pretrained = DualHeadFusion(seq_dim=6, static_dim=len(STATIC_KEYS)).to(device)
    pretrain_encoder(pretrained, pre_scaler, pre_packs, device, epochs=pretrain_epochs)
    pre_state = {k: v.detach().cpu().clone() for k, v in pretrained.state_dict().items()}
    print("pretrain done; starting LOOCV finetune")

    for left_out in train_ids:
        fold_ids = [i for i in train_ids if i != left_out]
        slopes_tr, g_tr = build_policy_slopes(cycles, fold_ids, exclude=None)
        train_packs = [
            pack_one_cell(summary, cycles, i, K, slopes_tr, g_tr, with_future=True) for i in fold_ids
        ]
        slopes_te, g_te = build_policy_slopes(cycles, fold_ids, exclude=None)
        test_pack = pack_one_cell(summary, cycles, left_out, K, slopes_te, g_te, with_future=True)
        scaler = Standardizer().fit(train_packs)
        model = DualHeadFusion(seq_dim=6, static_dim=len(STATIC_KEYS)).to(device)
        model.load_state_dict(pre_state)
        finetune_dual_head(model, scaler, train_packs, device, epochs=finetune_epochs)
        pred = predict_pack(model, scaler, test_pack, device)
        y_true = np.asarray(test_pack["y_true"], float)
        y_lin = np.asarray(test_pack["baseline"], float)
        stats_f = cell_metrics(y_true, pred["y_pred"])
        stats_l = cell_metrics(y_true, y_lin)
        rows.append(
            {
                "battery_id": left_out,
                "policy": test_pack["policy"],
                "model": "fusion_pbt_batterygpt",
                "k": K,
                **stats_f,
                "soh200_true": float(test_pack["soh200_true"]),
                "soh200_pred": float(pred["y_pred"][-1]),
                "n_eol": pred["n_eol"],
                "n_eol_life": pred["n_eol_life"],
                "n_eol_traj": pred["n_eol_traj"],
                "n_eol_lin": test_pack["n_eol_lin"],
                "recent_slope": test_pack["recent_slope"],
                "slope_hat": pred["slope_hat"],
            }
        )
        rows.append(
            {
                "battery_id": left_out,
                "policy": test_pack["policy"],
                "model": "linear_recent",
                "k": K,
                **stats_l,
                "soh200_true": float(test_pack["soh200_true"]),
                "soh200_pred": float(y_lin[-1]),
                "n_eol": test_pack["n_eol_lin"],
                "n_eol_life": test_pack["n_eol_lin"],
                "n_eol_traj": test_pack["n_eol_lin"],
                "n_eol_lin": test_pack["n_eol_lin"],
                "recent_slope": test_pack["recent_slope"],
                "slope_hat": test_pack["recent_slope"],
            }
        )
        future = np.arange(K + 1, HORIZON[1] + 1)
        for cycle, yt, yf, yl in zip(future, y_true, pred["y_pred"], y_lin):
            point_rows.append(
                {
                    "battery_id": left_out,
                    "policy": test_pack["policy"],
                    "cycle": int(cycle),
                    "soh_true": float(yt),
                    "soh_fusion": float(yf),
                    "soh_linear_recent": float(yl),
                }
            )
        print(f"LOOCV battery {left_out}: fusion RMSE={stats_f['rmse']:.6f}  linear RMSE={stats_l['rmse']:.6f}")

    cell_scores = pd.DataFrame(rows)
    points = pd.DataFrame(point_rows)
    summary_tbl = (
        cell_scores.groupby("model", as_index=False)
        .agg(
            n_cells=("battery_id", "nunique"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            soh200_mae=("soh200_abs_err", "mean"),
            bias_mean=("bias", "mean"),
            n_eol_median=("n_eol", "median"),
        )
        .sort_values("rmse_mean")
    )
    return cell_scores, points, summary_tbl


def train_final_and_predict_test(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    train_ids: list[int],
    device: torch.device,
    pretrain_epochs: int,
    finetune_epochs: int,
) -> tuple[pd.DataFrame, DualHeadFusion, Standardizer]:
    slopes, g = build_policy_slopes(cycles, train_ids, exclude=None)
    train_packs = [pack_one_cell(summary, cycles, i, K, slopes, g, with_future=True) for i in train_ids]
    scaler = Standardizer().fit(train_packs)
    model = DualHeadFusion(seq_dim=6, static_dim=len(STATIC_KEYS)).to(device)
    pretrain_encoder(model, scaler, train_packs, device, epochs=pretrain_epochs)
    finetune_dual_head(model, scaler, train_packs, device, epochs=finetune_epochs)
    test = collect_test_cells(summary, cycles)
    rows = []
    traj_rows = []
    for battery_id in test["battery_id"].astype(int):
        pack = pack_one_cell(summary, cycles, int(battery_id), K, slopes, g, with_future=False)
        pred = predict_pack(model, scaler, pack, device)
        y_lin = np.asarray(pack["baseline"], float)
        rows.append(
            {
                "battery_id": int(battery_id),
                "policy": pack["policy"],
                "soh_150": pack["soh_k"],
                "soh_200_fusion": float(pred["y_pred"][-1]),
                "soh_200_linear_recent": float(y_lin[-1]),
                "n_eol_fusion": pred["n_eol"],
                "n_eol_life_head": pred["n_eol_life"],
                "n_eol_traj_head": pred["n_eol_traj"],
                "n_eol_linear_recent": pack["n_eol_lin"],
                "slope_fusion": pred["slope_hat"],
                "slope_linear_recent": pack["recent_slope"],
            }
        )
        future = np.arange(K + 1, HORIZON[1] + 1)
        observed = cycles.loc[
            (cycles["battery_id"] == battery_id) & (cycles["cycle"] <= K),
            ["cycle", "SOH_smooth"],
        ].copy()
        observed["source"] = "observed"
        fut = pd.DataFrame(
            {
                "cycle": future,
                "SOH_smooth": pred["y_pred"],
                "source": "fusion_predicted",
            }
        )
        both = pd.concat([observed, fut], ignore_index=True)
        both["battery_id"] = int(battery_id)
        both["policy"] = pack["policy"]
        traj_rows.append(both)
    pred_df = pd.DataFrame(rows)
    traj_df = pd.concat(traj_rows, ignore_index=True)
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    traj_df.to_csv(OUT_DIR / "test_trajectories.csv", index=False)
    torch.save(
        {"model": model.state_dict(), "scaler": scaler.state_dict()},
        WEIGHT_DIR / "fusion_final_all40.pt",
    )
    return pred_df, model, scaler


def save_figures(cell_scores: pd.DataFrame, points: pd.DataFrame, test_pred: pd.DataFrame) -> None:
    configure_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    summary = cell_scores.groupby("model", as_index=False)["rmse"].mean()
    colors = ["#4c78a8" if m == "linear_recent" else "#e45756" for m in summary["model"]]
    ax.bar(summary["model"], summary["rmse"], color=colors)
    ax.set_ylabel("40 块电池 151–200 圈 RMSE 均值")
    ax.set_title("方案 B 融合 vs 近段线性")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rmse_fusion_vs_linear.png", dpi=200)
    plt.close(fig)

    wide = cell_scores.pivot(index="battery_id", columns="model", values="rmse")
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(wide["linear_recent"], wide["fusion_pbt_batterygpt"], c="#4c78a8")
    lims = [
        0,
        max(wide["linear_recent"].max(), wide["fusion_pbt_batterygpt"].max()) * 1.05,
    ]
    ax.plot(lims, lims, color="#888", linestyle="--", linewidth=1)
    ax.set_xlabel("linear_recent RMSE")
    ax.set_ylabel("fusion RMSE")
    ax.set_title("逐电池 RMSE：点在对角线下表示融合更好")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rmse_scatter.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    w2 = cell_scores.pivot(index="battery_id", columns="model", values="soh200_abs_err")
    ax.scatter(w2["linear_recent"], w2["fusion_pbt_batterygpt"], c="#f58518")
    lims = [0, max(w2["linear_recent"].max(), w2["fusion_pbt_batterygpt"].max()) * 1.05]
    ax.plot(lims, lims, color="#888", linestyle="--", linewidth=1)
    ax.set_xlabel("linear_recent |SOH_200 误差|")
    ax.set_ylabel("fusion |SOH_200 误差|")
    ax.set_title("第 200 圈 SOH 绝对误差")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "soh200_error_scatter.png", dpi=200)
    plt.close(fig)

    examples = [16, 9, 1, 41]
    have = set(points["battery_id"].unique())
    examples = [b for b in examples if b in have] or list(points["battery_id"].unique()[:4])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, bid in zip(axes.ravel(), examples):
        sub = points.loc[points["battery_id"] == bid]
        ax.plot(sub["cycle"], sub["soh_true"], color="black", label="真值")
        ax.plot(sub["cycle"], sub["soh_linear_recent"], color="#4c78a8", linestyle="--", label="近段线性")
        ax.plot(sub["cycle"], sub["soh_fusion"], color="#e45756", label="融合")
        ax.set_title(f"电池 {bid}")
        ax.set_ylabel("SOH")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("cycle")
    axes[1, 1].set_xlabel("cycle")
    fig.suptitle("截断验证：151–200 圈轨迹对照")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_trajectories.png", dpi=200)
    plt.close(fig)


def write_reports(
    summary_tbl: pd.DataFrame,
    cell_scores: pd.DataFrame,
    test_pred: pd.DataFrame,
    cycle_path: Path,
    n_better: int,
    n_cells: int,
) -> None:
    fus = summary_tbl.query("model == 'fusion_pbt_batterygpt'").iloc[0]
    lin = summary_tbl.query("model == 'linear_recent'").iloc[0]
    ratio = float(fus["rmse_mean"] / lin["rmse_mean"])
    if fus["rmse_mean"] < lin["rmse_mean"] * 0.97:
        verdict = "融合模型在 151–200 圈 RMSE 上优于近段线性。"
    elif fus["rmse_mean"] > lin["rmse_mean"] * 1.03:
        verdict = "近段线性在 151–200 圈 RMSE 上优于融合模型。"
    else:
        verdict = "两者在 151–200 圈 RMSE 上基本持平。"

    by_policy = (
        cell_scores.groupby(["policy", "model"], as_index=False)
        .agg(rmse=("rmse", "mean"), soh200_mae=("soh200_abs_err", "mean"))
        .sort_values(["policy", "model"])
    )

    report = [
        "# 方案 B：PBT × BatteryGPT 双头融合（脚本生成）",
        "",
        "本目录是问题三的**另存方案**，不覆盖 `A/问题三/output/` 中的近段线性结果。",
        "",
        f"- 循环数据：`{cycle_path.as_posix()}`",
        "- 训练数据：仅赛题附件（40 块验证电池的 1–200 圈；测试电池只用 1–150）。",
        "- 未加载 PBT / BatteryGPT 官方权重（输入是按圈汇总，与官方 V/I 曲线预训练不兼容）。",
        "- 结构对齐方案 B：共享编码器 + 轨迹头 + 寿命头 + 一致性损失。",
        "",
        "## 截断验证（k=150，预报 151–200）",
        "",
        markdown_table(summary_tbl),
        "",
        f"- 融合 / 近段线性 RMSE 比：{ratio:.3f}",
        f"- 40 块中融合 RMSE 更低的电池数：{n_better} / {n_cells}",
        f"- 判断：**{verdict}**",
        "",
        "## 按策略",
        "",
        markdown_table(by_policy),
        "",
        "## 测试集预报（无 151–200 真值）",
        "",
        markdown_table(test_pred),
        "",
        "## 如何读寿命头",
        "",
        "寿命标签由训练折内 150→200 的真实斜率外推到 SOH=0.80 得到，不是实测 80% 循环寿命。",
        "`n_eol_fusion` 为轨迹外推与寿命头的几何平均，仅作排序，不可写成已验证寿命。",
    ]
    (OUT_DIR / "output" / "report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT_DIR / "比较.md").write_text(
        "\n".join(
            [
                "# 融合方案 vs 近段线性",
                "",
                f"验证口径：40 块满 200 圈电池，只用 1–150 圈，预报 151–200。",
                "",
                markdown_table(summary_tbl),
                "",
                f"- RMSE：融合 {fus['rmse_mean']:.6g}，近段线性 {lin['rmse_mean']:.6g}，比值 {ratio:.3f}",
                f"- SOH_200 MAE：融合 {fus['soh200_mae']:.6g}，近段线性 {lin['soh200_mae']:.6g}",
                f"- 逐电池胜出：融合更好 {n_better}/{n_cells}",
                "",
                f"**结论：{verdict}**",
                "",
                "解释：151–200 仍处 LFP 平台，近段斜率已经很强。融合模型把跨电池协议信息（PBT）",
                "和生成式残差（BatteryGPT）叠在这条基线上；若残差学到的是噪声，就会略差于纯近段线性。",
                "寿命头没有 80% 真值，不能用 N_EOL 误差宣称融合获胜。",
                "",
                "原模型文件与 `A/问题三/output/` 未被本方案改写。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "output").mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    cycle_path = find_cycle_file()
    summary, cycles = load_tables()
    train = collect_train_cells(summary, cycles)
    train_ids = [int(x) for x in train["battery_id"]]

    pretrain_epochs = 20
    finetune_epochs = 40
    print(f"device={device}  train_cells={len(train_ids)}  pretrain={pretrain_epochs}  finetune={finetune_epochs}")

    cell_scores, points, summary_tbl = run_loocv(
        summary, cycles, train_ids, device, pretrain_epochs, finetune_epochs
    )
    test_pred, _model, _scaler = train_final_and_predict_test(
        summary, cycles, train_ids, device, pretrain_epochs, finetune_epochs
    )

    wide = cell_scores.pivot(index="battery_id", columns="model", values="rmse")
    n_better = int((wide["fusion_pbt_batterygpt"] < wide["linear_recent"]).sum())
    save_figures(cell_scores, points, test_pred)
    write_reports(summary_tbl, cell_scores, test_pred, cycle_path, n_better, len(train_ids))

    cell_scores.to_csv(OUT_DIR / "output" / "validation_cell_scores.csv", index=False)
    summary_tbl.to_csv(OUT_DIR / "output" / "validation_summary.csv", index=False)
    points.to_csv(OUT_DIR / "output" / "validation_points_k150.csv", index=False)
    wide.reset_index().to_csv(OUT_DIR / "output" / "rmse_by_cell.csv", index=False)

    payload = {
        "scheme": "B_dual_head_pbt_batterygpt",
        "n_train": len(train_ids),
        "n_test": int(len(collect_test_cells(summary, cycles))),
        "cycle_file": str(cycle_path),
        "overwrote_original": False,
        "original_output_dir": str(PROJECT_ROOT / "A" / "问题三" / "output"),
        "fusion_output_dir": str(OUT_DIR),
        "validation": summary_tbl.to_dict(orient="records"),
        "n_cells_fusion_better_rmse": n_better,
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
        "seed": SEED,
    }
    (OUT_DIR / "output" / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved separately to {OUT_DIR}")


if __name__ == "__main__":
    main()
