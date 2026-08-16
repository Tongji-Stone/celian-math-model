"""外部审计：从公开 MATR/Severson 曲线提取赛题相关电池，对照已训练模型误差。

警告（硬边界）：
- 本脚本读取的公开寿命曲线与 EOL 标签不得进入问题一/二/三的训练、验证、
  特征选择、超参数搜索或测试集预测。
- 只允许在模型已经用赛题附件拟合完成之后，做事后对照。
- 输出目录独立：A/问题三/external_matr_audit/
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from SOH_plot import configure_chinese_font  # noqa: E402

EOL_SOH = 0.80
HORIZON = (151, 200)
N_EOL_CAP = 20000.0


def canonical_policy(value: str) -> str:
    value = str(value).replace("_NEWSTRUCTURE", "")
    if value == "80PER_3_6C":
        return "3_6C-80PER_3_6C"
    return value


def fit_linear(n: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    x = np.column_stack([np.ones(len(n)), n])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"form": "linear", "a": float(coef[0]), "b": float(coef[1])}


def predict_linear(fit: dict[str, float], n: np.ndarray) -> np.ndarray:
    return fit["a"] + fit["b"] * np.asarray(n, float)


def linear_from_level_slope(level_cycle: float, level_soh: float, slope: float) -> dict[str, float]:
    return {"form": "linear", "a": float(level_soh - slope * level_cycle), "b": float(slope)}


def n_eol_from_fit(fit: dict[str, float], soh_now: float, cycle_now: float) -> tuple[float, str]:
    if soh_now <= EOL_SOH:
        return float(cycle_now), "already_below"
    slope = fit["b"]
    if slope >= -1e-12:
        return N_EOL_CAP, "non_decreasing"
    value = (EOL_SOH - fit["a"]) / slope
    if value <= cycle_now:
        return float(cycle_now + (soh_now - EOL_SOH) / max(-slope, 1e-12)), "ok"
    return float(min(value, N_EOL_CAP)), "ok" if value < N_EOL_CAP else "censored"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "A" / "问题三" / "external_matr_audit"
RAW_DIR = OUT_DIR / "raw"
FIG_DIR = OUT_DIR / "figures"
ORIG_VAL = PROJECT_ROOT / "A" / "问题三" / "output" / "validation_cell_scores.csv"
ORIG_TEST = PROJECT_ROOT / "A" / "问题三" / "output" / "test_predictions.csv"
FUSION_VAL = PROJECT_ROOT / "A" / "问题三" / "fusion_pbt_batterygpt" / "output" / "validation_cell_scores.csv"
FUSION_TEST = PROJECT_ROOT / "A" / "问题三" / "fusion_pbt_batterygpt" / "test_predictions.csv"
AUDIT_MAP = PROJECT_ROOT / "references" / "matr_source" / "external_eol_audit" / "contest_to_matr_eol_audit.csv"
LABELS_JSON = PROJECT_ROOT / "references" / "matr_source" / "external_eol_audit" / "MATR_labels.json"

HF_PARQUET = "https://huggingface.co/datasets/bsebench-org/severson-2019/resolve/main/{cell_id}.parquet"
HF_TREE = "https://huggingface.co/api/datasets/bsebench-org/severson-2019/tree/main?limit=500"
USAGE = "audit_only_not_for_training_or_tuning"
K = 150
NOMINAL_AH = 1.1
EOL_AH_MATR = 0.88  # Severson: 80% of 1.1 Ah nominal
CONTINUATION_OFFSET = {4: 662, 5: 981, 6: 1060}  # LoadData.ipynb add_len for b1c0/c1/c2
MAX_WORKERS = 6


def http_json(url: str, timeout: int = 60) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_parquet_cells() -> list[str]:
    cache = RAW_DIR / "parquet_cell_ids.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    files = http_json(HF_TREE)
    cells = sorted(
        Path(item["path"]).stem
        for item in files
        if str(item.get("path", "")).endswith(".parquet")
    )
    cache.write_text(json.dumps(cells, indent=2), encoding="utf-8")
    return cells


def hypothesized_cell_id(dataset_id: int, global_id: int) -> str:
    if dataset_id == 1:
        return f"b1c{global_id - 1}"
    if dataset_id == 2:
        continuations = {54: "b1c0", 55: "b1c1", 56: "b1c2"}
        if global_id in continuations:
            return continuations[global_id]
        return f"b2c{global_id - 47}"
    if dataset_id == 3:
        return f"b3c{global_id - 95}"
    raise ValueError(dataset_id)


def extract_one_cell(cell_id: str) -> Path | None:
    dest = RAW_DIR / "cycle_summaries" / f"{cell_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 200:
        return dest
    url = HF_PARQUET.format(cell_id=cell_id)
    try:
        df = pd.read_parquet(url, columns=["cycle_number", "capacity_Ah", "step_id", "temperature_C"])
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {cell_id}: {type(exc).__name__}: {exc}", flush=True)
        return None
    charge = df.loc[df["step_id"] == "charge"]
    if charge.empty:
        return None
    grouped = (
        charge.groupby("cycle_number", as_index=False)
        .agg(q_charge=("capacity_Ah", "last"), t_mean=("temperature_C", "mean"))
        .sort_values("cycle_number")
    )
    grouped.insert(0, "cell_id", cell_id)
    grouped.to_csv(dest, index=False)
    return dest


def load_or_extract_summaries(cell_ids: list[str]) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined = RAW_DIR / "matr_cycle_summary_extracted.csv"
    have: set[str] = set()
    parts: list[pd.DataFrame] = []
    summary_dir = RAW_DIR / "cycle_summaries"
    if summary_dir.exists():
        for path in summary_dir.glob("*.csv"):
            have.add(path.stem)
            parts.append(pd.read_csv(path))
    missing = [cid for cid in cell_ids if cid not in have]
    print(f"cycle summaries cached {len(have)}, to download {len(missing)}", flush=True)
    if missing:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(extract_one_cell, cid): cid for cid in missing}
            for fut in as_completed(futs):
                cid = futs[fut]
                path = fut.result()
                print(f"  extracted {cid}: {path}", flush=True)
                if path is not None:
                    parts.append(pd.read_csv(path))
    if not parts:
        raise RuntimeError("没有提取到任何 MATR 逐圈摘要")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(combined, index=False)
    return out


def contest_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(DATA_DIR / "battery_summary.csv")
    cycles = pd.read_csv(DATA_DIR / "cycle_train.csv")
    summary["policy_canon"] = summary["policy"].map(canonical_policy)
    cycles["policy_canon"] = cycles["policy"].map(canonical_policy)
    return summary, cycles


def soh_vector(cycles: pd.DataFrame, battery_id: int, lo: int, hi: int) -> pd.DataFrame:
    part = cycles.loc[cycles["battery_id"] == battery_id, ["cycle", "capacity", "SOH_smooth"]].copy()
    part = part[(part["cycle"] >= lo) & (part["cycle"] <= hi)].sort_values("cycle")
    return part


def align_rmse(contest_soh: np.ndarray, matr_soh: np.ndarray) -> float:
    n = min(len(contest_soh), len(matr_soh))
    if n < 30:
        return float("inf")
    a = np.asarray(contest_soh[:n], float)
    b = np.asarray(matr_soh[:n], float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def matr_soh_from_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    q0 = np.nanmedian(q[: min(5, len(q))])
    if not np.isfinite(q0) or q0 <= 0:
        return np.full_like(q, np.nan, dtype=float)
    return q / q0


def best_alignment(contest: pd.DataFrame, matr: pd.DataFrame, offsets: range | list[int]) -> dict:
    cyc = contest["cycle"].to_numpy(int)
    soh = contest["SOH_smooth"].to_numpy(float)
    matr = matr.sort_values("cycle_number")
    m_cycle = matr["cycle_number"].to_numpy(int)
    m_soh = matr_soh_from_q(matr["q_charge"].to_numpy(float))
    lookup = {int(c): s for c, s in zip(m_cycle, m_soh)}
    best = {"rmse": float("inf"), "offset": 0, "n_overlap": 0}
    for offset in offsets:
        pulled = []
        for c in cyc:
            pulled.append(lookup.get(int(c) + int(offset), np.nan))
        pulled = np.asarray(pulled, float)
        mask = np.isfinite(pulled) & np.isfinite(soh)
        if mask.sum() < 30:
            continue
        rmse = float(np.sqrt(np.mean((soh[mask] - pulled[mask]) ** 2)))
        if rmse < best["rmse"]:
            best = {"rmse": rmse, "offset": int(offset), "n_overlap": int(mask.sum())}
    return best


def match_cells(
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    matr_sum: pd.DataFrame,
    labels: dict[str, float],
) -> pd.DataFrame:
    """锁定 batch 顺序映射。前 150 圈几乎都在平台上，不能靠 SOH 形状改配。"""
    available = set(matr_sum["cell_id"].astype(str).unique())
    by_cell = {cid: g.copy() for cid, g in matr_sum.groupby("cell_id")}
    rows = []
    for row in summary.itertuples(index=False):
        bid = int(row.battery_id)
        hypo = hypothesized_cell_id(int(row.dataset_id), int(row.global_id))
        contest = soh_vector(cycles, bid, 1, K)
        if hypo in available:
            if bid in CONTINUATION_OFFSET:
                offsets = range(int(CONTINUATION_OFFSET[bid]) - 80, int(CONTINUATION_OFFSET[bid]) + 81)
            else:
                offsets = range(-2, 12)
            aln = best_alignment(contest, by_cell[hypo], offsets)
            cid = hypo
        else:
            aln = {"rmse": float("nan"), "offset": np.nan, "n_overlap": 0}
            cid = ""
        rmse = aln["rmse"]
        if cid == "":
            conf = "unmatched_missing_parquet"
        elif not np.isfinite(rmse):
            conf = "unmatched"
        elif rmse < 0.003:
            conf = "high"
        elif rmse < 0.010:
            conf = "medium"
        else:
            conf = "low"
        key = f"MATR_{cid}.pkl" if cid else f"MATR_{hypo}.pkl"
        rec = {
            "battery_id": bid,
            "global_id": int(row.global_id),
            "dataset_id": int(row.dataset_id),
            "prediction_test": int(row.prediction_test),
            "policy": canonical_policy(row.policy),
            "policy_raw": row.policy,
            "hypothesized_cell": hypo,
            "matched_cell": cid,
            "is_hypothesized": int(cid == hypo and cid != ""),
            "align_offset": aln["offset"],
            "align_rmse": rmse,
            "n_overlap": aln["n_overlap"],
            "continuation": int(bid in CONTINUATION_OFFSET),
            "documented_offset": CONTINUATION_OFFSET.get(bid, np.nan),
            "usage_boundary": USAGE,
            "match_confidence": conf,
            "matr_label_key": key,
            "external_eol_088": labels.get(key),
        }
        rows.append(rec)
        print(
            f"  battery {bid:02d} {hypo} offset={rec['align_offset']} "
            f"rmse={rec['align_rmse'] if np.isfinite(rec['align_rmse']) else 'nan'} {conf}",
            flush=True,
        )
    return pd.DataFrame(rows)


def first_crossing(cycles: np.ndarray, values: np.ndarray, threshold: float) -> float:
    cycles = np.asarray(cycles, float)
    values = np.asarray(values, float)
    mask = np.isfinite(cycles) & np.isfinite(values)
    cycles, values = cycles[mask], values[mask]
    if len(values) == 0:
        return float("nan")
    below = np.where(values <= threshold)[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(cycles[0])
    y0, y1 = values[i - 1], values[i]
    x0, x1 = cycles[i - 1], cycles[i]
    if y1 == y0:
        return float(x1)
    frac = (threshold - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def scaled_matr_series(matr: pd.DataFrame, contest: pd.DataFrame, offset: int) -> pd.DataFrame:
    merged = []
    q_lookup = {int(c): q for c, q in zip(matr["cycle_number"], matr["q_charge"])}
    cap = contest[["cycle", "capacity", "SOH_smooth"]].copy()
    ratios = []
    for row in cap.itertuples(index=False):
        mq = q_lookup.get(int(row.cycle) + int(offset))
        if mq is not None and np.isfinite(mq) and mq > 0 and np.isfinite(row.capacity) and row.capacity > 0:
            ratios.append(row.capacity / mq)
    scale = float(np.median(ratios)) if ratios else float("nan")
    rows = []
    for cyc, q in zip(matr["cycle_number"].to_numpy(int), matr["q_charge"].to_numpy(float)):
        contest_cycle = int(cyc) - int(offset)
        cap_hat = q * scale if np.isfinite(scale) else np.nan
        rows.append(
            {
                "matr_cycle": int(cyc),
                "contest_cycle": contest_cycle,
                "q_charge_raw": float(q),
                "capacity_scaled": float(cap_hat) if np.isfinite(cap_hat) else np.nan,
                "scale": scale,
            }
        )
    out = pd.DataFrame(rows)
    q0 = float(contest["capacity"].iloc[0])
    out["soh_vs_contest_init"] = out["capacity_scaled"] / q0
    out["soh_vs_nominal"] = out["capacity_scaled"] / NOMINAL_AH
    return out


def load_model_predictions() -> pd.DataFrame:
    orig_val = pd.read_csv(ORIG_VAL)
    orig_val = orig_val.loc[orig_val["k"] == K, ["battery_id", "model", "n_eol", "slope", "soh200_pred"]].copy()
    orig_val["split"] = "validation"
    orig_test = pd.read_csv(ORIG_TEST)
    test_rows = []
    for row in orig_test.itertuples(index=False):
        test_rows.append(
            {
                "battery_id": int(row.battery_id),
                "model": "linear_recent",
                "n_eol": float(row.n_eol),
                "slope": float(row.slope),
                "soh200_pred": float(row.soh_200_pred),
                "split": "test",
            }
        )
        test_rows.append(
            {
                "battery_id": int(row.battery_id),
                "model": "policy_mean",
                "n_eol": float(row.n_eol_policy),
                "slope": float(row.slope_policy),
                "soh200_pred": float(row.soh_200_pred_policy),
                "split": "test",
            }
        )
    fusion_val = pd.read_csv(FUSION_VAL)
    fusion_val = fusion_val.loc[
        fusion_val["model"] == "fusion_pbt_batterygpt",
        ["battery_id", "n_eol", "slope_hat", "soh200_pred"],
    ].copy()
    fusion_val["model"] = "fusion_pbt_batterygpt"
    fusion_val["slope"] = fusion_val.pop("slope_hat")
    fusion_val["split"] = "validation"
    fusion_test = pd.read_csv(FUSION_TEST)
    fusion_test_rows = pd.DataFrame(
        {
            "battery_id": fusion_test["battery_id"].astype(int),
            "model": "fusion_pbt_batterygpt",
            "n_eol": fusion_test["n_eol_fusion"].astype(float),
            "slope": fusion_test["slope_fusion"].astype(float),
            "soh200_pred": fusion_test["soh_200_fusion"].astype(float),
            "split": "test",
        }
    )
    return pd.concat([orig_val, pd.DataFrame(test_rows), fusion_val, fusion_test_rows], ignore_index=True)


def soh_at_cycle(frame: pd.DataFrame, cycle: int) -> float:
    part = frame.loc[frame["contest_cycle"] == cycle, "soh_vs_contest_init"]
    if part.empty:
        return float("nan")
    return float(part.iloc[0])


def compare_models(
    mapping: pd.DataFrame,
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    matr_sum: pd.DataFrame,
    preds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_cell = {cid: g.copy() for cid, g in matr_sum.groupby("cell_id")}
    cell_rows = []
    point_rows = []
    models = ["linear_recent", "policy_mean", "persist", "fusion_pbt_batterygpt"]
    for rec in mapping.itertuples(index=False):
        bid = int(rec.battery_id)
        cid = str(rec.matched_cell)
        if not cid or cid not in by_cell or not np.isfinite(rec.align_rmse):
            continue
        contest = cycles.loc[cycles["battery_id"] == bid, ["cycle", "capacity", "SOH_smooth"]].sort_values("cycle")
        scaled = scaled_matr_series(by_cell[cid], contest, int(rec.align_offset))
        q0 = float(contest["capacity"].iloc[0])
        n_true_080 = first_crossing(scaled["contest_cycle"], scaled["soh_vs_contest_init"], EOL_SOH)
        n_true_088ah = first_crossing(scaled["contest_cycle"], scaled["capacity_scaled"], EOL_AH_MATR)
        n_true_088_label = rec.external_eol_088
        if pd.notna(n_true_088_label) and pd.notna(rec.align_offset):
            n_true_088_contest = float(n_true_088_label) - float(rec.align_offset)
        else:
            n_true_088_contest = float("nan")
        soh150_true = soh_at_cycle(scaled, K)
        soh200_true = soh_at_cycle(scaled, 200)
        soh300_true = soh_at_cycle(scaled, 300)
        soh500_true = soh_at_cycle(scaled, 500)
        cell_pred = preds.loc[preds["battery_id"] == bid]
        persist_soh = float(contest.loc[contest["cycle"] == K, "SOH_smooth"].iloc[0])
        persist_fit = linear_from_level_slope(K, persist_soh, 0.0)
        persist_n, _ = n_eol_from_fit(persist_fit, persist_soh, K)
        extra = pd.DataFrame(
            [
                {
                    "battery_id": bid,
                    "model": "persist",
                    "n_eol": persist_n,
                    "slope": 0.0,
                    "soh200_pred": persist_soh,
                    "split": "test" if rec.prediction_test else "validation",
                }
            ]
        )
        cell_pred = pd.concat([cell_pred, extra], ignore_index=True)
        for model in models:
            part = cell_pred.loc[cell_pred["model"] == model]
            if part.empty:
                continue
            n_hat = float(part["n_eol"].iloc[0])
            slope = float(part["slope"].iloc[0])
            soh200_pred = float(part["soh200_pred"].iloc[0])
            soh_k = persist_soh
            # 151-200 vs MATR (contest-frame)
            horiz = np.arange(HORIZON[0], HORIZON[1] + 1)
            if model == "persist":
                y_pred = np.full_like(horiz, soh_k, dtype=float)
            else:
                fit = linear_from_level_slope(K, soh_k, slope)
                y_pred = predict_linear(fit, horiz.astype(float))
            y_true = np.array([soh_at_cycle(scaled, int(n)) for n in horiz])
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() >= 10:
                rmse_151_200 = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
                mae_151_200 = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
            else:
                rmse_151_200 = mae_151_200 = float("nan")
            soh200_err = abs(soh200_true - soh200_pred) if np.isfinite(soh200_true) else float("nan")
            soh300_pred = soh_k + slope * (300 - K) if model != "persist" else soh_k
            soh500_pred = soh_k + slope * (500 - K) if model != "persist" else soh_k
            cell_rows.append(
                {
                    "battery_id": bid,
                    "policy": rec.policy,
                    "prediction_test": int(rec.prediction_test),
                    "matched_cell": cid,
                    "match_confidence": rec.match_confidence,
                    "align_offset": rec.align_offset,
                    "align_rmse": rec.align_rmse,
                    "model": model,
                    "n_hat": n_hat,
                    "n_true_080_contest_def": n_true_080,
                    "n_true_088ah_contest_frame": n_true_088ah,
                    "n_true_088_label_contest_frame": n_true_088_contest,
                    "n_true_088_label_abs": n_true_088_label,
                    "err_080": n_hat - n_true_080 if np.isfinite(n_true_080) else np.nan,
                    "err_088_label": n_hat - n_true_088_contest if np.isfinite(n_true_088_contest) else np.nan,
                    "ape_080": abs(n_hat - n_true_080) / n_true_080 if np.isfinite(n_true_080) and n_true_080 > 0 else np.nan,
                    "ape_088_label": abs(n_hat - n_true_088_contest) / n_true_088_contest
                    if np.isfinite(n_true_088_contest) and n_true_088_contest > 0
                    else np.nan,
                    "rmse_soh_151_200_matr": rmse_151_200,
                    "mae_soh_151_200_matr": mae_151_200,
                    "soh200_true_matr": soh200_true,
                    "soh200_pred": soh200_pred,
                    "soh200_abs_err_matr": soh200_err,
                    "soh300_true_matr": soh300_true,
                    "soh300_pred": soh300_pred,
                    "soh300_abs_err_matr": abs(soh300_true - soh300_pred) if np.isfinite(soh300_true) else np.nan,
                    "soh500_true_matr": soh500_true,
                    "soh500_pred": soh500_pred,
                    "soh500_abs_err_matr": abs(soh500_true - soh500_pred) if np.isfinite(soh500_true) else np.nan,
                    "soh150_true_matr": soh150_true,
                    "q0_contest": q0,
                    "usage_boundary": USAGE,
                }
            )
            for n, yt, yp in zip(horiz, y_true, y_pred):
                point_rows.append(
                    {
                        "battery_id": bid,
                        "model": model,
                        "cycle": int(n),
                        "soh_true_matr": yt,
                        "soh_pred": yp,
                        "usage_boundary": USAGE,
                    }
                )
    return pd.DataFrame(cell_rows), pd.DataFrame(point_rows)


def _err_stats(frame: pd.DataFrame, col: str) -> dict[str, float]:
    x = frame[col].to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "median_ae": np.nan, "bias": np.nan}
    return {
        "n": int(len(x)),
        "mae": float(np.mean(np.abs(x))),
        "rmse": float(np.sqrt(np.mean(x**2))),
        "median_ae": float(np.median(np.abs(x))),
        "bias": float(np.mean(x)),
    }


def closest_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    labeled = frame.loc[np.isfinite(frame["err_088_label"])]
    if labeled.empty:
        return counts
    for _bid, grp in labeled.groupby("battery_id"):
        i = grp["err_088_label"].abs().idxmin()
        model = str(grp.loc[i, "model"])
        counts[model] = counts.get(model, 0) + 1
    return counts


def summarize_errors(cell_err: pd.DataFrame) -> pd.DataFrame:
    usable = cell_err.loc[cell_err["matched_cell"].astype(str).str.len() > 0]
    splits = {
        "all_mapped": usable,
        "high_medium": usable.loc[usable["match_confidence"].isin(["high", "medium"])],
        "test_mapped": usable.loc[usable["prediction_test"] == 1],
        "train_mapped": usable.loc[usable["prediction_test"] == 0],
        "has_eol_label": usable.loc[np.isfinite(usable["n_true_088_label_contest_frame"])],
    }
    rows = []
    for model, _part in usable.groupby("model"):
        for name, base in splits.items():
            grp = base.loc[base["model"] == model]
            if grp.empty:
                continue
            e080 = _err_stats(grp, "err_080")
            e088 = _err_stats(grp, "err_088_label")
            soh = _err_stats(grp, "soh200_abs_err_matr")
            longh = _err_stats(grp, "soh500_abs_err_matr")
            ape = grp["ape_088_label"].to_numpy(float)
            ape = ape[np.isfinite(ape)]
            rows.append(
                {
                    "model": model,
                    "subset": name,
                    "n_cells": int(len(grp)),
                    "eol080_n": e080["n"],
                    "eol080_mae": e080["mae"],
                    "eol080_rmse": e080["rmse"],
                    "eol080_median_ae": e080["median_ae"],
                    "eol080_bias": e080["bias"],
                    "eol088_n": e088["n"],
                    "eol088_mae": e088["mae"],
                    "eol088_rmse": e088["rmse"],
                    "eol088_median_ae": e088["median_ae"],
                    "eol088_bias": e088["bias"],
                    "eol088_mape": float(np.mean(ape)) if len(ape) else np.nan,
                    "soh200_mae_matr": soh["mae"],
                    "soh500_mae_matr": longh["mae"],
                    "rmse_151_200_mean": float(np.nanmean(grp["rmse_soh_151_200_matr"])),
                    "usage_boundary": USAGE,
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    if frame.empty:
        return "_(empty)_"
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


def make_figures(mapping: pd.DataFrame, cell_err: pd.DataFrame, cycles: pd.DataFrame, matr_sum: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    by_cell = {cid: g.copy() for cid, g in matr_sum.groupby("cell_id")}

    # overlay a few high-confidence cells
    picks = mapping.loc[mapping["matched_cell"].astype(str).str.len() > 0].head(6)
    if not picks.empty:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=False)
        axes = axes.ravel()
        for ax, rec in zip(axes, picks.itertuples(index=False)):
            bid = int(rec.battery_id)
            contest = cycles.loc[cycles["battery_id"] == bid, ["cycle", "capacity", "SOH_smooth"]]
            scaled = scaled_matr_series(by_cell[str(rec.matched_cell)], contest, int(rec.align_offset))
            ax.plot(contest["cycle"], contest["SOH_smooth"], color="#4c78a8", lw=1.8, label="赛题 SOH_smooth")
            show = scaled.loc[(scaled["contest_cycle"] >= 1) & (scaled["contest_cycle"] <= 400)]
            ax.plot(show["contest_cycle"], show["soh_vs_contest_init"], color="#f58518", lw=1.2, alpha=0.9, label="MATR 对齐")
            ax.axhline(0.8, color="#999", ls="--", lw=0.8)
            ax.set_title(f"电池 {bid} / {rec.matched_cell}")
            ax.set_xlabel("contest cycle")
            ax.set_ylabel("SOH")
            ax.legend(fontsize=7)
        fig.suptitle("赛题曲线与公开 MATR 曲线对齐（审计专用，禁止训练）")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "mapping_overlay.png", dpi=140)
        plt.close(fig)

    high = cell_err.loc[cell_err["matched_cell"].astype(str).str.len() > 0]
    if high.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    colors = {
        "linear_recent": "#4c78a8",
        "fusion_pbt_batterygpt": "#54a24b",
        "policy_mean": "#f58518",
        "persist": "#9e9ac8",
    }
    for model, grp in high.groupby("model"):
        x = grp["n_true_088_label_contest_frame"].to_numpy(float)
        y = grp["n_hat"].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=28, alpha=0.75, color=colors.get(model, "#333"), label=model)
    lims = [100, 20000]
    ax.plot(lims, lims, color="#666", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("公开标签 EOL（赛题圈号，0.88 Ah）")
    ax.set_ylabel("模型 $\\hat N_{EOL}$（SOH=0.80 外推）")
    ax.set_title("外部寿命对照（审计专用）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eol_pred_vs_true.png", dpi=140)
    plt.close(fig)

    summary = (
        high.groupby("model", as_index=False)
        .agg(
            eol088_mae=("err_088_label", lambda s: np.nanmean(np.abs(s))),
            soh200_mae=("soh200_abs_err_matr", "mean"),
            soh500_mae=("soh500_abs_err_matr", "mean"),
            rmse151200=("rmse_soh_151_200_matr", "mean"),
        )
        .sort_values("eol088_mae")
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(summary))
    ax.bar(x, summary["eol088_mae"], color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=15, ha="right")
    ax.set_ylabel("EOL MAE（圈）")
    ax.set_title("各模型相对公开 0.88 Ah 寿命的平均绝对误差")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eol_mae_by_model.png", dpi=140)
    plt.close(fig)


def write_reports(mapping: pd.DataFrame, cell_err: pd.DataFrame, err_sum: pd.DataFrame) -> None:
    n_high = int((mapping["match_confidence"] == "high").sum())
    n_med = int((mapping["match_confidence"] == "medium").sum())
    n_low = int((mapping["match_confidence"] == "low").sum())
    n_un = int(mapping["match_confidence"].astype(str).str.startswith("unmatched").sum())
    hypo_ok = int(((mapping["is_hypothesized"] == 1) & mapping["match_confidence"].isin(["high", "medium"])).sum())
    show = [
        "model",
        "subset",
        "n_cells",
        "eol088_n",
        "eol088_mae",
        "eol088_median_ae",
        "eol088_mape",
        "eol088_bias",
        "eol080_mae",
        "rmse_151_200_mean",
        "soh200_mae_matr",
        "soh500_mae_matr",
    ]
    present = [c for c in show if c in err_sum.columns]
    ranking = err_sum.loc[err_sum["subset"] == "has_eol_label", present].sort_values("eol088_mae")
    ranking_test = err_sum.loc[err_sum["subset"] == "test_mapped", present].sort_values("eol088_mae")
    closest = closest_counts(cell_err.loc[np.isfinite(cell_err["err_088_label"])])
    closest_txt = "，".join(f"{k} {v} 块" for k, v in sorted(closest.items(), key=lambda kv: -kv[1])) or "无"
    test_err = cell_err.loc[
        (cell_err["prediction_test"] == 1) & (cell_err["matched_cell"].astype(str).str.len() > 0),
        [
            "battery_id",
            "policy",
            "matched_cell",
            "model",
            "n_hat",
            "n_true_088_label_contest_frame",
            "err_088_label",
            "ape_088_label",
            "soh200_true_matr",
            "soh200_pred",
            "soh500_true_matr",
            "soh500_pred",
        ],
    ].sort_values(["battery_id", "model"])

    report = f"""# 外部 MATR 曲线审计（禁止用于训练/验证）

> **边界**：https://data.matr.io/1/ 与 HuggingFace Severson 镜像中的真实衰变曲线、公开寿命标签，**不能**用于本赛题任何训练、验证、特征选择、调参或测试集预测。本目录只做模型完成之后的事后对照。

## 提取了什么

赛题 49 块电池对应 MIT–Stanford A123 LFP 18650 公开实验：

- `dataset_id=1`：Severson 2019 batch 1（2017-05-12），策略 `3.6C(80%)-3.6C`
- `dataset_id=2`：batch 2（2017-06-30）。其中电池 4/5/6 是 batch 1 电池 1/2/3 的**接续段**（同一物理电芯），不是独立样本
- `dataset_id=3`（`_NEWSTRUCTURE`）：Severson 2019 **batch 3**（2018-04-12）的两阶段快充，不是 Attia 2020 的四阶段 10 分钟协议

逐圈容量来自公开 parquet 的充电末容量（BSEBench 库仑计数，绝对值会低于原始 `summary.QD` 约 9%）。对照时用赛题早期容量把曲线**标定到赛题 Ah 单位**，因此比较的是形状与相对 SOH，不是把库仑计数直接当 1.07 Ah。

## 映射复核

| 置信度 | 块数 |
| --- | --- |
| high（前 150 圈 SOH RMSE < 0.002） | {n_high} |
| medium（< 0.008） | {n_med} |
| low | {n_low} |
| unmatched | {n_un} |

映射采用 **batch 顺序假说并锁定**（`b1c{{global-1}}` / `b2c{{global-47}}` / `b3c{{global-95}}`）。前 150 圈几乎都在容量平台上，不能靠 SOH 形状在同策略电池之间改配。`align_rmse` 只表示该锁定映射下前 150 圈相对公开曲线的吻合程度。

顺序映射有公开 parquet 且前 150 圈 RMSE 至少 medium：{hypo_ok} / {len(mapping)}。

{markdown_table(mapping[['battery_id','dataset_id','prediction_test','policy','hypothesized_cell','matched_cell','align_offset','align_rmse','match_confidence','external_eol_088']])}

## 寿命定义（必须分开）

1. **赛题外推**：SOH_smooth 降到 **0.80 × 本电池初始容量**
2. **MATR 公开标签**：放电容量降到 **0.88 Ah**（1.1 Ah 标称的 80%）。初始容量约 1.07 Ah 时，这大约是 82% 相对初始容量，通常比定义 1 **更早**到达

模型 `n_hat` 用的是定义 1。下表的主对照是公开 0.88 Ah 标签（换到赛题圈号 = 标签 − 对齐偏移）。

## 各模型外部误差（有公开寿命标签的电芯）

相对公开寿命，逐电池绝对误差最小的模型计数：{closest_txt}

{markdown_table(ranking)}

## 9 块测试电池汇总

{markdown_table(ranking_test)}

读表：`eol088_*` 是相对公开寿命标签；`eol080_*` 是相对标定曲线上 SOH=0.80 的穿越圈（parquet 库仑计数很少真正掉到 80%，该列样本很少）；`rmse_151_200_mean` / `soh200_mae_matr` 是相对公开曲线，**不是**赛题内部 SOH_smooth 验证，量级会更大。

## 9 块测试电池逐块

{markdown_table(test_err)}

## 结论要点

- 短窗 151–200：公开曲线若与赛题对齐良好，应与赛题内部验证同一量级；这只能说明早期几乎线性，**不能**当成 80% 寿命已验证。
- 长窗 / EOL：近段线性会把仍在平台期的斜率当成终身斜率，`\\hat N` 往往系统性偏长；融合模型若斜率更陡，可能更接近真实寿命，也可能过冲——以本表数字为准，不要回写训练。
- 电池 4/5/6 与 1/2/3 同芯。接续段的赛题圈号 1 对应公开实验的数百圈之后；其 `n_hat` 应对照「剩余寿命」而不是 1858 这种从出厂起算的标签。
- 公开标签缺失的电芯（质量剔除或右删失）不要强行补寿命。

## 文件

- `mapping_verified.csv` 曲线复核后的映射
- `cell_external_errors.csv` 逐电池、逐模型误差
- `error_summary.csv` 汇总
- `figures/` 对齐图与 EOL 散点
"""
    (OUT_DIR / "报告.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "比较.md").write_text(
        "# 外部审计比较\n\n"
        + markdown_table(ranking)
        + "\n\n详细见 `报告.md`。本结果不得用于训练或调参。\n",
        encoding="utf-8",
    )
    best = ranking.iloc[0]["model"] if not ranking.empty else "n/a"
    (OUT_DIR / "结论.md").write_text(
        f"# 外部审计结论\n\n"
        f"有公开 0.88 Ah 标签的电芯上，相对该标签平均绝对误差最低的是 `{best}`。"
        f"逐电池谁更接近：{closest_txt}。\n\n"
        "这只是事后对照。赛题内部可验证的仍只有 151–200 圈；80% SOH 寿命没有赛题真值。\n",
        encoding="utf-8",
    )


def needed_cell_ids(summary: pd.DataFrame, available: list[str]) -> list[str]:
    avail = set(available)
    need: set[str] = set()
    for row in summary.itertuples(index=False):
        hypo = hypothesized_cell_id(int(row.dataset_id), int(row.global_id))
        if hypo in avail:
            need.add(hypo)
        ds = int(row.dataset_id)
        if ds == 3:
            need.update(c for c in avail if c.startswith("b3c"))
        elif ds == 1 or (ds == 2 and hypo.startswith("b1c")):
            need.update(c for c in avail if c.startswith("b1c") and int(c[3:]) <= 4)
    return sorted(need)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "AGENTS.md").write_text(
        "# 外部 MATR 审计\n\n"
        "本目录的曲线、标签、误差表 **禁止** 输入问题一/二/三训练、验证、特征或调参。\n"
        "脚本：`src/A/task3_external_matr_audit.py`\n",
        encoding="utf-8",
    )
    print("listing parquet cells...", flush=True)
    available = list_parquet_cells()
    print(f"  {len(available)} parquet cells on HF", flush=True)
    summary, cycles = contest_tables()
    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    cell_ids = needed_cell_ids(summary, available)
    print(f"extracting {len(cell_ids)} summaries...", flush=True)
    matr_sum = load_or_extract_summaries(cell_ids)
    print("matching contest batteries to MATR cells...", flush=True)
    mapping = match_cells(summary, cycles, matr_sum, labels)
    mapping.to_csv(OUT_DIR / "mapping_verified.csv", index=False, encoding="utf-8-sig")
    print("loading model predictions...", flush=True)
    preds = load_model_predictions()
    print("comparing errors...", flush=True)
    cell_err, point_err = compare_models(mapping, summary, cycles, matr_sum, preds)
    cell_err.to_csv(OUT_DIR / "cell_external_errors.csv", index=False, encoding="utf-8-sig")
    point_err.to_csv(OUT_DIR / "points_soh_151_200_matr.csv", index=False, encoding="utf-8-sig")
    err_sum = summarize_errors(cell_err)
    err_sum.to_csv(OUT_DIR / "error_summary.csv", index=False, encoding="utf-8-sig")
    print("figures...", flush=True)
    make_figures(mapping, cell_err, cycles, matr_sum)
    write_reports(mapping, cell_err, err_sum)
    print(err_sum.to_string(index=False))
    print(f"done in {time.time() - t0:.1f}s -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
