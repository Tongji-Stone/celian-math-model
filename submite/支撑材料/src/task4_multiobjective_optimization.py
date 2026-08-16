"""问题四：兼顾充电时间与寿命的两阶段快充策略优化。

运行（项目根目录）：
    python src/A/task4_multiobjective_optimization.py

默认读取本机原始 MATR EOL 审计表，只是为了复现当前的策略比较；寿命数值不写在
代码中。以后问题三给出预测寿命后，直接将 --life-input 指向预测 CSV 即可，例如：
    python src/A/task4_multiobjective_optimization.py \
        --life-input A/问题三/output/final/eol_predictions.csv \
        --life-column EOL_cycle_pred

输入表至少应包含 battery_id 和一个寿命列；可选列 dataset_id、prediction_test、
matr_source_key 用于样本筛选与重复样本去重。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data" / "battery_summary.csv"
DEFAULT_LIFE = ROOT / "references" / "matr_source" / "external_eol_audit" / "contest_to_matr_eol_audit.csv"
# 与团队的 Fade_200 主优化输出隔离；此脚本专门保存可替换 EOL 的补充评估。
DEFAULT_OUT = ROOT / "A" / "问题四" / "output_eol_audit"
SOC_END = 80.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="问题四两阶段快充多目标优化")
    p.add_argument("--life-input", type=Path, default=DEFAULT_LIFE, help="寿命标签或预测寿命 CSV")
    p.add_argument("--life-column", default=None, help="寿命列名；缺省时自动识别")
    p.add_argument("--include-test", action="store_true", help="保留 prediction_test=1 的行（默认不保留）")
    p.add_argument("--dataset-id", default="3", help="主分析批次；填 all 表示不按批次筛选（默认 3）")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def choose_life_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame:
            raise ValueError(f"未找到寿命列 {requested!r}；现有列为 {list(frame.columns)}")
        return requested
    for col in ("life_cycles", "EOL_cycle_pred", "external_eol_cycles", "eol_cycles", "EOL"):
        if col in frame:
            return col
    raise ValueError("无法自动识别寿命列。请用 --life-column 指定。")


def load_cells(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if not args.life_input.exists():
        raise FileNotFoundError(f"寿命输入不存在：{args.life_input}")
    life = pd.read_csv(args.life_input)
    if "battery_id" not in life:
        raise ValueError("寿命输入必须含 battery_id 列，才能与策略参数匹配。")
    life_col = choose_life_column(life, args.life_column)
    life = life.copy()
    life["life_cycles"] = pd.to_numeric(life[life_col], errors="coerce")
    life = life.loc[life["life_cycles"] > 0].copy()
    # 审计表的 unavailable/right-censored 行会在上一步自然剔除。
    if "eol_status" in life:
        life = life.loc[life["eol_status"].fillna("available").eq("available")].copy()
    if "prediction_test" in life and not args.include_test:
        life = life.loc[pd.to_numeric(life["prediction_test"], errors="coerce").fillna(0).eq(0)].copy()

    summary = pd.read_csv(SUMMARY)
    cells = summary.merge(life, on="battery_id", how="inner", suffixes=("", "_life"), validate="one_to_one")
    # 问题二的默认同批次主分析口径为 dataset 3；预测表若仅覆盖测试集，可显式改为 all。
    cells = cells.loc[cells[["C1", "Q1", "C2", "life_cycles"]].notna().all(axis=1)].copy()
    if str(args.dataset_id).lower() != "all":
        cells = cells.loc[cells["dataset_id"].eq(int(args.dataset_id))].copy()
    if not args.include_test:
        cells = cells.loc[cells["prediction_test"].eq(0)].copy()
    # 一个原始电池被赛题切成两段时，只保留一条，避免在策略均值中重复计数。
    dedup_key = "matr_source_key" if "matr_source_key" in cells and cells["matr_source_key"].notna().any() else "battery_id"
    cells = cells.sort_values("battery_id").drop_duplicates(dedup_key, keep="first").copy()
    cells["fade_rate_per_cycle"] = 0.20 / cells["life_cycles"]
    return cells, life_col


def nominal_cc_time(c1: np.ndarray, q1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """0--80% SOC 的恒流段理想时间（min）；忽略 CV 尾段和设备切换损耗。"""
    return 60.0 * (q1 / (100.0 * c1) + (SOC_END - q1) / (100.0 * c2))


def linear_time_fit(policy: pd.DataFrame) -> tuple[float, float, float]:
    x = policy["t_cc_ideal_min"].to_numpy(float)
    y = policy["charge_time_min"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot else np.nan
    return float(intercept), float(slope), float(r2)


def policy_table(cells: pd.DataFrame) -> pd.DataFrame:
    out = cells.groupby("policy", as_index=False).agg(
        n=("battery_id", "size"), C1=("C1", "mean"), Q1=("Q1", "mean"), C2=("C2", "mean"),
        charge_time_min=("mean_chargetime", "mean"), life_cycles=("life_cycles", "mean"),
        life_median=("life_cycles", "median"), fade_rate_per_cycle=("fade_rate_per_cycle", "mean"),
    )
    out["policy"] = out["policy"].astype(str).str.replace("_NEWSTRUCTURE", "", regex=False)
    out["t_cc_ideal_min"] = nominal_cc_time(out.C1.to_numpy(), out.Q1.to_numpy(), out.C2.to_numpy())
    out["soh_fade_200_equiv"] = 200.0 * out["fade_rate_per_cycle"]
    return out.sort_values("charge_time_min").reset_index(drop=True)


def pareto_flags(policy: pd.DataFrame) -> pd.DataFrame:
    out = policy.copy()
    time = out["charge_time_min"].to_numpy()
    fade = out["fade_rate_per_cycle"].to_numpy()
    flags = []
    for i in range(len(out)):
        dominates_i = (time <= time[i]) & (fade <= fade[i]) & ((time < time[i]) | (fade < fade[i]))
        flags.append(not bool(dominates_i.any()))
    out["pareto_existing"] = flags
    return out


def idw_predict(train: pd.DataFrame, candidates: pd.DataFrame, value_col: str = "life_cycles") -> np.ndarray:
    cols = ["C1", "Q1", "C2"]
    scale = train[cols].max() - train[cols].min()
    scale = scale.where(scale > 0, 1.0)
    a = train[cols].to_numpy(float) / scale.to_numpy(float)
    b = candidates[cols].to_numpy(float) / scale.to_numpy(float)
    d2 = ((b[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)
    weights = 1.0 / (d2 + 1e-6)
    values = train[value_col].to_numpy(float)
    if np.any(values <= 0):
        raise ValueError(f"IDW 输入 {value_col} 必须为正数")
    return np.exp((weights @ np.log(values)) / weights.sum(axis=1))


def lopo_idw(policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in policy.iterrows():
        train = policy.drop(index=i)
        pred = float(idw_predict(train, pd.DataFrame([row]))[0])
        rows.append({"held_out_policy": row.policy, "observed_life": row.life_cycles, "idw_life_pred": pred,
                     "ape_pct": abs(pred - row.life_cycles) / row.life_cycles * 100})
    return pd.DataFrame(rows)


def local_candidates(policy: pd.DataFrame, intercept: float, slope: float) -> pd.DataFrame:
    """仅在已有策略凸包内、且距任一实验点不远于 0.35（归一化距离）的网格探索。"""
    cols = ["C1", "Q1", "C2"]
    lower, upper = policy[cols].min(), policy[cols].max()
    grids = [np.round(np.arange(lower.C1, upper.C1 + 0.001, 0.1), 2),
             np.arange(int(lower.Q1), int(upper.Q1) + 1, 1),
             np.round(np.arange(lower.C2, upper.C2 + 0.001, 0.1), 2)]
    mesh = np.meshgrid(*grids, indexing="ij")
    cand = pd.DataFrame({c: v.ravel() for c, v in zip(cols, mesh)})
    pts = policy[cols].to_numpy(float)
    try:
        inside = Delaunay(pts).find_simplex(cand[cols].to_numpy(float)) >= 0
    except Exception:
        inside = np.zeros(len(cand), dtype=bool)
    cand = cand.loc[inside].copy()
    scale = (upper - lower).replace(0, 1.0)
    dist = np.sqrt((((cand[cols].to_numpy()[:, None, :] - pts[None, :, :]) / scale.to_numpy()) ** 2).sum(axis=2))
    cand["nearest_design_distance"] = dist.min(axis=1)
    cand = cand.loc[cand["nearest_design_distance"] <= 0.35].copy()
    cand["life_cycles_idw"] = idw_predict(policy, cand, "life_cycles")
    cand["fade_rate_idw"] = 0.20 / cand["life_cycles_idw"]
    cand["t_cc_ideal_min"] = nominal_cc_time(cand.C1.to_numpy(), cand.Q1.to_numpy(), cand.C2.to_numpy())
    # 弱线性校准不能可靠外推实际时间；邻域中采用同样受限的 IDW 时间近似。
    cand["charge_time_idw_min"] = idw_predict(policy, cand, "charge_time_min")
    return cand.sort_values(["charge_time_idw_min", "fade_rate_idw"]).reset_index(drop=True)


def choose_recommendation(policy: pd.DataFrame) -> pd.DataFrame:
    """在真实实验策略中按等权归一化折中，并保留帕累托点。"""
    out = pareto_flags(policy)
    t = out.charge_time_min
    r = out.fade_rate_per_cycle
    out["time_norm"] = (t - t.min()) / (t.max() - t.min())
    out["fade_norm"] = (r - r.min()) / (r.max() - r.min())
    out["equal_weight_score"] = 0.5 * out.time_norm + 0.5 * out.fade_norm
    out["recommended"] = False
    pool = out.loc[out.pareto_existing]
    if len(pool):
        out.loc[pool.equal_weight_score.idxmin(), "recommended"] = True
    return out.sort_values("equal_weight_score").reset_index(drop=True)


def md_table(frame: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    view = frame[cols].copy()
    for c in view.select_dtypes(include=[np.number]).columns:
        view[c] = view[c].map(lambda x: f"{x:.{digits}f}")
    return view.to_markdown(index=False)


def write_report(out_dir: Path, policy: pd.DataFrame, rec: pd.DataFrame, loo: pd.DataFrame,
                 candidates: pd.DataFrame, intercept: float, slope: float, r2: float,
                 args: argparse.Namespace, life_col: str, cells: pd.DataFrame) -> None:
    selected = rec.loc[rec.recommended].iloc[0]
    long = policy.loc[policy.life_cycles.idxmax()]
    short = policy.loc[policy.life_cycles.idxmin()]
    best_local = candidates.iloc[0] if len(candidates) else None
    lines = [
        "# 问题四：充电时间—寿命多目标优化（可替换寿命输入）", "",
        "## 数据口径", "",
        f"本次运行读取 `{args.life_input.as_posix()}` 的 `{life_col}` 列；在代码中未内置任一电池寿命值。",
        f"本次主样本为 dataset {args.dataset_id}、{'含测试集' if args.include_test else '非测试集'}，按原始来源去重后为 {len(cells)} 块电池、{len(policy)} 个实验策略。",
        "将来用问题三预测寿命替代时，只需改 `--life-input` 和（必要时）`--life-column`，其余计算不变。", "",
        "## 1. 充电时间近似", "",
        "对 0–80% SOC 的两段恒流部分，理论时间为", "",
        "$$t_{CC}=60\\left(\\frac{Q_1}{100C_1}+\\frac{80-Q_1}{100C_2}\\right)\\quad(\\mathrm{min}).$$", "",
        f"将其与数据集实际平均充电时间作线性核对，得到 `t_charge = {intercept:.3f} + {slope:.3f} t_CC`，但策略级 $R^2={r2:.3f}$ 很低。原因是实际时间还受恒压尾段、设备控制和切换损耗影响；故主比较直接使用各实测策略的平均时间，理论式只用于说明倍率与 SOC 区间的方向性关系。", "",
        "## 2. 寿命衰减估计", "",
        "将寿命输入 $L$ 换成统一的 SOH 损失率 $r=0.20/L$（SOH/循环）；其倒数越大表示越快到达 80% SOH。当前数值由外部原始 EOL 审计输入生成，后续可无缝替换为问题三预测值。", "",
        "", md_table(policy, ["policy", "n", "charge_time_min", "life_cycles", "fade_rate_per_cycle", "soh_fade_200_equiv"], 5), "",
        "## 3. 优化模型", "",
        "在已有实验策略集合上，采用帕累托优化：同时最小化 $f_1=t_{charge}$ 与 $f_2=r$。若需要一个单一选择，使用等权归一化目标 $J=0.5\\tilde t+0.5\\tilde r$；权重可以按使用场景调整。", "",
        "对于未实测策略，仅在六个设计点的三维凸包内、且到最近实验点的归一化距离不超过 0.35 时，用逆距离加权（IDW）作敏感性插值。该部分不作为主推荐依据。", "",
        f"IDW 留一策略的中位绝对百分比误差为 {loo.ape_pct.median():.1f}%（仅 6 个策略）；因此不把插值点当作确定结论。", "",
        "## 4. 推荐与典型对照", "",
        f"等权折中推荐为 **`{selected.policy}`**：平均充电时间 {selected.charge_time_min:.2f} min，平均寿命 {selected.life_cycles:.0f} 圈，损失率 {selected.fade_rate_per_cycle:.6f} SOH/圈；它属于实测策略帕累托前沿。", "",
        f"典型长寿命策略是 `{long.policy}`（{long.life_cycles:.0f} 圈，{long.charge_time_min:.2f} min）；典型短寿命策略是 `{short.policy}`（{short.life_cycles:.0f} 圈，{short.charge_time_min:.2f} min）。",
        "", md_table(rec, ["policy", "charge_time_min", "life_cycles", "fade_rate_per_cycle", "pareto_existing", "equal_weight_score", "recommended"], 5), "",
        "## 5. 适用范围与风险", "",
        "- 主结论只比较题目中已实际测试的六种策略，避免把稀疏样本的插值误认为最优配方。",
        "- 可探索范围受已有设计点凸包限制：C1 约 3.7–5.6C、Q1 19–80%、C2 4.0–5.9C；这只是数值边界，真正可用点还须满足“接近已有点”的距离限制。",
        "- EOL 若来自问题三外推，长期曲线模型误差会直接传递到优化结果；应同时报告 EOL 置信/敏感性区间，并在候选策略间差异小于该不确定度时不强行排序。",
        "- 充电时间模型未显式区分温度、容量标定、恒压截止逻辑；换电芯、换温控或跨批次时必须重新校准。",
    ]
    if best_local is not None:
        lines += ["", "## 邻域插值（仅敏感性参考）", "",
                  f"网格中充电时间最短的受限候选为 C1={best_local.C1:.1f}C、Q1={best_local.Q1:.0f}%、C2={best_local.C2:.1f}C；受限 IDW 时间 {best_local.charge_time_idw_min:.2f} min、IDW 寿命 {best_local.life_cycles_idw:.0f} 圈。由于这仍是小样本插值，不以它替代实测推荐。"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cells, life_col = load_cells(args)
    if len(cells) < 6:
        raise ValueError(f"可用于问题四的样本不足：{len(cells)}。请检查寿命输入和筛选条件。")
    policy = policy_table(cells)
    intercept, slope, r2 = linear_time_fit(policy)
    recommendation = choose_recommendation(policy)
    loo = lopo_idw(policy)
    candidates = local_candidates(policy, intercept, slope)
    # 仅留可复现的样本清单，不把逐电池原始 EOL 标签复制进版本库输出。
    cells[["battery_id", "global_id", "dataset_id", "policy", "C1", "Q1", "C2"]].to_csv(
        out_dir / "cell_cohort.csv", index=False
    )
    policy.to_csv(out_dir / "policy_summary.csv", index=False)
    recommendation.to_csv(out_dir / "optimization_existing_policies.csv", index=False)
    loo.to_csv(out_dir / "idw_leave_one_policy.csv", index=False)
    candidates.to_csv(out_dir / "local_neighbor_candidates.csv", index=False)
    meta = {"life_input": str(args.life_input), "life_column": life_col, "include_test": bool(args.include_test),
            "dataset_id": str(args.dataset_id), "n_cells": int(len(cells)), "n_policies": int(len(policy)), "time_intercept": intercept,
            "time_slope": slope, "time_r2": r2, "idw_lopo_median_ape_pct": float(loo.ape_pct.median())}
    (out_dir / "run_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, policy, recommendation, loo, candidates, intercept, slope, r2, args, life_col, cells)
    print((out_dir / "report.md").as_posix())


if __name__ == "__main__":
    main()
