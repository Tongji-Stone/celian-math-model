"""问题三：以公开 MATR 非测试 EOL 进行隔离标定。

本文件的边界非常重要：MATR_labels.json 是原始公开数据的寿命标签，
它不属于赛题附件。本脚本只将 prediction_test=0 的记录用于拟合和模型选择；
对 prediction_test=1 的九块电池，绝不读取其 external_eol_cycles 列。
因此输出是“带公开外部先验的寿命估计”，而不是仅靠附件可直接验证的答案。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "data" / "battery_summary.csv"
CYCLES_PATH = ROOT / "data" / "cycle_train.csv"
AUDIT_PATH = ROOT / "references" / "matr_source" / "external_eol_audit" / "contest_to_matr_eol_audit.csv"
OUT_DIR = ROOT / "A" / "问题三" / "output" / "external_eol_calibration"


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan


def feature_row(summary: pd.Series, cell: pd.DataFrame) -> dict[str, float]:
    """从截至第 150 圈的赛题可见信息提取少量、可解释的特征。"""
    cell = cell.sort_values("cycle").query("cycle <= 150")
    cyc = cell["cycle"].to_numpy(dtype=float)
    soh = cell["SOH_smooth"].to_numpy(dtype=float)
    row: dict[str, float] = {
        "battery_id": int(summary.battery_id),
        "C1": float(summary.C1) if pd.notna(summary.C1) else np.nan,
        "Q1": float(summary.Q1),
        "C2": float(summary.C2),
        "initial_capacity": float(summary.initial_capacity),
        "mean_chargetime": float(summary.mean_chargetime),
        "mean_IR": float(summary.mean_IR),
        "mean_Tavg": float(summary.mean_Tavg),
    }
    for point in (1, 25, 50, 75, 100, 125, 150):
        nearest = int(np.argmin(np.abs(cyc - point)))
        row[f"soh_{point}"] = float(soh[nearest])
    for left, right in ((1, 50), (51, 100), (101, 150), (1, 150)):
        mask = (cyc >= left) & (cyc <= right)
        row[f"slope_{left}_{right}"] = slope(cyc[mask], soh[mask])
    for col in ("IR", "Tavg", "chargetime"):
        values = cell[col].to_numpy(dtype=float)
        row[f"{col}_late_minus_early"] = float(np.nanmean(values[-25:]) - np.nanmean(values[:25]))
    return row


def metrics(y: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    pred = np.exp(pred_log)
    return {
        "rmse_cycles": float(mean_squared_error(y, pred) ** 0.5),
        "mae_cycles": float(mean_absolute_error(y, pred)),
        "median_ape_pct": float(np.median(np.abs(pred - y) / y) * 100),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_PATH)
    cycles = pd.read_csv(CYCLES_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    merged = summary.merge(
        audit[["battery_id", "matr_source_key", "external_eol_cycles", "eol_status"]],
        on="battery_id", how="left", validate="one_to_one",
    )
    # 审计文件可能保存了测试对象的公开标签；从这一行起完全脱敏，避免标签进入
    # 特征表、输出目录或任何预测调用。matr_source_key 保留，仅用于识别接续记录。
    test_mask = merged["prediction_test"].eq(1)
    merged.loc[test_mask, "external_eol_cycles"] = pd.NA
    merged.loc[test_mask, "eol_status"] = "withheld_test_label"

    records = []
    # 先对每个赛题记录提取特征；测试集中的续跑记录仍需各自给出预测。
    for row in merged.sort_values("global_id").itertuples(index=False):
        cell = cycles.loc[cycles.battery_id == row.battery_id]
        item = feature_row(row, cell)
        item.update({
            "global_id": int(row.global_id),
            "prediction_test": int(row.prediction_test),
            "matr_source_key": row.matr_source_key,
            "external_eol_cycles": row.external_eol_cycles,
            "eol_status": row.eol_status,
        })
        records.append(item)
    frame = pd.DataFrame(records)
    # 仅训练/验证时去重：batch-2 的续跑不是新的寿命样本。
    train = (frame.query("prediction_test == 0 and eol_status == 'available'")
             .sort_values("global_id").drop_duplicates("matr_source_key", keep="first").copy())
    test = frame.query("prediction_test == 1").copy()
    assert train.external_eol_cycles.notna().all()

    feature_cols = [c for c in frame.columns if c not in {
        "battery_id", "global_id", "prediction_test", "matr_source_key",
        "external_eol_cycles", "eol_status",
    }]
    x_train = train[feature_cols]
    y_train = train.external_eol_cycles.to_numpy(dtype=float)
    x_test = test[feature_cols]
    # 对数寿命会让 500 与 2000 圈的相对误差获得较公平的权重。
    log_y = np.log(y_train)
    candidates = {
        "ridge_log_eol": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "random_forest_log_eol": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.7, random_state=20260814)),
        "extra_trees_log_eol": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.8, random_state=20260814)),
    }
    loo = LeaveOneOut()
    summary_rows = []
    oof = pd.DataFrame({"battery_id": train.battery_id, "true_eol_cycles": y_train})
    for name, model in candidates.items():
        pred_log = cross_val_predict(model, x_train, log_y, cv=loo, n_jobs=-1)
        row = {"model": name, "samples": len(train), **metrics(y_train, pred_log)}
        summary_rows.append(row)
        oof[f"{name}_oof_eol"] = np.exp(pred_log)
    score = pd.DataFrame(summary_rows).sort_values("median_ape_pct")
    selected_name = str(score.iloc[0].model)
    selected = candidates[selected_name].fit(x_train, log_y)
    # 这是唯一用于测试对象的 fit/predict 调用：没有访问其 external_eol_cycles。
    test_pred_log = selected.predict(x_test)
    prediction = test[["battery_id", "global_id", "matr_source_key"]].copy()
    prediction["record_level_eol_prediction_for_diagnosis"] = np.exp(test_pred_log)
    # 同一原始电池的续跑记录对应同一个 EOL；用对数均值合并，等价于几何平均。
    prediction["estimated_eol_cycles_external_calibration"] = np.exp(
        prediction.groupby("matr_source_key")["record_level_eol_prediction_for_diagnosis"].transform(
            lambda values: np.log(values).mean()
        )
    )
    prediction["duplicate_source_resolution"] = "geometric_mean_within_same_matr_source_key"
    prediction["model"] = selected_name
    prediction["data_boundary"] = "trained_only_on_unique_non_test_external_labels"

    frame.to_csv(OUT_DIR / "feature_table.csv", index=False, encoding="utf-8-sig")
    score.to_csv(OUT_DIR / "loo_model_comparison.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUT_DIR / "loo_predictions_non_test_only.csv", index=False, encoding="utf-8-sig")
    prediction.to_csv(OUT_DIR / "estimated_test_eol.csv", index=False, encoding="utf-8-sig")
    readme = f"""# 外部 EOL 标定（问题三）

- 可用于训练和留一验证的样本：{len(train)} 个去重的非测试原电池。
- 9 块测试电池在读取后即从标签列排除；输出预测时未访问其 EOL。
- 模型按留一交叉验证的中位相对误差选择：`{selected_name}`。
- 这是一项公开 MATR 原始数据标签的外部标定审计，不能与“只用赛题附件”的短期 SOH 验证混为一谈。
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(score.to_string(index=False))
    print("selected:", selected_name)
    print(prediction.to_string(index=False))


if __name__ == "__main__":
    main()
