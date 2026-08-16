"""建立赛题记录与公开 MATR 外部 EOL 标签的审计映射。

警告：此脚本的输出仅能用于模型完成后的外部审计，绝不能作为问题三
训练、调参、特征构造或测试电池预测的输入。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data" / "battery_summary.csv"
LABELS = ROOT / "references" / "matr_source" / "external_eol_audit" / "MATR_labels.json"
OUTPUT = ROOT / "references" / "matr_source" / "external_eol_audit" / "contest_to_matr_eol_audit.csv"

# 这四条 key 不在公开 Life Labels 中不是下载遗漏：它们在原始作者的
# LoadData 流程里被作为噪声记录或未达到 0.88 Ah 阈值的右删失记录排除。
UNAVAILABLE_EVIDENCE = {
    "MATR_b3c2.pkl": (
        "unavailable_quality_exclusion",
        "原始 LoadData 的 nind=[3,40:41] 中第3条 batch-3 记录；噪声剔除。",
    ),
    "MATR_b3c37.pkl": (
        "unavailable_quality_exclusion",
        "原始 LoadData 先删除 batch3(38)：数据采集/连接问题。",
    ),
    "MATR_b3c42.pkl": (
        "unavailable_right_censored",
        "不在公开 Life Labels；原始流程将末容量仍高于0.885 Ah的记录视为未完成并剔除。",
    ),
    "MATR_b3c43.pkl": (
        "unavailable_right_censored",
        "不在公开 Life Labels；原始流程将末容量仍高于0.885 Ah的记录视为未完成并剔除。",
    ),
}


def source_key(dataset_id: int, global_id: int) -> tuple[str | None, str]:
    """按原始 batch 顺序及公开的 batch-2 接续说明生成候选原始 cell ID。"""
    if dataset_id == 1:
        return f"MATR_b1c{global_id - 1}.pkl", "batch-1 顺序映射"
    if dataset_id == 2:
        # b2c7--c9 是 b1c0--c2 的续跑，原始处理脚本将二者拼接。
        continuations = {54: "MATR_b1c0.pkl", 55: "MATR_b1c1.pkl", 56: "MATR_b1c2.pkl"}
        if global_id in continuations:
            return continuations[global_id], "batch-2 接续至 batch-1（公开说明）"
        return f"MATR_b2c{global_id - 47}.pkl", "batch-2 顺序映射（待曲线复核）"
    if dataset_id == 3:
        return f"MATR_b3c{global_id - 95}.pkl", "batch-3 顺序映射（待曲线复核）"
    return None, "未知 dataset_id"


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    cycles = pd.read_csv(ROOT / "data" / "cycle_train.csv")
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    records = []
    for row in summary.itertuples(index=False):
        key, method = source_key(int(row.dataset_id), int(row.global_id))
        eol = labels.get(key) if key else None
        status, evidence = (
            ("available", "公开 Life Labels 直接给出") if eol is not None else UNAVAILABLE_EVIDENCE.get(key, ("unavailable_unknown", "未找到对应公开标签；需人工复核"))
        )
        cell_cycles = cycles[cycles.battery_id == row.battery_id].sort_values("cycle")
        records.append(
            {
                "battery_id": int(row.battery_id),
                "global_id": int(row.global_id),
                "dataset_id": int(row.dataset_id),
                "prediction_test": int(row.prediction_test),
                "policy": row.policy,
                "matr_source_key": key,
                "external_eol_cycles": eol,
                "eol_status": status,
                "eol_evidence": evidence,
                "contest_last_cycle": int(cell_cycles.cycle.max()),
                "contest_last_soh": float(cell_cycles.SOH_smooth.iloc[-1]),
                "mapping_method": method,
                "usage_boundary": "audit_only_not_for_training_or_tuning",
            }
        )
    audit = pd.DataFrame(records)
    audit.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    matched = audit.external_eol_cycles.notna()
    print(f"matched {matched.sum()} / {len(audit)} contest records")
    print("unavailable records:")
    print(audit.loc[~matched, ["battery_id", "matr_source_key", "eol_status", "eol_evidence", "contest_last_soh"]].to_string(index=False))
    print(audit.loc[matched, ["battery_id", "matr_source_key", "external_eol_cycles", "prediction_test"]].to_string(index=False))


if __name__ == "__main__":
    main()
