# 方案 B 运行日志

- 运行时间：2026-08-15
- 脚本：`src/A/task3_pbt_batterygpt_fusion.py`
- Python：3.13 + torch 2.12 CPU；未改 `task3_degradation_model.py`，未写回 `A/问题三/output/`
- 数据：`A/问题一/output/A/cycle_train_cleaned.csv` + `data/battery_summary.csv`
- 输出：`A/问题三/fusion_pbt_batterygpt/`

## 训练设定

- 方案 B：共享 Transformer + 协议门控 MoE + 轨迹头 + 寿命头 + 一致性损失
- 预训练：掩码重建 1–150 圈 SOH，20 epoch
- 微调：留一电池，40 epoch；轨迹为近段线性基线 + 有界斜率/曲率修正
- 种子：20260815

## 验证（k=150）

- 融合 RMSE 0.000220，近段线性 0.000271，比值 0.813
- SOH_200 MAE：0.000327 vs 0.000466
- 40 块中融合更好：27

## 说明

官方 PBT / BatteryGPT 权重需要圈内 V/I 曲线，本题附件无法直接加载；本方案用赛题数据训练同构双头网络。
