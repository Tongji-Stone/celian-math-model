# 问题三另存方案：PBT × BatteryGPT 双头融合

本目录是方案 B 的独立实现，**不覆盖**原近段线性模型。

## 与原模型的关系

| | 原方案 | 本方案 |
|--|--------|--------|
| 代码 | `src/A/task3_degradation_model.py` | `src/A/task3_pbt_batterygpt_fusion.py` |
| 结果 | `A/问题三/output/` | `A/问题三/fusion_pbt_batterygpt/` |
| 主模型 | `linear_recent` | 共享编码器 + 轨迹头 + 寿命头 |

原脚本、原输出、论文 `paper/` 均未在本任务中改写。

## 方法摘要

1. BatteryGPT 路径：对 1–150 圈序列做掩码重建预训练，再生成 151–200 的 SOH 残差。
2. PBT 路径：协议特征门控的双专家 MoE + 寿命头预测 \(\log N_{\mathrm{EOL}}\)。
3. 方案 B：两头共享 Transformer 编码器，加轨迹–寿命一致性损失。
4. 轨迹接到近段线性基线上，并用 \(\tanh\) 限制斜率/曲率修正，避免小样本 50 维残差发散。

## 训练数据

只用赛题附件：`battery_summary.csv` + 清洗后的 `cycle_train`。
未加载 PBT / BatteryGPT 官方权重（官方输入为充电 V/I 曲线，与本题按圈表不兼容）。

## 验证

与原方案同一口径：40 块 `prediction_test=0` 且满 200 圈，留一电池交叉验证，预报 151–200。

## 运行

```text
python src/A/task3_pbt_batterygpt_fusion.py
```
