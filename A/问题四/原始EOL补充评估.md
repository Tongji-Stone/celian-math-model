# 原始 EOL 的可替换寿命评估补充

团队主优化器 `src/A/task4_optimize.py` 以问题二的 `Fade_200` 为寿命风险代理，适合在赛题早期数据内约束高 SOC 大电流。本补充不改动该主结论，而是提供一个单独的寿命输入接口，用此前校对得到的原始 EOL 作一次策略级核对；**EOL 不在代码中写死**。

## 运行与替换

默认复现（本机有原始审计表时）：

```powershell
python src/A/task4_multiobjective_optimization.py
```

将寿命来源换为问题三的预测表：

```powershell
python src/A/task4_multiobjective_optimization.py `
  --life-input A/问题三/output/final/eol_predictions.csv `
  --life-column EOL_cycle_pred `
  --include-test `
  --dataset-id all
```

输入只需要 `battery_id` 和一个寿命列；`--life-column` 可显式指定。默认输出到 `A/问题四/output_eol_audit/`，不会覆盖团队的 `output/`。

## 当前原始 EOL 复核

按 `dataset_id=3`、非测试集，并按原始 MATR 来源去重后，得到 30 块电池、6 个实验策略。策略级结果由脚本即时生成于 `output_eol_audit/report.md`：

- 等权时间—寿命折中：`5_3C_54PER_4C`，实测平均充电时间约 10.16 min、策略平均 EOL 约 1074 圈；
- 典型长寿：`4_8C_80PER_4_8C`，约 1570 圈、11.18 min；
- 典型短寿：`3_7C_31PER_5_9C`，约 654 圈、10.85 min。

这里的比较仅用于核对“更快”与“更长寿”确有权衡，不能把公开的完整寿命标签当作赛题测试集的可用信息。正式参赛版本应改用问题三预测 EOL，并随预测不确定性一同更新排序。

## 适用边界

寿命输入一旦来自问题三长期外推，其模型偏差会传递至优化；候选策略寿命差距小于预测不确定性时，不应强行给出严格排名。未实测策略只在已有设计点凸包内、且接近已有点时作 IDW 敏感性插值，主推荐仍优先选已有实验策略。
