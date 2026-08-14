# 问题三迭代二运行日志：特征提取与相关性检测

## 原始提示词

> 现在我要完成问题三的特征提取，相关文件为"D:\知识库\作业\数学建模国赛——校赛\celian-math-model\2026年度“策联杯”数学建模精英联赛-A题.pdf"，以及相关文件夹的已有工作。目前已经完成了部分数据集中特征的检测，相关工作见.\A\问题三\AGENTS.md。
> 现在你要去列举可能相关的更多特征，并且按照之前的流程验证相关性。将最后特征检测的结果输出为.\output\feature.md。
> 最后根据检测结果提出可行的建模和预测方案，输出为.\output\plan.md。

## 执行摘要

- 通读赛题 PDF 的问题三要求：基于 9 块测试电池截至第 150 圈的数据，预测第 151–200 圈 SOH 变化并预测 80% SOH 的寿命终止循环。
- 复用了迭代一（`src/问题三/iteration1_analysis.py`）的全量 + 分策略 pairplot/regplot/heatmap 工作流程，并新增**电池级（跨池）证据级相关性检验**，规避同一电池重复测量对相关性的虚增。
- 构造候选特征：42 个每循环特征 + 35 个电池级特征（策略参数、SOH 衰减趋势、内阻/温度/充电时间过程统计、曲线拟合参数、分段衰减量等）。
- 检测结论：`soh150` 对 `soh200` 的 Spearman ρ=0.973（水平主导）；`fade150`、线性斜率、分段衰减、指数拟合参数与 200 圈目标显著相关（趋势增量）；温度标准差、内阻均值显著但为二级；策略参数跨策略中等相关但受批次/组合混杂；`soh1`、`capacity` 无增量信息。
- 输出 `output/feature.md`（特征检测报告）与 `output/plan.md`（建模与预测方案）。

## 数据与口径

- 源文件：`data/battery_summary.csv`（49 池）、`data/cycle_train.csv`（9350 行）。
- 早期窗口：150 圈（与赛题测试口径一致）；开发电池：40 块完整 200 圈；测试电池：9 块 150 圈。
- C1 缺失处理：`80PER_3_6C` 的 3 块电池（含测试电池 5）按同义策略 `3_6C-80PER_3_6C` 补 3.6，并记录 `c1_imputed` 标记。

## 运行与产物

```powershell
python -X utf8 .\src\问题三\iteration2_features.py
```

- `output/问题三_特征/cycle_features.csv`：49 池全量每循环特征（9350 行）。
- `output/问题三_特征/battery_features.csv`：电池级特征 + 200 圈目标（49 行，40 行含目标）。
- `output/问题三_特征/feature_correlations_overall.csv`、`feature_correlations_by_policy.csv`、`battery_level_correlations.csv`：相关性检测表。
- 图：`heatmap_overall.png`、7 张 `heatmap_<策略>.png`、`regplot_overall.png`、`battery_scatter_soh200.png`。
- 验收：`output/问题三_特征/question3_features_validation.csv`（全部通过）。
- 报告：`output/feature.md`、`output/plan.md`。

## 关键检测结果

1. 每循环口径（描述性）：SOH 衰减趋势特征（`soh_loss`、尾部斜率、`cycle`）相关最强；温度波动 `tavg_std_w25` 是唯一中等相关的过程量特征（Spearman −0.46）。
2. 分策略一致性：衰减趋势特征与温度波动在 7/7 策略方向一致；内阻/充电时间局部斜率方向不稳定。
3. 电池级口径（证据级，n=40）：
   - `soh150` vs `soh200` Spearman 0.973（vs `late_mean` 0.987）；
   - `fade150` vs `fade200` Spearman 0.941；`exp_a/exp_b/exp_c`、分段衰减 `d1_50/d50_100/d100_150`、`soh_lin_slope` 均显著；
   - `tavg_std`（ρ=−0.517，p<0.001）、`ir_mean`（ρ=0.409，p=0.009）显著；
   - `soh1` 无预测力（ρ=−0.10，p≈0.5）；`capacity` 与 SOH 同源冗余。
4. 策略参数：`C1/C2/C1_over_C2/initial_capacity` 与衰减呈中等 Pearson 相关但 Spearman 弱（批次/组合混杂），作协变量并用正则化吸收。

## 数据质量备注

- 尾部窗口特征前 2 圈为 NaN（占 1.05%）；幂律拟合在 12 块电池上失败（早期 SOH 非单调），推荐以指数拟合为主；原始 1 条 SOH>1.1、2 条 IR=0 保留原值并注明。

## 后续

按 `output/plan.md` 执行迭代三：经验曲线外推 + 电池级特征回归 + 参数化混合模型，完成 151–200 圈预测、EOL 外推、不同窗口比较与策略信息消融。
