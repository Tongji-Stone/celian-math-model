# 问题二工作记录

## 目标

分析两阶段快充策略参数 `C1`、`Q1`、`C2` 与电池早期衰减的统计关联及显著性；赛题数据未观测 SOH=0.8，因此不得把 `SOH_200` 写成真实循环寿命。

## 统计口径

- 主响应：`SOH_200`、`fade_200 = SOH_1 - SOH_200` 与 1--200 圈 `SOH_smooth` 轨迹。
- 主分析队列：`dataset_id=3`、`prediction_test=0`、完整 200 圈。这样降低跨批次与 `NEWSTRUCTURE` 差异造成的混杂。
- 全体 40 块完整 200 圈电池仅用于敏感性对照，不作为因果证据。

## 方法

1. Kruskal--Wallis 检验策略间 `SOH_200` 差异，配合 Holm 校正的 Mann--Whitney 两两比较。
2. 以 `fade_200` 为响应做 `C1 + Q1 + C2` 的 HC3 稳健 OLS；报告 Spearman 相关、VIF 和诊断图。
3. 以电池为随机截距与随机循环斜率，拟合 `SOH_smooth ~ cycle * policy` 的混合效应模型；若不收敛，只记录失败原因，不输出不可靠结论。

## 文件约定

- 代码：`src/A/task2_statistical_analysis.py`
- 结果：`output/A/question2/`
- 图表：`output/A/question2/figures/`
- 论文：`paper/sections/02_problem2.tex`

## 限制

- 策略参数不是完全析因设计，策略组合数量有限；任何参数系数只能解释为样本内关联，不可表述为严格因果效应。
- 200 圈观测窗口的 SOH 约为 0.95--1.00，外推至 EOL=80% 的真实寿命不在本问题的结果范围内。
