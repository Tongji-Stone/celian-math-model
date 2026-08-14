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
4. **机制模型（log_2）**：固定高 SOC 门槛 \(Q^\ast\) 计算两窗口暴露 \(E^{L},E^{H}\) 与 \(C^{\mathrm{high}}\)，建立 `fade_200` 与暴露量的回归；详见 `模型.md`、`结论.md`。
5. **符号回归（PySR）**：一组受限搜索核对两窗口经验式；另一组只输入 `C1,Q1,C2` 并允许 `+,-,*,/` 自由组合。均以留一策略重搜检验公式结构，不将训练误差视为因果证据。

## 文件约定

- 代码：`src/A/task2_statistical_analysis.py`（组间检验与对照回归）、`src/A/task2_exposure_model.py`（暴露机制模型）
- 代码：`src/A/task2_statistical_analysis.py`（组间检验与对照回归）、`src/A/task2_exposure_model.py`（暴露机制模型）、`src/A/task2_pysr_symbolic.py`（两窗口 PySR）、`src/A/task2_pysr_raw_free.py` / `task2_pysr_raw_free_lopo.py`（原始参数自由搜索及验证）
- 结果：`output/A/question2/`（检验表）、`A/问题二/output/`（机制模型）、`A/问题二/output/pysr/`（两窗口 PySR）、`A/问题二/output/pysr_raw_free/`（原始参数自由搜索）
- 说明：`A/问题二/模型.md`、`A/问题二/结论.md`、`log_1.md`、`log_2.md`
- 论文：先写本目录 markdown，不要直接改 `paper/`

## 限制

- 策略参数不是完全析因设计，策略组合数量有限；任何参数系数只能解释为样本内关联，不可表述为严格因果效应。
- 200 圈观测窗口的 SOH 约为 0.95--1.00，外推至 EOL=80% 的真实寿命不在本问题的结果范围内。

## 已补充的稳健性模型

- 针对 `C1`、`Q1`、`C2` 的严重共线性，新增标准化岭回归及 1/2 潜变量 PLS 回归。
- 以电池为单位进行嵌套留一交叉验证；标准化和岭惩罚参数选择均只在训练折内进行。
- 交叉验证误差仅衡量当前 34 块电池上的样本内外推，不可作为参数因果效应的证据。
