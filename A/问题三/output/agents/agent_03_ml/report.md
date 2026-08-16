# Agent 03：基于健康特征与树模型的短期 SOH 预测

## 1. 任务边界与结论

本 Agent 只研究短期任务：已知电池前 (N) 个循环，预测未来 50 个循环的 SOH，其中重点为 (N=150)。全部模型选择与评价仅使用 40 块 `prediction_test=0` 电池；9 块测试电池从特征表构造、超参数选择、模型选择和误差计算中完全排除。当前 CSV 没有任何 `SOH <= 0.8`，因此本模型不用于 EOL 无限递推，也不报告 EOL 误差。

主要结论如下。

1. cutoff=150 时，SOH-only 树模型的 overall RMSE 为 0.000561，低于严格内层选窗的公共 Linear-Nested（0.000616，点估计改善 8.9%），也略低于固定 Linear-50（0.000568，改善 1.3%）。但是相对 Linear-Nested 的逐电池 RMSE 差异 bootstrap 95% 区间跨 0，不能称为“显著优于”线性基线。
2. 完整特征模型 C 的 RMSE 为 0.000577。加入充电策略后，相对 B 的逐电池 RMSE 平均变化为 (-4.07\times10^{-6})，40 块中 55% 改善，bootstrap 95% 区间为 ([-2.05\times10^{-5},1.25\times10^{-5}])。证据不足以认定 policy 带来稳定增益。
3. 置换重要性由近期 SOH 趋势与 horizon 主导；最重要的 `SOH_linear_delta_50` 在 95% 外层折上产生正的重要性。IR、温度、充电时间及策略变量的重要性远小于 SOH 趋势。
4. 本路线适合作为简单线性趋势之外的候选 expert，但不建议单独指定为主模型；最终 Judge 应同时考虑其最坏电池误差、内层选模方差和与线性基线差异的不确定性。

## 2. 特征构造

### 2.1 预测目标

对 cutoff (N) 和 horizon (h\in\{1,\ldots,50\})，统一回归

\[
X_{N,h}\longrightarrow \Delta SOH_h
=SOH(N+h)-SOH(N),
\]

并恢复

\[
\widehat{SOH}(N+h)=SOH(N)+\widehat{\Delta SOH_h}.
\]

训练表包含 40 块电池、5 个 cutoff、50 个 horizon，共 10,000 行；外层预测输出对 A/B/C 三套消融共 30,000 行。

### 2.2 SOH 特征（A）

- cutoff 时刻原始/稳健 SOH、历史方差；
- 近 10/20/30/50 cycle 的 slope、delta、局部线性拟合 RMSE；
- 近 50 cycle 二次项曲率、一阶与近似二阶导数；
- horizon 及其归一化值；
- `slope × horizon` 和局部二次外推量，用于让树模型直接看到趋势与预测距离的交互。

没有使用题目给定的 `SOH_smooth`。每个 cutoff 内只用 `cycle <= N` 的原始 SOH，以 median/MAD 的 8 倍阈值裁剪极端尖峰；该处理能抑制 battery 1 cycle 12 的异常值，并避免预先平滑可能隐含的未来窗口问题。

### 2.3 动态状态特征（B）

从 cutoff 内的 cycle 数据重新计算 IR、Tavg、chargetime 的 last、mean、slope、delta、standard deviation，并构造 (corr(SOH,IR))、(corr(SOH,T))、(corr(SOH,t_{charge}))。非正 IR 仅在当前 cutoff 内置为缺失并插值。没有使用 `battery_summary.csv` 中可能跨越 cutoff 的三个动态均值。

### 2.4 策略特征（C）

使用 (C_1,Q_1,C_2)、initial capacity、policy one-hot，并构造

\[
E_1=C_1Q_1,\qquad E_2=C_2(80-Q_1),
\]

以及 (C_1-C_2)、两阶段 SOC 宽度和

\[
C_{weighted}=\frac{C_1Q_1+C_2(80-Q_1)}{80}.
\]

`80PER_3_6C` 的 C1 是策略定义导致的结构性缺失，而非随机缺失。对该单阶段策略取 `C1_physical=C2=3.6`，并保留 `C1_missing_indicator=1`；没有采用总体均值填补。

## 3. 候选模型与验证协议

候选配置在实验前固定，以避免无意义的大搜索。

| 配置 | 主要超参数 |
|---|---|
| Extra Trees | 32 trees，`max_features=0.75`，`min_samples_leaf=2` |
| Random Forest | 32 trees，`max_features=0.75`，`min_samples_leaf=2` |
| HistGradientBoosting | 45 iterations，learning rate 0.05，15 leaves，L2=0.1 |
| XGBoost | 45 trees，depth 3，learning rate 0.04，subsample 0.85，L2=5 |

随机种子固定为 20260814，所有实现限制为单线程。由于小样本与资源约束，树数量只用于比较建模范式，不视为充分的大规模调参。

验证采用严格的 outer Leave-One-Battery-Out，共 40 折。每折中，目标电池的所有 cutoff 与 future rows 均不进入训练。内层在剩余 39 块电池上做一次确定性的 group holdout，分组键为 `battery_id`；用预先规定的 horizon 网格 1/10/20/30/40/50 比较四个配置，选择指标为

\[
0.5\,RMSE_{all\ cutoffs}+0.5\,RMSE_{cutoff=150}.
\]

随后在该 outer 折中固定同一配置，分别用 A、B、C 特征重新拟合，从而使消融差异主要反映特征增量而非模型家族变化。40 折选择次数为 Random Forest 21、Extra Trees 13、XGBoost 5、HistGradientBoosting 1。单次内层 holdout 比多折 GroupKFold 方差更大，这是结果的重要限制；outer 40 折预测本身仍完整且无 battery 泄漏。

## 4. 统一回测结果

### 4.1 A/B/C 消融与 early cutoff

| cutoff | A RMSE | B RMSE | C RMSE | 最优消融 |
|---:|---:|---:|---:|:---|
| 50 | 0.000532 | 0.000553 | 0.000554 | A |
| 75 | 0.000587 | 0.000541 | 0.000536 | C |
| 100 | 0.000666 | 0.000706 | 0.000693 | A |
| 125 | 0.000545 | 0.000584 | 0.000566 | A |
| 150 | 0.000561 | 0.000577 | 0.000577 | A |

误差并未随 early-cycle 长度单调下降。以 A 为例，cutoff=50 到 150 的 RMSE 反而上升约 5.5%；最佳为 cutoff=50。这里各 cutoff 预测的是不同真实区间（51–100、76–125 等），因此该曲线同时包含“信息长度”和“电池所处老化阶段”的变化，不能解释为多给 100 个 cycle 会导致因果上的性能下降。

### 4.2 cutoff=150 详细指标与线性基线

| 模型 | MAE | overall RMSE | mean battery RMSE | median battery RMSE | worst battery RMSE |
|:---|---:|---:|---:|---:|---:|
| A：SOH only | 0.000281 | 0.000561 | 0.000358 | 0.000238 | 0.002522 |
| B：+ dynamics | 0.000282 | 0.000577 | 0.000367 | 0.000199 | 0.002489 |
| C：+ policy | 0.000279 | 0.000577 | 0.000363 | 0.000200 | 0.002468 |
| Public Linear-Nested | 0.000336 | 0.000616 | 0.000410 | 0.000261 | 0.002332 |
| Public Linear-50 | 0.000310 | 0.000568 | 0.000389 | 0.000261 | 0.002332 |

A 相对 Linear-Nested 的逐电池平均 RMSE 变化为 (-5.22\times10^{-5})，40 块中 60% 改善，但 bootstrap 95% 区间约为 ([-1.47\times10^{-4},4.64\times10^{-5}])，跨过 0。树模型的 point estimate 有竞争力，但不能宣称统计上显著胜出；其 worst-battery RMSE 也略差于线性基线。

### 4.3 horizon 与 policy 稳定性

完整 C 模型在 cutoff=150 的 RMSE 随 horizon 分段为：1–10 为 0.000312，11–20 为 0.000468，21–30 为 0.000640，31–40 为 0.000491，41–50 为 0.000835。总体上远期误差扩大，但局部非单调，说明少数电池的真实波动和模型树分段共同影响结果。

C 模型的分 policy RMSE 从 0.000185 到 0.001918。训练同伴较多的新结构策略通常较稳定；只有两块训练电池的 `3_6C-80PER_3_6C`、`3_7C_31PER_5_9C_NEWSTRUCTURE` 和 `80PER_3_6C` 误差明显更高。策略层样本量仅 2 时，policy 指标不能视为稳定总体性能。

## 5. 充电策略是否增加预测能力

答案是：**当前数据不支持稳定增益**。

- cutoff=150 的 aggregate RMSE：B=0.000576951，C=0.000577053，几乎相同；
- paired mean battery RMSE change（C−B）为 (-4.07\times10^{-6})；
- 仅 55% 电池因 policy 特征改善；
- paired bootstrap 95% 区间跨 0；
- 置换重要性前列几乎全是 SOH trend 与 horizon，最佳 policy one-hot 的平均 RMSE 增量仅约 (1.43\times10^{-6})。

这不等价于“充电策略对寿命没有物理影响”。本实验只表明：在每策略训练电池仅 2–7 块、预测窗口仅 50 cycle、且近期 SOH 趋势已知时，策略变量没有提供可重复的额外短期预测信息。策略效应还与 (C_1,Q_1,C_2) 的离散组合和电池个体差异混杂，不能据此作因果结论。

## 6. 外层安全置换重要性

每个 outer 折先训练模型，再在完全未见的目标电池上置换非恒定特征；若静态特征在目标电池内恒定，则从该折训练电池的对应特征分布中抽取替代值。报告跨 40 个 outer 折的 RMSE 增量。

| 特征 | mean RMSE increase | 正增益折比例 |
|:---|---:|---:|
| SOH linear delta 50 | 0.000161 | 0.950 |
| SOH linear delta 30 | 0.000076 | 0.875 |
| forecast horizon | 0.000070 | 0.925 |
| horizon fraction | 0.000057 | 0.875 |
| SOH linear delta 20 | 0.000019 | 0.675 |
| cutoff | 0.000010 | 0.700 |

结果说明树模型的主要预测来源仍是近期退化斜率与预测距离。静态策略重要性较弱且跨折不稳定，因此未使用 SHAP；对高度相关的 slope、delta 和 slope×horizon 特征，单变量置换会分摊或重复计算重要性，数值应解释为模型敏感度而非物理因果效应。

## 7. 可复现性与产物

- 代码：`model.py`、`run.py`；
- 逐点 outer 预测：`predictions.csv`（30,000 行）；
- 统一消融：`ablation.csv`；
- 分 policy/horizon：`policy_metrics.csv`、`horizon_metrics.csv`；
- 外层置换重要性：`feature_importance.csv` 与明细 `feature_importance_outer_folds.csv`；
- 内层选模记录：`nested_selections.csv`；
- 机器可读摘要：`metrics.json`。

实际累计模型运行时间 373.4 秒。环境版本记录于 `metrics.json`：Python、NumPy、pandas、scikit-learn 和 XGBoost 均可追溯。树模型是非参数集成，固定神经网络式 parameter count 不适用；树数、深度/叶节点约束和所有候选超参数已完整记录。

## 8. 方法局限与推荐

1. 40 块训练电池的 battery-level 样本仍很小；一个 inner holdout 的模型选择可能随分组变化。
2. 每策略 2–7 块训练电池，分 policy 最坏指标高度不稳定。
3. 目标是原始 SOH，单点测量噪声会形成短期不可约误差；稳健特征仅处理输入，不平滑外层真值。
4. early cutoff 比较对应不同时间段，不是同一目标区间上的纯信息增量实验。
5. 本模型只验证 50-cycle 插值/近距外推；树模型没有可信的 0.8 长期外推机制。

因此建议 Judge 将 A（SOH-only tree）作为短期候选 expert，与 recent-linear、cohort 或 GP 比较/融合；不建议凭本 Agent 结果单独采用 C，也不建议把任何树模型用于 EOL。

## 9. 文献依据

特征工程遵循“利用早期循环可观测退化信号预测寿命，但必须在独立电池上验证”的思想。Severson 等在 fast-charged LFP/graphite cells 上展示了 early-cycle feature-based machine learning 的可行性；本 Agent 只借鉴方法，不访问其外部数据或测试电池未来轨迹。

- Kristen A. Severson, Peter M. Attia, Norman Jin, et al. “Data-driven prediction of battery cycle life before capacity degradation.” *Nature Energy*, 4, 383–391, 2019. DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8).
- Leo Breiman. “Random Forests.” *Machine Learning*, 45, 5–32, 2001. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).

