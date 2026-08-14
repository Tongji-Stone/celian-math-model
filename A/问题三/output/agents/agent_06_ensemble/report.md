# Agent 06：混合专家与自适应集成

## 1. 结论摘要

在严格的外层 Leave-One-Battery-Out（LOBO）验证下，cycle 150→151–200 的最佳单项为 **Ensemble Adaptive**，overall RMSE 为 **0.0005265**。三个集成中最佳的是 **Ensemble Adaptive**（RMSE **0.0005265**）；公共基线最佳为 **Linear-50**（RMSE **0.0005685**），相对变化为 **+7.38%**（正值表示集成降低 RMSE，负值表示未超过基线）。因此不以“集成”名称作为推荐依据；若集成没有稳定超过简单局部趋势，Judge 应保留更简单模型。

## 2. 数据边界与泄漏控制

- 数据源仅为赛题 PDF、`battery_summary.csv`、`cycle_train.csv`、公共审计和公共基线；未读取其他研究 Agent 的目录或结果，也未检索测试电池的外部未来数据。
- 9 块 `prediction_test==1` 电池完全未进入本报告的训练、调参或验证。40 块训练电池均按 battery_id 做外层 LOBO。
- 每个外层训练集内再进行 3 折 battery-grouped cross-fitting，生成权重学习所需的专家 OOF 预测。feature-ML 与 same-policy expert 在每个内层折均排除该折电池及外层目标电池的未来。
- 区间校准又对内层 OOF 预测做 3 折 battery-grouped 元层 cross-fitting，避免用拟合权重的同一条电池误差直接校准自身区间。
- cutoff=N 的全部动态特征、清洗、GP 拟合与相似度只读取 `cycle<=N`；没有使用汇总表中的动态均值。

## 3. 六个独立 Expert

1. recent 20-cycle linear；
2. recent 30-cycle linear；
3. recent 50-cycle linear；
4. exact same-policy mean future delta；若内层参考池中没有同策略电池才退化为全体参考均值；
5. Extra Trees 多步回归，输入 cutoff 内 SOH、IR、温度、充电时间和策略物理特征，并以 horizon 为输入预测相对末次观测的 SOH 增量；
6. linear mean + Matérn-3/2 GP residual。GP 使用固定的小规模核参数，避免在 40 块电池上做无意义的大搜索。

## 4. 集成方法

设六个专家输出为 $\hat y_k$。global 与 10-cycle horizon-bin 权重通过带约束最小二乘求解：

$$
\min_{w_k} \sum( y-\sum_k w_k\hat y_k)^2,
\qquad w_k\ge 0,\quad \sum_k w_k=1.
$$

自适应门控以 recent slope、curvature、局部残差波动、IR/温度/充电时间趋势、策略参数和同伴相似度预测各专家在五个 horizon bin 的 log-RMSE，再转为 softmax 权重，并与内层 horizon 权重各 50% 混合以降低小样本门控的不稳定性。整个门控训练仅用外层训练电池的 OOF 结果。

## 5. cycle 150 的统一指标

| model              |       MAE |   overall_RMSE |   mean_battery_RMSE |   median_battery_RMSE |   worst_battery_RMSE |
|:-------------------|----------:|---------------:|--------------------:|----------------------:|---------------------:|
| Ensemble Adaptive  | 0.0002613 |      0.0005265 |           0.0003382 |             0.0002048 |            0.0024178 |
| Ensemble Horizon   | 0.0002772 |      0.0005470 |           0.0003574 |             0.0002217 |            0.0024782 |
| Ensemble Global    | 0.0002613 |      0.0005473 |           0.0003410 |             0.0002000 |            0.0025558 |
| Expert GP          | 0.0003035 |      0.0005684 |           0.0003891 |             0.0002764 |            0.0023741 |
| Expert Linear-50   | 0.0003102 |      0.0005685 |           0.0003888 |             0.0002607 |            0.0023323 |
| Expert Linear-30   | 0.0003592 |      0.0005956 |           0.0004264 |             0.0003343 |            0.0022365 |
| Expert Feature-ML  | 0.0003046 |      0.0006272 |           0.0003903 |             0.0002201 |            0.0028626 |
| Expert Linear-20   | 0.0004715 |      0.0007350 |           0.0005544 |             0.0004656 |            0.0031349 |
| Expert Same-policy | 0.0006205 |      0.0013471 |           0.0007670 |             0.0002586 |            0.0041743 |

公共基线（相同 SOH 目标和 40 块训练电池）：

| model                  |       MAE |   overall_RMSE |   mean_battery_RMSE |   median_battery_RMSE |   worst_battery_RMSE |
|:-----------------------|----------:|---------------:|--------------------:|----------------------:|---------------------:|
| Linear-50              | 0.0003102 |      0.0005685 |           0.0003888 |             0.0002607 |            0.0023323 |
| Linear-30              | 0.0003592 |      0.0005956 |           0.0004264 |             0.0003343 |            0.0022365 |
| Linear-Nested          | 0.0003361 |      0.0006159 |           0.0004099 |             0.0002607 |            0.0023323 |
| Linear-20              | 0.0004715 |      0.0007350 |           0.0005544 |             0.0004656 |            0.0031349 |
| Same-policy mean delta | 0.0006205 |      0.0013471 |           0.0007670 |             0.0002586 |            0.0041743 |
| Linear-10              | 0.0009312 |      0.0016932 |           0.0010752 |             0.0006482 |            0.0074619 |
| Persistence            | 0.0013890 |      0.0021096 |           0.0016197 |             0.0011336 |            0.0064248 |

最佳集成的分策略误差：

| policy                       |   n_batteries |      RMSE |       MAE |
|:-----------------------------|--------------:|----------:|----------:|
| 5_6C_36PER_4_3C_NEWSTRUCTURE |             7 | 0.0001385 | 0.0001071 |
| 5_3C_54PER_4C_NEWSTRUCTURE   |             7 | 0.0001676 | 0.0001335 |
| 5_6C_19PER_4_6C_NEWSTRUCTURE |             7 | 0.0002382 | 0.0001974 |
| 5C_67PER_4C_NEWSTRUCTURE     |             6 | 0.0002480 | 0.0002011 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE |             5 | 0.0003356 | 0.0001899 |
| 3_7C_31PER_5_9C_NEWSTRUCTURE |             2 | 0.0003492 | 0.0002721 |
| 80PER_3_6C                   |             2 | 0.0006989 | 0.0004875 |
| 4_8C_80PER_4_8C              |             2 | 0.0010367 | 0.0008873 |
| 3_6C-80PER_3_6C              |             2 | 0.0017406 | 0.0009688 |

三个集成的 horizon-bin 误差：

| model             | horizon_bin   |      RMSE |       MAE |
|:------------------|:--------------|----------:|----------:|
| Ensemble Global   | 1-10          | 0.0002726 | 0.0001562 |
| Ensemble Global   | 11-20         | 0.0004510 | 0.0002221 |
| Ensemble Global   | 21-30         | 0.0006681 | 0.0002626 |
| Ensemble Global   | 31-40         | 0.0003811 | 0.0002380 |
| Ensemble Global   | 41-50         | 0.0007928 | 0.0004277 |
| Ensemble Horizon  | 1-10          | 0.0002699 | 0.0001586 |
| Ensemble Horizon  | 11-20         | 0.0004668 | 0.0002403 |
| Ensemble Horizon  | 21-30         | 0.0006681 | 0.0002665 |
| Ensemble Horizon  | 31-40         | 0.0004046 | 0.0002750 |
| Ensemble Horizon  | 41-50         | 0.0007716 | 0.0004456 |
| Ensemble Adaptive | 1-10          | 0.0002634 | 0.0001507 |
| Ensemble Adaptive | 11-20         | 0.0004578 | 0.0002405 |
| Ensemble Adaptive | 21-30         | 0.0006440 | 0.0002561 |
| Ensemble Adaptive | 31-40         | 0.0003799 | 0.0002467 |
| Ensemble Adaptive | 41-50         | 0.0007404 | 0.0004127 |

## 6. 早期数据长度

以下表格固定使用 cycle 150 时表现最好的集成类型 `Ensemble Adaptive`，避免按每个 cutoff 重新挑选赢家：

|   cutoff |       MAE |   overall_RMSE | best_public_baseline   |   baseline_RMSE |   RMSE_reduction_vs_baseline_pct |
|---------:|----------:|---------------:|:-----------------------|----------------:|---------------------------------:|
|       50 | 0.0003329 |      0.0006241 | Linear-Nested          |       0.0007014 |                       11.0285898 |
|       75 | 0.0003187 |      0.0005801 | Linear-30              |       0.0005351 |                       -8.4100243 |
|      100 | 0.0002834 |      0.0005555 | Linear-30              |       0.0005170 |                       -7.4458672 |
|      125 | 0.0002623 |      0.0005137 | Linear-Nested          |       0.0006313 |                       18.6356689 |
|      150 | 0.0002613 |      0.0005265 | Linear-50              |       0.0005685 |                        7.3758552 |

这组结果只表示当前 50-cycle 预测任务在不同 cutoff 的回测表现；不应把 cutoff 变长与训练电池老化阶段变化混为严格因果效应。Adaptive 在 cutoff 75 和 100 未超过当时最佳公共线性基线，说明其跨 cutoff 优势并不稳定；本 Agent 仅推荐把它保留为 cycle 150 的候选，而不是无条件替代局部线性模型。

## 7. 权重与自适应性

cycle 150 的外层折平均权重：

| ensemble   | horizon_bin   | expert      |   weight |
|:-----------|:--------------|:------------|---------:|
| adaptive   | 1-10          | feature_ml  |  0.17607 |
| adaptive   | 1-10          | gp          |  0.20492 |
| adaptive   | 1-10          | linear20    |  0.13096 |
| adaptive   | 1-10          | linear30    |  0.14666 |
| adaptive   | 1-10          | linear50    |  0.17373 |
| adaptive   | 1-10          | same_policy |  0.16766 |
| adaptive   | 11-20         | feature_ml  |  0.19225 |
| adaptive   | 11-20         | gp          |  0.17441 |
| adaptive   | 11-20         | linear20    |  0.12611 |
| adaptive   | 11-20         | linear30    |  0.16908 |
| adaptive   | 11-20         | linear50    |  0.18137 |
| adaptive   | 11-20         | same_policy |  0.15677 |
| adaptive   | 21-30         | feature_ml  |  0.11421 |
| adaptive   | 21-30         | gp          |  0.19970 |
| adaptive   | 21-30         | linear20    |  0.05375 |
| adaptive   | 21-30         | linear30    |  0.33177 |
| adaptive   | 21-30         | linear50    |  0.22253 |
| adaptive   | 21-30         | same_policy |  0.07805 |
| adaptive   | 31-40         | feature_ml  |  0.27086 |
| adaptive   | 31-40         | gp          |  0.19422 |
| adaptive   | 31-40         | linear20    |  0.11595 |
| adaptive   | 31-40         | linear30    |  0.14480 |
| adaptive   | 31-40         | linear50    |  0.19314 |
| adaptive   | 31-40         | same_policy |  0.08103 |
| adaptive   | 41-50         | feature_ml  |  0.20409 |
| adaptive   | 41-50         | gp          |  0.17775 |
| adaptive   | 41-50         | linear20    |  0.04747 |
| adaptive   | 41-50         | linear30    |  0.31774 |
| adaptive   | 41-50         | linear50    |  0.16136 |
| adaptive   | 41-50         | same_policy |  0.09159 |
| global     | all           | feature_ml  |  0.19633 |
| global     | all           | gp          |  0.24923 |
| global     | all           | linear20    |  0.00420 |
| global     | all           | linear30    |  0.30647 |
| global     | all           | linear50    |  0.23388 |
| global     | all           | same_policy |  0.00988 |
| horizon    | 1-10          | feature_ml  |  0.16667 |
| horizon    | 1-10          | gp          |  0.16667 |
| horizon    | 1-10          | linear20    |  0.16667 |
| horizon    | 1-10          | linear30    |  0.16667 |
| horizon    | 1-10          | linear50    |  0.16667 |
| horizon    | 1-10          | same_policy |  0.16667 |
| horizon    | 11-20         | feature_ml  |  0.15917 |
| horizon    | 11-20         | gp          |  0.17550 |
| horizon    | 11-20         | linear20    |  0.15000 |
| horizon    | 11-20         | linear30    |  0.17892 |
| horizon    | 11-20         | linear50    |  0.17945 |
| horizon    | 11-20         | same_policy |  0.15695 |
| horizon    | 21-30         | feature_ml  |  0.03456 |
| horizon    | 21-30         | gp          |  0.23049 |
| horizon    | 21-30         | linear20    |  0.00473 |
| horizon    | 21-30         | linear30    |  0.47291 |
| horizon    | 21-30         | linear50    |  0.25106 |
| horizon    | 21-30         | same_policy |  0.00625 |
| horizon    | 31-40         | feature_ml  |  0.30760 |
| horizon    | 31-40         | gp          |  0.21304 |
| horizon    | 31-40         | linear20    |  0.13866 |
| horizon    | 31-40         | linear30    |  0.14034 |
| horizon    | 31-40         | linear50    |  0.18501 |
| horizon    | 31-40         | same_policy |  0.01534 |
| horizon    | 41-50         | feature_ml  |  0.18584 |
| horizon    | 41-50         | gp          |  0.17661 |
| horizon    | 41-50         | linear20    |  0.00000 |
| horizon    | 41-50         | linear30    |  0.48243 |
| horizon    | 41-50         | linear50    |  0.13920 |
| horizon    | 41-50         | same_policy |  0.01592 |

`weights.csv` 保留每个外层目标、集成类型、horizon bin 与 expert 的完整权重，可检查非负性及和为 1。adaptive 权重在预测目标时只由截止时点特征决定，不会先查看该电池的 151–200 误差再选专家。

## 8. 不确定性

区间基础尺度由加权 expert disagreement、GP posterior standard deviation 和数值稳定下限共同构成。随后在元层 battery-grouped OOF 残差上，对每个 10-cycle horizon bin 的“单电池最大标准化残差”做有限样本 90%/95% conformal quantile 校准。

cycle 150 的 adaptive 区间表现：

|   cutoff | ensemble   |   nominal_level |   point_coverage |   full_50_cycle_path_coverage |   mean_interval_width |   median_interval_width |   calibration_batteries_per_outer_fold | calibration_note                                                                                            |
|---------:|:-----------|----------------:|-----------------:|------------------------------:|----------------------:|------------------------:|---------------------------------------:|:------------------------------------------------------------------------------------------------------------|
|      150 | adaptive   |         0.90000 |          0.97600 |                       0.80000 |               0.00254 |                 0.00182 |                                     39 | battery-grouped meta cross-fit; max normalized residual calibrated separately in five 10-cycle horizon bins |
|      150 | adaptive   |         0.95000 |          0.99000 |                       0.85000 |               0.00384 |                 0.00293 |                                     39 | battery-grouped meta cross-fit; max normalized residual calibrated separately in five 10-cycle horizon bins |

这里的 coverage 是 40 块训练电池的外层回测经验覆盖率，不是对 9 块测试电池的严格概率保证。策略组最小仅 2 块训练电池，且各 horizon 内同一电池误差相关；因此样本不足以声称有限样本条件在测试分布上完全成立。`full_50_cycle_path_coverage` 只是描述性指标，校准实际按五个 10-cycle bin 分别进行。

## 9. 复现配置

- seed：20260814
- inner/meta battery folds：3/3
- Extra Trees：n_estimators=40, max_depth=7, min_samples_leaf=8, max_features=0.80
- adaptive gate：Ridge alpha=5.0, temperature=0.75, horizon-prior blend=0.5
- interval scale floor：5e-05
- 复杂度：global 6 个权重（5 自由度）；horizon 30 个权重（25 自由度）；最终 adaptive gate 为 21 输入×30 输出加 30 个截距，共 660 个 Ridge 系数；Extra Trees 固定 40 棵、深度上限 7，节点数随外层折变化，不伪报固定“参数量”；GP 不优化核超参数，每次最多拟合 50 个残差观测。
- 总运行时间：277.3 s

## 10. 局限性与建议

- 这是一套 short-horizon 模型比较，不输出 0.8 EOL。当前 CSV 没有任何真实 SOH≤0.8；把本模型无限递推到 0.8 会混淆短期预测与长期外推。
- same-policy 组仅 2–7 块训练电池，分策略 RMSE 波动大；不能把组均值差异解释为充电策略的因果效应。
- feature-ML 和 adaptive gate 的有效独立样本单位是 battery，而不是 50 个 horizon rows；模型复杂度已据此限制。
- Judge 应依据完整 LOBO、worst-battery RMSE 与跨 cutoff 稳定性决定是否保留 ensemble；若相近，应优先 recent-linear。

## 11. 已核验方法文献

1. Severson, K. A., Attia, P. M., Jin, N., et al. *Data-driven prediction of battery cycle life before capacity degradation*. **Nature Energy**, 4, 383–391 (2019). DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)。用于早期循环特征与小样本寿命预测的设计依据。
2. Richardson, R. R., Osborne, M. A., & Howey, D. A. *Gaussian process regression for forecasting battery state of health*. **Journal of Power Sources**, 357, 209–219 (2017). DOI: [10.1016/j.jpowsour.2017.05.004](https://doi.org/10.1016/j.jpowsour.2017.05.004)。用于显式均值函数与 GP 残差不确定性的设计依据。
3. Zhang, H., Li, Y., Zheng, S., et al. *Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning*. **Nature Machine Intelligence**, 7, 270–277 (2025). DOI: [10.1038/s42256-024-00972-x](https://doi.org/10.1038/s42256-024-00972-x)。用于 inter-cell / reference-cell 思路；本研究没有下载或使用其数据与测试真值。
