# Agent 04：显式均值函数 + Gaussian Process 残差

## 结论摘要

在严格的外层 Leave-One-Battery-Out、内层其余电池选型下，cutoff=150 的 GP 总体 RMSE=0.000678，MAE=0.000396。同一 cutoff 最佳固定 recent-linear 基线是 Linear-50，RMSE=0.000568；GP 相对变化 +19.3%。严格内层选窗的 Linear-Nested RMSE=0.000616，GP 相对变化 +10.2%。

这一路线的主要价值是给出经过跨电池校准的预测区间，而不是保证点预测一定胜过局部线性。cutoff=150 的 90%/95% 区间经验覆盖率分别为 89.6% / 95.0%，平均宽度为 0.001730 / 0.002082。

## 方法与泄漏控制

对每块目标电池只使用 `cycle <= cutoff` 的原始 SOH。模型写作 $SOH(n)=m(n)+r(n)$；$m(n)$ 比较局部线性、幂律和加速指数形式，$r(n)$ 使用 RBF、Matérn 3/2 或 Matérn 5/2 核。核长度尺度候选为 10、30、60 cycles。共 27 个候选组合。所有候选在外层目标电池以外的 39 块电池上比较 50-step RMSE，目标电池未来值不参与均值函数、核或长度尺度选择。

原始 `SOH_smooth` 未用于拟合，因为题目未给出其平滑算法，无法排除 cutoff 边界处的双向平滑。非线性均值用稳健 soft-L1 拟合；GP 残差的信号/噪声尺度只从当前电池 cutoff 内残差估计。90% 和 95% 区间用内层电池的标准化绝对误差分位数校准，再应用于被留出的外层电池。

## Short-horizon 回测

| cutoff | GP RMSE | GP mean-battery RMSE | GP worst-battery RMSE | best fixed linear | fixed RMSE | nested linear RMSE | GP vs nested |
|---:|---:|---:|---:|:---|---:|---:|---:|
| 50 | 0.001013 | 0.000773 | 0.003337 | Linear-20 | 0.000701 | 0.000701 | +44.4% |
| 75 | 0.000538 | 0.000427 | 0.001483 | Linear-30 | 0.000535 | 0.000789 | -31.8% |
| 100 | 0.000614 | 0.000414 | 0.002214 | Linear-30 | 0.000517 | 0.000517 | +18.8% |
| 125 | 0.000546 | 0.000435 | 0.001883 | Linear-50 | 0.000631 | 0.000631 | -13.6% |
| 150 | 0.000678 | 0.000495 | 0.002879 | Linear-50 | 0.000568 | 0.000616 | +10.2% |

### Forecast horizon

| horizon after cutoff=150 | RMSE | MAE |
|:---|---:|---:|
| 1-10 | 0.000318 | 0.000203 |
| 11-20 | 0.000543 | 0.000293 |
| 21-30 | 0.000751 | 0.000365 |
| 31-40 | 0.000592 | 0.000464 |
| 41-50 | 0.000995 | 0.000657 |

误差总体随外推距离扩大；41–50 step 的 RMSE 明显高于 1–10 step，说明局部残差修正的作用会随距离衰减，长期部分更依赖均值函数。

### Early-data length

从 cutoff=50 增加到 cutoff=150，GP RMSE 由 0.001013 降至 0.000678，相对下降 33.0%。但性能并不单调：本实验最佳 cutoff=75，RMSE=0.000538。因此不能把更多早期循环机械地等同于更低误差；当前均值族与最近 50-cycle 线性基线的局部适配仍会影响结果。

### 均值函数与核的独立比较

下面给出 cutoff=150 时每类均值函数跨核/长度尺度的最佳候选；它们是候选诊断，最终逐电池预测仍使用严格内层选型。

| mean function | best kernel | length scale | RMSE |
|:---|:---|---:|---:|
| power | matern32 | 60 | 0.000617 |
| linear | matern52 | 10 | 0.000624 |
| exponential | matern52 | 60 | 0.001219 |

按核函数聚合后的 cutoff=150 最佳候选为：

| kernel | best mean | length scale | RMSE |
|:---|:---|---:|---:|
| matern32 | power | 60 | 0.000617 |
| matern52 | linear | 10 | 0.000624 |
| rbf | power | 60 | 0.000649 |

完整逐点预测、分 policy 指标和 horizon 分段区间指标分别见 `predictions.csv`、`metrics.json`、`horizon_metrics.csv` 与 `uncertainty_metrics.csv`；全部 27 个候选的统一指标另见 `candidate_metrics.csv`。原始 GP 标准差不直接当置信区间；报告区间均使用严格内层电池误差校准。

## EOL：外推而非监督预测

40 块训练电池的 200-cycle 数据和 9 块测试电池的 150-cycle 数据均没有 SOH<=0.8。因此没有可报告的真实 EOL RMSE。`eol_sensitivity.csv` 分别给出线性、幂律、指数均值函数的 0.8 crossing；区间来自 100 条移动块残差 bootstrap 轨迹后重新拟合均值参数。GP 残差在训练域外回归到 0，不能凭空提供远至 0.8 的物理信息，所以这些区间是给定函数族的条件敏感性区间，不是已校准寿命置信区间。

9 块测试电池三种函数形式的有限 point EOL 跨模型差异中位数为 7379.8 cycles。若某一模型在 20000 cycles 内不 crossing，CSV 保留 NA 并报告 bootstrap crossing probability，不以截断上限伪装成寿命。

### 伪阈值验证

伪阈值只保留至少 10 块训练电池发生连续 3-cycle crossing 的阈值；当前入选阈值及可评估子集结果如下：

- 阈值 0.995：在 cutoff=150 的可评估子集上，linear 的 crossing MAE=5.29 cycles (n=15)。
- 阈值 0.997：在 cutoff=150 的可评估子集上，power 的 crossing MAE=7.20 cycles (n=14)。

伪阈值离 0.8 很远，只能检验近域 crossing 排序和数值稳定性，不能证明真正 EOL 的绝对准确度。

## 不确定性解释

短期区间的覆盖率是 40 块训练电池的外层 LOBO 经验覆盖率；policy 级结果样本仅 2–7 块电池，不可把单个 policy 的覆盖率解释为严格概率保证。EOL bootstrap 同时暴露拟合残差与参数不确定性，但没有包含未知老化机理改变、knee onset 或训练域外分布漂移，因此必须与三种均值函数的模型分歧联合报告。

## 复杂度与复现

- Seed: 20260814。
- 运行时间: 91.7 s。
- Python 3.13.9；NumPy 2.3.5；Pandas 2.3.3；SciPy 1.16.3。
- 每块电池的精确 GP 保存 cutoff 个对偶系数；连续量包括 2 个（linear）或 3 个（power/exponential）均值参数以及 2 个从残差估计的信号/噪声尺度，因此 cutoff=150 时预测状态约为 154–155 个数。核种类和长度尺度是内层 battery-level validation 选择的离散超参数。
- GP 为每块电池独立的一维精确 GP；没有引入不稳定的多输出 GP 或策略核。由于本 Agent 的任务是检验单电池趋势+残差 GP，本结果不声称已量化 charging-policy 的增益。

## 文献依据

1. Richardson, R. R., Osborne, M. A., & Howey, D. A. (2017). *Gaussian process regression for forecasting battery state of health*. Journal of Power Sources, 357, 209–219. DOI: [10.1016/j.jpowsour.2017.05.004](https://doi.org/10.1016/j.jpowsour.2017.05.004). 该文明确讨论显式均值函数、核选择、SOH/RUL 预测与 GP 不确定性；本实现采用其“显式退化均值 + GP 残差”思想，但没有读取其任何电池未来数据。
2. Severson, K. A., Attia, P. M., Jin, N., et al. (2019). *Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 4, 383–391. DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8). 该文支持按电池而非按循环行验证早期寿命信息；本实现仅借鉴方法原则。

## 建议

若 GP 的 cutoff=150 点预测 RMSE 未优于 recent-linear，则不建议把 GP 单独作为 Stage I 主模型；可将其校准区间或均值+残差预测作为候选 expert。Stage II 必须继续采用跨函数族敏感性报告，不能把任一 GP/均值函数无限外推得到的单个 0.8 crossing 当作精确寿命。
