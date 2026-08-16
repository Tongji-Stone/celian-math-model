# Agent 02：半经验退化模型与长期 EOL 外推

## 1. 结论先行

短期 151–200 周期回测中，本路线最好的组合是 **power + cycle 50–150 拟合窗口**，overall RMSE 为 **0.000508963**。公共 baseline 的最佳 cutoff=150 模型为 Linear-50，RMSE=0.000568471；本路线最佳为 power（拟合起点 50），RMSE=0.000508963，差值为 -5.95082e-05。

0.8 EOL 没有真实 crossing 监督，因而本报告不提供 EOL RMSE，也不把任何单一外推值称为“真实寿命预测”。不同函数族和拟合起点产生的 EOL 估计在单块电池上的中位极差为 **11707.4 cycles**，最大极差为 **5735807.2 cycles**；中位最大/最小比为 **17.42**。这是模型假设不确定性，而不是可通过增加小型网络消除的普通噪声。

## 2. 数据边界与泄漏控制

- 仅使用赛题 PDF、`battery_summary.csv`、`cycle_train.csv`、公共审计与 baseline；未读取其他 Agent 的结果，未访问外部完整电池轨迹。
- 40 块非测试电池用于独立 rolling-origin battery-level 回测；9 块 `prediction_test=1` 电池没有参与训练、调参、伪阈值验证或本 Agent 的候选最终推理。
- 每个 cutoff 只读取 `cycle <= cutoff`。汇总表的 `mean_IR/mean_Tavg/mean_chargetime` 未被使用。
- 未直接使用赛题给出的 `SOH_smooth`：其平滑过程未知，且 battery 1 的早期异常会造成明显端点畸变。每个 cutoff 内仅对原始 SOH 使用固定 7 点 Hampel 异常抑制，参数不由未来数据选择。
- 所有模型允许观测点局部回升；物理约束只作用于潜在趋势的导数符号，避免把测量波动强行改成逐点单调。

## 3. 模型

比较三种可解释退化族：

\[
\text{Linear}:\quad SOH(n)=a-bn,\quad b\ge0,
\]

\[
\text{Power}:\quad SOH(n)=a-kn^p,\quad k\ge0,\ 0.25\le p\le3,
\]

\[
\text{Exponential}:\quad SOH(n)=a\exp(-kn),\quad k\ge0.
\]

参数用 `soft_l1` 稳健损失估计。另测试一个低容量半经验残差：在 power 曲线上叠加当前 10 周期中位残差，并以 25 cycle 时间常数指数衰减。它仅修正短期水平，不把残差趋势无限外推到 EOL。

在相同窗口下，damped residual 相对纯 power 的 RMSE 变化为 +6.13179e-06；没有改善，因此不支持增加残差网络。

## 4. 151–200 周期预测验证

### 4.1 cutoff=150 与窗口敏感性

| model | fit_start | MAE | overall_RMSE | mean_battery_RMSE | median_battery_RMSE | worst_battery_RMSE |
|---|---|---|---|---|---|---|
| power | 50 | 0.00028619 | 0.000508963 | 0.000354274 | 0.000240654 | 0.00219397 |
| power_damped_residual | 50 | 0.000292585 | 0.000515094 | 0.000359814 | 0.000251051 | 0.00222476 |
| power | 20 | 0.000334764 | 0.000555928 | 0.000414123 | 0.000323662 | 0.00219334 |
| power_damped_residual | 20 | 0.000333482 | 0.000558055 | 0.000415016 | 0.000332327 | 0.00222339 |
| power | 80 | 0.00034192 | 0.000578486 | 0.000413572 | 0.000297543 | 0.00218809 |
| power_damped_residual | 80 | 0.000353458 | 0.000588109 | 0.000425398 | 0.000307379 | 0.00221713 |
| linear | 80 | 0.000370826 | 0.000682986 | 0.000461846 | 0.000311264 | 0.00271122 |
| exponential | 80 | 0.000371274 | 0.000686308 | 0.000462334 | 0.000299543 | 0.00271255 |
| power_damped_residual | 1 | 0.000588798 | 0.0008327 | 0.000707051 | 0.000610119 | 0.00244707 |
| power | 1 | 0.000618977 | 0.000850239 | 0.000726558 | 0.000639161 | 0.00241699 |
| linear | 50 | 0.00048286 | 0.000861852 | 0.000576126 | 0.000427528 | 0.00293471 |
| exponential | 50 | 0.000486253 | 0.000868412 | 0.000580035 | 0.000425823 | 0.0029363 |
| linear | 20 | 0.000742984 | 0.00114762 | 0.000831483 | 0.000676747 | 0.00381797 |
| exponential | 20 | 0.000749952 | 0.00115756 | 0.000838578 | 0.000683832 | 0.00384717 |
| linear | 1 | 0.00113488 | 0.00152557 | 0.00121967 | 0.0010435 | 0.00479402 |
| exponential | 1 | 0.00114305 | 0.00153682 | 0.00122773 | 0.00104584 | 0.00482199 |

### 4.2 早期数据长度

下表在每个 cutoff 仅比较 cycle 1–N 拟合的三种退化族及小残差版本，并列出该 cutoff 的最优者。较晚 cutoff 不自动保证更好，因为全历史函数会平均掉最近退化速率；这也是单独检查 20/50/80 起点的原因。

| cutoff | model | fit_start | overall_RMSE | mean_battery_RMSE | worst_battery_RMSE |
|---|---|---|---|---|---|
| 50 | linear | 1 | 0.000900034 | 0.000773432 | 0.00189528 |
| 75 | power_damped_residual | 1 | 0.00104742 | 0.000912819 | 0.00238412 |
| 100 | power_damped_residual | 1 | 0.00102788 | 0.000899812 | 0.00231502 |
| 125 | power_damped_residual | 1 | 0.000905848 | 0.000716631 | 0.00302338 |
| 150 | power_damped_residual | 1 | 0.0008327 | 0.000707051 | 0.00244707 |

## 5. 伪阈值验证

以原始 SOH 的 5 点因果尾随中位数定义 crossing，并要求连续 3 点低于阈值。候选阈值为 [0.997, 0.995, 0.99, 0.98, 0.97]；只有在 cutoff=50 后至少 5 块训练电池发生 crossing 的阈值才保留，最终为 **[0.997, 0.995, 0.99]**。每个 case 的 cutoff 至少比真实 crossing 提前 5 cycles。

`crossing_MAE_within_range` 只评价落在 cutoff 与 2000 cycle 之间的 crossing；为防止极远预测或“无 crossing”被忽略，`penalized_crossing_MAE` 将缺失或超过 2000 的预测按 2000 cycle 计罚。90% 区间来自局部参数协方差抽样，仅用于诊断 calibration，不是严格覆盖保证。

| threshold | model | n_cases | n_batteries | within_evaluation_range_rate | crossing_MAE_within_range | penalized_crossing_MAE | interval_90_coverage | median_interval_width |
|---|---|---|---|---|---|---|---|---|
| 0.99 | power | 12 | 5 | 0.666667 | 7.18444 | 609.456 | 0.5 | 43.3014 |
| 0.99 | linear | 12 | 5 | 0.666667 | 242.756 | 766.504 | 0.0833333 | 409.511 |
| 0.99 | exponential | 12 | 5 | 0.666667 | 244.395 | 767.597 | 0.0833333 | 412.056 |
| 0.995 | power | 82 | 17 | 0.719512 | 28.6849 | 532.676 | 0.231707 | 37.365 |
| 0.995 | linear | 82 | 17 | 0.707317 | 173.518 | 656.745 | 0.0121951 | 200.888 |
| 0.995 | exponential | 82 | 17 | 0.707317 | 174.197 | 657.225 | 0.0121951 | 201.509 |
| 0.997 | power | 127 | 31 | 0.551181 | 14.6545 | 835.604 | 0.228346 | 54.035 |
| 0.997 | linear | 127 | 31 | 0.543307 | 221.525 | 960.852 | 0.0472441 | 1.95363e+07 |
| 0.997 | exponential | 127 | 31 | 0.543307 | 221.955 | 961.086 | 0.0472441 | 1.95758e+07 |

低阈值 crossing 数量很少，模型排序只能说明“接近当前数据范围的阈值外推”表现；它不能验证 0.8 的远距离寿命。

在三个保留阈值上，power 的惩罚 MAE 均为三类模型最低，但只有约 55%–72% 的 case 落入 2000-cycle 评价范围，90% 区间经验覆盖率也只有约 23%–50%。因此“power 排名第一”只是相对结论，绝不构成可靠校准或远程 EOL 准确性的证据。

## 6. 0.8 EOL 的模型与窗口敏感性

下表对每块训练电池仅使用 cycle 1–150，并分别从 cycle 1、20、50、80 开始拟合。区间为固定 8-cycle block 的移动块残差 bootstrap（40 次）得到的 90% 条件区间。

| fit_start | model | n_finite | median_EOL | p10_EOL | p90_EOL | min_EOL | max_EOL | median_bootstrap_interval_width |
|---|---|---|---|---|---|---|---|---|
| 1 | exponential | 40 | 11487.56 | 6664.303 | 27146.29 | 1084.742 | 37332.23 | 4500.111 |
| 1 | linear | 40 | 10308.46 | 5985.652 | 24341.25 | 985.7688 | 33465.39 | 3937.166 |
| 1 | power | 40 | 797.9599 | 643.3638 | 1222.47 | 473.5921 | 2214.473 | 1044.362 |
| 20 | exponential | 40 | 8974.203 | 5719.72 | 17266.47 | 1052.215 | 18961.91 | 1186.947 |
| 20 | linear | 40 | 8057.088 | 5225.706 | 15487.41 | 958.0863 | 17010.91 | 1221.677 |
| 20 | power | 40 | 1479.535 | 700.224 | 4320.488 | 555.1861 | 6883.502 | 784.122 |
| 50 | exponential | 40 | 8093.991 | 4667.246 | 13273.43 | 1013.749 | 17360.35 | 966.2812 |
| 50 | linear | 40 | 7272.49 | 4197.574 | 11911.4 | 926.1392 | 15578.19 | 798.9834 |
| 50 | power | 40 | 2967.756 | 764.6534 | 16349.24 | 577.1418 | 101918.3 | 4075.49 |
| 80 | exponential | 40 | 7547.582 | 3889.726 | 12299.92 | 990.8831 | 19041.12 | 1330.53 |
| 80 | linear | 40 | 6784.483 | 3503.087 | 11041.9 | 908.201 | 17100.82 | 998.4111 |
| 80 | power | 40 | 1881.726 | 671.7883 | 5465.005 | 584.1287 | 5736567 | 8128.619 |

bootstrap 区间只反映“给定函数族和窗口”的参数/残差不确定性；跨函数族、跨窗口的差异通常更大。因此论文应同时列出 linear/power/exponential，另以模型-窗口包络表示假设敏感性。没有 0.8 crossing 时，不应把 bootstrap 区间称为经覆盖率校准的置信区间。

## 7. PINN / Neural ODE 判断

本数据只有 40 个独立训练 cell、每个只有早期 200 cycles，且没有 0.8 crossing。cycle 行数并不等于独立物理实验数；把同一电池的相邻行当成大量监督样本会夸大有效样本量。题目数据也不含可辨识的电化学状态方程、反应速率或完整长期轨迹，PINN/Neural ODE 中的“物理项”和“神经残差”会高度互相补偿。鉴于低容量 damped residual 的实际增益及模型族敏感性，本 Agent **淘汰 PINN/Neural ODE，不推荐作为主模型或 EOL 外推器**。更合理的做法是短期预测与长期半经验外推分阶段，并把函数族分歧显式报告。

## 8. 可复现性

- seed：20260814
- Python：3.13.9
- NumPy：2.3.5
- pandas：2.3.3
- SciPy：1.16.3
- robust loss：`soft_l1`，`f_scale=0.0015`
- 参数量：linear=2，power=3，exponential=2，power+damped-residual=4（其中 decay=25 固定）
- power exponent bounds：`[0.25, 3.0]`
- EOL bootstrap：moving-block residual bootstrap，block=8，B=40
- runtime：99.21 s

运行命令：`python problem3/agents/agent_02_physics/run.py`

## 9. 文献依据（仅用于方法，不用于提取测试答案）

1. Kristen A. Severson, Peter M. Attia, Norman Jin, Nicholas Perkins, Benben Jiang, Zi Yang, Michael H. Chen, Muratahan Aykol, Patrick K. Herring, Dimitrios Fraggedakis, Martin Z. Bazant, Stephen J. Harris, William C. Chueh, Richard D. Braatz. *Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 2019, 4:383–391. DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8).
2. Ian A. Richardson, Michael A. Osborne, David A. Howey. *Gaussian process regression for forecasting battery state of health*. Journal of Power Sources, 2017, 357:209–219. DOI: [10.1016/j.jpowsour.2017.05.004](https://doi.org/10.1016/j.jpowsour.2017.05.004).
3. Pengfei Wen, Zhi-Sheng Ye, Yong Li, Shaowei Chen, Pu Xie, Shuai Zhao. *Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries*. IEEE Transactions on Intelligent Vehicles, 2024, 9(1):2276–2289. DOI: [10.1109/TIV.2023.3315548](https://doi.org/10.1109/TIV.2023.3315548).

这些论文分别支持早期信息建模、显式退化均值/不确定性、以及物理—数据融合的研究动机；本实验没有下载或使用其中任何电池未来数据。
