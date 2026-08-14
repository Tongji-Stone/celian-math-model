# Agent 05：Inter-cell / Same-policy Cohort Transfer

## 摘要

本 Agent 将目标电池在 cutoff 之后的变化写成其他电池未来变化的加权迁移，并在 40 块非测试电池上完成 outer Leave-One-Battery-Out、inner Leave-One-Battery-Out 的严格嵌套回测。所有目标电池未来值只用于最外层评分，从未进入 reference pool 或超参数选择。

在核心 `150 -> 151--200` 回测中，cohort 系列最低 overall RMSE 的方案为 **Slope-adapted cohort**，RMSE=0.0006927，MAE=0.0003071，最差单电池 RMSE=0.0027712。相对公共 nested recent-linear 的 RMSE 变化为 -12.48%（正值表示降低误差，负值表示更差）。结论：**不推荐单独作为主模型；公共局部线性基线更准确，cohort 输出更适合作为集成专家。**

## 1 模型

对 cutoff 为 $N$ 的目标电池 $i$ 与参考电池 $j$，定义

$$D_j(h)=SOH_j(N+h)-SOH_j(N),\qquad h=1,\ldots,50,$$

并预测

$$\widehat{SOH}_i(N+h)=SOH_i(N)+\sum_j w_{ij}D_j(h),\qquad w_{ij}\ge 0,\ \sum_jw_{ij}=1.$$

比较的 cohort 路线为：

1. **Exact-policy mean delta**：仅使用同策略参考电池并等权平均；公共 baseline 与本实现逐预测点一致。
2. **Trajectory similarity**：对最后 $L$ 个 `SOH_smooth` 相对轨迹计算 RMSE，并用高斯核加权。
3. **Slope-adapted cohort**：在轨迹核权重上，用平滑 SOH 的近期退化速率比缩放参考增量，并将比例截断；速率下限仅用于避免近零斜率除法不稳定。
4. **Multivariate similarity**：SOH 相对轨迹与 IR、Tavg、chargetime 的标准化距离共同决定权重；缩放参数只由当前 reference pool 的 `cycle <= N` 数据计算。
5. **Soft-policy neighbors**：允许所有训练策略参与，权重同时惩罚 SOH 轨迹距离与标准化 $(C_1,Q_1,C_2)$ 策略距离；结构性缺失的单阶段策略使用 $C_1^{effective}=C_2$，没有总体均值填补。

## 2 无泄漏验证协议

- 9 块 `prediction_test == 1` 电池完全未进入本 Agent 的训练、选参和回测。
- 外层对 40 块训练电池逐块留一；其余 39 块是唯一可用 reference pool，目标自身的 future 永不进入 pool。
- 对每个 outer fold、每个 cutoff、每个改进方法，在 39 块 outer-training 电池上再做 inner LOO；$L$、核尺度、斜率 clip、多变量距离权重和策略核尺度都由 inner RMSE 决定。
- inner fold 如果因 outer 留一导致某策略没有同策略同伴，使用当时仍可用的其他 inner reference，且仍不读取 inner target future 作为 reference。
- cutoff 分别为 50、75、100、125、150；每次只从 `cycle <= cutoff` 构造相似度与标准化量，预测后 50 个 cycle。
- 预测行与公共 baseline 的真实值、same-policy mean delta 均逐点对账，最大差异不超过 $10^{-12}$。

## 3 核心回测结果

### 3.1 cutoff=150 全模型

| model                   |       MAE |   overall_RMSE |   mean_battery_RMSE |   median_battery_RMSE |   worst_battery_RMSE |
|:------------------------|----------:|---------------:|--------------------:|----------------------:|---------------------:|
| Public Linear-Nested    | 0.0003361 |      0.0006159 |           0.0004099 |             0.0002607 |            0.0023323 |
| Slope-adapted cohort    | 0.0003071 |      0.0006927 |           0.0004017 |             0.0002183 |            0.0027712 |
| Soft-policy neighbors   | 0.0005161 |      0.0011482 |           0.0006275 |             0.0002533 |            0.004348  |
| Trajectory similarity   | 0.0005956 |      0.0013427 |           0.0007399 |             0.0002059 |            0.0041743 |
| Multivariate similarity | 0.0006065 |      0.0013455 |           0.0007521 |             0.0002446 |            0.0041743 |
| Exact-policy mean delta | 0.0006205 |      0.0013471 |           0.000767  |             0.0002586 |            0.0041743 |

### 3.2 early cutoff 比较

|   cutoff | model                   |       MAE |   overall_RMSE |   mean_battery_RMSE |   worst_battery_RMSE |
|---------:|:------------------------|----------:|---------------:|--------------------:|---------------------:|
|       50 | Public Linear-Nested    | 0.0003945 |      0.0007014 |           0.0004867 |            0.0023417 |
|       50 | Slope-adapted cohort    | 0.0003646 |      0.0007923 |           0.0004692 |            0.0039435 |
|       50 | Exact-policy mean delta | 0.0005041 |      0.0011184 |           0.000636  |            0.0042968 |
|       75 | Slope-adapted cohort    | 0.0003744 |      0.0007762 |           0.0004551 |            0.004148  |
|       75 | Public Linear-Nested    | 0.0003967 |      0.0007893 |           0.0004708 |            0.003999  |
|       75 | Exact-policy mean delta | 0.0005291 |      0.0011554 |           0.0006325 |            0.0043698 |
|      100 | Public Linear-Nested    | 0.0002894 |      0.000517  |           0.0003699 |            0.0020066 |
|      100 | Slope-adapted cohort    | 0.0003579 |      0.0006848 |           0.0004387 |            0.0031955 |
|      100 | Exact-policy mean delta | 0.0006183 |      0.0012623 |           0.0007391 |            0.0044357 |
|      125 | Public Linear-Nested    | 0.0003779 |      0.0006313 |           0.0004717 |            0.0022388 |
|      125 | Slope-adapted cohort    | 0.0003818 |      0.0007924 |           0.0004919 |            0.003387  |
|      125 | Exact-policy mean delta | 0.0005705 |      0.0012247 |           0.0007141 |            0.0040972 |
|      150 | Public Linear-Nested    | 0.0003361 |      0.0006159 |           0.0004099 |            0.0023323 |
|      150 | Slope-adapted cohort    | 0.0003071 |      0.0006927 |           0.0004017 |            0.0027712 |
|      150 | Exact-policy mean delta | 0.0006205 |      0.0013471 |           0.000767  |            0.0041743 |

各 cutoff 的 cohort 系列最优候选如下；这是跨 outer folds 的回测总结，不用于对某块 outer battery 看未来后择优：

|   cutoff | model                |   overall_RMSE |   worst_battery_RMSE |
|---------:|:---------------------|---------------:|---------------------:|
|       50 | Slope-adapted cohort |      0.0007923 |            0.0039435 |
|       75 | Slope-adapted cohort |      0.0007762 |            0.004148  |
|      100 | Slope-adapted cohort |      0.0006848 |            0.0031955 |
|      125 | Slope-adapted cohort |      0.0007924 |            0.003387  |
|      150 | Slope-adapted cohort |      0.0006927 |            0.0027712 |

### 3.3 cutoff=150 的 horizon 分段

| horizon_bin   |      RMSE |       MAE |
|:--------------|----------:|----------:|
| 1-10          | 0.000264  | 0.0001601 |
| 11-20         | 0.0005245 | 0.0002746 |
| 21-30         | 0.0008085 | 0.0003246 |
| 31-40         | 0.0006472 | 0.0002897 |
| 41-50         | 0.000991  | 0.0004866 |

### 3.4 cutoff=150 的分策略表现

| policy                       |   n_batteries |   mean_exact_peer_count |      RMSE |       MAE |
|:-----------------------------|--------------:|------------------------:|----------:|----------:|
| 5_6C_36PER_4_3C_NEWSTRUCTURE |             7 |                       6 | 0.0001488 | 0.000117  |
| 5_3C_54PER_4C_NEWSTRUCTURE   |             7 |                       6 | 0.000165  | 0.0001298 |
| 5_6C_19PER_4_6C_NEWSTRUCTURE |             7 |                       6 | 0.0002166 | 0.0001657 |
| 3_7C_31PER_5_9C_NEWSTRUCTURE |             2 |                       1 | 0.0002898 | 0.000218  |
| 5C_67PER_4C_NEWSTRUCTURE     |             6 |                       5 | 0.0003013 | 0.0002464 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE |             5 |                       4 | 0.000369  | 0.0002346 |
| 4_8C_80PER_4_8C              |             2 |                       1 | 0.0008625 | 0.0006678 |
| 3_6C-80PER_3_6C              |             2 |                       1 | 0.0019717 | 0.0011876 |
| 80PER_3_6C                   |             2 |                       1 | 0.001983  | 0.0012999 |

### 3.5 同伴数与误差

|   exact_peer_count |   n_batteries |      RMSE |   mean_battery_RMSE |
|-------------------:|--------------:|----------:|--------------------:|
|                  1 |             8 | 0.0014704 |           0.0011313 |
|                  4 |             5 | 0.000369  |           0.0003602 |
|                  5 |             6 | 0.0003013 |           0.0002857 |
|                  6 |            21 | 0.0001792 |           0.0001667 |

同伴数与误差并非随机分配：peer count 由 policy 样本量决定，因此该表只能描述鲁棒性，不能解释为同伴数量的因果效应。

## 4 消融与解释

- Exact-policy RMSE=0.0013471；SOH 轨迹相似度 RMSE=0.0013427；斜率自适应 RMSE=0.0006927。
- 加入 IR、温度与充电时间后的 multivariate RMSE=0.0013455。这只是预测增益消融，不应被解释为这些量对寿命的因果效应。
- soft-policy RMSE=0.0011482，相对 exact-policy 变化 +14.76%（正值表示误差降低）。若该值为负，说明在当前样本中跨策略借用引入的偏差大于扩充同伴的收益。
- cutoff=150 最常见的内层选参结果如下；每个 outer fold 都独立重选，因此不存在用 outer target 的 151--200 误差选择 $L$/核尺度/clip 的行为。

| method       | most_common_config                                       |   outer_fold_count |
|:-------------|:---------------------------------------------------------|-------------------:|
| multivariate | {"L":20,"dynamic_weight":0.25,"tau_scale":0.75}          |                 34 |
| slope        | {"L":30,"clip_high":2.0,"clip_low":0.25,"tau_scale":1.5} |                 35 |
| soft_policy  | {"L":20,"strategy_tau":2.0,"tau_scale":0.75}             |                 35 |
| trajectory   | {"L":50,"tau_scale":0.75}                                |                 36 |

## 5 适用性与局限

1. 小策略仅有 2 块训练电池，outer fold 中 exact-policy 实际只有 1 个同伴，相似度权重无法在同策略内部发挥作用；这会导致分策略误差高度离散。
2. cohort 方法迁移的是 50-cycle 真实变化模板，只适合本题的短期预测；它没有提供可信的 $SOH=0.8$ 长期外推机制，不报告 EOL。
3. 未给出形式化预测区间。不同同伴的分歧可作为后续集成的不确定性信号，但当前样本量不足以把它直接称为校准置信区间。
4. 策略参数是离散实验组合，soft-policy 的参数距离不等价于物理退化距离，且 `policy` 与 $(C_1,Q_1,C_2)$ 共线。
5. 本模型是非参数加权器，训练参数量为 0；主要复杂度来自 nested LOO。运行时间为 36.63 秒，随机种子记录为 20260814（算法本身确定性）。

## 6 文献依据

- Han Zhang, Yuqi Li, Shun Zheng, Ziheng Lu, Xiaofan Gui, Wei Xu, Jiang Bian. *Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning*. **Nature Machine Intelligence**, 7, 270--277, 2025. DOI: [10.1038/s42256-024-00972-x](https://doi.org/10.1038/s42256-024-00972-x). 该文把未知寿命电池视为 target、完整电池视为 reference，并学习 cell 间差异；本 Agent 采用不需要神经网络的核加权短期差分迁移作为小样本对应物。
- Kristen A. Severson, Peter M. Attia, Norman Jin, Nicholas Perkins, Benben Jiang, Zi Yang, Michael H. Chen, Muratahan Aykol, Patrick K. Herring, Dimitrios Fraggedakis, Martin Z. Bazant, Stephen J. Harris, William C. Chueh, Richard D. Braatz. *Data-driven prediction of battery cycle life before capacity degradation*. **Nature Energy**, 4, 383--391, 2019. DOI: [10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8). 该文支持从早期循环提取退化差异信息；本报告仅借鉴方法，不使用其公开数据补全本题测试电池未来。

## 7 可复现产物

- `model.py`：cohort 模型、距离、权重与 nested selection。
- `run.py`：全量回测、公共 baseline 对账、指标与报告生成。
- `predictions.csv`：10,000 个 outer-CV 预测点及逐折配置。
- `ablation.csv`、`policy_metrics.csv`、`horizon_metrics.csv`、`cohort_size_metrics.csv`：总体、策略、horizon 与同伴数指标。
- `selected_hyperparameters.csv`、`metrics.json`：逐折选参与完整机器可读摘要。
