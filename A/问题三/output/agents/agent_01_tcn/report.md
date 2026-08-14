# Agent 01：小样本时序神经网络研究

## 1. 结论先行

TCN 未显著优于 recent-linear baseline，不推荐作为问题三主模型。

在核心 `150 -> 151--200` 的严格外层 battery-level 五折交叉验证中，嵌套选择 TCN 的 overall RMSE 为 0.00216067，Linear-50 为 0.00056847。TCN 相对误差变化为 280.08%。40 块电池的配对比较中，TCN 仅在 18 块上更优；`TCN RMSE - Linear-50 RMSE` 的均值为 0.00043354，95% battery bootstrap 区间为 [0.00000844, 0.00114086]。单侧 Wilcoxon 检验（备择：TCN 更小）p=0.9107。

因此，对“神经网络是否显著优于 recent-linear baseline？”的回答是：**否**。当前数据仅有 40 块可监督训练电池，TCN 的参数共享没有转化为稳定的跨电池优势，**不推荐作为问题三短期预测主模型**。

## 2. 数据、任务与泄漏控制

- 数据审计给出 49 块电池，其中 40 块训练电池含 cycle 1--200，9 块 `prediction_test=1` 电池只含 cycle 1--150。本 Agent 完全未将这 9 块电池用于训练、调参或验证。
- 本路线只研究可回测的短期任务：在 cutoff `N` 后直接预测 50 维 `SOH(N+h)-SOH(N)`。数据中没有 `SOH <= 0.8`，所以不使用 TCN 无限递推 EOL，也不报告神经网络 EOL 精度。
- 外层采用 policy-aware 的五折 battery split；同一电池的 cycle 行绝不跨 train/validation。每个外层折内另设 battery-level inner holdout，先比较消融，再比较序列长度。外层验证电池的未来值从不参与选择。
- rolling cutoffs 为 50,60,...,150；它们只在已完成 battery 划分后从训练电池生成。每个 cutoff 的输入只读取 `cycle <= cutoff`，目标读取训练电池的后续 50 cycle。
- 汇总表的 `mean_IR/mean_Tavg/mean_chargetime` 完全未用。动态标准化、静态标准化和逐 horizon 目标标准化均只在对应训练折拟合。
- 题目未说明给定 `SOH_smooth` 的平滑方向。为避免预计算的平滑端点隐含未来信息，代码从当前 cutoff 内的原始 SOH 重建 5 点 trailing median + causal EWMA，记为 `SOH_causal`。

## 3. 模型结构

模型由四个 dilation=(1,2,4,8) 的因果 TCN residual blocks、可选静态 MLP、以及 direct 50-horizon decoder 组成。编码器拼接最后时刻表示和有效时间步的 masked mean，输出 50 个 SOH delta；不存在 autoregressive error accumulation。不同外层折所选模型参数量见 `predictions.csv`，全路线参数量保持在约 5630--6174，远小于大型 Transformer。

固定正则化为 dropout=0.2、AdamW weight decay=0.0001，inner training 最多 30 epochs、patience=5。最终外层模型在全部 outer-train batteries 上按 inner early stopping 选出的 epoch 数重新训练。

## 4. 消融与序列长度筛选

消融定义如下：

- A：仅 cutoff 内重建的因果 SOH 平滑序列；
- B：A + raw SOH、capacity、IR、Tavg、chargetime，并加入静态 initial_capacity；
- C：B + C1、Q1、C2 和 C1 结构性缺失指示。C1 缺失时使用审计确定的单阶段策略物理解释 `C1_effective=C2`，没有均值填补。

下表是五个外层折内部 holdout 的筛选误差，仅用于选参，不是外层泛化成绩。

| ablation              |   inner_validation_MAE |   inner_validation_RMSE |   folds |
|:----------------------|-----------------------:|------------------------:|--------:|
| A_SOH_only            |             0.00054960 |              0.00104100 |       5 |
| B_multivariate        |             0.00068818 |              0.00124414 |       5 |
| C_multivariate_policy |             0.00068550 |              0.00129263 |       5 |

筛选采用节省计算的顺序嵌套方案：先固定 length=50 比较 A/B/C，再仅对该折胜出的消融比较 20/30/50/75/100。这样避免 15 个组合的无意义网格搜索，也确保没有查看 outer future。

|   sequence_length |   inner_validation_MAE |   inner_validation_RMSE |     trials |
|------------------:|-----------------------:|------------------------:|-----------:|
|       20.00000000 |             0.00057670 |              0.00101536 | 5.00000000 |
|       30.00000000 |             0.00057883 |              0.00102015 | 5.00000000 |
|       50.00000000 |             0.00049807 |              0.00089637 | 5.00000000 |
|       75.00000000 |             0.00056288 |              0.00102458 | 5.00000000 |
|      100.00000000 |             0.00051797 |              0.00093951 | 5.00000000 |

各折最终选择计数：`{'A_SOH_only': 3, 'C_multivariate_policy': 1, 'B_multivariate': 1}`；序列长度计数：`{'50': 2, '30': 1, '75': 1, '20': 1}`。由于 inner holdout 规模小，选择在折间不稳定，这本身是小电池样本下神经模型方差较大的证据。

## 5. 外层验证结果

核心 cutoff=150：

| model         |        MAE |   overall_RMSE |   mean_battery_RMSE |   median_battery_RMSE |   worst_battery_RMSE |
|:--------------|-----------:|---------------:|--------------------:|----------------------:|---------------------:|
| Linear-50     | 0.00031025 |     0.00056847 |          0.00038879 |            0.00026068 |           0.00233229 |
| Linear-Nested | 0.00033613 |     0.00061589 |          0.00040986 |            0.00026068 |           0.00233229 |
| TCN-Nested    | 0.00067662 |     0.00216067 |          0.00082234 |            0.00035563 |           0.01288235 |

各 early cutoff 的 overall RMSE：

|       cutoff |   Linear-50 |   Linear-Nested |   TCN-Nested |
|-------------:|------------:|----------------:|-------------:|
|  50.00000000 |  0.00356107 |      0.00070144 |   0.00221039 |
|  75.00000000 |  0.00062609 |      0.00078935 |   0.00214856 |
| 100.00000000 |  0.00057054 |      0.00051704 |   0.00215161 |
| 125.00000000 |  0.00063133 |      0.00063133 |   0.00210260 |
| 150.00000000 |  0.00056847 |      0.00061589 |   0.00216067 |

完整 overall、per-policy、horizon-bin 指标在 `metrics.json`；逐点外层预测在 `predictions.csv`。公共 Linear-50 与 Linear-Nested 预测按 battery/cutoff/horizon 原样合并，以保证相同真值、相同折外样本的公平比较。

## 6. 解释与局限

1. rolling cutoff 增加的是同一电池内高度相关的训练窗口，并没有把 battery-level 样本数从 40 真正扩展到数百个独立样本。
2. TCN 可学习局部多变量形态，但 IR、温度、充电时间和策略变量的增益在 inner folds 间不稳定；网络容易把电池个体噪声当成跨电池规律。
3. 近期线性趋势与本数据 50-cycle 预测尺度高度匹配。更复杂网络只有在配对外层误差及其不确定区间均优于线性时才应升级；本实验未达到这一标准。
4. 五折 outer CV 严格隔离电池，但不是 40 次 LOO；每折含约 8 块电池。它给出可信的 battery-level 泛化估计，同时把 CPU 预算集中在严格嵌套而非扩大网络。
5. 充电策略贡献不能从本 Agent 单独作因果结论：策略只有 9 个离散组合，并与 C1/Q1/C2 共线；这里只能解释为预测消融。

## 7. 复现信息

- seed：20260814
- Python：3.13.9
- PyTorch：2.12.0+cpu（CPU，单线程确定性训练）
- NumPy / pandas / scikit-learn / SciPy：2.3.5 / 2.3.3 / 1.7.2 / 1.16.3
- 总运行时间：55.5 s
- 运行命令：`python problem3/agents/agent_01_tcn/run.py`

## 8. 方法文献

1. Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv. DOI: 10.48550/arXiv.1803.01271。用于因果扩张卷积与 residual TCN 的结构依据。
2. Severson, K. A., et al. (2019). *Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 4, 383--391. DOI: 10.1038/s41560-019-0356-8。用于早期循环信息可以支持寿命/退化预测的研究背景；本实验没有读取其公开电池未来轨迹。

以上论文只用于方法设计和结果解释，没有从外部数据、论文补充材料、`global_id` 或 GitHub 获取 9 块测试电池 cycle 151 后的真值。
