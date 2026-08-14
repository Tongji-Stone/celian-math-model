# 任务：使用 6 个子 Agent 并行完成问题三的模型探索、比较与最终实现

你现在负责完成“2026年度策联杯数学建模精英联赛 A题——问题三：电池寿命预测”。

这不是一个单模型实现任务。

你需要首先创建 **6 个相互独立的研究 Agent**，让它们从不同建模范式独立探索问题；随后建立统一评价体系比较六种方案；最后由主 Agent 综合实验结果，选择或融合最优方案，完成问题三最终代码、预测结果、图表和建模报告。

不要预设神经网络一定优于传统模型。

最终模型必须由严格的 battery-level validation 结果决定。

---

# 0. 首先读取项目与数据

在开始建模之前，主 Agent 和所有子 Agent 必须读取并理解：

- 比赛题目 PDF
- `battery_summary.csv`
- `cycle_train.csv` 或实际存在的对应文件
- 当前项目中已经完成的问题一、问题二代码和结果（如果存在）

先自动检查项目文件结构，不要假定路径。

读取数据后输出一份：

`problem3/data_audit.md`

记录：

- battery 数量
- prediction_test 数量
- policy 数量
- 每种 policy 的 battery 数量
- 测试 battery ID
- 每个 battery 最大 cycle
- 缺失值
- C1/Q1/C2 数据特点
- SOH / SOH_smooth 范围
- IR / Tavg / chargetime 范围
- 每个测试 battery 是否存在同 policy 的训练 battery
- 给定数据中是否真实存在 SOH <= 0.8

不得直接开始训练模型。

---

# 1. 问题定义

问题三要求完成：

1. 表征电池健康状态的特征提取；
2. 建立未来 SOH 衰减预测模型；
3. 根据测试电池前150 cycle预测 cycle151–200；
4. 预测 SOH 达到0.8时的循环寿命 EOL；
5. 比较不同早期循环数据长度对预测结果的影响；
6. 分析充电策略信息和早期衰减特征对寿命预测的影响。

必须明确区分两个任务：

## Task A：Short-horizon forecasting

已知：

\[
SOH(1),...,SOH(N)
\]

预测：

\[
SOH(N+1),...,SOH(N+50)
\]

最终核心任务：

\[
N=150
\]

预测：

\[
151\sim200
\]

这是可以利用训练电池真实数据进行严格回测的任务。

---

## Task B：Long-horizon EOL extrapolation

求：

\[
n_{\mathrm{EOL}}
=
\min\{n:SOH(n)\le0.8\}
\]

首先检查当前竞赛 CSV 是否包含真实 SOH<=0.8 的监督数据。

如果当前数据只提供早期约200 cycles、SOH仍明显高于0.8：

必须明确指出：

> EOL 不是普通监督预测问题，而是 long-horizon extrapolation。

禁止仅仅把短期模型无限递推到0.8，然后把结果当作精确寿命。

EOL必须单独进行：

- degradation model comparison
- pseudo-threshold validation
- uncertainty estimation
- sensitivity analysis

---

# 2. 严格的数据泄漏规则

这是整个问题三最高优先级约束。

## Rule 1

`prediction_test == 1`

的9块电池只能使用前150 cycles。

绝对禁止通过：

- 外部完整 MIT 数据集
- GitHub 中完整原始实验数据
- 论文 supplementary data
- global_id 查询
- 任何其他途径

取得这9块电池的151 cycle以后真实数据。

论文只允许用于借鉴方法。

不得获取比赛测试答案。

---

## Rule 2

模型验证必须按照：

**battery_id**

划分。

严禁随机划分 cycle rows。

同一个 battery 的 cycle 不允许同时出现在 train 和 validation 中。

---

## Rule 3

模拟 cutoff=N 时：

所有特征必须只使用：

\[
cycle\le N
\]

的数据。

例如：

cutoff=100

禁止使用 cycle101–200 计算：

- mean_IR
- mean_Tavg
- mean_chargetime
- normalization parameter
- smoothing parameter
- feature selection statistic

---

## Rule 4

`battery_summary.csv` 中：

- mean_IR
- mean_Tavg
- mean_chargetime

可能包含 cutoff 后的信息。

因此在 simulated cutoff validation 中：

不要直接使用这些字段。

从 `cycle_train.csv` 的：

\[
cycle\le cutoff
\]

重新计算。

可以直接使用的 static variables：

- C1
- Q1
- C2
- policy
- initial_capacity

---

# 3. 建立统一 Benchmark

在运行复杂模型之前，实现：

`problem3/common/baselines.py`

至少包括：

## Baseline 0

Persistence：

\[
\hat{SOH}_{N+h}=SOH_N
\]

## Baseline 1

recent 10-cycle linear regression

## Baseline 2

recent 20-cycle linear regression

## Baseline 3

recent 30-cycle linear regression

## Baseline 4

recent 50-cycle linear regression

## Baseline 5

same-policy mean future delta

即：

\[
\Delta SOH_j(h)
=
SOH_j(N+h)-SOH_j(N)
\]

然后：

\[
\hat{SOH}_{target}(N+h)
=
SOH_{target}(N)
+
\frac1K
\sum_j \Delta SOH_j(h)
\]

其中 j 只能来自其他训练 battery。

自动选择最佳 linear window 时必须使用内层 validation，不能根据测试 battery 决定。

---

# 4. 六个子 Agent

创建：

```text
problem3/
    agents/
        agent_01_tcn/
        agent_02_physics/
        agent_03_ml/
        agent_04_gp/
        agent_05_intercell/
        agent_06_ensemble/
```

六个 Agent 相互独立工作。

每个 Agent：

- 可以读取公共 data audit；
- 可以读取题目；
- 可以读取公共 baseline；
- 不允许读取其他 Agent 的最终实验结果，直到独立探索阶段结束。

避免 confirmation bias。

---

# Agent 01 — Small-data Temporal Neural Network

研究：

**TCN / GRU 多变量时序预测**

目标不是证明神经网络有效，而是检验神经网络是否能从多变量 cycle dynamics 中提取简单趋势模型无法获得的信息。

## Dynamic input

研究：

- SOH_smooth
- SOH
- capacity
- IR
- Tavg
- chargetime

## Static input

- C1
- Q1
- C2
- initial_capacity

## 推荐结构

优先：

TCN encoder

+

static strategy MLP encoder

+

direct multi-horizon decoder

不要首先使用大型 Transformer。

输出建议预测：

\[
\Delta SOH_h
=
SOH_{N+h}-SOH_N
\]

而不是绝对 SOH。

最终：

\[
\hat{SOH}_{N+h}
=
SOH_N+\widehat{\Delta SOH_h}
\]

## 数据扩充

允许使用 rolling cutoff：

例如：

60,65,70,...,150

构造训练样本。

但必须：

先按 battery 划分 train/validation，

再生成 windows。

禁止 window-level random split。

## 消融实验

比较：

A. SOH only

B. SOH + IR + Tavg + chargetime

C. B + C1/Q1/C2

研究：

sequence length：

20 / 30 / 50 / 75 / 100

网络参数量必须记录。

使用：

- early stopping
- dropout
- weight decay

控制过拟合。

输出：

`agents/agent_01_tcn/report.md`

必须明确回答：

> 神经网络是否显著优于 recent-linear baseline？

如果没有：

明确判定“不推荐作为主模型”。

---

# Agent 02 — Physics-informed / Semi-empirical Degradation Model

该 Agent 重点负责：

**长期外推稳定性。**

研究：

\[
SOH(n)=f(n;\theta)
\]

至少比较：

### Linear

\[
SOH=a-bn
\]

### Power law

\[
SOH=a-kn^p
\]

### Exponential family

选择有合理长期衰减行为的形式。

进一步研究：

**semi-empirical model + neural residual**

例如：

\[
\frac{dSOH}{dn}
=
-f_{\mathrm{empirical}}(n,\theta)
-f_{\mathrm{NN}}
(SOH,IR,T,C_1,Q_1,C_2)
\]

约束：

\[
\frac{dSOH}{dn}\le0
\]

作为软约束而非强制每个观测点严格单调。

研究：

- PINN
- Neural ODE
- physics-informed residual network

哪个在当前小数据条件下最合理。

不得为了“PINN”这一名称强行增加复杂度。

## 特别实验

分别用：

cycle 1–150

cycle 20–150

cycle 50–150

cycle 80–150

拟合。

比较：

- 151–200 RMSE
- EOL预测

如果仅改变 fitting window 就导致 EOL 巨幅变化：

必须把这个结果作为重要结论。

输出：

- EOL sensitivity
- model assumption sensitivity
- uncertainty interval

---

# Agent 03 — Feature Engineering + Tree Machine Learning

该 Agent 负责：

**小样本机器学习和可解释性。**

建立 battery health feature vector。

## SOH features

至少：

- last SOH
- recent slope 10/20/30/50
- recent delta
- variance
- local linear RMSE
- quadratic curvature
- first derivative
- approximate second derivative

## IR features

- IR_last
- IR_mean
- IR_slope
- IR_delta
- IR_std

## Temperature

- Tavg_mean
- Tavg_last
- Tavg_slope
- Tavg_std

## Chargetime

- mean
- last
- slope
- delta

## Cross-indicator features

例如：

\[
corr(SOH,IR)
\]

\[
corr(SOH,T)
\]

\[
corr(SOH,t_{charge})
\]

## Strategy features

原始：

- C1
- Q1
- C2

设计：

\[
E_1=C_1Q_1
\]

\[
E_2=C_2(80-Q_1)
\]

作为简单 charging exposure proxy。

再研究：

\[
C_1-C_2
\]

weighted C-rate

SOC-stage width

以及其他有合理解释的 interaction。

对于 C1 缺失：

不要盲目均值填充。

根据 policy 物理含义判断，并保留：

`C1_missing_indicator`

---

## Model

比较：

- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

若库不存在，不必全部安装。

优先选择最适合小数据的2–4个。

预测方法建议：

把：

`forecast_horizon`

作为输入：

\[
X_N,h
\rightarrow
SOH_{N+h}-SOH_N
\]

从而建立一个统一 multi-horizon regressor。

---

## 必须做 Ablation

A：

SOH features only

B：

SOH + IR + Tavg + chargetime

C：

B + charging policy

比较：

MAE / RMSE。

利用：

- permutation importance
- SHAP（若可靠）

回答：

> charging policy 信息是否真的增加预测能力？

---

# Agent 04 — Gaussian Process Forecasting

研究：

**Gaussian Process + uncertainty**

优先考虑：

\[
SOH(n)
=
m(n)+r(n)
\]

其中：

\[
r(n)\sim GP(0,k)
\]

不要默认 zero mean GP。

比较 mean function：

- local linear
- power law
- exponential

比较 kernel：

- RBF
- Matérn 3/2
- Matérn 5/2

重点研究：

**Linear mean + GP residual**

因为纯 GP 在远距离外推时可能向 prior mean 回归。

---

## Strategy-conditioned GP

如果可行，研究：

\[
k
=
k_{cycle}
\times
k_{strategy}
\]

其中 strategy features：

\[
(C_1,Q_1,C_2)
\]

或者：

shared GP

+

battery-specific residual。

不要为了 Multi-output GP 引入无法稳定训练的大规模复杂实现。

---

## Uncertainty

必须得到：

\[
\mu_{SOH}(n)
\]

以及：

\[
\sigma_{SOH}(n)
\]

报告：

- 90% / 95% interval
- empirical coverage
- interval width

EOL不能只从 posterior mean 求一次。

从 posterior trajectory sampling：

\[
SOH^{(m)}(n)
\]

得到：

\[
n_{\mathrm{EOL}}^{(m)}
\]

最终报告：

- median EOL
- lower quantile
- upper quantile

同时报告不同 mean function 导致的 EOL 差异。

---

# Agent 05 — Inter-cell / Same-policy Cohort Transfer

这是一个需要重点探索的创新方向。

首先验证：

每个 prediction_test battery 是否存在同 policy 的训练 peers。

如果成立：

定义：

\[
\mathcal P_i
=
\{
j:
policy_j=policy_i,
j\neq i
\}
\]

---

## Basic template

对 peer j：

\[
D_j(h)
=
SOH_j(N+h)-SOH_j(N)
\]

预测：

\[
\hat{SOH}_i(N+h)
=
SOH_i(N)
+
\sum_jw_{ij}D_j(h)
\]

其中：

\[
\sum_jw_{ij}=1
\]

---

## Innovation 1：trajectory similarity

利用最后 L 个 cycles：

\[
x_i=
SOH_i(N-L+1:N)-SOH_i(N)
\]

计算：

\[
d_{ij}
=
RMSE(x_i,x_j)
\]

权重可以：

\[
w_{ij}
\propto
\exp(-d_{ij}^2/\tau^2)
\]

---

## Innovation 2：degradation-rate adaptation

定义：

\[
s_i
=
\text{Slope}
(SOH_i(N-L+1:N))
\]

对 peer：

\[
r_{ij}
=
\frac{s_i}{s_j}
\]

使用合理 clipping：

\[
r_{\min}
\le r_{ij}\le
r_{\max}
\]

预测：

\[
D'_j(h)=r_{ij}D_j(h)
\]

clipping 范围必须通过 validation 决定。

---

## Innovation 3：multivariate similarity

trajectory distance 不仅使用 SOH。

研究：

- SOH
- IR
- Tavg
- chargetime

标准化后组合距离。

---

## Innovation 4：soft policy neighbors

如果某种 policy peers 数量太少：

不局限 exact policy。

建立：

\[
d_{strategy}(i,j)
=
\sqrt{
w_1(C_{1i}-C_{1j})^2+
w_2(Q_{1i}-Q_{1j})^2+
w_3(C_{2i}-C_{2j})^2
}
\]

允许从参数接近的其他策略借用信息。

比较：

exact-policy

vs

soft-policy neighbors。

---

## 最重要的 validation

对每一块非测试 battery：

把它假装成测试 battery。

只保留前 N cycles。

所有其他 battery 是 reference pool。

完成真正的：

**Leave-One-Battery-Out cohort backtest**

如果 target battery 属于某 policy：

不得使用 target 自己的 future trajectory。

报告：

- 每个 policy RMSE
- peer count
- performance vs cohort size

---

# Agent 06 — Mixture of Experts / Adaptive Ensemble

该 Agent 在独立阶段不能简单读取其他 Agent 的最终结果。

首先自行实现一组基础 Experts：

### Expert A

recent 20-cycle linear

### Expert B

recent 30-cycle linear

### Expert C

recent 50-cycle linear

### Expert D

same-policy cohort transfer

### Expert E

feature-based ML

### Expert F

GP（若计算量允许）

---

# Ensemble formulation

最终：

\[
\hat y
=
\sum_{k=1}^K
w_k\hat y_k
\]

满足：

\[
w_k\ge0,
\qquad
\sum_kw_k=1
\]

比较：

## Global weights

一个固定权重。

## Horizon-dependent weights

\[
w_k(h)
\]

例如：

短 horizon 更相信 local trend，

长 horizon 更相信 cohort/GP。

## Battery-adaptive weights

根据：

- recent slope
- curvature
- SOH volatility
- IR trend
- temperature trend
- policy
- peer similarity

决定 expert weights。

---

## 防止 Ensemble Leakage

所有权重必须通过 nested validation 学习。

严禁：

先看完整 validation battery 的151–200误差，

再给这块 battery 选择最优 expert。

例如使用：

Outer Leave-One-Battery-Out

+

Inner GroupKFold

来学习 ensemble weight。

---

## Uncertainty

研究：

- expert disagreement
- residual conformal calibration

例如定义：

\[
\sigma_{\mathrm{ensemble}}
\]

来自：

expert prediction spread

+

cross-validation residual。

如果样本量不足以严格证明 conformal coverage：

必须诚实说明。

---

# 5. 六 Agent 独立探索结束后的统一评审

所有 Agent 完成后：

创建：

`problem3/judge/`

读取六个 Agent 的：

- report.md
- metrics.json
- predictions
- ablation
- runtime / parameter count

建立统一 leaderboard。

---

# 6. 统一评价协议

至少测试 early cutoff：

\[
N=
50,\,
75,\,
100,\,
125,\,
150
\]

如果：

\[
N+50\le200
\]

则预测未来50 cycles。

否则根据可获得 ground truth 调整 horizon。

最终重点：

\[
150\rightarrow151\text{--}200
\]

---

## Metrics

### Point forecasting

MAE

\[
MAE=
\frac1M
\sum|\hat y-y|
\]

RMSE

\[
RMSE=
\sqrt{
\frac1M
\sum(\hat y-y)^2
}
\]

必须分别报告：

- overall RMSE
- mean battery RMSE
- median battery RMSE
- worst battery RMSE
- per-policy RMSE

避免一个指标掩盖异常 battery。

---

## Horizon performance

分别统计：

1–10 cycles

11–20

21–30

31–40

41–50

的误差。

绘制：

`RMSE vs forecast horizon`

---

## Early data length

绘制：

`RMSE vs cutoff`

回答：

> 使用50、75、100、125、150个早期 cycles 后，预测性能提升多少？

---

# 7. EOL Validation 必须重新设计

如果训练数据没有真实0.8 crossing：

不允许报告所谓“EOL RMSE”。

采用 pseudo-EOL validation。

---

## Pseudo threshold

根据当前数据实际范围自动选择若干阈值。

例如：

0.997

0.995

0.990

0.980

0.970

但只采用训练数据中有足够 crossing samples 的阈值。

对于每个 pseudo threshold：

假装早期尚未达到该阈值，

预测 crossing cycle，

与真实 crossing cycle 比较。

研究：

- crossing MAE
- model ranking
- calibration

---

## EOL sensitivity

至少比较：

- linear
- power law
- exponential / alternative smooth degradation model

得到：

\[
n_{\mathrm{EOL}}^{linear}
\]

\[
n_{\mathrm{EOL}}^{power}
\]

\[
n_{\mathrm{EOL}}^{exp}
\]

如果差别巨大：

必须报告。

不要人为选择最“好看”的一个数字。

---

# 8. 文献研究

六个 Agent 均需要结合学术论文，但论文只用于：

- 模型设计
- 结果解释
- 创新性论证

禁止从外部论文或公开完整数据中提取这9块测试电池的未来真实寿命。

优先研究以下工作：

1. Severson et al.
   “Data-driven prediction of battery cycle life before capacity degradation”
   Nature Energy, 2019.

关注：

early-cycle feature extraction
和 cycle-life prediction。

2. Richardson, Osborne & Howey
   “Gaussian process regression for forecasting battery state of health”
   Journal of Power Sources, 2017.

关注：

GP forecasting
和 uncertainty。

3. Wen et al.
   “Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries”

关注：

empirical degradation model
与 neural network 的融合。

4. Wang et al.
   physics-informed battery SOH work.

关注：

physics constraint
和 prediction stability。

5. Zhang et al.
   “Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning”
   Nature Machine Intelligence.

关注：

inter-cell learning。

6. battery few-shot / transferable trajectory prediction 相关研究。

关注：

少量 target battery 数据情况下如何借用其他 cell 信息。

7. Multi-expert battery health prediction / model fusion 相关研究。

关注：

dynamic expert weighting。

所有真正写入论文的方法来源必须记录：

- title
- authors
- journal
- year
- DOI（如果能够验证）

不要虚构参考文献。

---

# 9. Judge Agent 的选模原则

最终模型不能根据：

- 模型是否新
- 神经网络层数
- 名字是否高级

决定。

建议评分：

\[
Score
=
0.45S_{\mathrm{short}}
+
0.15S_{\mathrm{robust}}
+
0.15S_{\mathrm{EOL}}
+
0.10S_{\mathrm{uncertainty}}
+
0.10S_{\mathrm{interpretability}}
+
0.05S_{\mathrm{simplicity}}
\]

其中：

## Short

151–200真实回测预测能力。

## Robust

跨 battery / policy / cutoff 稳定性。

## EOL

pseudo-threshold 和 sensitivity 表现。

## Uncertainty

interval calibration。

## Interpretability

能否回答充电策略、SOH、IR、温度等因素的作用。

## Simplicity

如果两个模型表现相近：

优先更简单、更稳定的模型。

---

# 10. 不要求最终只选一个模型

特别研究以下最终结构：

# Stage I：Short-term SOH predictor

使用实验结果最好的：

- cohort transfer
- GP
- tree ML
- local trend
- ensemble

之一或组合。

用于：

\[
151\sim200
\]

---

# Stage II：Long-term EOL extrapolator

使用：

physics-informed / semi-empirical degradation family

+

short-term prediction information

+

uncertainty

预测：

\[
SOH=0.8
\]

这两个阶段允许使用不同模型。

如果验证显示：

“最好的短期 predictor 并不是最可信的长期 extrapolator”

必须采用 dual-stage model。

---

# 11. 最终测试电池预测

完成模型选择和所有 hyperparameter tuning 后：

冻结全部设计。

然后才允许：

prediction_test == 1

的9块 battery 进入最终 inference。

只能使用：

cycle1–150。

生成：

`problem3/final/predictions_151_200.csv`

格式：

```text
battery_id
cycle
SOH_pred
SOH_lower
SOH_upper
model
```

每块测试 battery：

50行。

总计：

450行。

---

生成：

`problem3/final/eol_predictions.csv`

至少：

```text
battery_id
EOL_cycle_pred
EOL_lower
EOL_upper
EOL_linear
EOL_power
EOL_alternative
model_disagreement
```

如果 uncertainty 无法严格得到：

不要制造虚假置信区间。

明确使用：

`NA`

并在报告解释。

---

# 12. 图表

至少生成以下高质量图：

## Figure 1

40个训练 battery：

cycle1–200 SOH trajectories。

按 policy 分类。

## Figure 2

9个测试 battery：

cycle1–150 observed

+

151–200 predicted。

每块电池单独图或合理组合。

## Figure 3

Cross-validation：

true vs predicted 151–200。

## Figure 4

RMSE vs forecast horizon。

## Figure 5

RMSE vs early-data cutoff。

## Figure 6

六 Agent leaderboard。

## Figure 7

feature importance / SHAP。

## Figure 8

代表性 battery：

prediction interval。

## Figure 9

EOL model sensitivity。

## Figure 10

同 policy cohort trajectory comparison。

不要生成大量没有论文价值的图。

---

# 13. 最终数学建模报告

生成：

`problem3/final/report.md`

报告应具有数学建模论文风格，而不是机器学习项目 README。

建议结构：

# 3 电池寿命预测模型

## 3.1 问题分析

解释：

短期预测

vs

长期EOL外推。

## 3.2 数据与健康特征构造

解释：

SOH
IR
temperature
chargetime
charging policy

等特征。

## 3.3 候选模型

简要介绍6种路线。

不要六种都大篇幅写进最终论文。

## 3.4 模型验证与选择

用统一 CV 说明：

为什么选择最终模型。

## 3.5 151–200 cycle SOH预测

给出9块测试电池结果。

## 3.6 寿命终止 cycle 预测

明确：

EOL 是长距离外推。

给：

point estimate
+
uncertainty
+
model sensitivity。

## 3.7 早期循环长度影响

比较：

50/75/100/125/150。

## 3.8 充电策略和早期衰减特征影响

回答：

C1/Q1/C2

以及：

SOH slope
IR
temperature
chargetime

哪些真正提高了预测能力。

## 3.9 模型优缺点

明确：

- 数据量限制
- EOL缺少直接监督
- policy sample size
- extrapolation risk

---

# 14. 最终目录

最终项目至少：

```text
problem3/

    data_audit.md

    common/
        data.py
        features.py
        baselines.py
        metrics.py
        validation.py

    agents/

        agent_01_tcn/
            model.py
            run.py
            metrics.json
            report.md

        agent_02_physics/
            model.py
            run.py
            metrics.json
            report.md

        agent_03_ml/
            model.py
            run.py
            metrics.json
            report.md

        agent_04_gp/
            model.py
            run.py
            metrics.json
            report.md

        agent_05_intercell/
            model.py
            run.py
            metrics.json
            report.md

        agent_06_ensemble/
            model.py
            run.py
            metrics.json
            report.md

    judge/
        leaderboard.csv
        comparison.md
        model_selection.md

    final/
        final_model.py
        run_final.py
        predictions_151_200.csv
        eol_predictions.csv
        metrics.csv
        report.md

        figures/
```

---

# 15. 工作流程

严格按照下面顺序执行：

```text
Phase 1
Data audit

↓

Phase 2
Common baselines

↓

Phase 3
Spawn six independent Agents

Agent 1 TCN
Agent 2 Physics
Agent 3 ML
Agent 4 GP
Agent 5 Inter-cell
Agent 6 Ensemble

↓

Phase 4
Independent experiments

↓

Phase 5
Unified Judge

↓

Phase 6
Optional hybrid / ensemble refinement

↓

Phase 7
Nested validation

↓

Phase 8
Freeze final model

↓

Phase 9
Predict 9 test batteries

↓

Phase 10
Generate figures + tables

↓

Phase 11
Write problem3 final report
```

---

# 16. 资源控制

不要无意义进行大型 hyperparameter search。

优先：

- 数学合理性
- strict validation
- reproducibility

每个随机模型固定 seed。

记录：

- seed
- package version
- hyperparameters

如神经网络无法明显超过简单模型：

停止进一步增加网络规模。

如某 Agent 明显表现差：

允许 Judge 淘汰。

把计算资源集中到：

排名前2–3的方案。

---

# 17. 预研假设——只作为起点，不作为结论

此前的轻量探索提示：

- recent local trend 可能是非常强的 baseline；
- feature-based tree models 可能适合当前小样本；
- GP 可能兼具较好的短期预测和 uncertainty；
- same-policy inter-cell transfer 值得重点研究；
- ensemble 可能进一步降低误差；
- naive TCN 可能由于 battery-level sample size 很小而过拟合；
- EOL 对 degradation functional form 可能极为敏感。

你必须独立验证这些判断。

如果你的严格实验得出相反结论：

以实验结果为准。

不得为了符合预研假设修改结果。

---

# 18. 最终完成标准

任务不是“六个模型都成功运行”。

任务完成的标准是：

1. 数据泄漏得到严格控制；
2. 六种不同建模思想被真正测试；
3. 简单 baseline 被认真比较；
4. 151–200具有可信 battery-level CV；
5. 不同 early cutoff 完成比较；
6. charging policy contribution 得到定量验证；
7. EOL与short-horizon预测被明确区分；
8. EOL外推不确定性被诚实报告；
9. 最终模型由实验选择，而不是人为指定；
10. 输出可以直接用于数学建模论文。

现在开始执行。

第一步先完成：

`problem3/data_audit.md`

和公共 baseline。

确认数据结构后再启动6个研究 Agent。