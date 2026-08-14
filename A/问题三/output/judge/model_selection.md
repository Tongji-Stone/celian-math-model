# 最终模型选择与冻结说明

## Stage I：151–200 周期 SOH

最终采用精简的两专家凸组合：

\[
\widehat{SOH}_{N+h}=w\widehat{SOH}^{power}_{N+h}+(1-w)\widehat{SOH}^{linear50}_{N+h},
\quad w=0.668396.
\]

Power 使用最后最多 101 个已观测循环的原始 SOH，经 cutoff 内稳健异常抑制后拟合 `a-k n^p`；核心 N=150 时窗口为 cycle 50–150。Linear-50 仅使用 cycle 101–150。该组合以 3–5 个有效参数完成预测，避免复杂 ensemble 的 6 experts、树模型和 660 个门控系数；其 worst-battery RMSE 也优于复杂 hybrid。

短期区间使用 40 块训练电池严格元层 LOO 残差，在每个 10-cycle horizon bin 上取 battery 内最大绝对误差，再做有限样本 90% 分位。它是分组经验预测区间，不是独立同分布条件下的严格 coverage 保证。

## Stage II：0.8 EOL

数据没有任何 SOH≤0.8 crossing。EOL 不由 Stage I 递推得到，而是分别拟合 linear、power、exponential，并比较 cycle 1/20/50/80–150 四个窗口。点估计取跨函数/窗口中位数；上下界取函数-窗口移动块残差 bootstrap 的包络。该包络表示模型假设敏感性，不称为经监督校准的置信区间，也不报告 EOL RMSE。

## 冻结与测试隔离

模型结构、窗口、权重学习规则、异常处理、区间规则和 EOL 函数族均在读取测试电池预测结果前冻结。最终测试推理只读取每块测试电池 cycle 1–150，不读取 `global_id` 对应外部数据，也不使用汇总表动态均值。
