# 问题一运行日志（log_1）

- 运行时间：2026-08-13T21:13:17+08:00
- IQR 系数：1.5
- critical_SOH：1.000000
- 绘图字体：Microsoft YaHei

## 输入质量检查

- battery_summary.csv：49 行，13 列，49 块电池。
- cycle_train.csv：9350 行，9 列。
- battery_id + cycle 重复记录：0。
- 汇总表缺失单元格：3（其中 C1 的结构性缺失保留，不参与循环数据清洗）。
- 循环表缺失单元格：0。
- 最大循环数分布：{150: 9, 200: 40}。

## IQR 清洗

- 方法：按 battery_id 分组，对 capacity、SOH、SOH_smooth、chargetime、IR、Tavg 分别计算 Q1、Q3 和 IQR。
- 判定：小于 Q1 - 1.5×IQR 或大于 Q3 + 1.5×IQR 的值标记为异常。
- 填补：使用同一电池、同一字段中距离最近的前后正常值均值；边界点仅有一侧正常值时使用该侧值。
- 共替换 559 个单元格；分字段数量：{'capacity': 36, 'SOH': 36, 'SOH_smooth': 17, 'IR': 142, 'Tavg': 142, 'chargetime': 186}。
- 原始 data 文件未改写；清洗结果另存为 output/A/cycle_train_cleaned.csv。

## 200 次循环样本

- 纳入：最大 cycle 恰为 200 的 40 块电池。
- 排除：仅记录至 150 次循环的 9 块测试电池。
- SOH 范围：0.949733–1.000235，均值 0.991423。
- 以 critical_SOH=1.000 分类：长寿命 1 块，短寿命 39 块。

## 输出文件

- `output/A/cycle_train_cleaned.csv`
- `output/A/charge_policy.csv`
- `output/A/battery_SOH_charge.csv`
- `output/A/SOH_1_example.png`
- `output/A/SOH_summary.png`
- `output/A/SOH_chargetime.png`
- `output/A/charge_policy.png`

## 说明

- 箱型图纵轴采用“第 200 次循环 SOH”作为使用寿命代理指标；本题数据尚未覆盖 SOH<80% 的真实寿命终止循环。
- IQR 是统计异常筛查，不等同于确认传感器故障；清洗结果用于完成本题指定分析。
