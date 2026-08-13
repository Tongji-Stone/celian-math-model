# 问题一第三次迭代运行日志（log_3）

- 运行时间：2026-08-14T02:37:22+08:00
- IQR 系数：1.5
- 上四分位线 Q3（75%分位数）：0.996103
- 下四分位线 Q1（25%分位数）：0.993433
- 绘图字体：Microsoft YaHei

## 输入质量检查

- battery_summary.csv：49 行，13 列，49 块电池。
- cycle_train.csv：9350 行，9 列。
- battery_id + cycle 重复记录：0。
- 汇总表缺失单元格：3（其中 C1 的结构性缺失保留，不参与循环数据清洗）。
- 循环表缺失单元格：0。
- 最大循环数分布：{150: 9, 200: 40}。

## 策略统一命名

- 原始策略数：9；统一后策略数：7；恰好减少 2 种。
- `80PER_3_6C` 统一为 `3_6C-80PER_3_6C`。
- `4_8C_80PER_4_8C_NEWSTRUCTURE` 统一为 `4_8C_80PER_4_8C`。
- 其余策略仅移除 `_NEWSTRUCTURE` 后缀。

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
- 按两个分位数分类：长寿命 10 块，中等寿命 20 块，短寿命 10 块。
- 寿命最长策略（按策略 SOH 中位数）：4_8C_80PER_4_8C。
- 寿命最短策略（按策略 SOH 中位数）：3_7C_31PER_5_9C。

## SOH 曲线

- 使用清洗后的 `SOH_smooth` 绘制49张单体曲线和1张全部电池曲线。
- 单体曲线命名为 `SOH_N.png`；总图命名为 `SOH_ALL.png`，颜色按电池编号由蓝到红渐变。

## 验收确认

- [x] 策略数量恰好减少2种：原始9种，统一后7种，减少2种。
- [x] 统一策略集合正确：实际策略：['3_6C-80PER_3_6C', '3_7C_31PER_5_9C', '4_8C_80PER_4_8C', '5C_67PER_4C', '5_3C_54PER_4C', '5_6C_19PER_4_6C', '5_6C_36PER_4_3C']。
- [x] 旧策略名称已消除：禁止名称残留：[]。
- [x] NEWSTRUCTURE后缀已全部移除：统一后的策略名不含NEWSTRUCTURE。
- [x] 全部处理表命名一致：清洗循环表、charge_policy和battery_SOH_charge均采用统一名称。
- [x] charge_policy覆盖49块电池：记录数=49，唯一电池数=49。
- [x] 核心CSV均写入标准文件名：cycle_train_cleaned.csv->cycle_train_cleaned.csv, charge_policy.csv->charge_policy.csv, battery_SOH_charge.csv->battery_SOH_charge.csv, policy.csv->policy.csv。
- [x] 策略报告覆盖全部统一策略：报告策略数=7。
- [x] 49张单体曲线与SOH_ALL齐全：单体曲线=49，SOH_ALL=1。

## 输出文件

- `output/A/cycle_train_cleaned.csv`
- `output/A/charge_policy.csv`
- `output/A/battery_SOH_charge.csv`
- `output/A/SOH_1_example.png`
- `output/A/SOH_summary.png`
- `output/A/SOH_chargetime.png`
- `output/A/charge_policy.png`
- `output/A/policy.csv`
- `output/doc/1.md`
- `output/A/acceptance_3.md`
- `A/问题一/SOH`

## 说明

- 箱型图纵轴采用“第 200 次循环 SOH”作为使用寿命代理指标；本题数据尚未覆盖 SOH<80% 的真实寿命终止循环。
- IQR 是统计异常筛查，不等同于确认传感器故障；清洗结果用于完成本题指定分析。
- 图中的两条参考线统一命名为上四分位线（Q3）和下四分位线（Q1）。
- 策略报告使用统一后的7种策略，最终SOH统计只纳入完整200循环的40块电池。

## 原始提示词

以下完整保留本次执行时 `A/问题一/AGENTS.md` 的内容：

```text
### 数据集介绍

在data文件夹中有 MIT–Stanford 公开锂离子电池循环老化数据集,其包含49个A123 18650型磷酸铁锂电池的循环老化实验数据，额定容量为1.1 Ah。数据覆盖电池从初始健康状态到容量衰减至约80% SOH的老化过程，可用于研究不同快充策略对电池寿命衰减的影响其中：

1. battery_summary.csv记录各电池的基本信息，包括充电策略参数（C1、Q1、C2）、初始容量、平均充电时间、平均内阻及平均温度等。其中，prediction_test 字段用于标记问题3的测试电池，取值为1表示该电池为问题3指定的测试对象，取值为0表示非测试电池。
2. cycle_train.csv记录49块电池的早期循环实验数据，包括循环次数、容量（Capacity）、电池健康状态（State of Health，SOH）、平滑后的SOH（SOH_smooth）、充电时间、内阻、平均温度及对应充电策略等信息。对于非问题3测试电池，数据记录截至第200次循环；对于问题3指定的9块测试电池，数据仅提供截至第150次循环的实验数据。



### 概念明细

电池寿命：通常用电池健康状态SOH表示：\[SOH = \frac{Q_{current}}{Q_{initial}}\]，当\[SOH < 80\%\]时认为电池达到寿命终止条件

影响电池寿命的两个指标：

1. 充电倍率
2. SOC（state of charge）：荷电状态电池当前剩余电量占额定容量的比例，如、\[0\%\]表示电池处于放空状态，\[100\%\]表示电池处于满电状态。

不同SOC区间内电池对充电电流的敏感程度不同。在快充过程中，充电倍率和充电SOC区间会共同影响电池老化。



### 任务目标

1. 进行数据清理，使用iqr方法寻找有无异常数据，然后使用前后均值填补异常数据

2. 提取不同电池的快充策略，输出为charge_policy.csv，格式为

   | battery_id | policy                       |
   | ---------- | ---------------------------- |
   | 1          | 3_6C-80PER_3_6C              |
   | ......     | ......                       |
   | 49         | 4_8C_80PER_4_8C_NEWSTRUCTURE |

3. 创建电池SOH曲线的绘制脚本SOH_plot.py，然后以1号电池为例绘制SOH曲线并且保存为SOH_1_example

4. 统计使用寿命，数据采用数据集中cycle截止为200的数据，150的数据不使用，先提取相应battery_id的第200个cycle的SOH和其充电时间的平均值，输出到battery_SOH_charge.csv中，然后根据所得数据绘制使用寿命柱状图，横坐标为第200个cycle的SOH，纵坐标为位于相应SOH的电池数量，然后以critical_SOH绘制一条分界线（红色虚线）用以区分长寿命电池和短寿命电池，输出为SOH_summary，设置critical_SOH为可变变量，默认为1.0

5. 同样使用batter_SOH_charge.csv，绘制散点图，横轴为SOH，纵轴为charge_time_mean，输出为SOH_chargetime

6. 分析不同充电策略下的寿命分布，绘制箱型图，横坐标为充电策略，纵坐标为使用寿命，并且在\[y = citical\_SOH\]处绘制一条红线，输出为charge_policy



### 第二次迭代

1. 调整citical\_SOH设置策略，将critical_SOH_high设置为SOH的75%分位数，critical_SOH_low设置为SOH的25%分位数，以这两个值为分界线分别画一条红色的和蓝色的虚线。
2. 将前一次的输出整理到.\output\A\1 中
3. 为SOH_chargetime散点图添加color_map，美化图表



### 第三次迭代

1. 重新组织一下数据，添加到数据清理的步骤中
   1. 将3_6C-80PER_3_6C，80PER_3_6C视作一种策略，统一命名为3_6C-80PER_3_6C
   2. 将4_8C_80PER_4_8C，4_8C_80PER_4_8C_NEWSTRUCTURE视作同一种策略，统一命名为4_8C_80PER_4_8C
   3. 所有策略后缀去掉newstructure，其余名称保持不变
2. 根据得到的SOH_plot.py模版，完成所有电池的SOH曲线绘制，命名为SOH_N，此外再将所有SOH曲线绘制在一张图上（不同曲线的颜色从蓝到红渐变），命名为SOH_ALL，输出到.\A\问题一\SOH，SOH数据采用smooth_SOH数据，

3. 将原来图中的长寿命分界线和短寿命分界线改成上四分位线和下四分位线

4. 增加验收确认

   1. 本次数据清理要求合并两个策略，检验是否少了两个策略
   2. 检验命名是否统一

5. 报告输出.\output\doc\，文件名为1.md，内容包括：

   1. 各个策略的最终SOH平均值和最大最小值，以及其他参数，包含的电池记号，充电时间的平均值和最大值最小值

   

### 注意事项

1. 绘图时注意中文兼容，图例在右上角显示

2. 代码放到.\src\A中,输出文件放到.\output\A中
3. 此前已有任务日志log_1-2 在.\A\问题一\ 中输出这次的日志 log_3，日志中要保留原始提示词
```
