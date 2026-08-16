# 问题三迭代一运行日志

## 原始提示词

> 根据.\A\问题三\AGENTS.md完成任务

本次执行依据 `A/问题三/AGENTS.md`，其任务原文如下：

> **迭代一**，目的：初步探究相关关系，便于确认后续模型。
>
> 1. 根据battery_summary.csv确认prediction_test为1的电池id，然后将cycle_train.csv中对应电池的循环数据提取出来，命名为data_train.csv，这是用来训练的数据集，其余的数据提取出来作为data_test.csv，这是测试集数据。
> 2. 使用Seaborn的pairplot,regplot和heatmap绘制不同策略下SOH与相关特征的关系图和总图。相关特征设置为：cycle，chargetime，IR，Tavg
>
> 验收检查：训练集数据是否cycle数都是150，测试集数据是否cycle数都是200。
>
> 注意事项：绘图时注意中文兼容，图例在右上角显示；代码放到`.\src\问题三`中，输出文件放到`.\output\`中；在`.\A\问题三\`中输出这次的日志 `log_1`，日志中要保留原始提示词。

## 数据拆分

- 源文件：`data/battery_summary.csv`、`data/cycle_train.csv`。
- `prediction_test=1` 的电池编号：`2, 5, 9, 10, 11, 14, 16, 24, 25`。
- `output/data_train.csv`：9 块电池，1,350 行；每块电池均完整覆盖第 1--150 圈。
- `output/data_test.csv`：40 块电池，8,000 行；每块电池均完整覆盖第 1--200 圈。
- 两个输出集合互斥，合计 9,350 行，与原始循环数据行数一致；CSV 保留原始列、原始值和按电池/周期排序后的记录。

说明：以上命名严格遵照任务原文，即把 `prediction_test=1` 的 150 圈数据命名为 `data_train.csv`，其余 200 圈数据命名为 `data_test.csv`。

## 绘图

- 分析变量：`SOH`、`cycle`、`chargetime`、`IR`、`Tavg`。
- 使用 Seaborn 生成 `pairplot`、`regplot` 和 Pearson 相关系数 `heatmap`。
- 策略绘图名称统一去除 `_NEWSTRUCTURE`；`80PER_3_6C` 合并为同义策略 `3_6C-80PER_3_6C`，最终得到 7 种规范化策略。此规范化仅用于绘图，两个拆分 CSV 不改写策略字段。
- 总图：`output/pairplot_overall.png`、`output/regplot_overall.png`、`output/heatmap_overall.png`。
- 分策略图：`output/问题三_分策略/` 下共 21 张，即 7 种策略各 3 张。
- 中文字体：Microsoft YaHei；图例放置于右上角。

## 验收与视觉检查

- 训练集 9 块电池逐块核验：周期序列均严格等于整数 `1..150`。
- 测试集 40 块电池逐块核验：周期序列均严格等于整数 `1..200`。
- 原始分析字段无缺失值，`(battery_id, cycle)` 无重复，两个源文件的电池编号集合一致。
- 已打开检查总 `pairplot`、总 `regplot`、总 `heatmap` 及代表性分策略图；中文、标题、坐标轴、图例和相关系数标注均正常，无裁切或乱码。
- 结构化验收结果：`output/question3_validation.csv`。

## 探索性分析限制

- 原始数据存在 1 条 `SOH>1.1` 的极端观测（battery 1, cycle 12, SOH=1.437443）及 2 条 `IR=0` 观测。为避免擅自修改赛题数据，本迭代全部保留；因此总图中的回归线和 Pearson 相关系数可能受极端值影响。
- 每块电池贡献多个循环观测，同一电池内记录并非相互独立；本迭代图形仅用于探索关联，不能据此作因果解释或显著性结论。
- 不同策略的电池数与周期窗口并不完全相同，跨策略总体相关性可能混入策略、批次和重复测量结构的影响；后续建模宜采用以电池为分组单位的纵向/混合效应方法并进行稳健性分析。

## 可复现运行

```powershell
python .\src\问题三\iteration1_analysis.py
```

