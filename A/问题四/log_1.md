# 问题四运行日志（log_1）

- 运行时间：2026-08-16
- 脚本：`src/A/task4_optimize.py`
- Python：Anaconda（`PYTHONNOUSERSITE=1`）；numpy / pandas / matplotlib；暴露量定义来自 `task2_exposure_model.py`（scipy）。
- 数据：`data/battery_summary.csv`、`data/cycle_train.csv`（主队列与问题二相同：dataset 3、非测试、满 200 圈，34 块 / 6 策略）；充电时间来自 `A/问题一/output/A/battery_SOH_charge.csv`
- 输出：`A/问题四/output/`；图同步到 `output/A/question4/figures/`
- 未改问题二、问题三的模型文件与 output。
- 全题口径：`A/口径.md`

## 关键数值

- 时间：\(\widehat T=T_{\mathrm{CC}}+0.045\) min，长寿四策略 RMSE=0.009 min（问题一口径）。
- 网格 11718 点，可行（\(C_1\ge C_2\)，\(C^{\mathrm{high}}\le 4.8\)，\(T\in[9.90,10.25]\)）1249 点。
- 正式推荐：`5_3C_54PER_4C`（实测 Fade=0.00256，\(C^{\mathrm{high}}=4.17\)，T≈10.04 min）。
- 邻域候选：5.3C(50%)-4.0C（\(C^{\mathrm{high}}=4.0\)，\(\widehat T\approx 10.21\) min，未实验）。不把该点上的负 PySR Fade 写成预期寿命。
- OLS 角点 5.6C(80%)-5.9C：OLS Fade=−0.013，两窗口 Fade=0.075。

## 输出文件

- `output/report.md`、`observed_policies.csv`、`recommendation.csv`、`contrast_optima.csv`、`pareto_front.csv`、`run_summary.json`
- `output/figures/discrete_time_fade.png`、`pareto_grid.png`、`chigh_constraint.png`、`wrong_model_optima.png`
- `模型.md`、`结论.md`、`比较.md`

## 原始提示词

```text
tongjistone的问题二的模型来做问题四如何
```
