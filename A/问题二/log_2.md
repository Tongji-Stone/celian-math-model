# 问题二暴露模型运行日志（log_2）

- 运行时间：2026-08-14
- 脚本：`src/A/task2_exposure_model.py`
- Python：Anaconda 3.11（`PYTHONNOUSERSITE=1`）；回归用 numpy 实现，不调用 statsmodels
- 主门槛：Q* = 50%
- 主分析样本：34 块，6 个策略
- 输出目录：`A/问题二/output/`

## 数据口径

与 `log_1.md` 相同：`dataset_id=3`、`prediction_test=0`、最大循环 200。响应为 `fade_200 = SOH_1 - SOH_200`（平滑 SOH），反向代理寿命，不是 \(N_{\mathrm{EOL}}\)。

## 本次完成的模型

1. 固定 SOC 窗的低/高暴露 \(E^{L},E^{H}\) 与高 SOC 等效倍率 \(C^{\mathrm{high}}\)。
2. 主回归 \(Y\sim E^{L}+E^{H}\)；嵌套 \(Y\sim E^{H}\)；简化 \(Y\sim C^{\mathrm{high}}\)；对照 \(Y\sim C_1+Q_1+C_2\)。
3. 电池级 HC3 / 策略聚类标准误，以及策略加权 WLS（主推断）。
4. 稳健性：Q* = 40/50/60；去掉 `3_7C`；去掉 `SOH_1<0.98` 的电池 41；留一策略；幂次网格 p,q ∈ {1, 1.5, 2}。

## 关键数值（策略 WLS）

- 主模型：β_L = 0.001017，β_H = 0.001356（p ≈ 2.3e-5），调整 R² = 0.996。
- 简化模型：γ_1 = 0.01516（p ≈ 0.004），调整 R² = 0.775。
- 去掉 `3_7C` 后：β_H p ≈ 0.059，调整 R² = 0.153。高 SOC 分离主要靠该组 2 块电池。
- Kruskal–Wallis（Fade_200）：全部 p = 0.011；去掉短寿组后 p = 0.033。

## 输出文件

- `output/report.md`：脚本生成的数值报告
- `output/cell_exposure.csv` / `policy_exposure.csv`
- `output/coefficients.csv` / `fit_stats.csv` / `robustness.csv` / `loo_policy.csv` / `power_grid.csv`
- `output/figures/fade_vs_E_high.png`、`fade_vs_C_high.png`、`main_residuals.png`
- `模型.md`、`结论.md`

未修改 `paper/` 下任何文件。

## 原始提示词

```text
按照你刚刚讲述的模型，完成问题2的全部建模。然后再写入仓库。注意，不要更改paper文件夹中的文件，用markdown文件来说明模型和原理，以及对各种参数的考量。并且请先在本地编译，最后将成果一次性上传仓库，不要边写边传，以免冲突。
```
