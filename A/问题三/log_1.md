# 问题三运行日志（log_1）

- 运行时间：2026-08-15
- 脚本：`src/A/task3_degradation_model.py`
- Python：Anaconda（`PYTHONNOUSERSITE=1`）；numpy / pandas / scipy / matplotlib。导入时有 NumPy 1.x/2.x 与 pyarrow 的警告，脚本仍正常结束。
- 循环数据：`A/问题一/output/A/cycle_train_cleaned.csv`
- 输出：`A/问题三/output/`
- 未修改 `paper/`，未改动任何他人机器学习文件（仓库中尚无问题三 ML 脚本）。

## 样本

- 验证：40 块 `prediction_test=0`、满 200 圈。
- 测试：9 块 `prediction_test=1`。
- 主模型（k=150，RMSE 最优）：`linear_recent`。

## 关键数值（k=150）

- linear_recent：RMSE 均值 0.000271，SOH_200 MAE 0.000466，\(\hat N\) 中位 6436（验证集）。
- policy_mean：RMSE 0.000867。
- persist：RMSE 0.001530。
- 测试电池 16（短寿）：预报 SOH_200=0.962，\(\hat N\approx 1030\)。
- 测试电池 9（4.8C）：近段明显快于组均值，\(\hat N\approx 1679\)（组均值约 10100）。

## 输出文件

- `output/report.md`、`validation_summary.csv`、`validation_cell_scores.csv`、`test_predictions.csv`
- `output/figures/val_rmse_models.png`、`val_rmse_by_window.png`、`val_soh200_scatter.png`、`val_example_trajectories.png`、`test_forecast_151_200.png`
- `模型.md`、`结论.md`、`比较.md`

## 原始提示词

```text
那就按照你原本说的思路做一下问题三，不要改变其他人原本的机器学习的文件，自己重新建立一个文件来建模，然后验证比较一下哪个思路更加可靠，最后输出对两种模型的比较说明。如果机器学习的思路尚且没有完成，那就只完成你自己的模型就可以了。
```
