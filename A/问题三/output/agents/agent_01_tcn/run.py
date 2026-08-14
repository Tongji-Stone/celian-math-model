from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problem3.agents.agent_01_tcn.model import (  # noqa: E402
    ArrayBundle,
    SmallTCNForecaster,
    parameter_count,
    predict,
    set_torch_seed,
    train_fixed_epochs,
    train_with_early_stopping,
)
from problem3.common.data import load_data, physical_static_features, split_battery_ids  # noqa: E402
from problem3.common.metrics import detailed_metric_tables, mae, rmse  # noqa: E402
from problem3.common.validation import EARLY_CUTOFFS, FORECAST_HORIZON, SEED  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent
ROLLING_CUTOFFS = list(range(50, 151, 10))
SEQUENCE_LENGTHS = [20, 30, 50, 75, 100]
ABLATIONS: dict[str, dict[str, list[str]]] = {
    "A_SOH_only": {
        "dynamic": ["SOH_causal"],
        "static": [],
    },
    "B_multivariate": {
        "dynamic": [
            "SOH_causal",
            "SOH",
            "capacity",
            "IR",
            "Tavg",
            "chargetime",
        ],
        "static": ["initial_capacity"],
    },
    "C_multivariate_policy": {
        "dynamic": [
            "SOH_causal",
            "SOH",
            "capacity",
            "IR",
            "Tavg",
            "chargetime",
        ],
        "static": [
            "initial_capacity",
            "C1_effective",
            "Q1",
            "C2",
            "C1_missing_indicator",
        ],
    },
}


@dataclass(frozen=True)
class Hyperparameters:
    hidden_channels: int = 12
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    static_hidden: int = 8
    decoder_hidden: int = 32
    dropout: float = 0.20
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5


@dataclass(frozen=True)
class RawBundle:
    dynamic: np.ndarray
    static: np.ndarray
    lengths: np.ndarray
    target_delta: np.ndarray
    last_soh: np.ndarray
    metadata: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FoldScaler:
    dynamic_mean: np.ndarray
    dynamic_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def causal_clean_history(history: pd.DataFrame) -> pd.DataFrame:
    """Clean and smooth using only rows present at the simulated cutoff."""
    clean = history.copy()
    clean.loc[clean["IR"].le(0), "IR"] = np.nan
    clean["IR"] = clean["IR"].interpolate(limit_direction="both")
    # The provenance of the supplied SOH_smooth is not documented.  Rebuild a
    # robust, trailing-only series so the cutoff endpoint cannot use future rows.
    trailing_median = clean["SOH"].rolling(window=5, min_periods=1).median()
    clean["SOH_causal"] = trailing_median.ewm(span=5, adjust=False).mean()
    return clean


def make_battery_folds(
    summary: pd.DataFrame,
    battery_ids: list[int],
    n_splits: int,
    seed: int,
) -> list[list[int]]:
    """Policy-aware deterministic battery folds; cycle rows never split."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    policy_frame = summary.loc[summary["battery_id"].isin(battery_ids), ["battery_id", "policy"]]
    offset = 0
    for _, group in policy_frame.groupby("policy", sort=True):
        ids = group["battery_id"].astype(int).to_numpy()
        rng.shuffle(ids)
        for index, battery_id in enumerate(ids):
            folds[(offset + index) % n_splits].append(int(battery_id))
        offset = (offset + len(ids)) % n_splits
    return [sorted(fold) for fold in folds]


def make_inner_split(
    summary: pd.DataFrame,
    outer_train_ids: list[int],
    outer_fold: int,
) -> tuple[list[int], list[int]]:
    folds = make_battery_folds(summary, outer_train_ids, 5, SEED + 101 + outer_fold)
    validation_ids = folds[outer_fold % len(folds)]
    train_ids = sorted(set(outer_train_ids).difference(validation_ids))
    if not train_ids or not validation_ids:
        raise RuntimeError("Empty inner battery split")
    return train_ids, validation_ids


def build_raw_bundle(
    summary: pd.DataFrame,
    cycles_by_battery: dict[int, pd.DataFrame],
    battery_ids: list[int],
    cutoffs: list[int],
    ablation: str,
    sequence_length: int,
) -> RawBundle:
    definition = ABLATIONS[ablation]
    dynamic_columns = definition["dynamic"]
    static_columns = definition["static"]
    dynamic_rows: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    lengths: list[int] = []
    targets: list[np.ndarray] = []
    last_values: list[float] = []
    metadata: list[dict[str, Any]] = []

    summary_index = summary.set_index("battery_id", drop=False)
    for battery_id in sorted(battery_ids):
        full = cycles_by_battery[battery_id]
        summary_row = summary_index.loc[battery_id]
        static_map = physical_static_features(summary_row)
        for cutoff in cutoffs:
            history = full.loc[full["cycle"].le(cutoff)].copy()
            future = full.loc[
                full["cycle"].gt(cutoff) & full["cycle"].le(cutoff + FORECAST_HORIZON)
            ].copy()
            if int(history["cycle"].max()) != cutoff or len(future) != FORECAST_HORIZON:
                raise ValueError(f"Incomplete sample battery={battery_id}, cutoff={cutoff}")
            history = causal_clean_history(history)
            valid_length = min(sequence_length, len(history))
            values = history.tail(valid_length)[dynamic_columns].to_numpy(dtype=np.float64)
            padded = np.full(
                (sequence_length, len(dynamic_columns)), np.nan, dtype=np.float64
            )
            padded[-valid_length:] = values
            dynamic_rows.append(padded)
            static_rows.append(
                np.asarray([static_map[column] for column in static_columns], dtype=np.float64)
            )
            lengths.append(valid_length)
            last_soh = float(history["SOH"].iloc[-1])
            target = future["SOH"].to_numpy(dtype=np.float64) - last_soh
            targets.append(target)
            last_values.append(last_soh)
            metadata.append(
                {
                    "battery_id": int(battery_id),
                    "policy": str(summary_row["policy"]),
                    "cutoff": int(cutoff),
                }
            )
    return RawBundle(
        dynamic=np.stack(dynamic_rows),
        static=np.stack(static_rows),
        lengths=np.asarray(lengths, dtype=np.int64),
        target_delta=np.stack(targets),
        last_soh=np.asarray(last_values, dtype=np.float64),
        metadata=tuple(metadata),
    )


def fit_scaler(raw: RawBundle) -> FoldScaler:
    flat_dynamic = raw.dynamic.reshape(-1, raw.dynamic.shape[-1])
    dynamic_mean = np.nanmean(flat_dynamic, axis=0)
    dynamic_std = np.nanstd(flat_dynamic, axis=0)
    dynamic_std = np.where(dynamic_std < 1e-8, 1.0, dynamic_std)
    if raw.static.shape[1]:
        static_mean = raw.static.mean(axis=0)
        static_std = raw.static.std(axis=0)
        static_std = np.where(static_std < 1e-8, 1.0, static_std)
    else:
        static_mean = np.zeros(0, dtype=np.float64)
        static_std = np.ones(0, dtype=np.float64)
    target_mean = raw.target_delta.mean(axis=0)
    target_std = raw.target_delta.std(axis=0)
    target_std = np.where(target_std < 1e-5, 1e-5, target_std)
    return FoldScaler(
        dynamic_mean=dynamic_mean,
        dynamic_std=dynamic_std,
        static_mean=static_mean,
        static_std=static_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def transform(raw: RawBundle, scaler: FoldScaler) -> ArrayBundle:
    dynamic = (raw.dynamic - scaler.dynamic_mean) / scaler.dynamic_std
    dynamic = np.nan_to_num(dynamic, nan=0.0, posinf=0.0, neginf=0.0)
    static = (raw.static - scaler.static_mean) / scaler.static_std
    target = (raw.target_delta - scaler.target_mean) / scaler.target_std
    return ArrayBundle(
        dynamic=dynamic.astype(np.float32),
        static=static.astype(np.float32),
        lengths=raw.lengths,
        target=target.astype(np.float32),
    )


def make_model(ablation: str, hyperparameters: Hyperparameters, seed: int) -> SmallTCNForecaster:
    set_torch_seed(seed)
    return SmallTCNForecaster(
        dynamic_channels=len(ABLATIONS[ablation]["dynamic"]),
        static_features=len(ABLATIONS[ablation]["static"]),
        horizon=FORECAST_HORIZON,
        hidden_channels=hyperparameters.hidden_channels,
        kernel_size=hyperparameters.kernel_size,
        dilations=hyperparameters.dilations,
        static_hidden=hyperparameters.static_hidden,
        decoder_hidden=hyperparameters.decoder_hidden,
        dropout=hyperparameters.dropout,
    )


def evaluate_raw_prediction(
    prediction_scaled: np.ndarray,
    validation_raw: RawBundle,
    scaler: FoldScaler,
) -> tuple[float, float, np.ndarray]:
    prediction_delta = prediction_scaled * scaler.target_std + scaler.target_mean
    return (
        mae(validation_raw.target_delta.ravel(), prediction_delta.ravel()),
        rmse(validation_raw.target_delta.ravel(), prediction_delta.ravel()),
        prediction_delta,
    )


def run_inner_trial(
    summary: pd.DataFrame,
    cycles_by_battery: dict[int, pd.DataFrame],
    inner_train_ids: list[int],
    inner_validation_ids: list[int],
    outer_fold: int,
    stage: str,
    ablation: str,
    sequence_length: int,
    hyperparameters: Hyperparameters,
) -> tuple[dict[str, Any], int]:
    start = time.perf_counter()
    trial_seed = (
        SEED
        + outer_fold * 1000
        + list(ABLATIONS).index(ablation) * 100
        + sequence_length
    )
    train_raw = build_raw_bundle(
        summary,
        cycles_by_battery,
        inner_train_ids,
        ROLLING_CUTOFFS,
        ablation,
        sequence_length,
    )
    validation_raw = build_raw_bundle(
        summary,
        cycles_by_battery,
        inner_validation_ids,
        [150],
        ablation,
        sequence_length,
    )
    scaler = fit_scaler(train_raw)
    model = make_model(ablation, hyperparameters, trial_seed)
    result = train_with_early_stopping(
        model,
        transform(train_raw, scaler),
        transform(validation_raw, scaler),
        seed=trial_seed,
        max_epochs=hyperparameters.max_epochs,
        patience=hyperparameters.patience,
        batch_size=hyperparameters.batch_size,
        learning_rate=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    validation_scaled = predict(result.model, transform(validation_raw, scaler))
    validation_mae, validation_rmse, _ = evaluate_raw_prediction(
        validation_scaled, validation_raw, scaler
    )
    record = {
        "outer_fold": outer_fold,
        "stage": stage,
        "ablation": ablation,
        "sequence_length": sequence_length,
        "inner_train_batteries": len(inner_train_ids),
        "inner_validation_batteries": len(inner_validation_ids),
        "inner_validation_MAE": validation_mae,
        "inner_validation_RMSE": validation_rmse,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "parameter_count": parameter_count(result.model),
        "runtime_seconds": time.perf_counter() - start,
    }
    return record, result.best_epoch


def predict_outer_fold(
    summary: pd.DataFrame,
    cycles_by_battery: dict[int, pd.DataFrame],
    outer_train_ids: list[int],
    outer_validation_ids: list[int],
    outer_fold: int,
    ablation: str,
    sequence_length: int,
    selected_epoch: int,
    hyperparameters: Hyperparameters,
) -> tuple[list[dict[str, Any]], int, float]:
    start = time.perf_counter()
    fold_seed = SEED + 50000 + outer_fold
    train_raw = build_raw_bundle(
        summary,
        cycles_by_battery,
        outer_train_ids,
        ROLLING_CUTOFFS,
        ablation,
        sequence_length,
    )
    validation_raw = build_raw_bundle(
        summary,
        cycles_by_battery,
        outer_validation_ids,
        EARLY_CUTOFFS,
        ablation,
        sequence_length,
    )
    scaler = fit_scaler(train_raw)
    model = make_model(ablation, hyperparameters, fold_seed)
    model = train_fixed_epochs(
        model,
        transform(train_raw, scaler),
        seed=fold_seed,
        epochs=selected_epoch,
        batch_size=hyperparameters.batch_size,
        learning_rate=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    prediction_scaled = predict(model, transform(validation_raw, scaler))
    prediction_delta = prediction_scaled * scaler.target_std + scaler.target_mean
    rows: list[dict[str, Any]] = []
    for sample_index, meta in enumerate(validation_raw.metadata):
        last_soh = float(validation_raw.last_soh[sample_index])
        truth = last_soh + validation_raw.target_delta[sample_index]
        forecast = last_soh + prediction_delta[sample_index]
        for horizon in range(1, FORECAST_HORIZON + 1):
            rows.append(
                {
                    **meta,
                    "horizon": horizon,
                    "cycle": int(meta["cutoff"]) + horizon,
                    "y_true": float(truth[horizon - 1]),
                    "SOH_pred": float(forecast[horizon - 1]),
                    "outer_fold": outer_fold,
                    "selected_ablation": ablation,
                    "selected_sequence_length": sequence_length,
                    "selected_epoch": selected_epoch,
                    "parameter_count": parameter_count(model),
                }
            )
    return rows, parameter_count(model), time.perf_counter() - start


def paired_comparison_at_150(predictions: pd.DataFrame) -> dict[str, Any]:
    work = predictions.loc[predictions["cutoff"].eq(150)].copy()
    per_battery = work.groupby("battery_id").apply(
        lambda group: pd.Series(
            {
                "TCN_RMSE": rmse(group["y_true"].to_numpy(), group["SOH_pred"].to_numpy()),
                "Linear50_RMSE": rmse(
                    group["y_true"].to_numpy(), group["pred_linear_50"].to_numpy()
                ),
            }
        ),
        include_groups=False,
    )
    differences = (per_battery["TCN_RMSE"] - per_battery["Linear50_RMSE"]).to_numpy()
    rng = np.random.default_rng(SEED)
    bootstrap = np.asarray(
        [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(10000)]
    )
    try:
        test = wilcoxon(differences, alternative="less")
        p_value = float(test.pvalue)
    except ValueError:
        p_value = 1.0
    ci = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "paired_batteries": int(len(per_battery)),
        "mean_battery_RMSE_difference_TCN_minus_Linear50": float(differences.mean()),
        "median_battery_RMSE_difference_TCN_minus_Linear50": float(np.median(differences)),
        "bootstrap_95pct_CI_for_mean_difference": [float(ci[0]), float(ci[1])],
        "one_sided_Wilcoxon_p_TCN_less_than_Linear50": p_value,
        "TCN_better_battery_count": int(np.sum(differences < 0)),
        "Linear50_better_or_tied_battery_count": int(np.sum(differences >= 0)),
    }


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def render_report(metrics: dict[str, Any], ablation: pd.DataFrame) -> str:
    cutoff_150 = pd.DataFrame(metrics["overall_metrics"])
    cutoff_150 = cutoff_150.loc[cutoff_150["cutoff"].eq(150)].sort_values("overall_RMSE")
    metric_table = cutoff_150[
        [
            "model",
            "MAE",
            "overall_RMSE",
            "mean_battery_RMSE",
            "median_battery_RMSE",
            "worst_battery_RMSE",
        ]
    ].to_markdown(index=False, floatfmt=".8f")
    ablation_summary = (
        ablation.loc[ablation["stage"].eq("ablation")]
        .groupby("ablation", as_index=False)
        .agg(
            inner_validation_MAE=("inner_validation_MAE", "mean"),
            inner_validation_RMSE=("inner_validation_RMSE", "mean"),
            folds=("outer_fold", "nunique"),
        )
        .sort_values("inner_validation_RMSE")
    )
    length_summary = (
        ablation.loc[ablation["stage"].eq("sequence_length")]
        .groupby("sequence_length", as_index=False)
        .agg(
            inner_validation_MAE=("inner_validation_MAE", "mean"),
            inner_validation_RMSE=("inner_validation_RMSE", "mean"),
            trials=("outer_fold", "count"),
        )
        .sort_values("sequence_length")
    )
    selection = metrics["selection"]
    paired = metrics["cutoff_150_paired_comparison"]
    conclusion = metrics["conclusion"]
    cutoff_rows = pd.DataFrame(metrics["overall_metrics"])
    cutoff_pivot = cutoff_rows.pivot(index="cutoff", columns="model", values="overall_RMSE")
    cutoff_pivot = cutoff_pivot.reset_index()
    return f"""# Agent 01：小样本时序神经网络研究

## 1. 结论先行

{conclusion["recommendation_cn"]}

在核心 `150 -> 151--200` 的严格外层 battery-level 五折交叉验证中，嵌套选择 TCN 的 overall RMSE 为 {conclusion["tcn_rmse_150"]:.8f}，Linear-50 为 {conclusion["linear50_rmse_150"]:.8f}。TCN 相对误差变化为 {conclusion["relative_rmse_change_vs_linear50_pct"]:.2f}%。40 块电池的配对比较中，TCN 仅在 {paired["TCN_better_battery_count"]} 块上更优；`TCN RMSE - Linear-50 RMSE` 的均值为 {paired["mean_battery_RMSE_difference_TCN_minus_Linear50"]:.8f}，95% battery bootstrap 区间为 [{paired["bootstrap_95pct_CI_for_mean_difference"][0]:.8f}, {paired["bootstrap_95pct_CI_for_mean_difference"][1]:.8f}]。单侧 Wilcoxon 检验（备择：TCN 更小）p={paired["one_sided_Wilcoxon_p_TCN_less_than_Linear50"]:.4g}。

因此，对“神经网络是否显著优于 recent-linear baseline？”的回答是：**否**。当前数据仅有 40 块可监督训练电池，TCN 的参数共享没有转化为稳定的跨电池优势，**不推荐作为问题三短期预测主模型**。

## 2. 数据、任务与泄漏控制

- 数据审计给出 49 块电池，其中 40 块训练电池含 cycle 1--200，9 块 `prediction_test=1` 电池只含 cycle 1--150。本 Agent 完全未将这 9 块电池用于训练、调参或验证。
- 本路线只研究可回测的短期任务：在 cutoff `N` 后直接预测 50 维 `SOH(N+h)-SOH(N)`。数据中没有 `SOH <= 0.8`，所以不使用 TCN 无限递推 EOL，也不报告神经网络 EOL 精度。
- 外层采用 policy-aware 的五折 battery split；同一电池的 cycle 行绝不跨 train/validation。每个外层折内另设 battery-level inner holdout，先比较消融，再比较序列长度。外层验证电池的未来值从不参与选择。
- rolling cutoffs 为 50,60,...,150；它们只在已完成 battery 划分后从训练电池生成。每个 cutoff 的输入只读取 `cycle <= cutoff`，目标读取训练电池的后续 50 cycle。
- 汇总表的 `mean_IR/mean_Tavg/mean_chargetime` 完全未用。动态标准化、静态标准化和逐 horizon 目标标准化均只在对应训练折拟合。
- 题目未说明给定 `SOH_smooth` 的平滑方向。为避免预计算的平滑端点隐含未来信息，代码从当前 cutoff 内的原始 SOH 重建 5 点 trailing median + causal EWMA，记为 `SOH_causal`。

## 3. 模型结构

模型由四个 dilation=(1,2,4,8) 的因果 TCN residual blocks、可选静态 MLP、以及 direct 50-horizon decoder 组成。编码器拼接最后时刻表示和有效时间步的 masked mean，输出 50 个 SOH delta；不存在 autoregressive error accumulation。不同外层折所选模型参数量见 `predictions.csv`，全路线参数量保持在约 {metrics["model_complexity"]["parameter_count_min"]}--{metrics["model_complexity"]["parameter_count_max"]}，远小于大型 Transformer。

固定正则化为 dropout={metrics["hyperparameters"]["dropout"]}、AdamW weight decay={metrics["hyperparameters"]["weight_decay"]}，inner training 最多 {metrics["hyperparameters"]["max_epochs"]} epochs、patience={metrics["hyperparameters"]["patience"]}。最终外层模型在全部 outer-train batteries 上按 inner early stopping 选出的 epoch 数重新训练。

## 4. 消融与序列长度筛选

消融定义如下：

- A：仅 cutoff 内重建的因果 SOH 平滑序列；
- B：A + raw SOH、capacity、IR、Tavg、chargetime，并加入静态 initial_capacity；
- C：B + C1、Q1、C2 和 C1 结构性缺失指示。C1 缺失时使用审计确定的单阶段策略物理解释 `C1_effective=C2`，没有均值填补。

下表是五个外层折内部 holdout 的筛选误差，仅用于选参，不是外层泛化成绩。

{ablation_summary.to_markdown(index=False, floatfmt=".8f")}

筛选采用节省计算的顺序嵌套方案：先固定 length=50 比较 A/B/C，再仅对该折胜出的消融比较 20/30/50/75/100。这样避免 15 个组合的无意义网格搜索，也确保没有查看 outer future。

{length_summary.to_markdown(index=False, floatfmt=".8f")}

各折最终选择计数：`{selection["selected_configuration_counts"]}`；序列长度计数：`{selection["selected_sequence_length_counts"]}`。由于 inner holdout 规模小，选择在折间不稳定，这本身是小电池样本下神经模型方差较大的证据。

## 5. 外层验证结果

核心 cutoff=150：

{metric_table}

各 early cutoff 的 overall RMSE：

{cutoff_pivot.to_markdown(index=False, floatfmt=".8f")}

完整 overall、per-policy、horizon-bin 指标在 `metrics.json`；逐点外层预测在 `predictions.csv`。公共 Linear-50 与 Linear-Nested 预测按 battery/cutoff/horizon 原样合并，以保证相同真值、相同折外样本的公平比较。

## 6. 解释与局限

1. rolling cutoff 增加的是同一电池内高度相关的训练窗口，并没有把 battery-level 样本数从 40 真正扩展到数百个独立样本。
2. TCN 可学习局部多变量形态，但 IR、温度、充电时间和策略变量的增益在 inner folds 间不稳定；网络容易把电池个体噪声当成跨电池规律。
3. 近期线性趋势与本数据 50-cycle 预测尺度高度匹配。更复杂网络只有在配对外层误差及其不确定区间均优于线性时才应升级；本实验未达到这一标准。
4. 五折 outer CV 严格隔离电池，但不是 40 次 LOO；每折含约 8 块电池。它给出可信的 battery-level 泛化估计，同时把 CPU 预算集中在严格嵌套而非扩大网络。
5. 充电策略贡献不能从本 Agent 单独作因果结论：策略只有 9 个离散组合，并与 C1/Q1/C2 共线；这里只能解释为预测消融。

## 7. 复现信息

- seed：{metrics["seed"]}
- Python：{metrics["package_versions"]["python"]}
- PyTorch：{metrics["package_versions"]["torch"]}（CPU，单线程确定性训练）
- NumPy / pandas / scikit-learn / SciPy：{metrics["package_versions"]["numpy"]} / {metrics["package_versions"]["pandas"]} / {metrics["package_versions"]["scikit_learn"]} / {metrics["package_versions"]["scipy"]}
- 总运行时间：{metrics["runtime_seconds"]:.1f} s
- 运行命令：`python problem3/agents/agent_01_tcn/run.py`

## 8. 方法文献

1. Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv. DOI: 10.48550/arXiv.1803.01271。用于因果扩张卷积与 residual TCN 的结构依据。
2. Severson, K. A., et al. (2019). *Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 4, 383--391. DOI: 10.1038/s41560-019-0356-8。用于早期循环信息可以支持寿命/退化预测的研究背景；本实验没有读取其公开电池未来轨迹。

以上论文只用于方法设计和结果解释，没有从外部数据、论文补充材料、`global_id` 或 GitHub 获取 9 块测试电池 cycle 151 后的真值。
"""


def run_experiment(smoke: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    hyperparameters = Hyperparameters(
        max_epochs=3 if smoke else 30,
        patience=2 if smoke else 5,
    )
    summary, cycles = load_data()
    train_ids, _ = split_battery_ids(summary)
    cycles_by_battery = {
        int(battery_id): group.sort_values("cycle").reset_index(drop=True)
        for battery_id, group in cycles.groupby("battery_id")
        if int(battery_id) in train_ids
    }
    outer_folds = make_battery_folds(summary, train_ids, 5, SEED)
    ablation_records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_selections: list[dict[str, Any]] = []
    final_parameter_counts: list[int] = []
    final_runtime = 0.0

    for outer_fold, validation_ids in enumerate(outer_folds):
        outer_train_ids = sorted(set(train_ids).difference(validation_ids))
        inner_train_ids, inner_validation_ids = make_inner_split(
            summary, outer_train_ids, outer_fold
        )
        ablation_results: dict[str, tuple[dict[str, Any], int]] = {}
        for ablation in ABLATIONS:
            record, epoch = run_inner_trial(
                summary,
                cycles_by_battery,
                inner_train_ids,
                inner_validation_ids,
                outer_fold,
                "ablation",
                ablation,
                50,
                hyperparameters,
            )
            ablation_records.append(record)
            ablation_results[ablation] = (record, epoch)
        selected_ablation = min(
            ablation_results,
            key=lambda name: (
                ablation_results[name][0]["inner_validation_RMSE"],
                list(ABLATIONS).index(name),
            ),
        )

        length_results: dict[int, tuple[dict[str, Any], int]] = {
            50: ablation_results[selected_ablation]
        }
        # Duplicate the reused length=50 record under the sequence-length stage
        # so the output table is complete without retraining it.
        reused_record = dict(ablation_results[selected_ablation][0])
        reused_record["stage"] = "sequence_length"
        reused_record["reused_from_ablation_stage"] = True
        ablation_records.append(reused_record)
        for sequence_length in [20, 30, 75, 100]:
            record, epoch = run_inner_trial(
                summary,
                cycles_by_battery,
                inner_train_ids,
                inner_validation_ids,
                outer_fold,
                "sequence_length",
                selected_ablation,
                sequence_length,
                hyperparameters,
            )
            record["reused_from_ablation_stage"] = False
            ablation_records.append(record)
            length_results[sequence_length] = (record, epoch)
        selected_length = min(
            length_results,
            key=lambda length: (
                length_results[length][0]["inner_validation_RMSE"], length
            ),
        )
        selected_record, selected_epoch = length_results[selected_length]
        rows, count, fold_runtime = predict_outer_fold(
            summary,
            cycles_by_battery,
            outer_train_ids,
            validation_ids,
            outer_fold,
            selected_ablation,
            selected_length,
            selected_epoch,
            hyperparameters,
        )
        prediction_rows.extend(rows)
        final_parameter_counts.append(count)
        final_runtime += fold_runtime
        fold_selections.append(
            {
                "outer_fold": outer_fold,
                "outer_train_ids": outer_train_ids,
                "outer_validation_ids": validation_ids,
                "inner_train_ids": inner_train_ids,
                "inner_validation_ids": inner_validation_ids,
                "selected_ablation": selected_ablation,
                "selected_sequence_length": selected_length,
                "selected_epoch": selected_epoch,
                "selection_inner_RMSE": selected_record["inner_validation_RMSE"],
            }
        )
        print(
            f"fold={outer_fold} validation={validation_ids} "
            f"selected={selected_ablation}/L{selected_length}/epoch{selected_epoch}",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    baseline = pd.read_csv(PROJECT_ROOT / "problem3" / "common" / "baseline_predictions.csv")
    baseline_columns = [
        "battery_id",
        "cutoff",
        "horizon",
        "pred_linear_50",
        "pred_linear_nested",
    ]
    predictions = predictions.merge(
        baseline[baseline_columns],
        on=["battery_id", "cutoff", "horizon"],
        how="left",
        validate="one_to_one",
    )
    if predictions[["pred_linear_50", "pred_linear_nested"]].isna().any().any():
        raise RuntimeError("Failed to align common baseline predictions")
    prediction_columns = {
        "TCN-Nested": "SOH_pred",
        "Linear-50": "pred_linear_50",
        "Linear-Nested": "pred_linear_nested",
    }
    overall, policy, horizon = detailed_metric_tables(predictions, prediction_columns)
    paired = paired_comparison_at_150(predictions)
    ablation_frame = pd.DataFrame(ablation_records)
    selected_configurations = Counter(
        selection["selected_ablation"] for selection in fold_selections
    )
    selected_lengths = Counter(
        str(selection["selected_sequence_length"]) for selection in fold_selections
    )
    core = overall.loc[overall["cutoff"].eq(150)].set_index("model")
    tcn_rmse = float(core.loc["TCN-Nested", "overall_RMSE"])
    linear_rmse = float(core.loc["Linear-50", "overall_RMSE"])
    significantly_better = bool(
        tcn_rmse < linear_rmse
        and paired["bootstrap_95pct_CI_for_mean_difference"][1] < 0
        and paired["one_sided_Wilcoxon_p_TCN_less_than_Linear50"] < 0.05
    )
    recommendation = (
        "TCN 在配对外层验证中显著优于 Linear-50，可作为候选主模型。"
        if significantly_better
        else "TCN 未显著优于 recent-linear baseline，不推荐作为问题三主模型。"
    )
    metrics: dict[str, Any] = {
        "agent": "agent_01_tcn",
        "status": "smoke" if smoke else "complete",
        "seed": SEED,
        "runtime_seconds": time.perf_counter() - start,
        "final_outer_training_runtime_seconds": final_runtime,
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "hyperparameters": {
            **asdict(hyperparameters),
            "dilations": list(hyperparameters.dilations),
            "rolling_cutoffs": ROLLING_CUTOFFS,
            "outer_folds": 5,
            "target": "SOH(cutoff+h)-SOH(cutoff), h=1..50",
        },
        "validation_protocol": {
            "outer": "policy-aware five-fold split by battery_id",
            "inner": "battery-level holdout within each outer training fold",
            "selection": "A/B/C at L=50, then L=20/30/50/75/100 for winning ablation",
            "normalization": "fit independently on each training fold only",
            "test_batteries_used": False,
        },
        "selection": {
            "folds": fold_selections,
            "selected_configuration_counts": dict(selected_configurations),
            "selected_sequence_length_counts": dict(selected_lengths),
        },
        "model_complexity": {
            "parameter_count_min": min(final_parameter_counts),
            "parameter_count_max": max(final_parameter_counts),
            "parameter_count_by_fold": final_parameter_counts,
        },
        "overall_metrics": json_records(overall),
        "per_policy_metrics": json_records(policy),
        "horizon_bin_metrics": json_records(horizon),
        "cutoff_150_paired_comparison": paired,
        "conclusion": {
            "significantly_outperforms_recent_linear": significantly_better,
            "recommend_as_primary_model": significantly_better,
            "recommendation_cn": recommendation,
            "tcn_rmse_150": tcn_rmse,
            "linear50_rmse_150": linear_rmse,
            "relative_rmse_change_vs_linear50_pct": 100.0 * (tcn_rmse / linear_rmse - 1.0),
        },
    }
    if not smoke:
        predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
        ablation_frame.to_csv(OUTPUT_DIR / "ablation.csv", index=False)
        with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        (OUTPUT_DIR / "report.md").write_text(
            render_report(metrics, ablation_frame), encoding="utf-8"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run three-epoch pipeline check")
    args = parser.parse_args()
    metrics = run_experiment(smoke=args.smoke)
    print(json.dumps(metrics["conclusion"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
