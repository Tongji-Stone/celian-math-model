"""问题二：用 PySR 搜索低复杂度的早期衰减候选经验式。

运行：
    conda run -n pysr python src/A/task2_pysr_symbolic.py

这不是寿命外推：目标为 Fade_200 = SOH_1 - SOH_200。由于只有 6 个独立策略点，
输出的公式只能作为机制假设；脚本会以留一策略重搜检验其不稳定性。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "A" / "问题二" / "output" / "cell_exposure.csv"
OUT = ROOT / "A" / "问题二" / "output" / "pysr"

FEATURE_SETS = {
    "raw_parameters": ["C1", "Q1", "C2"],
    "two_window_exposure": ["E_L", "E_H"],
}
TARGET_SCALE = 1000.0  # 搜索 milli-Fade，避免目标约 1e-3 时常数搜索变慢。


def make_model(run_id: str, iterations: int, timeout: int, seed: int) -> PySRRegressor:
    return PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=12,
        niterations=iterations,
        timeout_in_seconds=timeout,
        precision=64,
        model_selection="best",
        deterministic=True,
        random_state=seed,
        parallelism="serial",
        output_directory=str(OUT / "runs"),
        run_id=run_id,
    )


def fit_full(data: pd.DataFrame, name: str) -> tuple[PySRRegressor, pd.DataFrame, dict]:
    features = FEATURE_SETS[name]
    X = data[features].to_numpy(dtype=float)
    y = data["fade_200"].to_numpy(dtype=float) * TARGET_SCALE
    model = make_model(f"full_{name}", iterations=80, timeout=20, seed=20260814)
    model.fit(X, y, variable_names=features)
    equations = model.equations_.copy()
    pred = model.predict(X) / TARGET_SCALE
    metrics = {
        "feature_set": name,
        "n_batteries": int(len(data)),
        "n_independent_policies": int(data["policy"].nunique()),
        "target": "1000 * Fade_200",
        "selected_equation_milli_fade": str(model.get_best()["equation"]),
        "selected_equation_fade": f"({model.get_best()['equation']}) / {TARGET_SCALE:g}",
        "train_rmse_fade": float(np.sqrt(np.mean((data["fade_200"].to_numpy() - pred) ** 2))),
        "train_mae_fade": float(np.mean(np.abs(data["fade_200"].to_numpy() - pred))),
    }
    pred_table = data[["battery_id", "policy", "fade_200"]].copy()
    pred_table["prediction_selected"] = pred
    pred_table["residual"] = pred_table["fade_200"] - pred
    return model, equations, {**metrics, "predictions": pred_table}


def leave_one_policy_search(data: pd.DataFrame) -> pd.DataFrame:
    """A deliberately strict diagnostic: every held-out policy is unseen in formula search."""
    features = FEATURE_SETS["two_window_exposure"]
    rows = []
    for number, policy in enumerate(sorted(data["policy"].unique()), start=1):
        train = data.loc[data["policy"] != policy]
        test = data.loc[data["policy"] == policy]
        model = make_model(f"lopo_{number}", iterations=35, timeout=8, seed=20260814 + number)
        model.fit(
            train[features].to_numpy(dtype=float),
            train["fade_200"].to_numpy(dtype=float) * TARGET_SCALE,
            variable_names=features,
        )
        prediction = model.predict(test[features].to_numpy(dtype=float)) / TARGET_SCALE
        observed = test["fade_200"].to_numpy(dtype=float)
        best = model.get_best()
        rows.append(
            {
                "held_out_policy": policy,
                "n_train_batteries": int(len(train)),
                "n_train_policies": int(train["policy"].nunique()),
                "selected_equation_milli_fade": str(best["equation"]),
                "observed_mean_fade": float(observed.mean()),
                "predicted_mean_fade": float(prediction.mean()),
                "mae_fade": float(np.mean(np.abs(observed - prediction))),
                "rmse_fade": float(np.sqrt(np.mean((observed - prediction) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(lambda value: f"{value:.6g}")
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    data = data.query("dataset_id == 3 and prediction_test == 0 and max_cycle == 200").copy()
    if len(data) != 34 or data["policy"].nunique() != 6:
        raise ValueError("Expected the fixed primary cohort: 34 batteries / 6 policies.")

    summaries = []
    for name in FEATURE_SETS:
        _, equations, result = fit_full(data, name)
        equations.to_csv(OUT / f"equations_{name}.csv", index=False, encoding="utf-8-sig")
        predictions = result.pop("predictions")
        pd.Series(result).to_json(OUT / f"summary_{name}.json", force_ascii=False, indent=2)
        predictions.to_csv(OUT / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")
        summaries.append(result)

    lopo = leave_one_policy_search(data)
    lopo.to_csv(OUT / "leave_one_policy_two_window.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 问题二：PySR 符号回归候选式",
        "",
        "## 使用口径",
        "",
        "- 目标是 `Fade_200`，不是 SOH=80% 的真实寿命。为改善数值搜索，PySR 拟合的是 `1000 * Fade_200`；报告的公式已除以 1000。",
        "- 34 块电池只有 6 个独立策略配方，策略内的重复不能增加公式结构的识别信息。所有公式是候选经验式，而非因果定律。",
        "- 算子严格限制为 `+,-,*`，最大复杂度 12；不允许任意幂、指数或三角函数，防止小样本产生花哨的插值式。",
        "",
        "## 全样本 Pareto 前沿的自动选择式",
        "",
        markdown_table(summary.drop(columns=["target"])),
        "",
        "完整 Pareto 前沿见 `equations_raw_parameters.csv` 和 `equations_two_window_exposure.csv`；不应只根据训练误差选择最复杂的式子。",
        "",
        "## 留一策略重搜（两窗口暴露）",
        "",
        markdown_table(lopo),
        "",
        "## 解释边界",
        "",
        "- 若留一策略时所选公式结构改变、或被留出 `3_7C_31PER_5_9C` 时误差明显变大，即表明公式依赖该短寿组，不能作为普遍规律。",
        "- 两窗口变量 `E_L,E_H` 来自先验定义的 SOC 暴露，而不是 PySR 自发证明的物理机理；PySR 的作用是检验低复杂度经验式是否足以描述当前数据。",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary[["feature_set", "selected_equation_fade", "train_rmse_fade"]].to_string(index=False))
    print(lopo[["held_out_policy", "rmse_fade"]].to_string(index=False))
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
