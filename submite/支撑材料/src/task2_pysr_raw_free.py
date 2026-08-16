"""问题二：只用 C1、Q1、C2 的自由符号回归。

运行：
    conda activate pysr
    python src/A/task2_pysr_raw_free.py

本脚本刻意不输入两窗口暴露量，目的是检验原始策略参数本身能否稳定地产生候选规律。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "A" / "问题二" / "output" / "cell_exposure.csv"
OUT = ROOT / "A" / "问题二" / "output" / "pysr_raw_free"
FEATURES = ["C1", "Q1", "C2"]
SCALE = 1000.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA).query("dataset_id == 3 and prediction_test == 0 and max_cycle == 200").copy()
    if len(data) != 34 or data["policy"].nunique() != 6:
        raise ValueError("Expected primary cohort with 34 batteries and 6 policies.")

    # The denominator is intentionally restricted to a single variable/constant.
    # This keeps the search free over C1,Q1,C2 while rejecting nested rational interpolation.
    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        constraints={"/": (-1, 1)},
        maxsize=20,
        niterations=180,
        timeout_in_seconds=55,
        precision=64,
        deterministic=True,
        random_state=20260815,
        parallelism="serial",
        model_selection="best",
        output_directory=str(OUT / "runs"),
        run_id="raw_free_full",
    )
    X = data[FEATURES].to_numpy(dtype=float)
    y = data["fade_200"].to_numpy(dtype=float) * SCALE
    model.fit(X, y, variable_names=FEATURES)

    equations = model.equations_.copy()
    equations.to_csv(OUT / "equations.csv", index=False, encoding="utf-8-sig")
    selected_index = int(model.get_best().name)
    prediction = model.predict(X, index=selected_index) / SCALE
    prediction_table = data[["battery_id", "policy", "C1", "Q1", "C2", "fade_200"]].copy()
    prediction_table["prediction_selected"] = prediction
    prediction_table["residual"] = prediction_table["fade_200"] - prediction
    prediction_table.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

    rmse = float(np.sqrt(np.mean((prediction_table["residual"]) ** 2)))
    mae = float(np.mean(np.abs(prediction_table["residual"])))
    selected = model.get_best()
    summary = pd.DataFrame([{
        "n_batteries": len(data),
        "n_policies": data["policy"].nunique(),
        "operators": "+,-,*,/ (denominator complexity <= 1)",
        "selected_complexity": int(selected["complexity"]),
        "selected_equation_milli_fade": str(selected["equation"]),
        "selected_equation_fade": f"({selected['equation']}) / {SCALE:g}",
        "train_rmse_fade": rmse,
        "train_mae_fade": mae,
    }])
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(equations[["complexity", "loss", "score", "equation"]].to_string(index=False))


if __name__ == "__main__":
    main()
