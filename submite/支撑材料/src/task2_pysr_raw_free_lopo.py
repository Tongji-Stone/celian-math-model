"""对原始 C1,Q1,C2 自由符号回归做留一策略重搜。"""

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


def model_for(run_id: str, seed: int) -> PySRRegressor:
    return PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        constraints={"/": (-1, 1)},
        maxsize=20,
        niterations=70,
        timeout_in_seconds=12,
        precision=64,
        deterministic=True,
        random_state=seed,
        parallelism="serial",
        model_selection="best",
        output_directory=str(OUT / "runs"),
        run_id=run_id,
    )


def main() -> None:
    data = pd.read_csv(DATA).query("dataset_id == 3 and prediction_test == 0 and max_cycle == 200").copy()
    rows = []
    for number, held in enumerate(sorted(data["policy"].unique()), start=1):
        train = data.loc[data["policy"] != held]
        test = data.loc[data["policy"] == held]
        model = model_for(f"raw_free_lopo_{number}", 20260815 + number)
        model.fit(train[FEATURES].to_numpy(float), train["fade_200"].to_numpy(float) * SCALE, variable_names=FEATURES)
        pred = model.predict(test[FEATURES].to_numpy(float)) / SCALE
        observed = test["fade_200"].to_numpy(float)
        chosen = model.get_best()
        rows.append({
            "held_out_policy": held,
            "selected_complexity": int(chosen["complexity"]),
            "selected_equation_milli_fade": str(chosen["equation"]),
            "observed_mean_fade": float(observed.mean()),
            "predicted_mean_fade": float(pred.mean()),
            "rmse_fade": float(np.sqrt(np.mean((observed - pred) ** 2))),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "leave_one_policy.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(out[["rmse_fade"]].agg(["mean", "max"]).to_string())


if __name__ == "__main__":
    main()
