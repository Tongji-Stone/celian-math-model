"""问题二：以可由题目数据直接计算的物理代理量进行 PySR 搜索。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "A" / "问题二" / "output" / "cell_exposure.csv"
CYCLES = ROOT / "data" / "cycle_train.csv"
OUT = ROOT / "A" / "问题二" / "output" / "pysr_physical_priors"
Q_STAR = 50.0
SCALE = 1000.0
FEATURES = ["Pi_high", "tau_high", "H_all"]


def high_soc_time(c1: float, q1: float, c2: float) -> float:
    """Relative charging time above Q_STAR, in percentage-point/C units."""
    if q1 >= Q_STAR:
        return (q1 - Q_STAR) / c1 + (80.0 - q1) / c2
    return (80.0 - Q_STAR) / c2


def add_physical_priors(cells: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    ir1 = cycles.loc[cycles["cycle"] == 1, ["battery_id", "IR"]].rename(columns={"IR": "IR_1"})
    out = cells.merge(ir1, on="battery_id", validate="one_to_one").copy()
    out["tau_high"] = [high_soc_time(r.C1, r.Q1, r.C2) for r in out.itertuples()]
    # I*R is a first-order polarization proxy; use baseline IR to avoid outcome leakage.
    out["Pi_high"] = out["C_high"] * out["IR_1"]
    # Integral I^2 R dt is proportional to R * C * DeltaSOC in CC segments.
    out["H_all"] = out["IR_1"] * (out["C1"] * out["Q1"] + out["C2"] * (80.0 - out["Q1"]))
    return out


def build_model(run_id: str, seed: int, iterations: int, timeout: int) -> PySRRegressor:
    return PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=12,
        niterations=iterations,
        timeout_in_seconds=timeout,
        precision=64,
        deterministic=True,
        random_state=seed,
        parallelism="serial",
        model_selection="best",
        output_directory=str(OUT / "runs"),
        run_id=run_id,
    )


def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame | None, run_id: str, seed: int, iterations: int, timeout: int):
    model = build_model(run_id, seed, iterations, timeout)
    model.fit(train[FEATURES].to_numpy(float), train["fade_200"].to_numpy(float) * SCALE, variable_names=FEATURES)
    selected = model.get_best()
    if test is None:
        test = train
    pred = model.predict(test[FEATURES].to_numpy(float), index=int(selected.name)) / SCALE
    return model, selected, pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(DATA).query("dataset_id == 3 and prediction_test == 0 and max_cycle == 200").copy()
    cycles = pd.read_csv(CYCLES)
    data = add_physical_priors(cells, cycles)
    data.to_csv(OUT / "physical_features.csv", index=False, encoding="utf-8-sig")

    model, selected, prediction = fit_and_predict(data, None, "full", 20260815, 100, 28)
    equations = model.equations_.copy()
    equations.to_csv(OUT / "equations.csv", index=False, encoding="utf-8-sig")
    pred_table = data[["battery_id", "policy", "fade_200", *FEATURES]].copy()
    pred_table["prediction_selected"] = prediction
    pred_table["residual"] = pred_table["fade_200"] - prediction
    pred_table.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

    rows = []
    for number, policy in enumerate(sorted(data["policy"].unique()), start=1):
        train = data.loc[data["policy"] != policy]
        test = data.loc[data["policy"] == policy]
        _, chosen, pred = fit_and_predict(train, test, f"lopo_{number}", 20260900 + number, 42, 8)
        err = test["fade_200"].to_numpy(float) - pred
        rows.append({
            "held_out_policy": policy,
            "selected_complexity": int(chosen["complexity"]),
            "selected_equation_milli_fade": str(chosen["equation"]),
            "observed_mean_fade": float(test["fade_200"].mean()),
            "predicted_mean_fade": float(pred.mean()),
            "rmse_fade": float(np.sqrt(np.mean(err**2))),
        })
    lopo = pd.DataFrame(rows)
    lopo.to_csv(OUT / "leave_one_policy.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "selected_complexity": int(selected["complexity"]),
        "selected_equation_milli_fade": str(selected["equation"]),
        "selected_equation_fade": f"({selected['equation']}) / {SCALE:g}",
        "train_rmse_fade": float(np.sqrt(np.mean(pred_table["residual"] ** 2))),
        "train_mae_fade": float(np.mean(np.abs(pred_table["residual"]))),
        "lopo_mean_rmse": float(lopo["rmse_fade"].mean()),
        "lopo_max_rmse": float(lopo["rmse_fade"].max()),
    }])
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(equations[["complexity", "loss", "score", "equation"]].to_string(index=False))
    print(lopo.to_string(index=False))


if __name__ == "__main__":
    main()
