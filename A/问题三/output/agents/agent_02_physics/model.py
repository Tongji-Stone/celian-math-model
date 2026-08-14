from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


BASE_MODELS = ("linear", "power", "exponential")
PARAM_BOUNDS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "linear": (np.array([0.80, 1e-10]), np.array([1.20, 1e-2])),
    "power": (
        np.array([0.80, 1e-12, 0.25]),
        np.array([1.20, 2e-2, 3.00]),
    ),
    "exponential": (np.array([0.80, 1e-10]), np.array([1.20, 2e-2])),
}


@dataclass(frozen=True)
class DegradationFit:
    model: str
    params: np.ndarray
    covariance: np.ndarray
    success: bool
    fit_rmse: float
    n_observations: int
    message: str


def robust_asof_soh(history: pd.DataFrame, window: int = 7) -> np.ndarray:
    """Clean SOH using only rows available in the supplied cutoff view.

    The supplied SOH_smooth is intentionally not used: its smoothing provenance is
    unknown, and its endpoint at an earlier simulated cutoff could encode later rows.
    A fixed Hampel-style rule suppresses only high-impact measurement spikes.
    """

    values = history["SOH"].astype(float).copy()
    local_median = values.rolling(window, center=True, min_periods=3).median()
    local_median = local_median.fillna(values.expanding(min_periods=1).median())
    absolute_deviation = (values - local_median).abs()
    local_mad = absolute_deviation.rolling(window, center=True, min_periods=3).median()
    global_mad = float(np.median(np.abs(values - np.median(values))))
    robust_scale = 1.4826 * local_mad.fillna(global_mad).clip(lower=max(global_mad, 1e-6))
    tolerance = np.maximum(6.0 * robust_scale.to_numpy(dtype=float), 0.01)
    cleaned = values.to_numpy(dtype=float)
    replacement = local_median.to_numpy(dtype=float)
    is_outlier = np.abs(cleaned - replacement) > tolerance
    cleaned[is_outlier] = replacement[is_outlier]
    return cleaned


def model_values(model: str, cycles: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(cycles, dtype=float)
    if model == "linear":
        a, b = params
        return a - b * x
    if model == "power":
        a, k, p = params
        return a - k * np.power(x, p)
    if model == "exponential":
        a, k = params
        return a * np.exp(-k * x)
    raise ValueError(f"Unknown degradation model: {model}")


def model_derivative(model: str, cycles: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(cycles, dtype=float)
    if model == "linear":
        return np.repeat(-float(params[1]), len(x))
    if model == "power":
        _, k, p = params
        return -k * p * np.power(np.maximum(x, 1e-9), p - 1.0)
    if model == "exponential":
        a, k = params
        return -a * k * np.exp(-k * x)
    raise ValueError(f"Unknown degradation model: {model}")


def _initial_guesses(model: str, x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    decline = max(1e-8, -float(slope))
    level = float(np.clip(intercept, 0.82, 1.18))
    if model == "linear":
        return [np.array([level, decline])]
    if model == "exponential":
        a0 = float(np.clip(np.median(y[: min(10, len(y))]), 0.82, 1.18))
        return [np.array([a0, max(1e-9, decline / max(a0, 1e-6))])]
    if model == "power":
        guesses = []
        for p0 in (0.5, 1.0, 1.5, 2.0):
            scale = max(1e-10, decline / (p0 * max(float(np.median(x)), 1.0) ** (p0 - 1.0)))
            a0 = float(np.clip(np.median(y[: min(10, len(y))]) + scale * np.median(x[: min(10, len(x))]) ** p0, 0.82, 1.18))
            guesses.append(np.array([a0, scale, p0]))
        return guesses
    raise ValueError(f"Unknown degradation model: {model}")


def fit_degradation_model(
    cycles: np.ndarray,
    soh: np.ndarray,
    model: str,
    *,
    initial_params: np.ndarray | None = None,
    calculate_covariance: bool = True,
) -> DegradationFit:
    """Fit a monotone degradation family with robust residual loss.

    The sign bounds impose a non-positive model derivative, but do not force the
    noisy observations themselves to be monotone. This is the requested soft
    physics treatment at the data level.
    """

    x = np.asarray(cycles, dtype=float)
    y = np.asarray(soh, dtype=float)
    lower, upper = PARAM_BOUNDS[model]
    guesses = [initial_params] if initial_params is not None else _initial_guesses(model, x, y)
    best = None
    for guess in guesses:
        guess = np.clip(np.asarray(guess, dtype=float), lower + 1e-12, upper - 1e-12)
        result = least_squares(
            lambda params: model_values(model, x, params) - y,
            x0=guess,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.0015,
            max_nfev=3000,
        )
        objective = float(np.sum((model_values(model, x, result.x) - y) ** 2))
        if best is None or objective < best[0]:
            best = (objective, result)
    if best is None:
        raise RuntimeError(f"No fit attempt completed for model={model}")
    objective, result = best
    fitted = model_values(model, x, result.x)
    covariance = np.full((len(result.x), len(result.x)), np.nan)
    if calculate_covariance and len(y) > len(result.x):
        try:
            jacobian = np.asarray(result.jac, dtype=float)
            sigma2 = objective / max(1, len(y) - len(result.x))
            covariance = sigma2 * np.linalg.pinv(jacobian.T @ jacobian)
        except np.linalg.LinAlgError:
            pass
    return DegradationFit(
        model=model,
        params=np.asarray(result.x, dtype=float),
        covariance=covariance,
        success=bool(result.success),
        fit_rmse=float(np.sqrt(np.mean((fitted - y) ** 2))),
        n_observations=int(len(y)),
        message=str(result.message),
    )


def forecast(fit: DegradationFit, future_cycles: np.ndarray) -> np.ndarray:
    return model_values(fit.model, np.asarray(future_cycles, dtype=float), fit.params)


def power_damped_residual_forecast(
    fit: DegradationFit,
    history_cycles: np.ndarray,
    history_soh: np.ndarray,
    future_cycles: np.ndarray,
    decay_cycles: float = 25.0,
) -> np.ndarray:
    """A low-capacity residual correction that decays back to the power law.

    This is deliberately much smaller than a neural residual model: it estimates
    only the current level mismatch and does not extrapolate an unconstrained
    residual slope. Its long-horizon limit is the monotone power-law family.
    """

    if fit.model != "power":
        raise ValueError("Damped residual correction is defined for the power model")
    x_hist = np.asarray(history_cycles, dtype=float)
    y_hist = np.asarray(history_soh, dtype=float)
    tail = min(10, len(x_hist))
    residual_level = float(np.median(y_hist[-tail:] - model_values("power", x_hist[-tail:], fit.params)))
    future_x = np.asarray(future_cycles, dtype=float)
    horizon = future_x - float(x_hist[-1])
    correction = residual_level * np.exp(-horizon / float(decay_cycles))
    return model_values("power", future_x, fit.params) + correction


def crossing_cycle_from_params(model: str, params: np.ndarray, threshold: float) -> float:
    params = np.asarray(params, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if model == "linear":
            a, b = params
            value = (a - threshold) / b
        elif model == "power":
            a, k, p = params
            value = np.power((a - threshold) / k, 1.0 / p) if a > threshold else np.nan
        elif model == "exponential":
            a, k = params
            value = np.log(a / threshold) / k if a > threshold else np.nan
        else:
            raise ValueError(f"Unknown degradation model: {model}")
    value = float(value)
    return value if np.isfinite(value) and value >= 0.0 else float("nan")


def crossing_cycle(fit: DegradationFit, threshold: float) -> float:
    return crossing_cycle_from_params(fit.model, fit.params, threshold)


def covariance_crossing_interval(
    fit: DegradationFit,
    threshold: float,
    *,
    seed: int,
    draws: int = 500,
    interval: float = 0.90,
) -> tuple[float, float, int]:
    if not np.isfinite(fit.covariance).all():
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    try:
        samples = rng.multivariate_normal(fit.params, fit.covariance, size=draws, check_valid="ignore")
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), float("nan"), 0
    lower, upper = PARAM_BOUNDS[fit.model]
    samples = np.clip(samples, lower + 1e-12, upper - 1e-12)
    crossings = np.array(
        [crossing_cycle_from_params(fit.model, params, threshold) for params in samples],
        dtype=float,
    )
    crossings = crossings[np.isfinite(crossings)]
    if len(crossings) < max(20, draws // 10):
        return float("nan"), float("nan"), int(len(crossings))
    alpha = (1.0 - interval) / 2.0
    return (
        float(np.quantile(crossings, alpha)),
        float(np.quantile(crossings, 1.0 - alpha)),
        int(len(crossings)),
    )


def moving_block_bootstrap_crossing_interval(
    fit: DegradationFit,
    cycles: np.ndarray,
    soh: np.ndarray,
    threshold: float,
    *,
    seed: int,
    repetitions: int = 40,
    block_length: int = 8,
    interval: float = 0.90,
) -> tuple[float, float, int]:
    x = np.asarray(cycles, dtype=float)
    y = np.asarray(soh, dtype=float)
    fitted = model_values(fit.model, x, fit.params)
    residuals = y - fitted
    residuals = residuals - float(np.mean(residuals))
    rng = np.random.default_rng(seed)
    crossings: list[float] = []
    n = len(residuals)
    for _ in range(repetitions):
        sampled_parts = []
        while sum(len(part) for part in sampled_parts) < n:
            start = int(rng.integers(0, n))
            index = (start + np.arange(block_length)) % n
            sampled_parts.append(residuals[index])
        sampled = np.concatenate(sampled_parts)[:n]
        bootstrap_soh = fitted + sampled
        try:
            bootstrap_fit = fit_degradation_model(
                x,
                bootstrap_soh,
                fit.model,
                initial_params=fit.params,
                calculate_covariance=False,
            )
            value = crossing_cycle(bootstrap_fit, threshold)
            if np.isfinite(value):
                crossings.append(float(value))
        except (RuntimeError, ValueError, FloatingPointError):
            continue
    if len(crossings) < max(10, repetitions // 2):
        return float("nan"), float("nan"), int(len(crossings))
    alpha = (1.0 - interval) / 2.0
    return (
        float(np.quantile(crossings, alpha)),
        float(np.quantile(crossings, 1.0 - alpha)),
        int(len(crossings)),
    )


def persistent_crossing_cycle(
    trajectory: pd.DataFrame,
    threshold: float,
    *,
    median_window: int = 5,
    persistence: int = 3,
) -> float:
    """Observed pseudo-threshold crossing from a causal trailing median."""

    ordered = trajectory.sort_values("cycle")
    filtered = ordered["SOH"].astype(float).rolling(median_window, min_periods=median_window).median()
    below = filtered.le(threshold).fillna(False).to_numpy(dtype=bool)
    cycles = ordered["cycle"].to_numpy(dtype=float)
    for index in range(0, len(below) - persistence + 1):
        if bool(np.all(below[index : index + persistence])):
            return float(cycles[index])
    return float("nan")
