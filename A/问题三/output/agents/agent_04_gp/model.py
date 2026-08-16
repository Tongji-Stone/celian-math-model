from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import least_squares


MEAN_KINDS = ("linear", "power", "exponential")
KERNEL_KINDS = ("rbf", "matern32", "matern52")
LENGTH_SCALES = (10.0, 30.0, 60.0)
MAX_EOL_CYCLE = 20_000


@dataclass(frozen=True, order=True)
class GPConfig:
    mean_kind: str
    kernel_kind: str
    length_scale: float

    @property
    def name(self) -> str:
        return f"{self.mean_kind}+{self.kernel_kind}(ell={self.length_scale:g})"


@dataclass(frozen=True)
class MeanModel:
    kind: str
    params: np.ndarray
    cutoff: float

    def predict(self, cycles: np.ndarray) -> np.ndarray:
        x = np.asarray(cycles, dtype=float)
        if self.kind == "linear":
            level, slope = self.params
            return level + slope * (x - self.cutoff)
        if self.kind == "power":
            level, scale, exponent = self.params
            return level - scale * np.power(np.maximum(x, 0.0) / 100.0, exponent)
        if self.kind == "exponential":
            level, scale, time_constant = self.params
            z = np.clip(x / time_constant, 0.0, 50.0)
            return level - scale * np.expm1(z)
        raise ValueError(f"Unknown mean kind: {self.kind}")

    def crossing_cycle(
        self,
        threshold: float,
        after_cycle: int,
        max_cycle: int = MAX_EOL_CYCLE,
    ) -> float:
        if self.kind == "linear":
            level, slope = self.params
            if slope >= -1e-12:
                return np.nan
            crossing = self.cutoff + (threshold - level) / slope
        elif self.kind == "power":
            level, scale, exponent = self.params
            if scale <= 1e-12 or level <= threshold:
                crossing = float(after_cycle + 1)
            else:
                crossing = 100.0 * ((level - threshold) / scale) ** (1.0 / exponent)
        elif self.kind == "exponential":
            level, scale, time_constant = self.params
            if scale <= 1e-12 or level <= threshold:
                crossing = float(after_cycle + 1)
            else:
                crossing = time_constant * np.log1p((level - threshold) / scale)
        else:
            raise ValueError(f"Unknown mean kind: {self.kind}")
        crossing = max(float(after_cycle + 1), float(crossing))
        return crossing if np.isfinite(crossing) and crossing <= max_cycle else np.nan


@dataclass(frozen=True)
class GPForecast:
    mean: np.ndarray
    std: np.ndarray
    mean_model: MeanModel
    signal_std: float
    noise_std: float


def candidate_configs() -> list[GPConfig]:
    return [
        GPConfig(mean_kind, kernel_kind, length_scale)
        for mean_kind in MEAN_KINDS
        for kernel_kind in KERNEL_KINDS
        for length_scale in LENGTH_SCALES
    ]


def _robust_scale(values: np.ndarray, floor: float = 1e-5) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, floor)


def _fit_scale(y: np.ndarray) -> float:
    diff = np.diff(np.asarray(y, dtype=float))
    return max(_robust_scale(diff, floor=2e-5), 8e-5)


def fit_mean(cycles: np.ndarray, soh: np.ndarray, kind: str) -> MeanModel:
    x = np.asarray(cycles, dtype=float)
    y = np.asarray(soh, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise ValueError("cycles and soh must be aligned one-dimensional arrays")
    cutoff = float(x[-1])
    robust_loss_scale = _fit_scale(y)

    if kind == "linear":
        use = min(50, len(x))
        x_fit, y_fit = x[-use:], y[-use:]
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        initial = np.array([intercept + slope * cutoff, slope])

        def residual(params: np.ndarray) -> np.ndarray:
            level, local_slope = params
            return level + local_slope * (x_fit - cutoff) - y_fit

        result = least_squares(
            residual,
            initial,
            bounds=([0.70, -0.01], [1.50, 0.002]),
            loss="soft_l1",
            f_scale=robust_loss_scale,
            max_nfev=300,
        )
    elif kind == "power":
        slope = float(np.polyfit(x[-min(50, len(x)):], y[-min(50, len(y)):], 1)[0])
        scale0 = float(np.clip(-slope * 100.0, 1e-5, 0.1))
        level0 = float(np.clip(np.median(y[: min(10, len(y))]) + scale0 * (x[0] / 100.0), 0.8, 1.4))
        initial = np.array([level0, scale0, 1.0])

        def residual(params: np.ndarray) -> np.ndarray:
            level, scale, exponent = params
            return level - scale * np.power(x / 100.0, exponent) - y

        result = least_squares(
            residual,
            initial,
            bounds=([0.70, 1e-9, 0.2], [1.50, 0.50, 3.0]),
            loss="soft_l1",
            f_scale=robust_loss_scale,
            max_nfev=600,
        )
    elif kind == "exponential":
        slope = float(np.polyfit(x[-min(50, len(x)):], y[-min(50, len(y)):], 1)[0])
        time0 = 1000.0
        scale0 = float(np.clip(-slope * time0, 1e-5, 0.1))
        level0 = float(np.clip(np.median(y[: min(10, len(y))]), 0.8, 1.4))
        initial = np.array([level0, scale0, time0])

        def residual(params: np.ndarray) -> np.ndarray:
            level, scale, time_constant = params
            return level - scale * np.expm1(np.clip(x / time_constant, 0.0, 50.0)) - y

        result = least_squares(
            residual,
            initial,
            bounds=([0.70, 1e-9, 30.0], [1.50, 0.50, 10_000.0]),
            loss="soft_l1",
            f_scale=robust_loss_scale,
            max_nfev=800,
        )
    else:
        raise ValueError(f"Unknown mean kind: {kind}")
    return MeanModel(kind=kind, params=result.x.astype(float), cutoff=cutoff)


def _kernel_matrix(x1: np.ndarray, x2: np.ndarray, kind: str, length_scale: float) -> np.ndarray:
    distance = np.abs(np.subtract.outer(np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)))
    scaled = distance / float(length_scale)
    if kind == "rbf":
        return np.exp(-0.5 * scaled**2)
    if kind == "matern32":
        root3 = np.sqrt(3.0) * scaled
        return (1.0 + root3) * np.exp(-root3)
    if kind == "matern52":
        root5 = np.sqrt(5.0) * scaled
        return (1.0 + root5 + 5.0 * scaled**2 / 3.0) * np.exp(-root5)
    raise ValueError(f"Unknown kernel kind: {kind}")


def fit_predict_gp(
    history_cycles: np.ndarray,
    history_soh: np.ndarray,
    future_cycles: np.ndarray,
    config: GPConfig,
) -> GPForecast:
    x = np.asarray(history_cycles, dtype=float)
    y = np.asarray(history_soh, dtype=float)
    x_future = np.asarray(future_cycles, dtype=float)
    mean_model = fit_mean(x, y, config.mean_kind)
    residual = y - mean_model.predict(x)
    residual_center = float(np.median(residual))
    residual_scale = _robust_scale(residual, floor=5e-5)
    clipped = np.clip(residual, residual_center - 6.0 * residual_scale, residual_center + 6.0 * residual_scale)
    noise_std = max(_robust_scale(np.diff(clipped), floor=2e-5) / np.sqrt(2.0), 5e-5)
    total_var = float(np.var(clipped))
    signal_std = max(np.sqrt(max(total_var - noise_std**2, 0.0)), 5e-5)

    train_kernel = _kernel_matrix(x, x, config.kernel_kind, config.length_scale)
    covariance = signal_std**2 * train_kernel
    covariance.flat[:: len(x) + 1] += noise_std**2 + 1e-12
    factor, lower = cho_factor(covariance, lower=True, check_finite=False)
    alpha = cho_solve((factor, lower), clipped, check_finite=False)

    cross = signal_std**2 * _kernel_matrix(x, x_future, config.kernel_kind, config.length_scale)
    prediction = mean_model.predict(x_future) + cross.T @ alpha
    solved = solve_triangular(factor, cross, lower=lower, check_finite=False)
    variance = signal_std**2 + noise_std**2 - np.sum(solved**2, axis=0)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return GPForecast(
        mean=prediction.astype(float),
        std=std.astype(float),
        mean_model=mean_model,
        signal_std=float(signal_std),
        noise_std=float(noise_std),
    )


def finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else np.nan


def block_bootstrap_eol(
    cycles: np.ndarray,
    soh: np.ndarray,
    mean_kind: str,
    threshold: float = 0.8,
    n_samples: int = 100,
    seed: int = 20260814,
    block_length: int = 10,
) -> dict[str, float]:
    x = np.asarray(cycles, dtype=float)
    y = np.asarray(soh, dtype=float)
    fitted = fit_mean(x, y, mean_kind)
    fitted_values = fitted.predict(x)
    residual = y - fitted_values
    center = float(np.median(residual))
    scale = _robust_scale(residual, floor=5e-5)
    residual = np.clip(residual, center - 6.0 * scale, center + 6.0 * scale)
    rng = np.random.default_rng(seed)
    sampled_crossings: list[float] = []
    block = min(block_length, len(residual))
    max_start = len(residual) - block
    blocks_needed = int(np.ceil(len(residual) / block))
    for _ in range(n_samples):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        sampled_residual = np.concatenate([residual[start : start + block] for start in starts])[: len(residual)]
        synthetic = fitted_values + sampled_residual
        sampled_model = fit_mean(x, synthetic, mean_kind)
        crossing = sampled_model.crossing_cycle(threshold, int(x[-1]))
        sampled_crossings.append(crossing)
    values = np.asarray(sampled_crossings, dtype=float)
    finite = values[np.isfinite(values)]
    result = {
        "point": fitted.crossing_cycle(threshold, int(x[-1])),
        "median": finite_quantile(finite, 0.50),
        "lower_90": finite_quantile(finite, 0.05),
        "upper_90": finite_quantile(finite, 0.95),
        "lower_95": finite_quantile(finite, 0.025),
        "upper_95": finite_quantile(finite, 0.975),
        "crossing_probability": float(len(finite) / n_samples),
        "n_samples": float(n_samples),
    }
    return result
