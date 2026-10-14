"""Automatic calibration: noise floor, edge threshold and sensor lag.

Pure Python + numpy, no Home Assistant imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from .const import (
    DEFAULT_EDGE_THRESHOLD,
    MAX_EDGE_THRESHOLD,
    MAX_LAG_SECONDS,
    MIN_EDGE_THRESHOLD,
)

FloatArray = npt.NDArray[np.float64]

# A difference larger than this is considered an edge, not noise.
_QUIET_DIFF_LIMIT = 300.0
# Maximum share of "loud" differences for a window to count as quiet.
_QUIET_MAX_LOUD_FRACTION = 0.02
# Edge threshold = NOISE_FACTOR * robust sigma of span differences.
NOISE_FACTOR = 10.0
_NOISE_EMA = 0.1
_LAG_EMA = 0.25
_LAG_OUTLIER_S = 3.0
_LAG_HISTORY = 15
_LAG_MAJORITY = 8
_MIN_LAG_CORRELATION = 0.3
_MIN_LAG_MARGIN = 0.1
_MIN_GEN_ACTIVITY_W = 15.0

# Calibration is considered complete after this many quiet windows / lag fits.
CALIBRATION_NOISE_WINDOWS = 20
CALIBRATION_LAG_ESTIMATES = 8


def span_diff(x: FloatArray, span: int) -> FloatArray:
    """Return ``x[t] - x[t - span]`` (NaN for the first ``span`` samples)."""
    out = np.full_like(x, math.nan)
    if span < x.shape[0]:
        out[span:] = x[span:] - x[:-span]
    return out


def robust_sigma(values: FloatArray) -> float | None:
    """Return the MAD based standard deviation estimate of ``values``."""
    values = values[np.isfinite(values)]
    if values.size < 30:
        return None
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return 1.4826 * mad


def quiet_noise_sigma(residual: FloatArray, span: int) -> float | None:
    """Estimate the noise sigma of a window, or ``None`` if it is not quiet."""
    diffs = span_diff(residual, span)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 60:
        return None
    loud = np.abs(diffs) > _QUIET_DIFF_LIMIT
    if float(np.mean(loud)) > _QUIET_MAX_LOUD_FRACTION:
        return None
    return robust_sigma(diffs[~loud])


def threshold_from_sigma(sigma: float | None) -> float:
    """Map a noise sigma onto the clamped edge threshold."""
    if sigma is None:
        return DEFAULT_EDGE_THRESHOLD
    return float(np.clip(NOISE_FACTOR * sigma, MIN_EDGE_THRESHOLD, MAX_EDGE_THRESHOLD))


def estimate_lag(
    grid: FloatArray, gen: FloatArray, max_lag: int = MAX_LAG_SECONDS
) -> tuple[float, float] | None:
    """Estimate how many seconds the generation signal lags behind the grid signal.

    In a zero-export system every change of the inverter output is mirrored by
    an opposite change of the grid power at the same physical instant. The
    lag ``L`` that maximises ``-corr(d_grid[t], d_gen[t + L])`` (cross
    correlation of the difference signals) is therefore the reporting offset
    between both sensors.

    Load steps that are larger than anything the inverter does carry no
    information about that offset but would dominate the correlation (the
    inverter reacts to them a few seconds later). Grid differences beyond the
    inverter's range are therefore ignored. Only a clear interior peak is
    accepted. Returns ``(lag, correlation)`` or ``None``.
    """
    n = grid.shape[0]
    if n < 4 * max_lag or gen.shape[0] != n:
        return None
    d_grid = np.diff(grid)
    d_gen = np.diff(gen)
    valid = np.isfinite(d_grid) & np.isfinite(d_gen)
    if float(np.mean(valid)) < 0.8:
        return None
    d_grid = np.where(valid, d_grid, 0.0)
    d_gen = np.where(valid, d_gen, 0.0)
    if float(np.std(d_gen)) < _MIN_GEN_ACTIVITY_W:
        return None
    gen_range = (
        float(np.percentile(np.abs(d_gen[d_gen != 0]), 95))
        if np.any(d_gen != 0)
        else 0.0
    )
    if gen_range <= 0:
        return None
    d_grid = np.where(np.abs(d_grid) > 1.3 * gen_range, 0.0, d_grid)
    # Smooth both spike trains a little so that sensors with different update
    # cadences still overlap.
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    d_grid = np.convolve(d_grid, kernel, mode="same")
    d_gen = np.convolve(d_gen, kernel, mode="same")
    core = slice(max_lag, d_grid.shape[0] - max_lag)
    g = d_grid[core]
    norm_g = float(np.sqrt(np.dot(g, g)))
    if norm_g <= 0:
        return None
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(lags.shape[0], dtype=np.float64)
    m = g.shape[0]
    for i, shift in enumerate(lags.tolist()):
        seg = d_gen[max_lag + shift : max_lag + shift + m]
        norm_s = float(np.sqrt(np.dot(seg, seg)))
        corr[i] = -float(np.dot(g, seg)) / (norm_g * norm_s) if norm_s > 0 else 0.0
    best = int(np.argmax(corr))
    peak = float(corr[best])
    if peak < _MIN_LAG_CORRELATION or best in (0, corr.shape[0] - 1):
        return None
    away = np.abs(lags - lags[best]) > 4
    if np.any(away) and peak - float(np.max(corr[away])) < _MIN_LAG_MARGIN:
        return None
    lag = float(lags[best])
    # parabolic interpolation for sub-second resolution
    y0, y1, y2 = corr[best - 1], corr[best], corr[best + 1]
    denom = y0 - 2 * y1 + y2
    if denom < 0:
        lag += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return lag, peak


@dataclass(slots=True)
class Calibrator:
    """Hold and update the calibration state (persisted)."""

    noise_sigma: float | None = None
    noise_windows: int = 0
    lag_s: float | None = None
    lag_estimates: int = 0
    recent_lags: list[float] = field(default_factory=list)

    @property
    def edge_threshold(self) -> float:
        """Return the current edge threshold in W."""
        return threshold_from_sigma(self.noise_sigma)

    @property
    def lag_int(self) -> int:
        """Return the lag rounded to whole raster steps (0 if unknown)."""
        if self.lag_s is None:
            return 0
        return round(self.lag_s)

    def is_calibrated(self, has_generation: bool) -> bool:
        """Return True once enough data has been seen."""
        if self.noise_windows < CALIBRATION_NOISE_WINDOWS:
            return False
        return not has_generation or self.lag_estimates >= CALIBRATION_LAG_ESTIMATES

    def update_noise(self, residual: FloatArray, span: int) -> bool:
        """Feed a window; returns True if it was quiet and used."""
        sigma = quiet_noise_sigma(residual, span)
        if sigma is None:
            return False
        if self.noise_sigma is None:
            self.noise_sigma = sigma
        else:
            # fast convergence at the start, slow drift afterwards
            alpha = max(_NOISE_EMA, 1.0 / (self.noise_windows + 1))
            self.noise_sigma += alpha * (sigma - self.noise_sigma)
        self.noise_windows += 1
        return True

    def update_lag(self, grid: FloatArray, gen: FloatArray) -> bool:
        """Feed a window of raw grid and generation data; True if the lag changed.

        Every raw estimate is kept in a short history. Estimates far from the
        median of that history are outliers and discarded; accepted ones are
        smoothed with an EMA. If the majority of the recent estimates moved
        to a new value (for example after changing the inverter polling
        interval), the EMA follows the median immediately.
        """
        result = estimate_lag(grid, gen)
        if result is None:
            return False
        lag, _corr = result
        self.recent_lags.append(lag)
        del self.recent_lags[:-_LAG_HISTORY]
        reference = float(np.median(self.recent_lags))
        if self.lag_s is None:
            self.lag_s = lag
            self.lag_estimates = 1
            return True
        if len(self.recent_lags) >= _LAG_MAJORITY and (
            abs(self.lag_s - reference) > _LAG_OUTLIER_S
        ):
            self.lag_s = reference
            self.lag_estimates += 1
            return True
        if len(self.recent_lags) >= 3 and abs(lag - reference) > _LAG_OUTLIER_S:
            return False  # outlier
        alpha = max(_LAG_EMA, 1.0 / (self.lag_estimates + 1))
        self.lag_s += alpha * (lag - self.lag_s)
        self.lag_estimates += 1
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage."""
        return {
            "noise_sigma": self.noise_sigma,
            "noise_windows": self.noise_windows,
            "lag_s": self.lag_s,
            "lag_estimates": self.lag_estimates,
            "recent_lags": [round(v, 2) for v in self.recent_lags],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibrator:
        """Restore from storage."""
        noise = data.get("noise_sigma")
        lag = data.get("lag_s")
        return cls(
            noise_sigma=float(noise) if noise is not None else None,
            noise_windows=int(data.get("noise_windows", 0)),
            lag_s=float(lag) if lag is not None else None,
            lag_estimates=int(data.get("lag_estimates", 0)),
            recent_lags=[float(v) for v in data.get("recent_lags", [])][-_LAG_HISTORY:],
        )
