"""Tests for noise floor, edge threshold and lag estimation."""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.herdwaechter.calibration import (
    CALIBRATION_LAG_ESTIMATES,
    CALIBRATION_NOISE_WINDOWS,
    Calibrator,
    estimate_lag,
    quiet_noise_sigma,
    robust_sigma,
    span_diff,
    threshold_from_sigma,
)
from custom_components.herdwaechter.const import (
    DEFAULT_EDGE_THRESHOLD,
    MAX_EDGE_THRESHOLD,
    MIN_EDGE_THRESHOLD,
)

from .helpers import cycling


def _zero_export(lag_s: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Grid and generation signals of a zero-export system.

    The generation sensor reports ``lag_s`` seconds later than the grid
    sensor and both sample-and-hold with different cadences.
    """
    rng = np.random.default_rng(seed)
    n = 600
    load = cycling(n, power=1500, period=35, duty=0.5, seed=seed) + rng.normal(0, 5, n)
    inv = np.zeros(n)
    for t in range(2, n):
        target = min(800.0, max(0.0, load[t - 2] - 20))
        step = np.clip(target - inv[t - 1], -150, 150)
        inv[t] = inv[t - 1] + step
    grid_true = load - inv

    def hold(sig: np.ndarray, every: int, phase: int, delay: int) -> np.ndarray:
        out = np.empty(n)
        last = sig[0]
        for t in range(n):
            if (t - phase) % every == 0 and t >= delay:
                last = sig[t - delay]
            out[t] = last
        return out

    grid = hold(grid_true, 3, 1, max(0, -lag_s))
    gen = hold(inv, 3, 2, max(0, lag_s))
    return grid, gen


def test_span_diff_and_robust_sigma() -> None:
    x = np.arange(10.0)
    d = span_diff(x, 3)
    assert np.all(np.isnan(d[:3]))
    assert np.all(d[3:] == 3)
    assert np.all(np.isnan(span_diff(np.arange(2.0), 3)))
    rng = np.random.default_rng(1)
    assert robust_sigma(rng.normal(0, 30, 5000)) == pytest.approx(30, rel=0.1)
    assert robust_sigma(np.ones(5)) is None


def test_quiet_noise_and_threshold() -> None:
    rng = np.random.default_rng(2)
    quiet = 200 + rng.normal(0, 20, 600)
    sigma = quiet_noise_sigma(quiet, 3)
    assert sigma is not None
    assert sigma == pytest.approx(20 * np.sqrt(2), rel=0.2)
    busy = cycling(600, period=20)
    assert quiet_noise_sigma(busy, 3) is None
    assert quiet_noise_sigma(np.full(10, 1.0), 3) is None
    assert threshold_from_sigma(None) == DEFAULT_EDGE_THRESHOLD
    assert threshold_from_sigma(1.0) == MIN_EDGE_THRESHOLD
    assert threshold_from_sigma(1000.0) == MAX_EDGE_THRESHOLD
    assert MIN_EDGE_THRESHOLD < threshold_from_sigma(70.0) < MAX_EDGE_THRESHOLD


@pytest.mark.parametrize("lag", [0, 2, 5, 9, -3])
def test_cross_correlation_finds_known_lag(lag: int) -> None:
    grid, gen = _zero_export(lag)
    result = estimate_lag(grid, gen)
    assert result is not None
    assert result[0] == pytest.approx(lag, abs=1.0)


def test_estimate_lag_rejects_unusable_windows() -> None:
    grid, gen = _zero_export(2)
    assert estimate_lag(grid[:30], gen[:30]) is None
    assert estimate_lag(grid, np.zeros_like(gen)) is None
    assert estimate_lag(grid, gen[:-1]) is None
    holes = grid.copy()
    holes[::2] = np.nan
    assert estimate_lag(holes, gen) is None


def test_calibrator_converges_and_rejects_outliers() -> None:
    cal = Calibrator()
    assert cal.edge_threshold == DEFAULT_EDGE_THRESHOLD
    assert cal.lag_int == 0
    rng = np.random.default_rng(3)
    for _ in range(CALIBRATION_NOISE_WINDOWS):
        assert cal.update_noise(200 + rng.normal(0, 10, 600), 3)
    assert not cal.update_noise(cycling(600, period=20), 3)
    assert cal.is_calibrated(has_generation=False)
    assert not cal.is_calibrated(has_generation=True)
    assert cal.edge_threshold == MIN_EDGE_THRESHOLD

    grid, gen = _zero_export(3)
    for _ in range(CALIBRATION_LAG_ESTIMATES):
        assert cal.update_lag(grid, gen)
    assert cal.lag_int == 3
    assert cal.is_calibrated(has_generation=True)
    # a single outlier is ignored ...
    far_grid, far_gen = _zero_export(12)
    assert not cal.update_lag(far_grid, far_gen)
    assert cal.lag_int == 3
    # ... but once most recent estimates agree on a new offset, it is followed
    for _ in range(10):
        cal.update_lag(far_grid, far_gen)
    assert cal.lag_int == 12
    assert not cal.update_lag(grid, np.zeros_like(gen))

    restored = Calibrator.from_dict(cal.to_dict())
    assert restored.lag_s == pytest.approx(cal.lag_s)
    assert len(restored.recent_lags) == len(cal.recent_lags)
    assert restored.noise_windows == cal.noise_windows
    empty = Calibrator.from_dict({})
    assert empty.noise_sigma is None
    assert empty.lag_s is None
