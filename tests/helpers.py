"""Synthetic signal helpers shared by the unit tests (no Home Assistant)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def cycling(
    n: int,
    power: float = 1800.0,
    period: float = 40.0,
    duty: float = 0.5,
    start: int = 0,
    heat_up: int = 0,
    base: float = 200.0,
    noise: float = 8.0,
    seed: int = 0,
    jitter: float = 0.0,
) -> FloatArray:
    """Return a base load with one cycling heating element."""
    rng = np.random.default_rng(seed)
    x = np.full(n, base) + rng.normal(0, noise, n)
    x[start : start + heat_up] += power
    t = float(start + heat_up)
    while t < n:
        p = period * (1 + rng.uniform(-jitter, jitter))
        t += p * (1 - duty)
        on = round(p * duty)
        a = round(t)
        x[a : a + on] += power
        t += p * duty
    return x


def block(
    n: int,
    power: float,
    start: int,
    duration: int,
    base: float = 200.0,
    noise: float = 8.0,
    seed: int = 0,
) -> FloatArray:
    """Return a base load with one single block (kettle-like)."""
    rng = np.random.default_rng(seed)
    x = np.full(n, base) + rng.normal(0, noise, n)
    x[start : start + duration] += power
    return x
