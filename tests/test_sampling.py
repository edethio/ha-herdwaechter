"""Tests for the 1 s sample-and-hold raster."""

from __future__ import annotations

import math

import numpy as np
import pytest

from custom_components.herdwaechter.sampling import (
    ROLE_GEN,
    ROLE_GRID,
    ROLE_SUB,
    Resampler,
    RingBuffer,
)

T0 = 1_800_000_000.0


def test_ring_buffer_wraps_and_pads() -> None:
    buf = RingBuffer(5)
    assert np.all(np.isnan(buf.last(3)))
    buf.push_constant(1.0, 3)
    np.testing.assert_array_equal(buf.last(4)[1:], [1.0, 1.0, 1.0])
    assert math.isnan(buf.last(4)[0])
    buf.push_constant(2.0, 4)  # wraps around
    np.testing.assert_array_equal(buf.last(5), [1.0, 2.0, 2.0, 2.0, 2.0])
    buf.push_constant(3.0, 10)  # longer than capacity
    np.testing.assert_array_equal(buf.last(5), [3.0] * 5)
    buf.push_constant(4.0, 0)
    assert len(buf) == 5


def test_sample_and_hold_on_one_second_raster() -> None:
    rs = Resampler(0, 0)
    rs.update(ROLE_GRID, 0, T0, 100.0)
    rs.update(ROLE_GRID, 0, T0 + 3.0, 200.0)
    win = rs.snapshot(T0 + 5.2, 6)
    np.testing.assert_array_equal(win.grid, [100, 100, 100, 200, 200, 200])
    assert win.end_ts == T0 + 5
    assert win.length == 6


def test_unavailable_creates_gap_not_zero() -> None:
    rs = Resampler(0, 0)
    rs.update(ROLE_GRID, 0, T0, 500.0)
    rs.update(ROLE_GRID, 0, T0 + 2, None)
    rs.update(ROLE_GRID, 0, T0 + 5, 600.0)
    win = rs.snapshot(T0 + 6, 7)
    assert win.grid[0] == 500.0
    assert np.all(np.isnan(win.grid[2:5]))
    assert win.grid[-1] == 600.0
    assert not np.any(win.grid == 0.0)


def test_non_finite_value_is_a_gap() -> None:
    rs = Resampler(0, 0)
    rs.update(ROLE_GRID, 0, T0, float("nan"))
    rs.update(ROLE_GRID, 0, T0 + 1, float("inf"))
    win = rs.snapshot(T0 + 3, 3)
    assert np.all(np.isnan(win.grid))


def test_grid_value_goes_stale_after_max_hold() -> None:
    rs = Resampler(0, 0, max_grid_hold_s=10)
    rs.update(ROLE_GRID, 0, T0, 300.0)
    win = rs.snapshot(T0 + 20, 21)
    assert np.all(win.grid[:11] == 300.0)
    assert np.all(np.isnan(win.grid[11:]))


def test_auxiliary_sensors_sum_and_hold_when_unavailable() -> None:
    rs = Resampler(2, 1, aux_hold_s=5)
    rs.update(ROLE_GRID, 0, T0, 0.0)
    rs.update(ROLE_GEN, 0, T0, 300.0)
    rs.update(ROLE_GEN, 1, T0, 200.0)
    rs.update(ROLE_SUB, 0, T0, 50.0)
    rs.update(ROLE_GEN, 0, T0 + 2, None)
    win = rs.snapshot(T0 + 10, 11)
    assert win.gen[0] == 500.0
    # unavailable sensor keeps its last value for the hold time ...
    assert np.all(win.gen[2:7] == 500.0)
    assert not np.any(win.aux_missing[2:7])
    # ... and then contributes nothing, flagged as missing
    assert np.all(win.gen[7:] == 200.0)
    assert np.all(win.aux_missing[7:])
    assert np.all(win.sub == 50.0)


def test_unknown_role_raises() -> None:
    rs = Resampler(0, 0)
    with pytest.raises(ValueError, match="unknown role"):
        rs.update("battery", 0, T0, 1.0)


def test_effective_update_interval_and_reports() -> None:
    rs = Resampler(1, 0)
    assert rs.effective_update_interval() is None
    assert rs.seconds_since_grid_update(T0) is None
    for i in range(10):
        rs.update(ROLE_GRID, 0, T0 + 3 * i, float(i))
        rs.update(ROLE_GEN, 0, T0 + 5 * i, float(i))
    assert rs.effective_update_interval() == pytest.approx(3.0)
    assert rs.generation_update_interval() == pytest.approx(5.0)
    # unchanged values reported via state_reported events still count
    for i in range(10, 20):
        rs.report(ROLE_GRID, 0, T0 + 3 * i)
    assert rs.effective_update_interval() == pytest.approx(3.0)
    assert rs.seconds_since_grid_update(T0 + 60) == pytest.approx(3.0)


def test_report_without_value_is_ignored() -> None:
    rs = Resampler(0, 0)
    rs.update(ROLE_GRID, 0, T0, None)
    rs.report(ROLE_GRID, 0, T0 + 1)
    assert rs.seconds_since_grid_update(T0 + 1) == pytest.approx(1.0)
    assert rs.effective_update_interval() is None


def test_long_gap_between_updates_fills_buffer() -> None:
    rs = Resampler(0, 0, capacity_s=60, max_grid_hold_s=10_000)
    rs.update(ROLE_GRID, 0, T0, 10.0)
    rs.update(ROLE_GRID, 0, T0 + 500, 20.0)
    win = rs.snapshot(T0 + 501, 60)
    assert win.grid[-1] == 20.0
    assert np.all(win.grid[:-2] == 10.0)
