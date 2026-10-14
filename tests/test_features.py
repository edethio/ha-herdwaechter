"""Tests for edge detection, pairing, periodicity and the feature vector."""

from __future__ import annotations

import math

import numpy as np
import pytest

from custom_components.herdwaechter.features import (
    FEATURE_INDEX as FI,
    N_FEATURES,
    Cycles,
    Edges,
    analyse_window,
    detect_edges,
    dominant_period,
    group_by_height,
    interval_cv,
    largest_equal_group,
    match_levels,
    off_durations,
    pair_edges,
    reconstruct_residual,
    remove_generation_artifacts,
    robust_cv,
    rolling_median,
    still_running,
    time_of_day,
    zone_stats,
)

from .helpers import block, cycling


def test_reconstruct_residual_shifts_generation() -> None:
    grid = np.array([10.0, 20, 30, 40, 50])
    gen = np.array([1.0, 2, 3, 4, 5])
    sub = np.ones(5)
    np.testing.assert_array_equal(
        reconstruct_residual(grid, gen, sub, 0), grid + gen - sub
    )
    res = reconstruct_residual(grid, gen, sub, 2)
    np.testing.assert_array_equal(res[:3], [10 + 3 - 1, 20 + 4 - 1, 30 + 5 - 1])
    assert np.all(np.isnan(res[3:]))
    res = reconstruct_residual(grid, gen, sub, -1)
    assert math.isnan(res[0])
    assert res[1] == 20 + 1 - 1


def test_rolling_median_handles_borders_and_gaps() -> None:
    x = np.array([1.0, 1, 9, 1, 1, np.nan, 1, 1])
    out = rolling_median(x, 3)
    assert math.isnan(out[0])
    assert out[2] == 1.0  # single spike removed
    assert math.isnan(out[5])
    np.testing.assert_array_equal(rolling_median(x, 1), x)
    even = rolling_median(np.arange(10.0), 4)
    assert np.isfinite(even[3])


def test_detect_edges_finds_step_time_and_height() -> None:
    x = np.r_[np.full(50, 200.0), np.full(50, 2000.0), np.full(50, 200.0)]
    edges = detect_edges(x, 500.0, span=3, smooth=3)
    assert edges.count == 2
    np.testing.assert_allclose(edges.idx, [50, 100], atol=1)
    np.testing.assert_allclose(edges.height, [1800, -1800], rtol=0.01)


def test_detect_edges_ignores_short_transients_and_small_steps() -> None:
    x = np.full(100, 200.0)
    x[40:42] -= 800  # 2 s regulation dip
    x[70:] += 300  # below threshold
    assert detect_edges(x, 500.0, span=3, smooth=5).count == 0
    assert detect_edges(np.full(2, 1.0), 500.0, span=3).count == 0


def test_pair_edges_matches_similar_heights() -> None:
    edges = Edges(
        np.array([10.0, 20, 30, 40, 60]),
        np.array([1800.0, 1200, -1250, -1790, -3000]),
    )
    cycles, open_on = pair_edges(edges)
    assert cycles.count == 2
    np.testing.assert_allclose(cycles.on, [10, 20])
    np.testing.assert_allclose(cycles.off, [40, 30])
    assert open_on.shape[0] == 0
    # outside ±20 % → not paired
    cycles, open_on = pair_edges(Edges(np.array([0.0, 5]), np.array([1000.0, -1500])))
    assert cycles.count == 0
    assert open_on.shape[0] == 1


def test_grouping_helpers() -> None:
    vals = np.array([1000.0, 1050, 1800, 1850, 2300])
    groups = group_by_height(vals)
    assert groups[0] == groups[1] != groups[2] == groups[3] != groups[4]
    assert group_by_height(np.empty(0)).shape[0] == 0
    assert largest_equal_group(np.array([1800.0, 1810, 1790, 1200])) == 3
    assert largest_equal_group(np.empty(0)) == 0


def test_robust_interval_cv_ignores_heat_up_block() -> None:
    # heat-up block of 200 s followed by regular 40 s cycles
    ups = [0.0, 220, 260, 300]
    downs = [200.0, 240, 280, 320]
    edges = Edges(
        np.array(sorted(ups + downs)),
        np.array([1800.0 if t in ups else -1800.0 for t in sorted(ups + downs)]),
    )
    assert interval_cv(edges) < 0.15
    assert interval_cv(Edges(np.array([1.0]), np.array([1.0]))) == 2.0
    assert robust_cv(np.array([1.0])) == 2.0
    assert robust_cv(np.array([0.0, 0.0])) == 2.0


def test_dominant_period_of_cycling_and_block() -> None:
    x = cycling(600, period=40, duty=0.5, jitter=0.05)
    edges = detect_edges(x, 500.0)
    period, strength = dominant_period(edges, 600)
    assert period == pytest.approx(40, abs=3)
    assert strength > 0.6
    kettle = detect_edges(block(600, 2000, 100, 180), 500.0)
    assert dominant_period(kettle, 600) == (0.0, 0.0)


def test_zone_stats_separates_superimposed_zones() -> None:
    x = cycling(600, power=1200, period=30, duty=0.5, seed=1) + cycling(
        600, power=2300, period=55, duty=0.4, base=0, noise=0, seed=2
    )
    edges = detect_edges(x, 500.0)
    cycles, _ = pair_edges(edges)
    zone = zone_stats(edges, cycles, 600)
    assert zone.height in (pytest.approx(1200, rel=0.1), pytest.approx(2300, rel=0.1))
    assert zone.strength > 0.5
    assert zone.interval_cv < 0.3
    assert zone_stats(Edges.empty(), cycles, 600).n_edges == 0


def test_off_durations_and_still_running() -> None:
    cycles = Cycles(np.array([0.0, 40]), np.array([20.0, 60]), np.array([1800.0, 1800]))
    np.testing.assert_allclose(off_durations(cycles), [20.0])
    assert off_durations(Cycles(np.empty(0), np.empty(0), np.empty(0))).shape[0] == 0
    x = block(600, 2000, 400, 300)
    edges = detect_edges(x, 500.0)
    _, open_on = pair_edges(edges)
    assert still_running(x, open_on, edges).shape[0] == 1
    assert still_running(x, np.empty(0), edges).shape[0] == 0
    gaps = np.full(600, np.nan)
    assert still_running(gaps, open_on, edges).shape[0] == 0


def test_match_levels_and_time_of_day() -> None:
    levels = np.array([1200.0, 1800.0])
    np.testing.assert_array_equal(
        match_levels(np.array([1190.0, -1850.0, 2500.0]), levels), [True, True, False]
    )
    assert not np.any(match_levels(np.array([1200.0]), np.empty(0)))
    s, c = time_of_day(6.0)
    assert s == pytest.approx(1.0)
    assert c == pytest.approx(0.0, abs=1e-9)


def test_analyse_window_cycling_vs_kettle() -> None:
    cook = cycling(1200, power=1800, period=40, duty=0.5, start=700, heat_up=120)
    a = analyse_window(cook, 600, 500.0, 3, 3, np.array([1800.0]), 12.0)
    f = a.features
    assert f.shape == (N_FEATURES,)
    assert f[FI["n_cycles"]] >= 8
    assert f[FI["n_equal_cycles"]] >= 8
    assert f[FI["period_s"]] == pytest.approx(40, abs=3)
    assert f[FI["level_match_frac"]] == pytest.approx(1.0)
    assert f[FI["duty_cycle"]] == pytest.approx(0.5, abs=0.15)
    assert f[FI["baseline_rise"]] > 500
    assert f[FI["longest_block_s"]] >= 100
    assert f[FI["zone_height"]] == pytest.approx(1800, rel=0.05)

    kettle = block(1200, 2100, 900, 180)
    k = analyse_window(kettle, 600, 500.0, 3, 3, np.empty(0), 8.0)
    assert k.features[FI["n_cycles"]] == 1
    assert k.features[FI["period_strength"]] == 0.0
    assert k.features[FI["longest_block_s"]] == pytest.approx(180, abs=3)

    quiet = analyse_window(np.full(1200, 150.0), 600, 500.0, 3, 3, np.empty(0), 3.0)
    assert quiet.edges.count == 0
    assert quiet.features[FI["n_on"]] == 0


def test_generation_artifacts_are_removed() -> None:
    # true load: one 2000 W step at t=100. Residual shows an 800 W dip from
    # t=103 to t=107 because the inverter report arrives late.
    residual = np.full(300, 200.0)
    residual[100:] += 2000
    residual[103:108] -= 800
    gen = np.zeros(300)
    gen[108:] = 800.0  # late generation report closes the dip
    edges = detect_edges(residual, 500.0, span=3, smooth=3)
    assert edges.count == 3
    cleaned = remove_generation_artifacts(edges, gen, 3, 3, max_gap_s=16)
    assert cleaned.count == 1
    assert cleaned.height[0] == pytest.approx(2000, rel=0.1)
    # nothing to do without generation data
    assert remove_generation_artifacts(edges, np.empty(0), 3, 3, 16).count == 3
    nan_gen = np.full(300, np.nan)
    assert remove_generation_artifacts(edges, nan_gen, 3, 3, 16).count == 3
