"""Tests for the synthetic load generator and the simulation runner."""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.herdwaechter.sampling import ROLE_GEN, ROLE_GRID, Resampler

from .sim import HouseholdConfig, SensorConfig, build_scenario, run_simulation
from .sim.household import DAY, generate_household, moving_average, pv_potential
from .sim.runner import _ROLE_NAMES, BASE_TS
from .sim.sensors import periodic_reports, simulate_inverter


@pytest.fixture(scope="module")
def two_days() -> tuple:
    return build_scenario(HouseholdConfig(days=2, seed=1), SensorConfig(seed=5))


def test_household_generator(two_days: tuple) -> None:
    house, _ = two_days
    assert house.seconds == 2 * DAY
    assert np.all(house.total >= 0)
    assert house.sessions
    for s in house.sessions:
        assert 1 <= s.zones <= 3
        assert 5 * 60 <= s.end - s.start <= 61 * 60
        assert 30 <= s.heat_up_s <= 480
        assert 20 <= s.period_s <= 70
    names = {name for name, _, _ in house.activities}
    assert "cooking" in names
    assert "kettle" in names
    assert house.active_at(house.sessions[0].start, house.sessions[0].start + 1)
    # cooking adds the zone power on top of the base load
    s = house.sessions[0]
    assert np.max(house.total[int(s.start) : int(s.end)]) > min(house.hob_levels)


def test_microwave_differs_from_cooktop_zones() -> None:
    house = generate_household(HouseholdConfig(days=1, seed=2, big_loads=False))
    assert len(house.hob_levels) == 3
    quiet = generate_household(
        HouseholdConfig(days=1, seed=2, disturbers=False, big_loads=False)
    )
    assert quiet.events.get("kettle") is None


def test_moving_average_and_pv() -> None:
    np.testing.assert_allclose(moving_average(np.ones(10), 3), np.ones(10))
    pv = pv_potential(DAY, seed=1)
    hour = np.arange(DAY) / 3600
    assert np.all(pv[hour < 6] == 0)
    assert pv[(13 * 3600)] > 0


def test_zero_export_inverter() -> None:
    load = np.r_[np.full(300, 300.0), np.full(300, 3000.0)]
    grid, out = simulate_inverter(load, np.zeros(600), SensorConfig())
    assert abs(grid[290]) < 50  # regulated to ~0 W import
    assert out[590] == pytest.approx(800.0)  # inverter limit reached
    assert grid[590] == pytest.approx(2200.0, abs=5)
    # the step is visible at the meter only briefly
    assert grid[300] > 2500
    grid, out = simulate_inverter(load, np.zeros(600), SensorConfig(zero_export=False))
    np.testing.assert_array_equal(grid, load)
    empty = SensorConfig(battery_wh=1.0)
    grid, _ = simulate_inverter(np.full(600, 500.0), np.zeros(600), empty)
    assert grid[-1] == pytest.approx(500.0)  # battery empty: no compensation


def test_periodic_reports_change_only() -> None:
    rng = np.random.default_rng(0)
    signal = np.r_[np.full(100, 5.0), np.full(100, 7.0)]
    unavailable = np.zeros(200, dtype=bool)
    unavailable[150:160] = True
    ts, values = periodic_reports(signal, 3.0, 0.5, rng, unavailable)
    assert np.all(np.diff(ts) > 0)
    finite = values[np.isfinite(values)]
    assert list(dict.fromkeys(finite.tolist())) == [5.0, 7.0]
    assert np.isnan(values).sum() == 1  # one "unavailable" event, no repeats


def test_events_feed_the_resampler(two_days: tuple) -> None:
    _house, events = two_days
    assert np.all(np.diff(events.ts) >= 0)
    assert set(np.unique(events.role).tolist()) == {0, 1, 2}
    # replay one hour and compare with an independent sample-and-hold
    end = 3600.0 * 14
    sel = events.ts <= end
    rs = Resampler(events.n_generation, events.n_subtract)
    for t, r, i, v in zip(
        events.ts[sel],
        events.role[sel],
        events.index[sel],
        events.value[sel],
        strict=True,
    ):
        rs.update(_ROLE_NAMES[int(r)], int(i), BASE_TS + t, None if np.isnan(v) else v)
    window = rs.snapshot(BASE_TS + end, 600)
    grid_sel = sel & (events.role == 0)
    gts, gval = events.ts[grid_sel], events.value[grid_sel]
    seconds = np.arange(end - 599, end + 1)
    idx = np.searchsorted(gts, seconds, side="right") - 1
    expected = gval[idx]
    np.testing.assert_allclose(window.grid, expected, equal_nan=True)
    assert rs.effective_update_interval() == pytest.approx(3.0, abs=0.5)
    assert ROLE_GRID != ROLE_GEN


def test_short_run_produces_metrics(two_days: tuple) -> None:
    house, events = two_days
    result = run_simulation(house, events)
    metrics = result.metrics()
    assert metrics.sessions == len(house.sessions)
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert result.probability.shape == result.is_on.shape
    summary = metrics.as_dict()
    assert {"precision", "recall", "median_latency_min", "false_alarms_per_day"} <= set(
        summary
    )
