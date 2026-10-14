"""End-to-end learning test: 30 simulated days, 3 s updates, zero export.

The virtual user labels only 50 % of the cooking sessions and presses the
false alarm button for 50 % of the false alarms. The trained state (last
seven days) must meet the minimum requirements, the prior alone must already
be useful from day 0, and learning must improve the result over time.

Takes a few minutes; deselect with ``-m "not slow"``.
"""

from __future__ import annotations

import pytest

from .sim import (
    HouseholdConfig,
    RunResult,
    SensorConfig,
    build_scenario,
    run_simulation,
)

pytestmark = pytest.mark.slow

DAYS = 30


@pytest.fixture(scope="module")
def runs() -> tuple[RunResult, RunResult]:
    house, events = build_scenario(
        HouseholdConfig(days=DAYS, seed=7), SensorConfig(seed=11, grid_interval_s=3.0)
    )
    learned = run_simulation(house, events, learning=True)
    prior = run_simulation(house, events, learning=False)
    return learned, prior


def test_trained_state_meets_requirements(runs: tuple[RunResult, RunResult]) -> None:
    learned, _ = runs
    final = learned.metrics(DAYS - 7, DAYS)
    assert final.sessions >= 5
    assert final.precision >= 0.90
    assert final.recall >= 0.90
    assert final.median_latency_s <= 4 * 60
    assert final.false_alarms_per_day <= 0.2


def test_prior_is_useful_from_day_zero(runs: tuple[RunResult, RunResult]) -> None:
    learned, prior = runs
    assert prior.metrics().recall >= 0.6
    assert prior.metrics(0, 1).recall >= 0.6
    assert learned.metrics(0, 1).recall >= 0.6


def test_learning_improves_over_time(runs: tuple[RunResult, RunResult]) -> None:
    learned, prior = runs
    week1 = learned.metrics(0, 7)
    final = learned.metrics(DAYS - 7, DAYS)
    prior_final = prior.metrics(DAYS - 7, DAYS)
    # fewer (or equally few) false alarms and faster detection than in week 1
    assert final.false_alarms <= week1.false_alarms
    assert final.median_latency_s <= week1.median_latency_s
    # the trained state is at least as good as the prior alone
    assert final.false_alarms <= prior_final.false_alarms
    assert final.precision >= prior_final.precision
    assert final.recall >= prior_final.recall
    engine = learned.engine
    assert engine.quality.known
    # the learned power levels are the cooktop's zones
    levels = sorted(engine.clusterer.levels().tolist())
    for truth in learned.house.hob_levels:
        assert any(abs(level - truth) <= 0.05 * truth for level in levels)
    assert engine.calibrator.lag_s is not None
