"""Tests for the profile registry and the radiant cooktop prior."""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.herdwaechter.features import FEATURE_INDEX as FI, N_FEATURES
from custom_components.herdwaechter.profiles import PROFILES, get_profile


def _vec(**values: float) -> np.ndarray:
    f = np.zeros(N_FEATURES)
    for name, value in values.items():
        f[FI[name]] = value
    return f


def test_registry() -> None:
    assert "ceran" in PROFILES
    assert get_profile("ceran").key == "ceran"
    with pytest.raises(ValueError, match="unknown profile"):
        get_profile("induction")


def test_prior_rules_and_range() -> None:
    prof = get_profile("ceran")
    quiet = prof.prior_probability(_vec(zone_heatup_seen=1))
    cooking = prof.prior_probability(
        _vec(
            n_on=10,
            n_off=10,
            n_cycles=9,
            n_equal_cycles=9,
            heatup_cycles=8,
            period_s=40,
            period_strength=0.8,
            interval_cv=0.05,
            zone_height=1800,
            height_median=1800,
            duty_cycle=0.5,
            zone_heatup_seen=1,
        )
    )
    kettle = prof.prior_probability(
        _vec(n_on=1, n_off=1, n_cycles=1, zone_height=2100, longest_block_s=180)
    )
    assert quiet < 0.05
    assert kettle < 0.05
    assert 0.75 < cooking <= 0.86  # cautious: the prior alone stays below 86 %
    water_heater = prof.rule_scores(
        _vec(n_on=8, n_off=8, zone_height=1000, height_cv=0.6, peak_above_base=9000)
    )
    assert water_heater["very_large_load"] == -4.0
    washer_artifact = prof.rule_scores(_vec(n_on=4, n_off=4, zone_height=550))
    assert washer_artifact["plausible_height"] < 0
    microwave = prof.rule_scores(
        _vec(n_on=10, n_off=10, n_equal_cycles=8, period_s=14, period_strength=0.8)
    )
    assert microwave["cycling_without_heatup"] < 0
    assert microwave["short_period"] < 0
    unknown = prof.rule_scores(
        _vec(n_on=8, n_off=8, n_equal_cycles=6, levels_known=3, zone_heatup_seen=1)
    )
    assert unknown["unknown_power_level"] < 0


def test_model_inputs_shape_and_errors() -> None:
    prof = get_profile("ceran")
    x = prof.model_inputs(np.zeros((3, N_FEATURES)), np.array([1200.0]))
    assert x.shape == (3, prof.n_model_inputs)
    assert np.all(np.isfinite(x))
    with pytest.raises(ValueError, match="features"):
        prof.model_inputs(np.zeros((1, N_FEATURES - 1)))
    assert prof.prior_offsets(np.zeros((2, N_FEATURES))).shape == (2,)
    assert not prof.is_cycling_window(np.zeros(N_FEATURES))
