"""Profile for radiant glass-ceramic cooktops ("Ceran" type).

Radiant heating elements are switched by an energy regulator: each zone
turns on and off with a fixed power step (typically 1.0-2.5 kW) and a period
of roughly 10-90 s for many minutes. Several active zones superimpose. The
prior below scores exactly that signature with a handful of explainable
rules. All rule weights are logit contributions; the output is squeezed into
``[0.01, 0.86]`` so that the prior alone rarely exceeds 85 %.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..const import PROFILE_CERAN
from ..features import FEATURE_INDEX
from .base import DeviceProfile

FloatArray = npt.NDArray[np.float64]

_FI = FEATURE_INDEX

PERIOD_RANGE_S = (8.0, 120.0)
# radiant zones (incl. inner rings of dual-circuit zones) draw 0.7-3.6 kW
PLAUSIBLE_HEIGHT_W = (700.0, 3600.0)
SUSPICIOUS_HEIGHT_W = 6000.0
# a cooktop alone never raises the load this far above the base load
SUSPICIOUS_PEAK_W = 7000.0
# regulator periods below this are rare (microwave programmes cycle faster)
SHORT_PERIOD_S = 18.0


class CeranProfile(DeviceProfile):
    """Radiant cooktop profile."""

    key: ClassVar[str] = PROFILE_CERAN
    min_level_w: ClassVar[float] = 700.0
    max_level_w: ClassVar[float] = 3600.0

    def rule_scores(self, features: FloatArray) -> dict[str, float]:
        """Return the contribution (logit units) of every rule for one window."""
        f = features
        n_edges = float(f[_FI["n_on"]] + f[_FI["n_off"]])
        n_cycles = float(f[_FI["n_cycles"]])
        n_equal = float(f[_FI["n_equal_cycles"]])
        period = float(f[_FI["period_s"]])
        strength = float(f[_FI["period_strength"]])
        interval_cv = float(f[_FI["interval_cv"]])
        zone_h = float(f[_FI["zone_height"]])
        h_med = zone_h if zone_h > 0 else float(f[_FI["height_median"]])
        h_max = float(f[_FI["height_max"]])
        h_cv = float(f[_FI["height_cv"]])
        duty = float(f[_FI["duty_cycle"]])
        longest = float(f[_FI["longest_block_s"]])
        level_frac = float(f[_FI["level_match_frac"]])
        heatup = float(f[_FI["heatup_cycles"]])
        peak = float(f[_FI["peak_above_base"]])
        heatup_seen = float(f[_FI["zone_heatup_seen"]])
        levels_known = float(f[_FI["levels_known"]])

        scores = {"base": -4.0}

        # Rule 1: repeated on/off cycles of the same height (±15 %).
        if n_equal >= 3:
            scores["equal_cycles"] = 2.6 + 0.25 * min(n_equal - 3.0, 4.0)
        elif n_equal == 2:
            scores["equal_cycles"] = 1.6

        # Rule 1b: continuous heat-up followed by restarts at the same power
        # (energy regulator taking over); kettles and toasters never do this.
        if heatup >= 2:
            scores["heatup_then_cycling"] = 2.5 if interval_cv <= 0.3 else 1.5
        elif heatup >= 1:
            scores["heatup_then_cycling"] = 1.0

        # Rule 1c: a radiant element starts cold and runs continuously
        # first. A zone that cycles although its power never ran continuously
        # during the current activity is unusual for a cooktop (typical for
        # microwave programmes).
        if n_equal >= 3 and heatup < 1 and heatup_seen < 0.5:
            scores["cycling_without_heatup"] = -2.5

        # Rule 2: a regular rhythm in the typical regulator period range.
        if PERIOD_RANGE_S[0] <= period <= PERIOD_RANGE_S[1] and strength > 0.3:
            scores["periodic"] = 2.0 * min(1.0, (strength - 0.3) / 0.3)
            if period < SHORT_PERIOD_S:
                scores["short_period"] = -0.8

        # Rule 3: intervals between same-kind edges of the zone are regular.
        if interval_cv <= 0.1 and n_edges >= 6:
            scores["regular_intervals"] = 1.5
        elif interval_cv <= 0.2:
            scores["regular_intervals"] = 0.8
        elif interval_cv >= 0.6 and n_edges >= 6:
            scores["regular_intervals"] = -0.6

        # Rule 4: step heights typical for heating elements.
        if n_edges > 0:
            if h_med < PLAUSIBLE_HEIGHT_W[0]:
                scores["plausible_height"] = -3.0
            elif h_med > PLAUSIBLE_HEIGHT_W[1]:
                scores["plausible_height"] = -1.5
            else:
                scores["plausible_height"] = 0.4
            if h_max > SUSPICIOUS_HEIGHT_W or peak > SUSPICIOUS_PEAK_W:
                # a modulating large load (instantaneous water heater, wallbox)
                # produces many steps of varying height on a high base load
                scores["very_large_load"] = -4.0 if h_cv > 0.35 else -2.5

        # Rule 5: a plausible duty cycle (neither a single pulse nor always on).
        if 0.08 <= duty <= 0.92:
            scores["duty_cycle"] = 0.3

        # Rule 6: single blocks without repetition (kettle, hair dryer, toaster).
        if n_cycles <= 1 and n_edges <= 3 and heatup < 1:
            scores["single_block"] = -2.0
        if longest >= 600 and n_equal < 2:
            scores["long_block_only"] = -0.8

        # Rule 7: steps match learned power levels of this cooktop. Once the
        # levels are known, regular cycling at a power that matches none of
        # them is most likely another appliance (zones have fixed powers).
        if level_frac > 0:
            scores["learned_levels"] = 1.0 * level_frac - 0.3
        if levels_known >= 2 and n_equal >= 3 and level_frac < 0.2:
            scores["unknown_power_level"] = -1.0
        # A known zone power that restarts after a continuous heat-up is the
        # most specific signature of this cooktop.
        if levels_known >= 1 and heatup >= 1 and level_frac >= 0.6:
            scores["known_zone_restart"] = 1.0

        return scores

    def is_cycling_window(self, features: FloatArray) -> bool:
        """Return True for windows that clearly show regulator cycling."""
        f = features
        return bool(
            f[_FI["n_equal_cycles"]] >= 3
            and f[_FI["period_strength"]] >= 0.45
            and PERIOD_RANGE_S[0] <= f[_FI["period_s"]] <= PERIOD_RANGE_S[1]
        )
