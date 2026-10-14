"""Device profile interface.

A profile bundles everything that is specific to one kind of appliance:
the explainable prior (rules that work from day one), the plausible power
range used for learning power levels, and the transformation of the raw
feature vector into inputs for the personal model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..features import FEATURE_INDEX, LEVEL_TOLERANCE, N_FEATURES, level_distance

FloatArray = npt.NDArray[np.float64]


def sigmoid(z: FloatArray) -> FloatArray:
    """Numerically safe logistic function."""
    out: FloatArray = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    return out


class DeviceProfile(ABC):
    """Base class for appliance profiles."""

    key: ClassVar[str]
    # plausible single heating element power in W (used to accept levels)
    min_level_w: ClassVar[float] = 500.0
    max_level_w: ClassVar[float] = 4000.0
    # output range of the prior probability
    prior_floor: ClassVar[float] = 0.01
    prior_span: ClassVar[float] = 0.85

    @abstractmethod
    def rule_scores(self, features: FloatArray) -> dict[str, float]:
        """Return the contribution (logit units) of every rule for one window."""

    def prior_logit(self, features: FloatArray) -> float:
        """Return the summed rule score of one feature vector."""
        return float(sum(self.rule_scores(features).values()))

    def prior_logits(self, matrix: FloatArray) -> FloatArray:
        """Return the summed rule score for each row."""
        x = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
        return np.fromiter(
            (self.prior_logit(row) for row in x), dtype=np.float64, count=x.shape[0]
        )

    def prior_probabilities(self, matrix: FloatArray) -> FloatArray:
        """Return the cautious prior probability for each row."""
        out: FloatArray = self.prior_floor + self.prior_span * sigmoid(
            self.prior_logits(matrix)
        )
        return out

    def prior_probability(self, features: FloatArray) -> float:
        """Return the cautious prior probability for one feature vector."""
        z = min(40.0, max(-40.0, self.prior_logit(features)))
        return self.prior_floor + self.prior_span / (1.0 + math.exp(-z))

    def is_cycling_window(self, features: FloatArray) -> bool:
        """Return True if the window shows the typical cycling pattern."""
        return False

    def prior_offsets(self, matrix: FloatArray) -> FloatArray:
        """Return the (clipped) prior logits used as model offset."""
        out: FloatArray = np.clip(self.prior_logits(matrix), -12.0, 8.0)
        return out

    def model_inputs(
        self, matrix: FloatArray, levels: FloatArray | None = None
    ) -> FloatArray:
        """Transform raw feature vectors (N x F) into model inputs.

        Counts and durations are log-compressed and powers scaled to kW. The
        prior logit is not an input but the model's offset (see
        :meth:`prior_offsets`), so the model learns corrections of the prior.

        ``levels`` are the currently learned power levels. Whether the zone of
        a window matches one of them is recomputed here (from the stored zone
        height) instead of being frozen at evaluation time, so that windows
        stored before a level was learned are judged consistently.
        """
        x = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
        lv = np.empty(0) if levels is None else np.asarray(levels, dtype=np.float64)
        if x.shape[1] != N_FEATURES:
            msg = f"expected {N_FEATURES} features, got {x.shape[1]}"
            raise ValueError(msg)
        fi = FEATURE_INDEX
        cols = [
            np.log1p(x[:, fi["n_on"]]),
            np.log1p(x[:, fi["n_off"]]),
            x[:, fi["height_median"]] / 1000.0,
            np.minimum(x[:, fi["height_cv"]], 2.0),
            x[:, fi["level_match_frac"]],
            np.minimum(x[:, fi["interval_cv"]], 2.0),
            x[:, fi["period_s"]] / 60.0,
            x[:, fi["period_strength"]],
            np.log1p(x[:, fi["on_median_s"]]),
            np.log1p(x[:, fi["off_median_s"]]),
            x[:, fi["duty_cycle"]],
            np.log1p(x[:, fi["n_cycles"]]),
            np.clip(x[:, fi["baseline_rise"]] / 1000.0, -3.0, 6.0),
            np.log1p(x[:, fi["longest_block_s"]] / 60.0),
            x[:, fi["tod_sin"]],
            x[:, fi["tod_cos"]],
            np.log1p(x[:, fi["height_max"]] / 1000.0),
            np.log1p(np.minimum(x[:, fi["activity_minutes"]], 90.0)),
            np.log1p(x[:, fi["n_equal_cycles"]]),
            x[:, fi["zone_height"]] / 1000.0,
            np.log1p(x[:, fi["heatup_cycles"]]),
            np.log1p(np.maximum(x[:, fi["peak_above_base"]], 0.0) / 1000.0),
            x[:, fi["zone_heatup_seen"]],
            (level_distance(x[:, fi["zone_height"]], lv) <= LEVEL_TOLERANCE).astype(
                np.float64
            ),
            level_distance(x[:, fi["zone_height"]], lv),
        ]
        return np.column_stack(cols)

    @property
    def n_model_inputs(self) -> int:
        """Return the number of model inputs."""
        return int(self.model_inputs(np.zeros((1, N_FEATURES))).shape[1])
