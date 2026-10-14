"""Detection engine: ties calibration, features, prior, model and learning.

The engine is pure Python + numpy and has no Home Assistant dependency. The
coordinator feeds it with raster windows and label state; the simulation in
``tests/sim`` drives exactly the same code.

All methods are synchronous and may run in an executor thread. The caller
must serialise calls (the coordinator uses an ``asyncio.Lock``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from .calibration import Calibrator
from .const import (
    CONSECUTIVE_EVALUATIONS,
    DEFAULT_PROFILE,
    MAX_LAG_SECONDS,
    MIN_EDGE_THRESHOLD,
    MIN_VALID_FRACTION,
    NEUTRAL_EVALUATIONS_AFTER_LABEL,
    NO_CYCLE_TIMEOUT_SECONDS,
    QUESTION_P_HIGH,
    QUESTION_P_LOW,
    QUESTION_TIMEOUT_SECONDS,
    STATE_CALIBRATING,
    STATE_INSUFFICIENT_DATA,
    STATE_LEARNING,
    STATE_PRIOR_ONLY,
    STATE_TRAINED,
    UPDATE_INTERVAL_BAD,
    UPDATE_INTERVAL_OK,
)
from .features import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    N_FEATURES,
    PAIR_TOLERANCE,
    WindowAnalysis,
    analyse_window,
    reconstruct_residual,
)
from .learning import (
    ANSWER_WEIGHT,
    MAX_MODEL_WEIGHT,
    STRONG_NEGATIVE_WEIGHT,
    LabelDataset,
    LabelPolicy,
    LabelSource,
    PowerLevelClusterer,
    QualityReport,
    evaluate_holdout,
    maturity,
    mixing_weight,
)
from .model import OnlineLogisticRegression
from .profiles import DeviceProfile, get_profile
from .sampling import WindowData

FloatArray = npt.NDArray[np.float64]

ENGINE_VERSION = 1
HISTORY_SECONDS = 24 * 3600
ACTIVITY_GAP_SECONDS = 10 * 60
RETRAIN_EVERY_LABELS = 25
RETRAIN_INTERVAL_SECONDS = 6 * 3600
REPLAY_BATCH = 15
FALSE_ALARM_LOOKBACK_SECONDS = 60 * 60
FALSE_ALARM_SUPPRESS_SECONDS = 30 * 60
AGE_HALF_LIFE_DAYS = 180.0
PERIODIC_LEVEL_WEIGHT = 0.3
EDGE_DEDUP_SECONDS = 2.0
# a continuous block of at least this length counts as a heat-up phase
EPISODE_HEATUP_MIN_S = 30.0
# activities starting this close to the first analysed window may have begun
# earlier (e.g. after a restart); their heat-up is treated as unknown
UNOBSERVED_START_S = 60.0
TRAINED_MATURITY = 0.6
# extra evaluations (see EvaluationScheduler)
EXTRA_EVAL_SETTLE_S = 8.0
EXTRA_EVAL_MIN_SPACING_S = 10.0
CONFIRM_DELAY_S = 10.0
# the model may raise p only where the prior is at least this convinced
UPWARD_GATE_LOW = 0.75
UPWARD_GATE_HIGH = 0.85


def edge_parameters(
    update_interval: float | None, has_generation: bool
) -> tuple[int, int, int]:
    """Return ``(span, smooth, period_smooth)`` for edge detection.

    ``span``: number of raster steps a step may take to complete (3, or more
    for slowly updating sensors). ``smooth``: rolling median width that
    suppresses short regulation transients of zero-export inverters.
    ``period_smooth``: jitter tolerance of the autocorrelation, which grows
    with the timing uncertainty of the sensor updates.
    """
    interval = 3.0 if update_interval is None else update_interval
    span = int(np.clip(math.ceil(interval) + 1, 3, 12))
    smooth = 5 if has_generation or interval >= 3 else 3
    period_smooth = int(np.clip(2 * math.ceil(interval) + 3, 5, 15))
    return span, smooth, period_smooth


def effective_weight(weight: float, p_prior: float, p_model: float) -> float:
    """Return the model weight used for one window (positive-unlabelled safe).

    The model may always lower the probability: it learns that from real
    negative labels (false alarm button, "no" answers). Raising it is only
    trusted where the prior itself sees a cooktop signature, because active
    but unlabelled windows (kettle, hair dryer, ...) are never used as
    negatives, so the model has no counter examples there and is biased
    towards "on".
    """
    if p_model <= p_prior:
        return weight
    gate = (p_prior - UPWARD_GATE_LOW) / (UPWARD_GATE_HIGH - UPWARD_GATE_LOW)
    return weight * float(np.clip(gate, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Static configuration of one engine instance."""

    profile: str = DEFAULT_PROFILE
    has_generation: bool = False
    on_threshold: float = 0.75
    off_threshold: float = 0.35
    eval_interval_s: float = 30.0
    window_s: int = 600
    max_label_s: float = 7200.0
    questions_enabled: bool = False
    max_questions_per_day: int = 3

    @property
    def buffer_seconds(self) -> int:
        """Return how many raster seconds one evaluation needs."""
        return 2 * self.window_s + MAX_LAG_SECONDS + 30


@dataclass(frozen=True, slots=True)
class EvaluationInput:
    """Everything one evaluation needs from the outside world."""

    now: float
    window: WindowData
    local_hour: float
    local_day: int
    update_interval_s: float | None = None
    generation_interval_s: float | None = None
    label_active: bool = False
    label_group: int = 0
    label_source: LabelSource = LabelSource.SWITCH


@dataclass(frozen=True, slots=True)
class Question:
    """An active question to the user."""

    question_id: str
    created: float
    probability: float

    @property
    def expires(self) -> float:
        """Return the expiry timestamp."""
        return self.created + QUESTION_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Output of one evaluation (immutable snapshot for the entities)."""

    evaluated: bool
    probability: float | None
    prior_probability: float | None
    model_probability: float | None
    model_weight: float
    confidence: float
    is_on: bool
    learning_state: str
    residual_w: float | None
    detected_period_s: float | None
    learned_levels: list[float]
    last_cycle: dict[str, Any] | None
    edge_threshold: float
    lag_s: float | None
    update_interval_s: float | None
    valid_fraction: float
    counts: dict[str, int]
    quality: dict[str, Any]
    question: Question | None = None

    @classmethod
    def initial(cls) -> EvaluationResult:
        """Return the state before the first evaluation."""
        return cls(
            evaluated=False,
            probability=None,
            prior_probability=None,
            model_probability=None,
            model_weight=0.0,
            confidence=0.0,
            is_on=False,
            learning_state=STATE_CALIBRATING,
            residual_w=None,
            detected_period_s=None,
            learned_levels=[],
            last_cycle=None,
            edge_threshold=0.0,
            lag_s=None,
            update_interval_s=None,
            valid_fraction=0.0,
            counts={},
            quality={},
        )


@dataclass(slots=True)
class _HistoryEntry:
    ts: float
    features: npt.NDArray[np.float32]
    p_prior: float
    p: float
    is_on: bool
    n_edges: int
    new_heights: npt.NDArray[np.float32]


@dataclass(slots=True)
class _Episode:
    start: float
    end: float | None = None
    heights: list[float] = field(default_factory=list)


class EvaluationScheduler:
    """Decide when an extra evaluation besides the regular interval is due.

    The regular evaluation runs every ``eval_interval_s``. To react faster,
    an extra evaluation is scheduled a few seconds after a large grid power
    step (so that the step and late auxiliary reports have settled), and a
    confirming evaluation shortly after the probability first crosses the
    switch-on threshold. Evaluations are never closer than
    ``EXTRA_EVAL_MIN_SPACING_S`` to each other.
    """

    def __init__(self, step_threshold_w: float = MIN_EDGE_THRESHOLD) -> None:
        """Start without a pending extra evaluation."""
        self.step_threshold_w = step_threshold_w
        self.settle_s = EXTRA_EVAL_SETTLE_S
        self._due: float | None = None
        self._last_grid: float | None = None
        self._last_eval: float | None = None

    def grid_update(self, ts: float, value: float | None) -> None:
        """Register a new grid power value."""
        if value is None:
            self._last_grid = None
            return
        if (
            self._last_grid is not None
            and abs(value - self._last_grid) >= self.step_threshold_w
            and self._due is None
        ):
            self._due = ts + self.settle_s
        self._last_grid = value

    def evaluated(
        self,
        ts: float,
        result: EvaluationResult,
        on_threshold: float,
        auxiliary_interval_s: float | None = None,
    ) -> None:
        """Register a finished evaluation."""
        self._last_eval = ts
        self._due = None
        self.settle_s = max(EXTRA_EVAL_SETTLE_S, (auxiliary_interval_s or 0.0) + 4.0)
        if (
            result.evaluated
            and not result.is_on
            and result.probability is not None
            and result.probability >= on_threshold
        ):
            self._due = ts + CONFIRM_DELAY_S

    def next_due(self) -> float | None:
        """Return the time of the next extra evaluation, if any."""
        if self._due is None:
            return None
        if self._last_eval is None:
            return self._due
        return max(self._due, self._last_eval + EXTRA_EVAL_MIN_SPACING_S)

    def regular_due(self, ts: float) -> bool:
        """Return False if a regular tick comes too soon after an extra one."""
        return (
            self._last_eval is None or ts - self._last_eval >= EXTRA_EVAL_MIN_SPACING_S
        )


class Hysteresis:
    """Binary output with two thresholds, a confirmation count and a timeout."""

    def __init__(
        self,
        on_threshold: float,
        off_threshold: float,
        consecutive: int = CONSECUTIVE_EVALUATIONS,
        no_cycle_timeout_s: float = NO_CYCLE_TIMEOUT_SECONDS,
    ) -> None:
        """Start in the off state."""
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.consecutive = consecutive
        self.no_cycle_timeout_s = no_cycle_timeout_s
        self.is_on = False
        self.last_match_ts: float | None = None
        self._above = 0
        self._below = 0

    def update(
        self, p: float | None, now: float, matching_edge_ts: float | None
    ) -> bool:
        """Feed one evaluation (``p`` None = not evaluable); return the state."""
        if matching_edge_ts is not None:
            self.last_match_ts = (
                matching_edge_ts
                if self.last_match_ts is None
                else max(self.last_match_ts, matching_edge_ts)
            )
        if p is not None:
            self._above = self._above + 1 if p >= self.on_threshold else 0
            self._below = self._below + 1 if p < self.off_threshold else 0
        if not self.is_on:
            if self._above >= self.consecutive:
                self.is_on = True
                self._below = 0
                self.last_match_ts = max(self.last_match_ts or now, now)
        elif self._below >= self.consecutive or (
            self.last_match_ts is not None
            and now - self.last_match_ts > self.no_cycle_timeout_s
        ):
            self.force_off()
        return self.is_on

    def force_off(self) -> None:
        """Switch off immediately."""
        self.is_on = False
        self._above = 0
        self._below = 0


def compute_confidence(
    p: float,
    model_weight: float,
    valid_fraction: float,
    update_interval: float | None,
    calibrated: bool,
) -> float:
    """Return the 0..1 confidence of the current statement.

    Combines the decisiveness of the decision, the maturity of the model and
    the current data quality.
    """
    decisiveness = abs(p - 0.5) * 2.0
    maturity = 0.5 + 0.5 * min(1.0, model_weight / (0.75 * MAX_MODEL_WEIGHT))
    data = 0.5 + 0.5 * float(
        np.clip((valid_fraction - MIN_VALID_FRACTION) / (1 - MIN_VALID_FRACTION), 0, 1)
    )
    if update_interval is not None:
        if update_interval > UPDATE_INTERVAL_BAD:
            data *= 0.15
        elif update_interval > UPDATE_INTERVAL_OK:
            span = UPDATE_INTERVAL_BAD - UPDATE_INTERVAL_OK
            data *= 0.8 - 0.4 * (update_interval - UPDATE_INTERVAL_OK) / span
        elif update_interval > 5.0:
            data *= 1.0 - 0.2 * (update_interval - 5.0) / (UPDATE_INTERVAL_OK - 5.0)
    if not calibrated:
        data *= 0.8
    return float(np.clip(decisiveness * maturity * data, 0.0, 1.0))


class HerdwaechterEngine:
    """Stateful detector for one monitored appliance."""

    def __init__(self, config: EngineConfig, seed: int = 0) -> None:
        """Create a fresh engine (no learned state)."""
        self.config = config
        self.profile: DeviceProfile = get_profile(config.profile)
        self._n_inputs = self.profile.n_model_inputs
        self.calibrator = Calibrator()
        self.model = OnlineLogisticRegression(self._n_inputs)
        self.dataset = LabelDataset(N_FEATURES)
        self.clusterer = PowerLevelClusterer(
            tolerance=0.12, min_support=10.0, max_clusters=12
        )
        self.quality = QualityReport()
        self.model_weight = 0.0
        self.learning_paused = False
        # learned state changed (labels, model, levels) / calibration changed
        self.dirty = False
        self.calibration_dirty = False
        self._rng = np.random.default_rng(seed)
        self._policy = LabelPolicy(NEUTRAL_EVALUATIONS_AFTER_LABEL)
        self._hysteresis = Hysteresis(config.on_threshold, config.off_threshold)
        self._history: deque[_HistoryEntry] = deque(
            maxlen=max(1, int(HISTORY_SECONDS / max(config.eval_interval_s, 1.0)))
        )
        self._episodes: deque[_Episode] = deque(maxlen=50)
        self._labels_since_retrain = 0
        self._eval_counter = 0
        self._last_retrain_ts: float | None = None
        self._last_edge_ts: float = -math.inf
        self._last_cycle_off_ts: float = -math.inf
        self._activity_start: float | None = None
        self._last_activity_ts: float | None = None
        self._first_window_start: float | None = None
        self._episode_blocks: list[float] = []
        self._suppress_until: float | None = None
        self._questions: dict[str, Question] = {}
        self._asked_activity: float | None = None
        self._question_day: int | None = None
        self._questions_today = 0
        self._question_counter = 0
        self._last_cycle: dict[str, Any] | None = None
        self._last_result = EvaluationResult.initial()
        self._event_heights: list[float] = []
        self._levels: FloatArray = np.empty(0)

    # ================================================================ public

    @property
    def is_on(self) -> bool:
        """Return the current binary output."""
        return self._hysteresis.is_on

    @property
    def last_result(self) -> EvaluationResult:
        """Return the most recent evaluation result."""
        return self._last_result

    @property
    def pending_questions(self) -> list[Question]:
        """Return unanswered questions."""
        return list(self._questions.values())

    def evaluate(self, inp: EvaluationInput) -> EvaluationResult:
        """Run one evaluation."""
        cfg = self.config
        now = inp.now
        win = inp.window
        self._expire_questions(now)

        lag = self.calibrator.lag_int if cfg.has_generation else 0
        residual_full = reconstruct_residual(win.grid, win.gen, win.sub, lag)
        cut = max(lag, 0)
        residual = (
            residual_full[: residual_full.shape[0] - cut] if cut else residual_full
        )
        # non-grid part of the residual (shifted generation minus metered
        # devices); its late reports cause transients in the residual
        shifted_gen: FloatArray | None = None
        if cfg.has_generation or np.any(win.sub != 0):
            shifted_full = reconstruct_residual(
                np.zeros_like(win.gen), win.gen, win.sub, lag
            )
            shifted_gen = (
                shifted_full[: shifted_full.shape[0] - cut] if cut else shifted_full
            )
        end_ts = win.end_ts - cut
        window_s = min(cfg.window_s, residual.shape[0])
        current = residual[residual.shape[0] - window_s :]
        valid_fraction = float(np.mean(np.isfinite(current))) if window_s else 0.0
        finite = residual_full[np.isfinite(residual_full)]
        residual_w = float(finite[-1]) if finite.shape[0] else None

        interval = inp.update_interval_s
        span, smooth, period_smooth = edge_parameters(interval, cfg.has_generation)
        evaluable = valid_fraction >= MIN_VALID_FRACTION and window_s >= 120

        self._eval_counter += 1
        if evaluable:
            # calibration does not need every window; keeps CPU usage low
            if self._eval_counter % 4 == 0 and self.calibrator.update_noise(
                current, span
            ):
                self.calibration_dirty = True
            if (
                cfg.has_generation
                and self._eval_counter % 2 == 0
                and self.calibrator.update_lag(win.grid[-300:], win.gen[-300:])
            ):
                self.calibration_dirty = True

        analysis: WindowAnalysis | None = None
        p: float | None = None
        p_prior: float | None = None
        p_model: float | None = None
        new_edges_ts = np.empty(0)
        new_edges_h = np.empty(0)
        new_cycle_heights = np.empty(0)
        if evaluable:
            levels = self.clusterer.levels()
            self._levels = levels
            if self._first_window_start is None:
                # edges before the first analysed window were never seen
                self._first_window_start = end_ts - window_s
            activity_minutes = self._activity_minutes(end_ts)
            analysis = analyse_window(
                residual[-(2 * window_s) :],
                window_s,
                self.calibrator.edge_threshold,
                span,
                smooth,
                levels,
                inp.local_hour,
                activity_minutes,
                period_smooth,
                shifted_gen[-(2 * window_s) :] if shifted_gen is not None else None,
                inp.generation_interval_s,
            )
            edge_ts = end_ts - (window_s - analysis.edges.idx)
            fresh = edge_ts > self._last_edge_ts + EDGE_DEDUP_SECONDS
            new_edges_ts = edge_ts[fresh]
            new_edges_h = analysis.edges.height[fresh]
            if new_edges_ts.shape[0]:
                self._last_edge_ts = float(np.max(new_edges_ts))
            cycle_off_ts = end_ts - (window_s - analysis.cycles.off)
            fresh_cycles = cycle_off_ts > self._last_cycle_off_ts + EDGE_DEDUP_SECONDS
            new_cycle_heights = analysis.cycles.height[fresh_cycles]
            if np.any(fresh_cycles):
                self._last_cycle_off_ts = float(np.max(cycle_off_ts[fresh_cycles]))
            self._update_activity(end_ts, analysis, new_edges_ts)
            # recompute with the updated activity duration
            analysis.features[FEATURE_INDEX["activity_minutes"]] = (
                self._activity_minutes(end_ts)
            )
            analysis.features[FEATURE_INDEX["zone_heatup_seen"]] = (
                self._zone_heatup_seen(analysis)
            )
            self._debug_analysis = analysis
            p_prior, p_model, p = self._probability(analysis.features)
            self._update_last_cycle(analysis, end_ts, window_s)

        suppressed = self._suppress_until is not None and now < self._suppress_until
        if self._suppress_until is not None and (
            not suppressed or self._activity_start is None
        ):
            self._suppress_until = None
            suppressed = False
        was_on = self._hysteresis.is_on
        matching_ts = self._matching_edge_ts(new_edges_ts, new_edges_h)
        is_on = self._hysteresis.update(None if suppressed else p, now, matching_ts)
        if suppressed and is_on:
            self._hysteresis.force_off()
            is_on = False
        self._track_episodes(was_on, is_on, now, analysis)

        if analysis is not None and p is not None and p_prior is not None:
            if not self.learning_paused:
                self._learn_from_window(inp, analysis, end_ts, p, new_cycle_heights)
            self._history.append(
                _HistoryEntry(
                    ts=end_ts,
                    features=analysis.features.astype(np.float32),
                    p_prior=p_prior,
                    p=p,
                    is_on=is_on,
                    n_edges=analysis.edges.count,
                    new_heights=np.abs(new_cycle_heights).astype(np.float32),
                )
            )

        question = None
        if p is not None and analysis is not None:
            question = self._maybe_ask(inp, p)

        calibrated = self.calibrator.is_calibrated(cfg.has_generation)
        state = self._learning_state(evaluable, interval, calibrated)
        confidence = (
            compute_confidence(
                p, self.model_weight, valid_fraction, interval, calibrated
            )
            if p is not None
            else 0.0
        )
        period = (
            analysis.period_s
            if analysis is not None and analysis.period_strength > 0.3
            else None
        )
        result = EvaluationResult(
            evaluated=evaluable,
            probability=p if p is not None else self._last_result.probability,
            prior_probability=p_prior,
            model_probability=p_model,
            model_weight=self.model_weight,
            confidence=confidence,
            is_on=is_on,
            learning_state=state,
            residual_w=residual_w,
            detected_period_s=period,
            learned_levels=[round(float(v)) for v in self.clusterer.levels()],
            last_cycle=(
                {k: v for k, v in self._last_cycle.items() if not k.startswith("_")}
                if self._last_cycle is not None
                else None
            ),
            edge_threshold=self.calibrator.edge_threshold,
            lag_s=self.calibrator.lag_s if cfg.has_generation else None,
            update_interval_s=interval,
            valid_fraction=valid_fraction,
            counts=self.dataset.counts(),
            quality=self.quality.to_dict(),
            question=question,
        )
        self._last_result = result
        return result

    def mark_false_alarm(self, now: float) -> int:
        """Label the current or last detected event as negative.

        Returns the number of windows that were labelled.
        """
        episode = self._false_alarm_episode(now)
        if self._hysteresis.is_on:
            self._hysteresis.force_off()
            if self._episodes and self._episodes[-1].end is None:
                self._episodes[-1].end = now
        self._suppress_until = now + FALSE_ALARM_SUPPRESS_SECONDS
        if episode is None:
            return 0
        end = episode.end if episode.end is not None else now
        count = 0
        if not self.learning_paused:
            for entry in self._history:
                if episode.start - 120 <= entry.ts <= end:
                    self._add_label(
                        entry.features,
                        0,
                        STRONG_NEGATIVE_WEIGHT,
                        entry.ts,
                        LabelSource.FALSE_ALARM,
                        int(episode.start),
                    )
                    for h in entry.new_heights.tolist():
                        self.clusterer.add_negative(h, 1.0)
                    count += 1
            if count:
                self._retrain(now)
        self.dirty = True
        return count

    def answer_question(self, question_id: str, cooking: bool, now: float) -> bool:
        """Apply the user's answer to a pending question."""
        question = self._questions.pop(question_id, None)
        if question is None:
            return False
        if self.learning_paused:
            return True
        lo = question.created - (120 if cooking else 300)
        hi = question.created + 60
        count = 0
        for entry in self._history:
            if lo <= entry.ts <= hi and (not cooking or entry.n_edges > 0):
                self._add_label(
                    entry.features,
                    1 if cooking else 0,
                    ANSWER_WEIGHT,
                    entry.ts,
                    LabelSource.ANSWER_YES if cooking else LabelSource.ANSWER_NO,
                    int(question.created),
                )
                for h in entry.new_heights.tolist():
                    if cooking:
                        self.clusterer.add(h, 1.0)
                    else:
                        self.clusterer.add_negative(h, 1.0)
                count += 1
        if count:
            self._retrain(now)
        self.dirty = True
        return True

    def add_label(self, start: float, end: float, cooking: bool, now: float) -> int:
        """Label stored windows (last 24 h) whose end lies within [start, end]."""
        if self.learning_paused or end < start:
            return 0
        count = 0
        for entry in self._history:
            if start <= entry.ts <= end and (not cooking or entry.n_edges > 0):
                self._add_label(
                    entry.features,
                    1 if cooking else 0,
                    ANSWER_WEIGHT if not cooking else 1.0,
                    entry.ts,
                    LabelSource.SERVICE,
                    int(start),
                )
                for h in entry.new_heights.tolist():
                    if cooking:
                        self.clusterer.add(h, 1.0)
                    else:
                        self.clusterer.add_negative(h, 1.0)
                count += 1
        if count:
            self._retrain(now)
        self.dirty = True
        return count

    def reset_model(self) -> None:
        """Forget the learned model, labels and power levels (keeps calibration)."""
        self.model = OnlineLogisticRegression(self._n_inputs)
        self.dataset.clear()
        self.clusterer.clear()
        self._levels = np.empty(0)
        self.quality = QualityReport()
        self.model_weight = 0.0
        self._labels_since_retrain = 0
        self._last_retrain_ts = None
        self.dirty = True

    def retrain(self, now: float) -> None:
        """Force a full refit over the stored labels."""
        self._retrain(now)

    # ============================================================ internals

    def _probability(self, features: FloatArray) -> tuple[float, float | None, float]:
        p_prior = self.profile.prior_probability(features)
        p_model: float | None = None
        p = p_prior
        if self.model.is_trained and self.model_weight > 0:
            inputs = self.profile.model_inputs(features[None, :], self._levels)
            offset = self.profile.prior_offsets(features[None, :])
            p_model = float(self.model.predict_proba(inputs, offset)[0])
            w = effective_weight(self.model_weight, p_prior, p_model)
            p = w * p_model + (1.0 - w) * p_prior
        return p_prior, p_model, float(np.clip(p, 0.0, 1.0))

    def _activity_minutes(self, end_ts: float) -> float:
        if self._activity_start is None:
            return 0.0
        return max(0.0, (end_ts - self._activity_start) / 60.0)

    def _update_activity(
        self, end_ts: float, analysis: WindowAnalysis, new_edges_ts: FloatArray
    ) -> None:
        if new_edges_ts.shape[0]:
            if self._activity_start is None or (
                self._last_activity_ts is not None
                and float(np.min(new_edges_ts)) - self._last_activity_ts
                > ACTIVITY_GAP_SECONDS
            ):
                self._activity_start = float(np.min(new_edges_ts))
                self._episode_blocks = []
            self._last_activity_ts = float(np.max(new_edges_ts))
        elif (
            self._last_activity_ts is not None
            and end_ts - self._last_activity_ts > ACTIVITY_GAP_SECONDS
        ):
            self._activity_start = None
            self._last_activity_ts = None
            self._episode_blocks = []
        # remember the powers that ran continuously during this activity
        cycles = analysis.cycles
        for height in cycles.height[cycles.on_duration >= EPISODE_HEATUP_MIN_S]:
            h = float(height)
            if not any(abs(h - b) <= 0.1 * b for b in self._episode_blocks):
                self._episode_blocks.append(h)

    def _zone_heatup_seen(self, analysis: WindowAnalysis) -> float:
        """Return 1 if the cycling zone ran continuously earlier in the activity.

        A radiant zone starts cold and runs continuously first; a microwave
        programme cycles right away. If the start of the activity was not
        observed (e.g. right after a restart) the answer is "yes".
        """
        zone = float(analysis.features[FEATURE_INDEX["zone_height"]])
        if zone <= 0:
            return 1.0
        start = self._activity_start
        if (
            start is None
            or self._first_window_start is None
            or start - self._first_window_start < UNOBSERVED_START_S
        ):
            return 1.0
        matched = any(abs(zone - b) <= 0.15 * b for b in self._episode_blocks)
        return 1.0 if matched else 0.0

    def _update_last_cycle(
        self, analysis: WindowAnalysis, end_ts: float, window_s: int
    ) -> None:
        cycles = analysis.cycles
        if cycles.count == 0:
            return
        i = int(np.argmax(cycles.off))
        off_ts = end_ts - (window_s - float(cycles.off[i]))
        on_ts = end_ts - (window_s - float(cycles.on[i]))
        if self._last_cycle is not None and off_ts <= self._last_cycle["_off_ts"] + 2:
            return
        self._last_cycle = {
            "power_w": round(float(cycles.height[i])),
            "on_s": round(off_ts - on_ts, 1),
            "ended_ago_s": round(end_ts - off_ts, 1),
            "_off_ts": off_ts,
        }

    def _matching_edge_ts(self, ts: FloatArray, heights: FloatArray) -> float | None:
        """Return the newest edge that matches the current event's power levels."""
        if ts.shape[0] == 0:
            return None
        refs = list(self.clusterer.levels()) + self._event_heights
        mags = np.abs(heights)
        if not refs:
            return float(np.max(ts))
        ref = np.asarray(refs)[None, :]
        ok = np.any(np.abs(mags[:, None] - ref) <= PAIR_TOLERANCE * ref, axis=1)
        return float(np.max(ts[ok])) if np.any(ok) else None

    def _track_episodes(
        self, was_on: bool, is_on: bool, now: float, analysis: WindowAnalysis | None
    ) -> None:
        if is_on and not was_on:
            heights = (
                [float(h) for h in analysis.cycles.height]
                if analysis is not None
                else []
            )
            self._event_heights = heights
            start = self._activity_start if self._activity_start is not None else now
            self._episodes.append(_Episode(start=min(start, now), heights=heights))
        elif was_on and not is_on:
            if self._episodes and self._episodes[-1].end is None:
                self._episodes[-1].end = now
            self._event_heights = []
        elif is_on and analysis is not None and analysis.cycles.count:
            # follow the zones that are actually cycling during the event
            self._event_heights = [float(h) for h in analysis.cycles.height]

    def _false_alarm_episode(self, now: float) -> _Episode | None:
        if not self._episodes:
            return None
        last = self._episodes[-1]
        if last.end is None or now - last.end <= FALSE_ALARM_LOOKBACK_SECONDS:
            return last
        return None

    def _learn_from_window(
        self,
        inp: EvaluationInput,
        analysis: WindowAnalysis,
        end_ts: float,
        p: float,
        new_heights: FloatArray,
    ) -> None:
        decision = self._policy.decide(
            inp.now,
            inp.label_active,
            inp.label_group,
            inp.label_source,
            analysis.edges.count,
            self._hysteresis.is_on,
        )
        if decision is not None:
            self._add_label(
                analysis.features,
                decision.label,
                decision.weight,
                end_ts,
                decision.source,
                decision.group,
            )
            self._incremental_update(analysis.features, decision.label, decision.weight)
        # learn power levels from labelled or clearly cycling windows
        if new_heights.shape[0]:
            if inp.label_active:
                weight = 1.0
            elif p >= self.config.on_threshold and self.profile.is_cycling_window(
                analysis.features
            ):
                weight = PERIODIC_LEVEL_WEIGHT
            else:
                weight = 0.0
            if weight > 0:
                for h in np.abs(new_heights).tolist():
                    if self.profile.min_level_w <= h <= self.profile.max_level_w:
                        self.clusterer.add(h, weight)
                self.dirty = True
        if self._labels_since_retrain >= RETRAIN_EVERY_LABELS or (
            self._labels_since_retrain > 0
            and (
                self._last_retrain_ts is None
                or inp.now - self._last_retrain_ts > RETRAIN_INTERVAL_SECONDS
            )
        ):
            self._retrain(inp.now)

    def _add_label(
        self,
        features: FloatArray | npt.NDArray[np.float32],
        label: int,
        weight: float,
        ts: float,
        source: LabelSource,
        group: int,
    ) -> None:
        self.dataset.add(
            np.asarray(features, dtype=np.float64), label, weight, ts, source, group
        )
        self._labels_since_retrain += 1
        self.dirty = True

    def _class_weights(self, y: FloatArray, w: FloatArray) -> FloatArray:
        out = w.copy()
        total = float(w.sum())
        for cls in (0.0, 1.0):
            sel = y == cls
            cls_total = float(w[sel].sum())
            if cls_total > 0:
                out[sel] *= total / (2.0 * cls_total)
        return out

    def _incremental_update(
        self, features: FloatArray, label: int, weight: float
    ) -> None:
        """Mini-batch update: the new window plus a replay of stored windows."""
        n = len(self.dataset)
        raw = [features[None, :]]
        ys = [np.array([float(label)])]
        ws = [np.array([weight])]
        if n > 1:
            idx = self._rng.choice(n, size=min(REPLAY_BATCH, n), replace=False)
            raw.append(self.dataset.x[idx])
            ys.append(self.dataset.y[idx])
            ws.append(self.dataset.weight[idx])
        matrix = np.vstack(raw)
        x = self.profile.model_inputs(matrix, self._levels)
        offset = self.profile.prior_offsets(matrix)
        y = np.concatenate(ys)
        w = self._class_weights(y, np.concatenate(ws))
        self.model.partial_fit(x, y, w, steps=3, new_rows=1, offset=offset)

    def _retrain(self, now: float) -> None:
        self._labels_since_retrain = 0
        self._last_retrain_ts = now
        n = len(self.dataset)
        counts = self.dataset.counts()
        if n == 0 or counts["positive"] == 0 or counts["negative"] == 0:
            self.model_weight = 0.0
            return
        raw = self.dataset.x
        self._levels = self.clusterer.levels()
        x = self.profile.model_inputs(raw, self._levels)
        y = self.dataset.y
        age_days = np.maximum(0.0, (now - self.dataset.ts) / 86400.0)
        w = self.dataset.weight * np.power(0.5, age_days / AGE_HALF_LIFE_DAYS)
        holdout = self.dataset.holdout_mask()
        train = ~holdout
        if np.unique(y[train]).shape[0] < 2:
            self.model_weight = 0.0
            return
        offset = self.profile.prior_offsets(raw)
        self.model.fit(
            x[train],
            y[train],
            self._class_weights(y[train], w[train]),
            offset=offset[train],
        )
        if np.any(holdout):
            p_model = self.model.predict_proba(x[holdout], offset[holdout])
            p_prior = self.profile.prior_probabilities(raw[holdout])
            # evaluate the model as it is used: with the PU-safe gate
            gate = np.where(
                p_model > p_prior,
                np.clip(
                    (p_prior - UPWARD_GATE_LOW) / (UPWARD_GATE_HIGH - UPWARD_GATE_LOW),
                    0.0,
                    1.0,
                ),
                1.0,
            )
            p_used = p_prior + gate * (p_model - p_prior)
            self.quality = evaluate_holdout(p_used, p_prior, y[holdout], w[holdout])
        else:
            self.quality = QualityReport()
        self.model_weight = mixing_weight(
            counts["positive_groups"], counts["negative_groups"], self.quality
        )
        self.dirty = True

    def _learning_state(
        self, evaluable: bool, interval: float | None, calibrated: bool
    ) -> str:
        if not evaluable or (interval is not None and interval > UPDATE_INTERVAL_BAD):
            return STATE_INSUFFICIENT_DATA
        if not calibrated:
            return STATE_CALIBRATING
        if self.model_weight <= 0 and not self.quality.known:
            return STATE_PRIOR_ONLY
        counts = self.dataset.counts()
        if (
            self.quality.known
            and maturity(counts["positive_groups"], counts["negative_groups"])
            >= TRAINED_MATURITY
        ):
            return STATE_TRAINED
        return STATE_LEARNING

    # ------------------------------------------------------------ questions

    def _expire_questions(self, now: float) -> None:
        for qid in [q.question_id for q in self._questions.values() if now > q.expires]:
            del self._questions[qid]

    def _maybe_ask(self, inp: EvaluationInput, p: float) -> Question | None:
        cfg = self.config
        if not cfg.questions_enabled or inp.label_active or self._questions:
            return None
        if not QUESTION_P_LOW <= p <= QUESTION_P_HIGH:
            return None
        if self._activity_start is None or self._asked_activity == self._activity_start:
            return None
        if self._question_day != inp.local_day:
            self._question_day = inp.local_day
            self._questions_today = 0
        if self._questions_today >= cfg.max_questions_per_day:
            return None
        self._question_counter += 1
        question = Question(
            question_id=f"{int(inp.now):x}{self._question_counter:02x}",
            created=inp.now,
            probability=p,
        )
        self._questions[question.question_id] = question
        self._questions_today += 1
        self._asked_activity = self._activity_start
        return question

    # ---------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Serialise the learned state (no raw power data)."""
        self.dirty = False
        self.calibration_dirty = False
        return {
            "engine_version": ENGINE_VERSION,
            "profile": self.profile.key,
            "calibration": self.calibrator.to_dict(),
            "model": self.model.to_dict(),
            "levels": self.clusterer.to_dict(),
            "dataset": self.dataset.to_dict(),
            "quality": self.quality.to_dict(),
            "model_weight": self.model_weight,
            "learning_paused": self.learning_paused,
            "last_retrain_ts": self._last_retrain_ts,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        """Restore the learned state written by :meth:`to_dict`."""
        if data.get("profile", self.profile.key) != self.profile.key:
            return
        self.calibrator = Calibrator.from_dict(data.get("calibration", {}))
        self.model = OnlineLogisticRegression.from_dict(
            data.get("model", {}), self._n_inputs
        )
        self.clusterer = PowerLevelClusterer.from_dict(data.get("levels", {}))
        self._levels = self.clusterer.levels()
        self.dataset = LabelDataset.from_dict(data.get("dataset", {}), N_FEATURES)
        self.quality = QualityReport.from_dict(data.get("quality", {}))
        self.model_weight = float(data.get("model_weight", 0.0))
        if not self.model.is_trained:
            self.model_weight = 0.0
        self.learning_paused = bool(data.get("learning_paused", False))
        last = data.get("last_retrain_ts")
        self._last_retrain_ts = float(last) if last is not None else None
        self.dirty = False
        self.calibration_dirty = False

    # ------------------------------------------------------- introspection

    def diagnostics(self) -> dict[str, Any]:
        """Return model parameters, statistics and recent feature vectors."""
        recent = list(self._history)[-10:]
        return {
            "config": {
                "profile": self.config.profile,
                "has_generation": self.config.has_generation,
                "on_threshold": self.config.on_threshold,
                "off_threshold": self.config.off_threshold,
                "eval_interval_s": self.config.eval_interval_s,
                "window_s": self.config.window_s,
            },
            "calibration": self.calibrator.to_dict(),
            "edge_threshold_w": self.calibrator.edge_threshold,
            "model": self.model.to_dict(),
            "model_weight": self.model_weight,
            "learned_levels_w": [round(float(v)) for v in self.clusterer.levels()],
            "level_clusters": self.clusterer.to_dict()["clusters"],
            "label_counts": self.dataset.counts(),
            "quality": self.quality.to_dict(),
            "learning_paused": self.learning_paused,
            "pending_questions": len(self._questions),
            "recent_windows": [
                {
                    "age_s": round(recent[-1].ts - e.ts, 1),
                    "p_prior": round(e.p_prior, 4),
                    "p": round(e.p, 4),
                    "is_on": e.is_on,
                    "features": {
                        name: round(float(v), 4)
                        for name, v in zip(FEATURE_NAMES, e.features, strict=True)
                    },
                }
                for e in recent
            ],
        }

    def export_training_data(self) -> dict[str, Any]:
        """Return the anonymised label data set (no timestamps, no entity ids)."""
        holdout = self.dataset.holdout_mask()
        rows = [
            {
                "label": int(y),
                "weight": round(float(w), 4),
                "source": LabelSource(int(src)).name.lower(),
                "holdout": bool(h),
                "features": [round(float(v), 5) for v in x],
            }
            for x, y, w, src, h in zip(
                self.dataset.x,
                self.dataset.y,
                self.dataset.weight,
                self.dataset.source,
                holdout,
                strict=True,
            )
        ]
        return {
            "format": 1,
            "profile": self.profile.key,
            "feature_names": list(FEATURE_NAMES),
            "learned_levels_w": [round(float(v)) for v in self.clusterer.levels()],
            "samples": rows,
        }
