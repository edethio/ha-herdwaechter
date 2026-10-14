"""Drive the real engine with simulated sensor events and a virtual user.

This is the same pipeline as in Home Assistant (event -> Resampler ->
HerdwaechterEngine) without Home Assistant itself. The virtual user labels a
share of the cooking sessions with the label switch and presses the false
alarm button for a share of the false alarms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from custom_components.herdwaechter.engine import (
    EngineConfig,
    EvaluationInput,
    EvaluationScheduler,
    HerdwaechterEngine,
)
from custom_components.herdwaechter.learning import LabelSource
from custom_components.herdwaechter.sampling import (
    ROLE_GEN,
    ROLE_GRID,
    ROLE_SUB,
    Resampler,
)

from .household import DAY, Household, HouseholdConfig, Session, generate_household
from .sensors import SensorConfig, SensorEvents, generate_events

FloatArray = npt.NDArray[np.float64]

BASE_TS = 1_800_000_000.0  # arbitrary epoch offset of simulation second 0
MATCH_TAIL_S = 10 * 60  # an on-period up to 10 min after a session still counts
_ROLE_NAMES = {0: ROLE_GRID, 1: ROLE_GEN, 2: ROLE_SUB}


@dataclass(frozen=True, slots=True)
class UserConfig:
    """Behaviour of the virtual user."""

    seed: int = 3
    label_share: float = 0.5
    false_alarm_press_share: float = 0.5
    label_delay_s: tuple[float, float] = (30.0, 240.0)
    label_off_delay_s: tuple[float, float] = (0.0, 180.0)
    forget_off_share: float = 0.1
    max_label_s: float = 7200.0


@dataclass(slots=True)
class Episode:
    """One on-period of the binary output."""

    start: float
    end: float
    true_positive: bool = False


@dataclass(slots=True)
class Metrics:
    """Event level evaluation for a time range."""

    days: float
    sessions: int
    detected: int
    episodes: int
    true_episodes: int
    false_alarms: int
    latencies_s: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Return TP episodes / all episodes."""
        return self.true_episodes / self.episodes if self.episodes else 1.0

    @property
    def recall(self) -> float:
        """Return detected sessions / sessions."""
        return self.detected / self.sessions if self.sessions else 1.0

    @property
    def false_alarms_per_day(self) -> float:
        """Return false alarms per simulated day."""
        return self.false_alarms / self.days if self.days else 0.0

    @property
    def median_latency_s(self) -> float:
        """Return the median detection latency."""
        return float(np.median(self.latencies_s)) if self.latencies_s else math.nan

    def as_dict(self) -> dict[str, float]:
        """Return the headline numbers."""
        return {
            "days": self.days,
            "sessions": self.sessions,
            "detected": self.detected,
            "episodes": self.episodes,
            "false_alarms": self.false_alarms,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "median_latency_min": round(self.median_latency_s / 60.0, 2),
            "false_alarms_per_day": round(self.false_alarms_per_day, 3),
        }


@dataclass(slots=True)
class RunResult:
    """Everything a simulation run produced."""

    house: Household
    events: SensorEvents
    eval_ts: FloatArray
    probability: FloatArray
    confidence: FloatArray
    is_on: npt.NDArray[np.bool_]
    episodes: list[Episode]
    labels_pressed: int
    false_alarm_presses: int
    engine: HerdwaechterEngine
    states: list[str]

    def metrics(self, day_from: float = 0.0, day_to: float | None = None) -> Metrics:
        """Evaluate sessions / episodes that start within [day_from, day_to)."""
        day_to = self.house.config.days if day_to is None else day_to
        lo = day_from * DAY
        hi = day_to * DAY
        sessions = [s for s in self.house.sessions if lo <= s.start < hi]
        episodes = [e for e in self.episodes if lo <= e.start < hi]
        detected = 0
        latencies: list[float] = []
        for s in sessions:
            hits = [
                e
                for e in self.episodes
                if e.end >= s.start and e.start <= s.end + MATCH_TAIL_S
            ]
            if hits:
                detected += 1
                latencies.append(max(0.0, min(e.start for e in hits) - s.start))
        true_eps = sum(1 for e in episodes if e.true_positive)
        return Metrics(
            days=day_to - day_from,
            sessions=len(sessions),
            detected=detected,
            episodes=len(episodes),
            true_episodes=true_eps,
            false_alarms=len(episodes) - true_eps,
            latencies_s=latencies,
        )


def _overlaps_session(start: float, end: float, sessions: list[Session]) -> bool:
    return any(end >= s.start and start <= s.end + MATCH_TAIL_S for s in sessions)


class _VirtualUser:
    """Label switch and false alarm button behaviour."""

    def __init__(self, house: Household, cfg: UserConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        self.intervals: list[tuple[float, float]] = []
        for s in house.sessions:
            if rng.random() >= cfg.label_share:
                continue
            on = s.start + rng.uniform(*cfg.label_delay_s)
            if rng.random() < cfg.forget_off_share:
                off = on + cfg.max_label_s
            else:
                off = min(
                    on + cfg.max_label_s, s.end + rng.uniform(*cfg.label_off_delay_s)
                )
            if off > on:
                self.intervals.append((on, off))
        self._i = 0
        self.pending_press: float | None = None

    def label_state(self, t: float) -> tuple[bool, int]:
        while self._i < len(self.intervals) and self.intervals[self._i][1] < t:
            self._i += 1
        if self._i < len(self.intervals):
            on, off = self.intervals[self._i]
            if on <= t <= off:
                return True, int(BASE_TS + on)
        return False, 0


def build_scenario(
    house_cfg: HouseholdConfig | None = None, sensor_cfg: SensorConfig | None = None
) -> tuple[Household, SensorEvents]:
    """Generate household and sensor events."""
    house = generate_household(house_cfg or HouseholdConfig())
    events = generate_events(house, sensor_cfg or SensorConfig())
    return house, events


def run_simulation(
    house: Household,
    events: SensorEvents,
    user_cfg: UserConfig | None = None,
    engine_cfg: EngineConfig | None = None,
    learning: bool = True,
    progress: Callable[[float], None] | None = None,
) -> RunResult:
    """Run the engine over the whole scenario."""
    user_cfg = user_cfg or UserConfig()
    cfg = engine_cfg or EngineConfig(has_generation=events.n_generation > 0)
    engine = HerdwaechterEngine(cfg, seed=user_cfg.seed)
    engine.learning_paused = not learning
    resampler = Resampler(events.n_generation, events.n_subtract)
    user = _VirtualUser(house, user_cfg)
    rng = np.random.default_rng(user_cfg.seed + 99)

    scheduler = EvaluationScheduler()
    n_regular = int(house.seconds // cfg.eval_interval_s)
    eval_list: list[float] = []
    prob_list: list[float] = []
    conf_list: list[float] = []
    on_list: list[bool] = []
    states: list[str] = []
    episodes: list[Episode] = []
    current: Episode | None = None
    press_at: float | None = None
    presses = 0
    labelled_windows = 0

    ts_l = events.ts.tolist()
    role_l = events.role.tolist()
    idx_l = events.index.tolist()
    val_l = events.value.tolist()
    n_ev = len(ts_l)
    ptr = 0
    buffer_s = cfg.buffer_seconds

    next_regular = cfg.eval_interval_s
    k = 0
    while True:
        due = scheduler.next_due()
        t = next_regular if due is None else min(next_regular, due - BASE_TS)
        if t >= house.seconds:
            break
        while ptr < n_ev and ts_l[ptr] <= t:
            v = val_l[ptr]
            value = None if math.isnan(v) else v  # NaN -> unavailable
            ts_abs = BASE_TS + ts_l[ptr]
            resampler.update(_ROLE_NAMES[role_l[ptr]], idx_l[ptr], ts_abs, value)
            if role_l[ptr] == 0:
                scheduler.grid_update(ts_abs, value)
                due = scheduler.next_due()
                if due is not None and due - BASE_TS < t:
                    t = due - BASE_TS
            ptr += 1
        if t >= next_regular:
            next_regular += cfg.eval_interval_s
            if not scheduler.regular_due(BASE_TS + t):
                continue
        k += 1
        now = BASE_TS + t
        window = resampler.snapshot(now, buffer_s)
        label_active, label_group = user.label_state(t)
        if label_active:
            labelled_windows += 1
        result = engine.evaluate(
            EvaluationInput(
                now=now,
                window=window,
                local_hour=(t % DAY) / 3600.0,
                local_day=int(t // DAY),
                update_interval_s=resampler.effective_update_interval(),
                generation_interval_s=resampler.generation_update_interval(),
                label_active=label_active,
                label_group=label_group,
                label_source=LabelSource.SWITCH,
            )
        )
        scheduler.evaluated(
            now, result, cfg.on_threshold, resampler.generation_update_interval()
        )
        eval_list.append(t)
        prob_list.append(math.nan if result.probability is None else result.probability)
        conf_list.append(result.confidence)
        on_list.append(result.is_on)
        states.append(result.learning_state)

        if result.is_on and current is None:
            current = Episode(start=t, end=t)
            tp = _overlaps_session(t, t, house.sessions)
            current.true_positive = tp
            if not tp and learning and rng.random() < user_cfg.false_alarm_press_share:
                press_at = t + rng.uniform(60, 360)
        elif result.is_on and current is not None:
            current.end = t
        elif not result.is_on and current is not None:
            current.end = t
            episodes.append(current)
            current = None
        if press_at is not None and t >= press_at:
            engine.mark_false_alarm(now)
            presses += 1
            press_at = None
            if current is not None:
                episodes.append(current)
                current = None
        if progress is not None and k % 2880 == 0:
            progress(min(1.0, t / (n_regular * cfg.eval_interval_s)))

    if current is not None:
        episodes.append(current)
    # re-evaluate true positives with the full episode extent
    for e in episodes:
        e.true_positive = _overlaps_session(e.start, e.end, house.sessions)
    return RunResult(
        house=house,
        events=events,
        eval_ts=np.asarray(eval_list),
        probability=np.asarray(prob_list),
        confidence=np.asarray(conf_list),
        is_on=np.asarray(on_list, dtype=bool),
        episodes=episodes,
        labels_pressed=len(user.intervals),
        false_alarm_presses=presses,
        engine=engine,
        states=states,
    )


def summary(result: RunResult) -> dict[str, Any]:
    """Return a compact summary of a run."""
    days = result.house.config.days
    weekly = [result.metrics(d, min(d + 7, days)).as_dict() for d in range(0, days, 7)]
    return {
        "overall": result.metrics().as_dict(),
        "day0": result.metrics(0, 1).as_dict(),
        "last_week": result.metrics(max(0, days - 7), days).as_dict(),
        "weekly": weekly,
        "labelled_sessions": result.labels_pressed,
        "false_alarm_presses": result.false_alarm_presses,
        "learned_levels": [round(float(v)) for v in result.engine.clusterer.levels()],
        "hob_levels": list(result.house.hob_levels),
        "model_weight": round(result.engine.model_weight, 3),
        "quality": result.engine.quality.to_dict(),
        "lag_s": result.engine.calibrator.lag_s,
        "edge_threshold": result.engine.calibrator.edge_threshold,
        "events": result.house.events,
    }
