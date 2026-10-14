"""Home Assistant glue: listens to sensors and drives the detection engine."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import dataclasses
from datetime import datetime, timedelta
import logging
import math
from typing import TYPE_CHECKING, Any

from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    EventStateReportedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_state_report_event,
    async_track_time_interval,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.translation import async_get_translations
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import PowerConverter

from .const import (
    ATTR_ENTRY_ID,
    ATTR_PROBABILITY,
    ATTR_QUESTION_ID,
    CONF_EVAL_INTERVAL,
    CONF_GENERATION_SENSORS,
    CONF_GRID_SENSOR,
    CONF_INVERT_GRID,
    CONF_LABEL_ENTITY,
    CONF_MAX_LABEL_MINUTES,
    CONF_MAX_QUESTIONS_PER_DAY,
    CONF_NOTIFY_SERVICE,
    CONF_OFF_THRESHOLD,
    CONF_ON_THRESHOLD,
    CONF_PROFILE,
    CONF_QUESTIONS_ENABLED,
    CONF_SUBTRACT_SENSORS,
    CONF_WINDOW_MINUTES,
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_MAX_LABEL_MINUTES,
    DEFAULT_MAX_QUESTIONS_PER_DAY,
    DEFAULT_OFF_THRESHOLD,
    DEFAULT_ON_THRESHOLD,
    DEFAULT_PROFILE,
    DEFAULT_QUESTIONS_ENABLED,
    DEFAULT_WINDOW_MINUTES,
    DOMAIN,
    EVENT_MOBILE_APP_ACTION,
    EVENT_QUESTION,
    QUESTION_ACTION_PREFIX,
    RING_BUFFER_SECONDS,
    STORAGE_MINOR_VERSION,
    STORAGE_SAVE_DELAY,
    STORAGE_VERSION,
)
from .engine import (
    EngineConfig,
    EvaluationInput,
    EvaluationResult,
    EvaluationScheduler,
    HerdwaechterEngine,
    Question,
)
from .learning import LabelSource
from .sampling import ROLE_GEN, ROLE_GRID, ROLE_SUB, Resampler

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry

_LOGGER = logging.getLogger(__name__)

# Limit storage writes (SD cards): routine learning changes are persisted at
# most every 15 minutes, pure calibration updates at most hourly. User actions
# and unloading always save.
_SNAPSHOT_LEARNING_INTERVAL = 15 * 60.0
_SNAPSHOT_CALIBRATION_INTERVAL = 60 * 60.0
_USER_ACTION_SAVE_DELAY = 10.0


def entry_options(entry: ConfigEntry) -> dict[str, Any]:
    """Return the effective configuration (data overlaid with options)."""
    return {**entry.data, **entry.options}


def engine_config(options: dict[str, Any]) -> EngineConfig:
    """Translate config entry options into an engine configuration."""
    return EngineConfig(
        profile=str(options.get(CONF_PROFILE, DEFAULT_PROFILE)),
        has_generation=bool(options.get(CONF_GENERATION_SENSORS)),
        on_threshold=float(options.get(CONF_ON_THRESHOLD, DEFAULT_ON_THRESHOLD)) / 100,
        off_threshold=float(options.get(CONF_OFF_THRESHOLD, DEFAULT_OFF_THRESHOLD))
        / 100,
        eval_interval_s=float(options.get(CONF_EVAL_INTERVAL, DEFAULT_EVAL_INTERVAL)),
        window_s=int(
            float(options.get(CONF_WINDOW_MINUTES, DEFAULT_WINDOW_MINUTES)) * 60
        ),
        max_label_s=float(
            options.get(CONF_MAX_LABEL_MINUTES, DEFAULT_MAX_LABEL_MINUTES)
        )
        * 60,
        questions_enabled=bool(
            options.get(CONF_QUESTIONS_ENABLED, DEFAULT_QUESTIONS_ENABLED)
        ),
        max_questions_per_day=int(
            options.get(CONF_MAX_QUESTIONS_PER_DAY, DEFAULT_MAX_QUESTIONS_PER_DAY)
        ),
    )


def parse_power(state: State | None) -> float | None:
    """Return the power of ``state`` in W, or None if it is not usable.

    ``unavailable``/``unknown``, non-numeric states and non-power units yield
    None (a gap), never 0 W. Units known to Home Assistant (W, kW, MW, ...)
    are converted; a missing unit is interpreted as W.
    """
    if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN, ""):
        return None
    try:
        value = float(state.state)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    if unit in (None, ""):
        return value
    if unit not in PowerConverter.VALID_UNITS:
        return None
    return float(PowerConverter.convert(value, unit, "W"))


class HerdwaechterStore(Store[dict[str, Any]]):
    """Versioned storage of the learned state of one config entry."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Create the store for ``entry_id``."""
        super().__init__(
            hass,
            STORAGE_VERSION,
            f"{DOMAIN}.{entry_id}",
            minor_version=STORAGE_MINOR_VERSION,
            serialize_in_event_loop=False,
        )

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
        old_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Migrate stored data written by an older version.

        Version 1.1 is the first storage format; future format changes add
        their migration steps here.
        """
        data = dict(old_data)
        if old_major_version == 1 and old_minor_version < 1:  # pragma: no cover
            data.setdefault("engine", {})
        return data


class HerdwaechterCoordinator:
    """Owns the engine of one config entry and connects it to Home Assistant."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialise from the config entry."""
        self.hass = hass
        self.entry = entry
        self.options = entry_options(entry)
        self.config = engine_config(self.options)
        self.grid_entity: str = self.options[CONF_GRID_SENSOR]
        self.invert_grid = bool(self.options.get(CONF_INVERT_GRID, False))
        self.generation_entities: list[str] = list(
            self.options.get(CONF_GENERATION_SENSORS) or []
        )
        self.subtract_entities: list[str] = list(
            self.options.get(CONF_SUBTRACT_SENSORS) or []
        )
        self.label_entity: str | None = self.options.get(CONF_LABEL_ENTITY) or None
        self.notify_service: str | None = self.options.get(CONF_NOTIFY_SERVICE) or None
        self.engine = HerdwaechterEngine(self.config)
        self.resampler = Resampler(
            len(self.generation_entities),
            len(self.subtract_entities),
            capacity_s=max(RING_BUFFER_SECONDS, self.config.buffer_seconds),
        )
        self.store = HerdwaechterStore(hass, entry.entry_id)
        self.data: EvaluationResult = EvaluationResult.initial()
        self.label_on = False
        self.label_since: float | None = None
        self.external_label_on = False
        self._roles: dict[str, tuple[str, int]] = {self.grid_entity: (ROLE_GRID, 0)}
        for i, eid in enumerate(self.generation_entities):
            self._roles[eid] = (ROLE_GEN, i)
        for i, eid in enumerate(self.subtract_entities):
            self._roles[eid] = (ROLE_SUB, i)
        self._lock = asyncio.Lock()
        self._listeners: list[CALLBACK_TYPE] = []
        self._unsubs: list[CALLBACK_TYPE] = []
        self._label_timer: CALLBACK_TYPE | None = None
        self._extra_timer: CALLBACK_TYPE | None = None
        self._extra_due: float | None = None
        self.scheduler = EvaluationScheduler()
        self._snapshot: dict[str, Any] | None = None
        self._snapshot_ts = 0.0
        self._bad_unit_logged: set[str] = set()
        self._shutdown = False

    # ------------------------------------------------------------ lifecycle

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device that groups all entities of this entry."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry.entry_id)},
            name=self.entry.title,
            manufacturer="Herdwächter",
            model=self.config.profile,
            sw_version=str(self.entry.version),
        )

    async def async_setup(self) -> None:
        """Load the stored state and start listening."""
        stored = await self.store.async_load()
        if stored:
            engine_data = stored.get("engine")
            if isinstance(engine_data, dict):
                await self.hass.async_add_executor_job(
                    self.engine.load_dict, engine_data
                )
                # keep the loaded state as the current snapshot so that a save
                # before the first evaluation never drops the learned data
                self._snapshot = engine_data
            label = stored.get("label") or {}
            since = label.get("since")
            if label.get("on") and since is not None:
                remaining = float(since) + self.config.max_label_s - self._now()
                if remaining > 0:
                    self.label_on = True
                    self.label_since = float(since)
                    self._label_timer = async_call_later(
                        self.hass, remaining, self._async_label_timeout
                    )
        now = self._now()
        for eid in self._roles:
            self._feed(eid, self.hass.states.get(eid), now)
        if self.label_entity is not None:
            state = self.hass.states.get(self.label_entity)
            self.external_label_on = state is not None and state.state == STATE_ON

        tracked = list(self._roles)
        if self.label_entity is not None:
            tracked.append(self.label_entity)
        self._unsubs.append(
            async_track_state_change_event(self.hass, tracked, self._handle_state_event)
        )
        self._unsubs.append(
            async_track_state_report_event(
                self.hass, list(self._roles), self._handle_report_event
            )
        )
        self._unsubs.append(
            async_track_time_interval(
                self.hass,
                self._async_tick,
                timedelta(seconds=self.config.eval_interval_s),
                name=f"{DOMAIN} evaluation",
            )
        )
        self._unsubs.append(
            self.hass.bus.async_listen(
                EVENT_MOBILE_APP_ACTION, self._handle_mobile_action
            )
        )

    async def async_shutdown(self) -> None:
        """Stop all listeners and timers and persist the learned state."""
        self._shutdown = True
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
        if self._label_timer is not None:
            self._label_timer()
            self._label_timer = None
        self._cancel_extra_timer()
        async with self._lock:
            snapshot = await self.hass.async_add_executor_job(self.engine.to_dict)
        self._snapshot = snapshot
        await self.store.async_save(self._store_data())

    # ------------------------------------------------------------ listeners

    @callback
    def async_add_listener(self, update_callback: CALLBACK_TYPE) -> CALLBACK_TYPE:
        """Register an entity update callback; returns the remove function."""
        self._listeners.append(update_callback)

        @callback
        def remove() -> None:
            if update_callback in self._listeners:
                self._listeners.remove(update_callback)

        return remove

    @callback
    def _notify(self) -> None:
        for update_callback in list(self._listeners):
            update_callback()

    # ------------------------------------------------------------- sensors

    @staticmethod
    def _now() -> float:
        return dt_util.utcnow().timestamp()

    def _feed(self, entity_id: str, state: State | None, ts: float) -> None:
        role, index = self._roles[entity_id]
        value = parse_power(state)
        if (
            value is None
            and state is not None
            and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN)
            and entity_id not in self._bad_unit_logged
        ):
            self._bad_unit_logged.add(entity_id)
            _LOGGER.warning(
                "Ignoring state of %s: not a numeric power value in a known unit",
                entity_id,
            )
        if value is not None and role == ROLE_GRID and self.invert_grid:
            value = -value
        self.resampler.update(role, index, ts, value)
        if role == ROLE_GRID:
            self.scheduler.grid_update(ts, value)
            self._schedule_extra_evaluation()

    @callback
    def _handle_state_event(self, event: Event[EventStateChangedData]) -> None:
        entity_id = event.data["entity_id"]
        new_state = event.data["new_state"]
        if entity_id == self.label_entity:
            self.external_label_on = (
                new_state is not None and new_state.state == STATE_ON
            )
            return
        if entity_id in self._roles:
            self._feed(entity_id, new_state, event.time_fired_timestamp)

    @callback
    def _handle_report_event(self, event: Event[EventStateReportedData]) -> None:
        entity_id = event.data["entity_id"]
        if entity_id in self._roles:
            role, index = self._roles[entity_id]
            self.resampler.report(role, index, event.time_fired_timestamp)

    # ----------------------------------------------------------- evaluation

    async def _async_tick(self, _now: datetime) -> None:
        if self.scheduler.regular_due(self._now()):
            await self.async_evaluate()

    @callback
    def _cancel_extra_timer(self) -> None:
        if self._extra_timer is not None:
            self._extra_timer()
            self._extra_timer = None
            self._extra_due = None

    @callback
    def _schedule_extra_evaluation(self) -> None:
        """(Re)schedule the extra evaluation requested by the scheduler."""
        if self._shutdown:
            return
        due = self.scheduler.next_due()
        if due is None:
            self._cancel_extra_timer()
            return
        if self._extra_due is not None and abs(self._extra_due - due) < 0.5:
            return
        self._cancel_extra_timer()
        self._extra_due = due
        self._extra_timer = async_call_later(
            self.hass, max(0.0, due - self._now()), self._async_extra_evaluation
        )

    async def _async_extra_evaluation(self, _now: datetime) -> None:
        self._extra_timer = None
        self._extra_due = None
        await self.async_evaluate()

    def _evaluate_job(
        self, inp: EvaluationInput
    ) -> tuple[EvaluationResult, dict[str, Any] | None]:
        """Run in the executor: evaluate and, if needed, snapshot the state."""
        result = self.engine.evaluate(inp)
        age = inp.now - self._snapshot_ts
        snapshot = None
        if (self.engine.dirty and age >= _SNAPSHOT_LEARNING_INTERVAL) or (
            self.engine.calibration_dirty and age >= _SNAPSHOT_CALIBRATION_INTERVAL
        ):
            snapshot = self.engine.to_dict()
        return result, snapshot

    @property
    def label_active(self) -> bool:
        """Return True if the internal or the external label is on."""
        return self.label_on or self.external_label_on

    async def async_evaluate(self) -> None:
        """Run one evaluation of the analysis window."""
        if self._shutdown:
            return
        now_dt = dt_util.utcnow()
        now = now_dt.timestamp()
        local = dt_util.as_local(now_dt)
        window = self.resampler.snapshot(now, self.config.buffer_seconds)
        label_source = LabelSource.SWITCH if self.label_on else LabelSource.EXTERNAL
        inp = EvaluationInput(
            now=now,
            window=window,
            local_hour=local.hour + local.minute / 60 + local.second / 3600,
            local_day=local.toordinal(),
            update_interval_s=self.resampler.effective_update_interval(),
            generation_interval_s=self.resampler.generation_update_interval(),
            label_active=self.label_active,
            label_group=int(self.label_since or now),
            label_source=label_source,
        )
        async with self._lock:
            result, snapshot = await self.hass.async_add_executor_job(
                self._evaluate_job, inp
            )
        self.data = result
        self.scheduler.evaluated(
            now,
            result,
            self.config.on_threshold,
            self.resampler.generation_update_interval(),
        )
        self._schedule_extra_evaluation()
        if snapshot is not None:
            self._schedule_save(snapshot, now)
        if result.question is not None:
            await self._async_ask(result.question)
        self._notify()

    # -------------------------------------------------------------- saving

    def _store_data(self) -> dict[str, Any]:
        return {
            "engine": self._snapshot or {},
            "label": {"on": self.label_on, "since": self.label_since},
        }

    def _schedule_save(
        self,
        snapshot: dict[str, Any] | None,
        now: float,
        delay: float = STORAGE_SAVE_DELAY,
    ) -> None:
        if snapshot is not None:
            self._snapshot = snapshot
            self._snapshot_ts = now
        if self._snapshot is None:
            return
        self.store.async_delay_save(self._store_data, delay)

    async def _async_run_and_save(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run an engine method in the executor and persist the result."""
        async with self._lock:
            result = await self.hass.async_add_executor_job(func, *args)
            snapshot = await self.hass.async_add_executor_job(self.engine.to_dict)
        self._schedule_save(snapshot, self._now(), _USER_ACTION_SAVE_DELAY)
        return result

    # --------------------------------------------------------- user actions

    async def async_set_label(self, on: bool) -> None:
        """Switch the built-in label "cooktop is on"."""
        if self._label_timer is not None:
            self._label_timer()
            self._label_timer = None
        if on:
            self.label_on = True
            self.label_since = self._now()
            self._label_timer = async_call_later(
                self.hass, self.config.max_label_s, self._async_label_timeout
            )
        else:
            self.label_on = False
            self.label_since = None
        self.store.async_delay_save(self._store_data, STORAGE_SAVE_DELAY)
        self._notify()

    async def _async_label_timeout(self, _now: datetime) -> None:
        self._label_timer = None
        _LOGGER.debug("Label switch reached its maximum duration, switching off")
        await self.async_set_label(False)

    async def async_false_alarm(self) -> int:
        """Mark the current or last detected event as a false alarm."""
        count: int = await self._async_run_and_save(
            self.engine.mark_false_alarm, self._now()
        )
        self.data = dataclasses.replace(self.data, is_on=False)
        self._notify()
        return count

    async def async_reset_model(self) -> None:
        """Forget the learned model and all labels."""
        await self._async_run_and_save(self.engine.reset_model)
        self.data = dataclasses.replace(
            self.data, model_weight=0.0, learned_levels=[], counts={}, quality={}
        )
        self._notify()

    async def async_set_learning_paused(self, paused: bool) -> None:
        """Pause or resume all model updates."""
        self.engine.learning_paused = paused
        async with self._lock:
            snapshot = await self.hass.async_add_executor_job(self.engine.to_dict)
        self._schedule_save(snapshot, self._now(), _USER_ACTION_SAVE_DELAY)
        self._notify()

    async def async_add_label(self, start: float, end: float, cooking: bool) -> int:
        """Label stored windows retroactively."""
        count: int = await self._async_run_and_save(
            self.engine.add_label, start, end, cooking, self._now()
        )
        return count

    async def async_answer_question(self, question_id: str, cooking: bool) -> bool:
        """Apply an answer to a pending question of this entry."""
        if question_id not in {q.question_id for q in self.engine.pending_questions}:
            return False
        ok: bool = await self._async_run_and_save(
            self.engine.answer_question, question_id, cooking, self._now()
        )
        return ok

    def has_question(self, question_id: str) -> bool:
        """Return True if ``question_id`` is pending for this entry."""
        return any(q.question_id == question_id for q in self.engine.pending_questions)

    async def async_engine_diagnostics(self) -> dict[str, Any]:
        """Return the engine diagnostics (serialised with the evaluations)."""
        async with self._lock:
            data: dict[str, Any] = await self.hass.async_add_executor_job(
                self.engine.diagnostics
            )
        return data

    async def async_export_training_data(self) -> dict[str, Any]:
        """Return the anonymised training data."""
        async with self._lock:
            data: dict[str, Any] = await self.hass.async_add_executor_job(
                self.engine.export_training_data
            )
        return data

    # ------------------------------------------------------------ questions

    async def _async_ask(self, question: Question) -> None:
        self.hass.bus.async_fire(
            EVENT_QUESTION,
            {
                ATTR_ENTRY_ID: self.entry.entry_id,
                ATTR_QUESTION_ID: question.question_id,
                ATTR_PROBABILITY: round(question.probability, 3),
            },
        )
        if not self.notify_service:
            return
        service = self.notify_service.removeprefix("notify.")
        if not self.hass.services.has_service("notify", service):
            _LOGGER.warning("Notification service notify.%s not found", service)
            return
        strings = await async_get_translations(
            self.hass, self.hass.config.language, "common", [DOMAIN]
        )

        def text(key: str) -> str:
            return strings.get(f"component.{DOMAIN}.common.{key}", key)

        prefix = f"{QUESTION_ACTION_PREFIX}_"
        await self.hass.services.async_call(
            "notify",
            service,
            {
                "title": text("question_title"),
                "message": text("question_message").replace("{name}", self.entry.title),
                "data": {
                    "tag": f"{DOMAIN}_{question.question_id}",
                    "actions": [
                        {
                            "action": f"{prefix}YES_{question.question_id}",
                            "title": text("answer_yes"),
                        },
                        {
                            "action": f"{prefix}NO_{question.question_id}",
                            "title": text("answer_no"),
                        },
                    ],
                },
            },
            blocking=False,
        )

    @callback
    def _handle_mobile_action(self, event: Event) -> None:
        action = str(event.data.get("action", ""))
        prefix = f"{QUESTION_ACTION_PREFIX}_"
        if not action.startswith(prefix):
            return
        answer, _, question_id = action.removeprefix(prefix).partition("_")
        if answer not in ("YES", "NO") or not self.has_question(question_id):
            return
        self.entry.async_create_background_task(
            self.hass,
            self.async_answer_question(question_id, answer == "YES"),
            f"{DOMAIN} answer {question_id}",
        )

    # ---------------------------------------------------------- diagnostics

    def diagnostics_config(self) -> dict[str, Any]:
        """Return the configuration without entity ids."""
        return {
            "profile": self.config.profile,
            "invert_grid": self.invert_grid,
            "generation_sensors": len(self.generation_entities),
            "subtract_sensors": len(self.subtract_entities),
            "external_label_source": self.label_entity is not None,
            "questions_enabled": self.config.questions_enabled,
            "notify_service_configured": self.notify_service is not None,
            "on_threshold": self.config.on_threshold,
            "off_threshold": self.config.off_threshold,
            "evaluation_interval_s": self.config.eval_interval_s,
            "window_s": self.config.window_s,
            "max_label_s": self.config.max_label_s,
            "max_questions_per_day": self.config.max_questions_per_day,
        }

    def data_quality(self) -> dict[str, Any]:
        """Return the current data quality figures."""
        now = self._now()
        age = self.resampler.seconds_since_grid_update(now)
        return {
            "update_interval_s": self.resampler.effective_update_interval(),
            "seconds_since_grid_update": round(age, 1) if age is not None else None,
            "valid_fraction": round(self.data.valid_fraction, 3),
            "evaluated": self.data.evaluated,
        }
