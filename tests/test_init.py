"""Integration tests: setup, entities, persistence, services, diagnostics."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
from typing import Any

from freezegun.api import FrozenDateTimeFactory
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, ServiceCall, State
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util
import numpy as np
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_capture_events,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.components.diagnostics import (
    get_diagnostics_for_config_entry,
)
from pytest_homeassistant_custom_component.typing import ClientSessionGenerator

from custom_components.herdwaechter.const import (
    CONF_EVAL_INTERVAL,
    CONF_GENERATION_SENSORS,
    CONF_GRID_SENSOR,
    CONF_INVERT_GRID,
    CONF_LABEL_ENTITY,
    CONF_MAX_LABEL_MINUTES,
    CONF_NOTIFY_SERVICE,
    CONF_PROFILE,
    CONF_QUESTIONS_ENABLED,
    CONF_SUBTRACT_SENSORS,
    DOMAIN,
    EVENT_QUESTION,
)
from custom_components.herdwaechter.coordinator import (
    HerdwaechterCoordinator,
    parse_power,
)
from custom_components.herdwaechter.engine import Question
from custom_components.herdwaechter.learning import LabelSource

GRID = "sensor.grid_power"
PV = "sensor.inverter_power"
DISH = "sensor.dishwasher_power"
LABEL = "input_boolean.cooking"


@pytest.fixture
def entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a configured (not yet loaded) entry."""
    hass.states.async_set(GRID, "350", {"unit_of_measurement": "W"})
    hass.states.async_set(PV, "0.2", {"unit_of_measurement": "kW"})
    hass.states.async_set(DISH, "0", {"unit_of_measurement": "W"})
    hass.states.async_set(LABEL, "off")
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Herd",
        data={CONF_PROFILE: "ceran"},
        options={
            CONF_GRID_SENSOR: GRID,
            CONF_INVERT_GRID: False,
            CONF_GENERATION_SENSORS: [PV],
            CONF_SUBTRACT_SENSORS: [DISH],
            CONF_LABEL_ENTITY: LABEL,
            CONF_EVAL_INTERVAL: 30,
            CONF_MAX_LABEL_MINUTES: 10,
        },
    )
    entry.add_to_hass(hass)
    return entry


async def _setup(
    hass: HomeAssistant, entry: MockConfigEntry
) -> HerdwaechterCoordinator:
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    coordinator: HerdwaechterCoordinator = entry.runtime_data
    return coordinator


EXPECTED = {
    "binary_sensor.herd": "cooktop",
    "sensor.herd_probability": "probability",
    "sensor.herd_confidence": "confidence",
    "sensor.herd_learning_state": "learning_state",
    "sensor.herd_residual_load": "residual_load",
    "switch.herd_label_cooktop_on": "cooking_label",
    "switch.herd_pause_learning": "pause_learning",
    "button.herd_false_alarm": "false_alarm",
    "button.herd_reset_model": "reset_model",
}


def test_parse_power() -> None:
    def st(value: str, unit: str | None = "W") -> State:
        attrs = {} if unit is None else {"unit_of_measurement": unit}
        return State("sensor.x", value, attrs)

    assert parse_power(st("1500")) == 1500.0
    assert parse_power(st("1.5", "kW")) == pytest.approx(1500.0)
    assert parse_power(st("-0.3", "kW")) == pytest.approx(-300.0)
    assert parse_power(st("2500000", "mW")) == pytest.approx(2500.0)
    assert parse_power(st("700", None)) == 700.0
    assert parse_power(st("12", "°C")) is None
    assert parse_power(st("unavailable")) is None
    assert parse_power(st("unknown")) is None
    assert parse_power(st("abc")) is None
    assert parse_power(st("nan")) is None
    assert parse_power(None) is None


async def test_setup_creates_entities_and_device(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    await _setup(hass, entry)
    ent_reg = er.async_get(hass)
    for entity_id, key in EXPECTED.items():
        reg = ent_reg.async_get(entity_id)
        assert reg is not None, entity_id
        assert reg.unique_id == f"{entry.entry_id}_{key}"
        assert reg.translation_key == key
    assert ent_reg.async_get("sensor.herd_residual_load").disabled_by is not None
    assert ent_reg.async_get("button.herd_reset_model").disabled_by is not None
    assert (
        ent_reg.async_get("sensor.herd_learning_state").entity_category == "diagnostic"
    )
    assert ent_reg.async_get("switch.herd_pause_learning").entity_category == "config"
    device = dr.async_get(hass).async_get_device_by_identifier(
        (DOMAIN, entry.entry_id), entry.entry_id
    )
    assert device is not None
    assert device.name == "Herd"
    state = hass.states.get("binary_sensor.herd")
    assert state is not None
    assert state.state == STATE_OFF
    assert state.attributes["device_class"] == "running"
    assert state.attributes["learning_state"] == "calibrating"
    assert hass.states.get("sensor.herd_probability").state == "unknown"


async def test_unload_removes_listeners(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    coordinator = await _setup(hass, entry)
    await coordinator.async_set_label(True)
    assert coordinator._label_timer is not None
    assert coordinator._unsubs
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.NOT_LOADED
    assert coordinator._unsubs == []
    assert coordinator._label_timer is None
    assert hass.states.get("binary_sensor.herd").state == STATE_UNAVAILABLE
    # sensor updates after unload must not raise
    hass.states.async_set(GRID, "4000", {"unit_of_measurement": "W"})
    await coordinator.async_evaluate()
    await hass.async_block_till_done()


async def test_evaluation_timer_and_sensor_values(
    hass: HomeAssistant, entry: MockConfigEntry, freezer: FrozenDateTimeFactory
) -> None:
    coordinator = await _setup(hass, entry)
    # feed a cycling 1.8 kW element with 3 s updates, including kW / gaps
    t = 0
    for step in range(450):
        t += 3
        on = (step // 7) % 2 == 1
        freezer.tick(timedelta(seconds=3))
        value = 300 + (1800 if on else 0) + step % 3
        hass.states.async_set(GRID, str(value), {"unit_of_measurement": "W"})
        if step % 5 == 0:
            hass.states.async_set(
                PV, str(0.1 + 0.001 * step), {"unit_of_measurement": "kW"}
            )
        if step == 200:
            hass.states.async_set(DISH, "unavailable")
        if step % 10 == 0:
            async_fire_time_changed(hass, dt_util.utcnow())
            await hass.async_block_till_done()
    await coordinator.async_evaluate()
    await hass.async_block_till_done()
    data = coordinator.data
    assert data.evaluated
    assert data.probability is not None
    assert data.update_interval_s == pytest.approx(3.0, abs=0.5)
    prob = hass.states.get("sensor.herd_probability")
    assert float(prob.state) == pytest.approx(100 * data.probability, abs=0.1)
    assert float(hass.states.get("sensor.herd_confidence").state) >= 0
    attrs = hass.states.get("sensor.herd_learning_state").attributes
    assert attrs["update_interval_s"] == pytest.approx(3.0, abs=0.5)
    assert "edge_threshold_w" in attrs
    cooktop = hass.states.get("binary_sensor.herd")
    assert cooktop.attributes["probability"] == pytest.approx(
        100 * data.probability, abs=0.1
    )
    assert data.detected_period_s == pytest.approx(42, abs=4)


async def test_label_switch_turns_off_automatically(
    hass: HomeAssistant, entry: MockConfigEntry, freezer: FrozenDateTimeFactory
) -> None:
    coordinator = await _setup(hass, entry)
    await hass.services.async_call(
        "switch",
        "turn_on",
        {"entity_id": "switch.herd_label_cooktop_on"},
        blocking=True,
    )
    state = hass.states.get("switch.herd_label_cooktop_on")
    assert state.state == STATE_ON
    assert 0 < state.attributes["auto_off_in_s"] <= 600
    assert coordinator.label_active
    freezer.tick(timedelta(minutes=11))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()
    assert hass.states.get("switch.herd_label_cooktop_on").state == STATE_OFF
    assert (
        hass.states.get("switch.herd_label_cooktop_on").attributes["auto_off_in_s"]
        is None
    )
    # manual off and the external label source
    await hass.services.async_call(
        "switch",
        "turn_on",
        {"entity_id": "switch.herd_label_cooktop_on"},
        blocking=True,
    )
    await hass.services.async_call(
        "switch",
        "turn_off",
        {"entity_id": "switch.herd_label_cooktop_on"},
        blocking=True,
    )
    assert not coordinator.label_active
    hass.states.async_set(LABEL, "on")
    await hass.async_block_till_done()
    assert coordinator.label_active


async def test_pause_reset_and_false_alarm(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    coordinator = await _setup(hass, entry)
    await hass.services.async_call(
        "switch", "turn_on", {"entity_id": "switch.herd_pause_learning"}, blocking=True
    )
    assert coordinator.engine.learning_paused
    assert hass.states.get("switch.herd_pause_learning").state == STATE_ON
    await hass.services.async_call(
        "switch", "turn_off", {"entity_id": "switch.herd_pause_learning"}, blocking=True
    )
    assert not coordinator.engine.learning_paused
    await hass.services.async_call(
        "button", "press", {"entity_id": "button.herd_false_alarm"}, blocking=True
    )
    assert not coordinator.data.is_on
    coordinator.engine.dataset.add(
        np.zeros(coordinator.engine.dataset.n_features),
        1,
        1.0,
        1.0,
        LabelSource.SWITCH,
        1,
    )
    ent_reg = er.async_get(hass)
    ent_reg.async_update_entity("button.herd_reset_model", disabled_by=None)
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    coordinator = entry.runtime_data
    assert len(coordinator.engine.dataset) == 1  # persisted over the reload
    await hass.services.async_call(
        "button", "press", {"entity_id": "button.herd_reset_model"}, blocking=True
    )
    assert len(coordinator.engine.dataset) == 0


async def test_persistence_over_restart(
    hass: HomeAssistant, entry: MockConfigEntry, hass_storage: dict[str, Any]
) -> None:
    coordinator = await _setup(hass, entry)
    coordinator.engine.calibrator.lag_s = 4.2
    coordinator.engine.calibrator.lag_estimates = 12
    coordinator.engine.clusterer.add(1800.0, 50.0)
    await coordinator.async_set_label(True)
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    stored = hass_storage[f"{DOMAIN}.{entry.entry_id}"]
    assert stored["version"] == 1
    assert stored["data"]["engine"]["calibration"]["lag_s"] == pytest.approx(4.2)
    assert stored["data"]["label"]["on"] is True
    assert "sensor." not in json.dumps(stored)  # no entity ids, no raw data

    coordinator = await _setup(hass, entry)
    assert coordinator.engine.calibrator.lag_s == pytest.approx(4.2)
    assert coordinator.engine.clusterer.levels().tolist() == [1800.0]
    assert coordinator.label_on
    assert coordinator._label_timer is not None
    attrs = hass.states.get("binary_sensor.herd").attributes
    assert attrs["learned_power_levels"] == []  # shown after the next evaluation

    # removing the entry deletes the stored data
    assert await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()
    assert f"{DOMAIN}.{entry.entry_id}" not in hass_storage


async def test_services(
    hass: HomeAssistant, entry: MockConfigEntry, tmp_path: Path
) -> None:
    coordinator = await _setup(hass, entry)
    hass.config.config_dir = str(tmp_path)
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            "answer_question",
            {"question_id": "nope", "answer": True},
            blocking=True,
        )
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            "export_training_data",
            {"config_entry_id": "missing"},
            blocking=True,
            return_response=True,
        )
    now = dt_util.utcnow()
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            "add_label",
            {
                "config_entry_id": entry.entry_id,
                "start": now,
                "end": now - timedelta(hours=1),
                "cooking": True,
            },
            blocking=True,
            return_response=True,
        )
    response = await hass.services.async_call(
        DOMAIN,
        "add_label",
        {
            "config_entry_id": entry.entry_id,
            # naive datetimes are interpreted in the configured time zone
            "start": dt_util.as_local(now - timedelta(hours=1)).replace(tzinfo=None),
            "end": now,
            "cooking": False,
        },
        blocking=True,
        return_response=True,
    )
    assert response == {"labelled_windows": 0}
    response = await hass.services.async_call(
        DOMAIN,
        "export_training_data",
        {"config_entry_id": entry.entry_id},
        blocking=True,
        return_response=True,
    )
    assert response["samples"] == 0

    def _read_export() -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in tmp_path.glob("herdwaechter_export_*.json")
        ]

    exports = await hass.async_add_executor_job(_read_export)
    assert len(exports) == 1
    payload = exports[0]
    assert payload["profile"] == "ceran"

    # questions: answered via service and via the mobile app action
    for qid in ("q1", "q2"):
        coordinator.engine._questions[qid] = Question(qid, coordinator._now(), 0.6)
    await hass.services.async_call(
        DOMAIN, "answer_question", {"question_id": "q1", "answer": True}, blocking=True
    )
    assert not coordinator.has_question("q1")
    hass.bus.async_fire(
        "mobile_app_notification_action", {"action": "HERDWAECHTER_NO_q2"}
    )
    hass.bus.async_fire("mobile_app_notification_action", {"action": "OTHER_ACTION"})
    hass.bus.async_fire(
        "mobile_app_notification_action", {"action": "HERDWAECHTER_X_q9"}
    )
    await hass.async_block_till_done()
    assert not coordinator.has_question("q2")
    assert not await coordinator.async_answer_question("unknown", True)

    # unloaded entries are rejected
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            "export_training_data",
            {"config_entry_id": entry.entry_id},
            blocking=True,
            return_response=True,
        )


async def test_question_event_and_notification(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    calls: list[ServiceCall] = []

    async def _notify(call: ServiceCall) -> None:
        calls.append(call)

    hass.services.async_register("notify", "mobile_app_phone", _notify)
    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_QUESTIONS_ENABLED: True,
            CONF_NOTIFY_SERVICE: "mobile_app_phone",
        },
    )
    coordinator = await _setup(hass, entry)
    events = async_capture_events(hass, EVENT_QUESTION)
    await coordinator._async_ask(Question("abc", coordinator._now(), 0.55))
    await hass.async_block_till_done()
    assert events[0].data == {
        "entry_id": entry.entry_id,
        "question_id": "abc",
        "probability": 0.55,
    }
    assert len(calls) == 1
    actions = calls[0].data["data"]["actions"]
    assert [a["action"] for a in actions] == [
        "HERDWAECHTER_YES_abc",
        "HERDWAECHTER_NO_abc",
    ]
    assert calls[0].data["title"] == "Is the cooktop on?"
    assert "Herd" in calls[0].data["message"]
    # a missing notify service only logs a warning
    coordinator.notify_service = "gone"
    await coordinator._async_ask(Question("def", coordinator._now(), 0.5))
    assert len(calls) == 1


async def test_diagnostics_without_entity_ids(
    hass: HomeAssistant, entry: MockConfigEntry, hass_client: ClientSessionGenerator
) -> None:
    coordinator = await _setup(hass, entry)
    await coordinator.async_evaluate()
    diag = await get_diagnostics_for_config_entry(hass, hass_client, entry)
    text = json.dumps(diag)
    assert "sensor." not in text
    assert "input_boolean." not in text
    assert diag["configuration"]["generation_sensors"] == 1
    assert diag["configuration"]["external_label_source"] is True
    assert "engine" in diag
    assert "model" in diag["engine"]


async def test_invalid_units_and_inverted_grid(
    hass: HomeAssistant, entry: MockConfigEntry, caplog: pytest.LogCaptureFixture
) -> None:
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_INVERT_GRID: True}
    )
    coordinator = await _setup(hass, entry)
    hass.states.async_set(GRID, "-500", {"unit_of_measurement": "W"})
    hass.states.async_set(PV, "12", {"unit_of_measurement": "°C"})
    hass.states.async_set(PV, "13", {"unit_of_measurement": "°C"})
    hass.states.async_set(GRID, "unknown")
    await hass.async_block_till_done()
    assert caplog.text.count("not a numeric power value") == 1
    await coordinator.async_evaluate()
    assert not coordinator.data.evaluated  # not enough data yet, but no error
