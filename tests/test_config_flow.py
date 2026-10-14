"""Tests for the config flow and the options flow."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import SOURCE_USER
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.herdwaechter.const import (
    CONF_EVAL_INTERVAL,
    CONF_GENERATION_SENSORS,
    CONF_GRID_SENSOR,
    CONF_INVERT_GRID,
    CONF_LABEL_ENTITY,
    CONF_MAX_LABEL_MINUTES,
    CONF_MAX_QUESTIONS_PER_DAY,
    CONF_NAME,
    CONF_NOTIFY_SERVICE,
    CONF_OFF_THRESHOLD,
    CONF_ON_THRESHOLD,
    CONF_PROFILE,
    CONF_QUESTIONS_ENABLED,
    CONF_SUBTRACT_SENSORS,
    CONF_WINDOW_MINUTES,
    DEFAULT_ON_THRESHOLD,
    DOMAIN,
)

GRID = "sensor.grid_power"
PV = "sensor.inverter_power"
DISH = "sensor.dishwasher_power"


def _states(hass: HomeAssistant) -> None:
    hass.states.async_set(GRID, "350", {"unit_of_measurement": "W"})
    hass.states.async_set(PV, "0.42", {"unit_of_measurement": "kW"})
    hass.states.async_set(DISH, "unavailable", {"unit_of_measurement": "W"})
    hass.states.async_set("sensor.temperature", "21.5", {"unit_of_measurement": "°C"})
    hass.states.async_set("sensor.text", "hello", {})
    hass.states.async_set("input_boolean.cooking", "off")


def _user_input(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        CONF_NAME: "Herd",
        CONF_GRID_SENSOR: GRID,
        CONF_INVERT_GRID: False,
        CONF_GENERATION_SENSORS: [PV],
        CONF_SUBTRACT_SENSORS: [DISH],
        CONF_LABEL_ENTITY: "input_boolean.cooking",
    }
    data.update(overrides)
    return data


async def test_user_flow_creates_entry(hass: HomeAssistant) -> None:
    _states(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _user_input()
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Herd"
    assert result["data"] == {CONF_PROFILE: "ceran"}
    options = result["options"]
    assert options[CONF_GRID_SENSOR] == GRID
    assert options[CONF_GENERATION_SENSORS] == [PV]
    assert options[CONF_ON_THRESHOLD] == DEFAULT_ON_THRESHOLD
    assert CONF_NAME not in options


async def test_user_flow_minimal_and_default_name(hass: HomeAssistant) -> None:
    _states(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_NAME: "  ", CONF_GRID_SENSOR: GRID}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Stove"
    assert result["options"][CONF_GENERATION_SENSORS] == []
    assert CONF_LABEL_ENTITY not in result["options"]


async def _submit(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    return dict(await hass.config_entries.flow.async_configure(result["flow_id"], data))


async def test_user_flow_errors(hass: HomeAssistant) -> None:
    _states(hass)
    cases = [
        ({CONF_GRID_SENSOR: "sensor.missing"}, {CONF_GRID_SENSOR: "entity_not_found"}),
        ({CONF_GRID_SENSOR: "sensor.text"}, {CONF_GRID_SENSOR: "not_numeric"}),
        ({CONF_GRID_SENSOR: "sensor.temperature"}, {CONF_GRID_SENSOR: "invalid_unit"}),
        (
            {CONF_GENERATION_SENSORS: [GRID]},
            {CONF_GRID_SENSOR: "grid_in_other_list"},
        ),
        ({CONF_SUBTRACT_SENSORS: [PV]}, {CONF_SUBTRACT_SENSORS: "duplicate_entity"}),
        (
            {CONF_GENERATION_SENSORS: [PV, PV]},
            {CONF_GENERATION_SENSORS: "duplicate_entity"},
        ),
        (
            {CONF_SUBTRACT_SENSORS: ["sensor.temperature"]},
            {CONF_SUBTRACT_SENSORS: "invalid_unit"},
        ),
        (
            {CONF_LABEL_ENTITY: "input_boolean.missing"},
            {CONF_LABEL_ENTITY: "entity_not_found"},
        ),
    ]
    for overrides, errors in cases:
        result = await _submit(hass, _user_input(**overrides))
        assert result["type"] is FlowResultType.FORM, overrides
        assert result["errors"] == errors, overrides


async def test_user_flow_aborts_on_duplicate(hass: HomeAssistant) -> None:
    _states(hass)
    MockConfigEntry(
        domain=DOMAIN,
        title="Herd",
        data={CONF_PROFILE: "ceran"},
        options={CONF_GRID_SENSOR: GRID},
    ).add_to_hass(hass)
    result = await _submit(hass, _user_input())
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_options_flow(hass: HomeAssistant) -> None:
    _states(hass)

    async def _notify(call: ServiceCall) -> None:
        """Do nothing."""

    hass.services.async_register("notify", "mobile_app_phone", _notify)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Herd",
        data={CONF_PROFILE: "ceran"},
        options={CONF_GRID_SENSOR: GRID, CONF_GENERATION_SENSORS: []},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    # errors are shown in the first step
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _user_input(**{CONF_GENERATION_SENSORS: [GRID]})
    )
    assert result["errors"] == {CONF_GRID_SENSOR: "grid_in_other_list"}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], _user_input(**{CONF_NAME: "Kitchen"})
    )
    assert result["step_id"] == "tuning"
    tuning = {
        CONF_ON_THRESHOLD: 70,
        CONF_OFF_THRESHOLD: 80,
        CONF_EVAL_INTERVAL: 20,
        CONF_WINDOW_MINUTES: 8,
        CONF_MAX_LABEL_MINUTES: 90,
        CONF_QUESTIONS_ENABLED: True,
        CONF_NOTIFY_SERVICE: "notify.unknown",
        CONF_MAX_QUESTIONS_PER_DAY: 2,
    }
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], tuning
    )
    assert result["errors"] == {
        CONF_OFF_THRESHOLD: "off_not_below_on",
        CONF_NOTIFY_SERVICE: "notify_not_found",
    }
    tuning[CONF_OFF_THRESHOLD] = 30
    tuning[CONF_NOTIFY_SERVICE] = "notify.mobile_app_phone"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], tuning
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.title == "Kitchen"
    assert entry.options[CONF_EVAL_INTERVAL] == 20
    assert entry.options[CONF_NOTIFY_SERVICE] == "mobile_app_phone"
    assert entry.options[CONF_GENERATION_SENSORS] == [PV]
    # the entry was reloaded with the new options
    coordinator = entry.runtime_data
    assert coordinator.config.eval_interval_s == 20
    assert coordinator.config.questions_enabled
