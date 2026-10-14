"""Config flow and options flow of the Herdwächter integration."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er, selector
from homeassistant.util.unit_conversion import PowerConverter
import voluptuous as vol

from .const import (
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
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_MAX_LABEL_MINUTES,
    DEFAULT_MAX_QUESTIONS_PER_DAY,
    DEFAULT_OFF_THRESHOLD,
    DEFAULT_ON_THRESHOLD,
    DEFAULT_PROFILE,
    DEFAULT_QUESTIONS_ENABLED,
    DEFAULT_WINDOW_MINUTES,
    DOMAIN,
    MAX_EVAL_INTERVAL,
    MAX_MAX_LABEL_MINUTES,
    MAX_QUESTIONS_LIMIT,
    MAX_WINDOW_MINUTES,
    MIN_EVAL_INTERVAL,
    MIN_MAX_LABEL_MINUTES,
    MIN_WINDOW_MINUTES,
)

SENSOR_KEYS = (
    CONF_GRID_SENSOR,
    CONF_INVERT_GRID,
    CONF_GENERATION_SENSORS,
    CONF_SUBTRACT_SENSORS,
    CONF_LABEL_ENTITY,
)

DEFAULT_OPTIONS: dict[str, Any] = {
    CONF_ON_THRESHOLD: DEFAULT_ON_THRESHOLD,
    CONF_OFF_THRESHOLD: DEFAULT_OFF_THRESHOLD,
    CONF_EVAL_INTERVAL: DEFAULT_EVAL_INTERVAL,
    CONF_WINDOW_MINUTES: DEFAULT_WINDOW_MINUTES,
    CONF_MAX_LABEL_MINUTES: DEFAULT_MAX_LABEL_MINUTES,
    CONF_QUESTIONS_ENABLED: DEFAULT_QUESTIONS_ENABLED,
    CONF_MAX_QUESTIONS_PER_DAY: DEFAULT_MAX_QUESTIONS_PER_DAY,
}

_POWER_SENSOR = selector.EntitySelectorConfig(
    domain="sensor", device_class=SensorDeviceClass.POWER
)
_POWER_SENSORS = selector.EntitySelectorConfig(
    domain="sensor", device_class=SensorDeviceClass.POWER, multiple=True
)
_LABEL_ENTITY = selector.EntitySelectorConfig(
    domain=["input_boolean", "switch", "binary_sensor"]
)


def _as_flow_schema(schema: vol.Schema) -> Any:
    """Return ``schema`` for Home Assistant's flow helpers.

    Home Assistant >= 2026.9 validates with probatio and aliases ``voluptuous``
    to it at runtime; older releases (the minimum is 2026.3) use voluptuous
    itself. Importing voluptuous works on both, only the static types differ.
    """
    return schema


def default_name(hass: HomeAssistant) -> str:
    """Return the default device name in the configured language."""
    return "Herd" if hass.config.language.startswith("de") else "Stove"


def sensor_schema(name_default: str) -> vol.Schema:
    """Return the schema of the sensor selection."""
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=name_default): selector.TextSelector(),
            vol.Required(CONF_GRID_SENSOR): selector.EntitySelector(_POWER_SENSOR),
            vol.Optional(CONF_INVERT_GRID, default=False): selector.BooleanSelector(),
            vol.Optional(CONF_GENERATION_SENSORS): selector.EntitySelector(
                _POWER_SENSORS
            ),
            vol.Optional(CONF_SUBTRACT_SENSORS): selector.EntitySelector(
                _POWER_SENSORS
            ),
            vol.Optional(CONF_LABEL_ENTITY): selector.EntitySelector(_LABEL_ENTITY),
        }
    )


def _number(minimum: float, maximum: float, unit: str) -> selector.NumberSelector:
    return selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=minimum,
            max=maximum,
            step=1,
            unit_of_measurement=unit,
            mode=selector.NumberSelectorMode.BOX,
        )
    )


def options_schema(hass: HomeAssistant) -> vol.Schema:
    """Return the schema of the tuning options."""
    notify_services = sorted(hass.services.async_services_for_domain("notify"))
    return vol.Schema(
        {
            vol.Required(CONF_ON_THRESHOLD): _number(5, 99, "%"),
            vol.Required(CONF_OFF_THRESHOLD): _number(1, 95, "%"),
            vol.Required(CONF_EVAL_INTERVAL): _number(
                MIN_EVAL_INTERVAL, MAX_EVAL_INTERVAL, "s"
            ),
            vol.Required(CONF_WINDOW_MINUTES): _number(
                MIN_WINDOW_MINUTES, MAX_WINDOW_MINUTES, "min"
            ),
            vol.Required(CONF_MAX_LABEL_MINUTES): _number(
                MIN_MAX_LABEL_MINUTES, MAX_MAX_LABEL_MINUTES, "min"
            ),
            vol.Required(CONF_QUESTIONS_ENABLED): selector.BooleanSelector(),
            vol.Optional(CONF_NOTIFY_SERVICE): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=notify_services,
                    custom_value=True,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(CONF_MAX_QUESTIONS_PER_DAY): _number(
                1, MAX_QUESTIONS_LIMIT, ""
            ),
        }
    )


def _entity_known(hass: HomeAssistant, entity_id: str) -> bool:
    return (
        hass.states.get(entity_id) is not None
        or er.async_get(hass).async_get(entity_id) is not None
    )


def _check_power_entity(hass: HomeAssistant, entity_id: str) -> str | None:
    """Return an error key if ``entity_id`` is not a usable power sensor."""
    state = hass.states.get(entity_id)
    if state is None:
        return None if _entity_known(hass, entity_id) else "entity_not_found"
    unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    if unit not in (None, "") and unit not in PowerConverter.VALID_UNITS:
        return "invalid_unit"
    if state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        return None
    try:
        float(state.state)
    except ValueError:
        return "not_numeric"
    return None


def validate_sensors(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, str]:
    """Validate the sensor selection; return a mapping field -> error key."""
    errors: dict[str, str] = {}
    grid: str = data.get(CONF_GRID_SENSOR) or ""
    generation: list[str] = list(data.get(CONF_GENERATION_SENSORS) or [])
    subtract: list[str] = list(data.get(CONF_SUBTRACT_SENSORS) or [])
    label: str = data.get(CONF_LABEL_ENTITY) or ""

    if not grid:
        errors[CONF_GRID_SENSOR] = "entity_not_found"
    elif error := _check_power_entity(hass, grid):
        errors[CONF_GRID_SENSOR] = error
    for key, entities in (
        (CONF_GENERATION_SENSORS, generation),
        (CONF_SUBTRACT_SENSORS, subtract),
    ):
        for entity_id in entities:
            if error := _check_power_entity(hass, entity_id):
                errors[key] = error
                break
        if len(set(entities)) != len(entities):
            errors[key] = "duplicate_entity"
    if grid and (grid in generation or grid in subtract):
        errors[CONF_GRID_SENSOR] = "grid_in_other_list"
    if set(generation) & set(subtract):
        errors[CONF_SUBTRACT_SENSORS] = "duplicate_entity"
    # the selector restricts the label source to on/off domains, so it can
    # never collide with one of the power sensors
    if label and not _entity_known(hass, label):
        errors[CONF_LABEL_ENTITY] = "entity_not_found"
    return errors


def validate_options(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, str]:
    """Validate the tuning options."""
    errors: dict[str, str] = {}
    if float(data[CONF_OFF_THRESHOLD]) >= float(data[CONF_ON_THRESHOLD]):
        errors[CONF_OFF_THRESHOLD] = "off_not_below_on"
    service = str(data.get(CONF_NOTIFY_SERVICE) or "").removeprefix("notify.")
    if service and not hass.services.has_service("notify", service):
        errors[CONF_NOTIFY_SERVICE] = "notify_not_found"
    return errors


def _clean_sensors(data: dict[str, Any]) -> dict[str, Any]:
    """Normalise the sensor selection (empty optional values are dropped)."""
    out = dict(data)
    if not out.get(CONF_LABEL_ENTITY):
        out.pop(CONF_LABEL_ENTITY, None)
    for key in (CONF_GENERATION_SENSORS, CONF_SUBTRACT_SENSORS):
        out[key] = list(out.get(key) or [])
    out[CONF_INVERT_GRID] = bool(out.get(CONF_INVERT_GRID, False))
    return out


def _clean_tuning(data: dict[str, Any]) -> dict[str, Any]:
    """Normalise the tuning options."""
    out = dict(data)
    if out.get(CONF_NOTIFY_SERVICE):
        out[CONF_NOTIFY_SERVICE] = str(out[CONF_NOTIFY_SERVICE]).removeprefix("notify.")
    else:
        out.pop(CONF_NOTIFY_SERVICE, None)
    for key in (
        CONF_ON_THRESHOLD,
        CONF_OFF_THRESHOLD,
        CONF_EVAL_INTERVAL,
        CONF_WINDOW_MINUTES,
        CONF_MAX_LABEL_MINUTES,
        CONF_MAX_QUESTIONS_PER_DAY,
    ):
        if key in out:
            out[key] = int(out[key])
    return out


class HerdwaechterConfigFlow(ConfigFlow, domain=DOMAIN):
    """Create one config entry per monitored appliance."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select name and sensors."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = validate_sensors(self.hass, user_input)
            if not errors:
                data = _clean_sensors(user_input)
                for entry in self._async_current_entries(include_ignore=False):
                    options = {**entry.data, **entry.options}
                    if (
                        options.get(CONF_GRID_SENSOR) == data[CONF_GRID_SENSOR]
                        and options.get(CONF_PROFILE) == DEFAULT_PROFILE
                    ):
                        return self.async_abort(reason="already_configured")
                name = str(data.pop(CONF_NAME)).strip() or default_name(self.hass)
                return self.async_create_entry(
                    title=name,
                    data={CONF_PROFILE: DEFAULT_PROFILE},
                    options={**DEFAULT_OPTIONS, **data},
                )
        schema = self.add_suggested_values_to_schema(
            _as_flow_schema(sensor_schema(default_name(self.hass))),
            user_input or {},
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> HerdwaechterOptionsFlow:
        """Return the options flow."""
        return HerdwaechterOptionsFlow()


class HerdwaechterOptionsFlow(OptionsFlowWithReload):
    """Change sensors and tuning options; the entry reloads afterwards."""

    def __init__(self) -> None:
        """Initialise the flow state."""
        self._sensors: dict[str, Any] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 1: name and sensors."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = validate_sensors(self.hass, user_input)
            if not errors:
                self._sensors = _clean_sensors(user_input)
                return await self.async_step_tuning()
        current = {
            CONF_NAME: self.config_entry.title,
            **{k: v for k, v in self.config_entry.options.items() if k in SENSOR_KEYS},
        }
        schema = self.add_suggested_values_to_schema(
            _as_flow_schema(sensor_schema(self.config_entry.title)),
            user_input or current,
        )
        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)

    async def async_step_tuning(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 2: thresholds, timing and active questions."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = validate_options(self.hass, user_input)
            if not errors:
                sensors = dict(self._sensors)
                name = str(sensors.pop(CONF_NAME, "")).strip()
                if name and name != self.config_entry.title:
                    self.hass.config_entries.async_update_entry(
                        self.config_entry, title=name
                    )
                return self.async_create_entry(
                    data={**sensors, **_clean_tuning(user_input)}
                )
        current = {**DEFAULT_OPTIONS, **self.config_entry.options}
        schema = self.add_suggested_values_to_schema(
            _as_flow_schema(options_schema(self.hass)),
            user_input or current,
        )
        return self.async_show_form(step_id="tuning", data_schema=schema, errors=errors)
