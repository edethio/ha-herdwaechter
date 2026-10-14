"""Services of the Herdwächter integration."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.util import dt as dt_util
import voluptuous as vol

from .const import (
    ATTR_ANSWER,
    ATTR_CONFIG_ENTRY_ID,
    ATTR_COOKING,
    ATTR_END,
    ATTR_QUESTION_ID,
    ATTR_START,
    DOMAIN,
    SERVICE_ADD_LABEL,
    SERVICE_ANSWER_QUESTION,
    SERVICE_EXPORT_TRAINING_DATA,
)

if TYPE_CHECKING:
    from .coordinator import HerdwaechterCoordinator

ANSWER_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_QUESTION_ID): cv.string,
        vol.Required(ATTR_ANSWER): cv.boolean,
    }
)
ADD_LABEL_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Required(ATTR_START): cv.datetime,
        vol.Required(ATTR_END): cv.datetime,
        vol.Required(ATTR_COOKING): cv.boolean,
    }
)
EXPORT_SCHEMA = vol.Schema({vol.Required(ATTR_CONFIG_ENTRY_ID): cv.string})


def _as_service_schema(schema: vol.Schema) -> Any:
    """Return ``schema`` for the service registry.

    Home Assistant >= 2026.9 validates with probatio and aliases ``voluptuous``
    to it at runtime; older releases (the minimum is 2026.3) use voluptuous
    itself. Importing voluptuous works on both, only the static types differ.
    """
    return schema


def _coordinator(hass: HomeAssistant, entry_id: str) -> HerdwaechterCoordinator:
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="entry_not_found",
            translation_placeholders={"entry_id": entry_id},
        )
    if entry.state is not ConfigEntryState.LOADED:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="entry_not_loaded",
            translation_placeholders={"entry_id": entry_id},
        )
    coordinator: HerdwaechterCoordinator = entry.runtime_data
    return coordinator


def _timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt_util.get_default_time_zone())
    return value.timestamp()


def _loaded_coordinators(hass: HomeAssistant) -> list[HerdwaechterCoordinator]:
    return [
        entry.runtime_data
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.state is ConfigEntryState.LOADED
    ]


def _write_export(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register all services (once per Home Assistant instance)."""

    async def answer_question(call: ServiceCall) -> None:
        question_id: str = call.data[ATTR_QUESTION_ID]
        for coordinator in _loaded_coordinators(hass):
            if coordinator.has_question(question_id):
                await coordinator.async_answer_question(
                    question_id, bool(call.data[ATTR_ANSWER])
                )
                return
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="question_not_found",
            translation_placeholders={"question_id": question_id},
        )

    async def add_label(call: ServiceCall) -> ServiceResponse:
        coordinator = _coordinator(hass, call.data[ATTR_CONFIG_ENTRY_ID])
        start = _timestamp(call.data[ATTR_START])
        end = _timestamp(call.data[ATTR_END])
        if end <= start:
            raise ServiceValidationError(
                translation_domain=DOMAIN, translation_key="invalid_time_range"
            )
        count = await coordinator.async_add_label(
            start, end, bool(call.data[ATTR_COOKING])
        )
        return {"labelled_windows": count}

    async def export_training_data(call: ServiceCall) -> ServiceResponse:
        entry_id: str = call.data[ATTR_CONFIG_ENTRY_ID]
        coordinator = _coordinator(hass, entry_id)
        payload = await coordinator.async_export_training_data()
        stamp = dt_util.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = Path(hass.config.path(f"{DOMAIN}_export_{stamp}.json"))
        await hass.async_add_executor_job(_write_export, path, payload)
        return {"path": str(path.name), "samples": len(payload["samples"])}

    hass.services.async_register(
        DOMAIN,
        SERVICE_ANSWER_QUESTION,
        answer_question,
        schema=_as_service_schema(ANSWER_SCHEMA),
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_ADD_LABEL,
        add_label,
        schema=_as_service_schema(ADD_LABEL_SCHEMA),
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_TRAINING_DATA,
        export_training_data,
        schema=_as_service_schema(EXPORT_SCHEMA),
        supports_response=SupportsResponse.OPTIONAL,
    )
