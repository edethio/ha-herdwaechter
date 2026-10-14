"""Diagnostics: configuration (without entity ids), model and statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from . import HerdwaechterConfigEntry


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: HerdwaechterConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry.

    Entity ids and names are intentionally left out: only counts of the
    configured sensors are reported.
    """
    coordinator = entry.runtime_data
    engine_diag = await coordinator.async_engine_diagnostics()
    data = coordinator.data
    return {
        "configuration": coordinator.diagnostics_config(),
        "data_quality": coordinator.data_quality(),
        "current": {
            "probability": data.probability,
            "prior_probability": data.prior_probability,
            "model_probability": data.model_probability,
            "confidence": data.confidence,
            "is_on": data.is_on,
            "learning_state": data.learning_state,
            "detected_period_s": data.detected_period_s,
        },
        "label_switch_on": coordinator.label_on,
        "engine": engine_diag,
    }
