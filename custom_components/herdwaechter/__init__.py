"""The Herdwächter integration.

Detects a running radiant cooktop from the total household power measured
at the electricity meter, learns continuously from optional user labels and
exposes the result as sensors. Everything runs locally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN
from .coordinator import HerdwaechterCoordinator, HerdwaechterStore
from .services import async_setup_services

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.typing import ConfigType

PLATFORMS: list[Platform] = [
    Platform.BINARY_SENSOR,
    Platform.BUTTON,
    Platform.SENSOR,
    Platform.SWITCH,
]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type HerdwaechterConfigEntry = ConfigEntry[HerdwaechterCoordinator]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the integration services."""
    async_setup_services(hass)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: HerdwaechterConfigEntry
) -> bool:
    """Set up one monitored appliance from a config entry."""
    coordinator = HerdwaechterCoordinator(hass, entry)
    await coordinator.async_setup()
    entry.runtime_data = coordinator
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception:
        await coordinator.async_shutdown()
        raise
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: HerdwaechterConfigEntry
) -> bool:
    """Unload a config entry: platforms, listeners, timers; save the state."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        await entry.runtime_data.async_shutdown()
    return unload_ok


async def async_remove_entry(
    hass: HomeAssistant, entry: HerdwaechterConfigEntry
) -> None:
    """Delete the stored model and labels when the entry is removed."""
    await HerdwaechterStore(hass, entry.entry_id).async_remove()
