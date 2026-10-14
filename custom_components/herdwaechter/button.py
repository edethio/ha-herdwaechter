"""Buttons: false alarm and model reset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.button import ButtonEntity
from homeassistant.const import EntityCategory

from .entity import HerdwaechterEntity

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from . import HerdwaechterConfigEntry
    from .coordinator import HerdwaechterCoordinator

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HerdwaechterConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the buttons."""
    coordinator = entry.runtime_data
    async_add_entities([FalseAlarmButton(coordinator), ResetModelButton(coordinator)])


class FalseAlarmButton(HerdwaechterEntity, ButtonEntity):
    """Marks the current or last detected event as "not the cooktop"."""

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "false_alarm")

    async def async_press(self) -> None:
        """Label the event as negative."""
        await self.coordinator.async_false_alarm()


class ResetModelButton(HerdwaechterEntity, ButtonEntity):
    """Deletes the learned model and all labels (disabled by default)."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "reset_model")

    async def async_press(self) -> None:
        """Reset the model."""
        await self.coordinator.async_reset_model()
