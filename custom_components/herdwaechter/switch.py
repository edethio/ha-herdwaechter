"""Switches: the "cooktop is on" label and "pause learning"."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.switch import SwitchEntity
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
    """Set up the switches."""
    coordinator = entry.runtime_data
    async_add_entities(
        [CookingLabelSwitch(coordinator), PauseLearningSwitch(coordinator)]
    )


class CookingLabelSwitch(HerdwaechterEntity, SwitchEntity):
    """Manual label: switch on while cooking. Turns itself off after a while."""

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "cooking_label")

    @property
    def is_on(self) -> bool:
        """Return the label state."""
        return self.coordinator.label_on

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return when the label will switch off automatically."""
        since = self.coordinator.label_since
        if not self.coordinator.label_on or since is None:
            return {"auto_off_in_s": None}
        remaining = (
            since + self.coordinator.config.max_label_s - self.coordinator._now()
        )
        return {"auto_off_in_s": max(0, round(remaining))}

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Start a positive label."""
        await self.coordinator.async_set_label(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """End the label."""
        await self.coordinator.async_set_label(False)


class PauseLearningSwitch(HerdwaechterEntity, SwitchEntity):
    """While on, no model updates happen."""

    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "pause_learning")

    @property
    def is_on(self) -> bool:
        """Return whether learning is paused."""
        return self.coordinator.engine.learning_paused

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Pause learning."""
        await self.coordinator.async_set_learning_paused(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Resume learning."""
        await self.coordinator.async_set_learning_paused(False)
