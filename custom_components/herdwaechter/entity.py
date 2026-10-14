"""Base entity of the Herdwächter integration."""

from __future__ import annotations

from homeassistant.helpers.entity import Entity

from .coordinator import HerdwaechterCoordinator


class HerdwaechterEntity(Entity):
    """Entity bound to one coordinator (one monitored appliance)."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, coordinator: HerdwaechterCoordinator, key: str) -> None:
        """Initialise the entity with a stable unique id."""
        self.coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = coordinator.device_info

    async def async_added_to_hass(self) -> None:
        """Subscribe to coordinator updates."""
        await super().async_added_to_hass()
        self.async_on_remove(
            self.coordinator.async_add_listener(self.async_write_ha_state)
        )
