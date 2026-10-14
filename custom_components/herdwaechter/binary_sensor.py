"""Binary sensor that reports whether the monitored cooktop is on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.binary_sensor import (  # type: ignore[attr-defined]
    BinarySensorDeviceClass,
    BinarySensorEntity,
)

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
    """Set up the binary sensor."""
    async_add_entities([CooktopBinarySensor(entry.runtime_data)])


class CooktopBinarySensor(HerdwaechterEntity, BinarySensorEntity):
    """On while the cooktop is detected as running."""

    _attr_device_class = BinarySensorDeviceClass.RUNNING
    _attr_name = None  # main feature: the entity carries the device name

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "cooktop")

    @property
    def is_on(self) -> bool:
        """Return the detection result."""
        return self.coordinator.data.is_on

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return probability, confidence and learned details."""
        data = self.coordinator.data
        return {
            "probability": _pct(data.probability),
            "confidence": _pct(data.confidence),
            "learned_power_levels": data.learned_levels,
            "last_cycle": data.last_cycle,
            "detected_period_s": data.detected_period_s,
            "model_weight": round(data.model_weight, 3),
            "learning_state": data.learning_state,
        }


def _pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 1)
