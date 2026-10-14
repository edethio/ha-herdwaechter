"""Sensors: probability, confidence, learning state and residual load."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfPower

from .const import LEARNING_STATES
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
    """Set up the sensors."""
    coordinator = entry.runtime_data
    async_add_entities(
        [
            ProbabilitySensor(coordinator),
            ConfidenceSensor(coordinator),
            LearningStateSensor(coordinator),
            ResidualLoadSensor(coordinator),
        ]
    )


class ProbabilitySensor(HerdwaechterEntity, SensorEntity):
    """Current probability that the cooktop is on."""

    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "probability")

    @property
    def native_value(self) -> float | None:
        """Return p in percent."""
        p = self.coordinator.data.probability
        return None if p is None else round(100.0 * p, 1)


class ConfidenceSensor(HerdwaechterEntity, SensorEntity):
    """Confidence of the current statement (separate from the probability)."""

    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "confidence")

    @property
    def native_value(self) -> float | None:
        """Return the confidence in percent."""
        data = self.coordinator.data
        if data.probability is None:
            return None
        return round(100.0 * data.confidence, 1)


class LearningStateSensor(HerdwaechterEntity, SensorEntity):
    """Learning / calibration state with model statistics."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = LEARNING_STATES
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "learning_state")

    @property
    def native_value(self) -> str:
        """Return the learning state."""
        return self.coordinator.data.learning_state

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return label counts, holdout quality and calibration values."""
        data = self.coordinator.data
        quality = data.quality or {}

        def rnd(value: Any, digits: int = 3) -> Any:
            return round(value, digits) if isinstance(value, float) else value

        return {
            "labels_positive": data.counts.get("positive", 0),
            "labels_negative": data.counts.get("negative", 0),
            "precision": rnd(quality.get("precision")),
            "recall": rnd(quality.get("recall")),
            "brier": rnd(quality.get("brier"), 4),
            "estimated_lag_s": rnd(data.lag_s, 1),
            "update_interval_s": rnd(data.update_interval_s, 1),
            "edge_threshold_w": round(data.edge_threshold),
        }


class ResidualLoadSensor(HerdwaechterEntity, SensorEntity):
    """Reconstructed household load minus separately metered devices."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_entity_registry_enabled_default = False
    _attr_suggested_display_precision = 0

    def __init__(self, coordinator: HerdwaechterCoordinator) -> None:
        """Initialise."""
        super().__init__(coordinator, "residual_load")

    @property
    def native_value(self) -> float | None:
        """Return the residual load in W."""
        value = self.coordinator.data.residual_w
        return None if value is None else round(value, 1)
