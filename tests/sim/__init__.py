"""Synthetic load profile generator and simulation runner (no Home Assistant)."""

from .household import Household, HouseholdConfig, Session, generate_household
from .runner import (
    Metrics,
    RunResult,
    UserConfig,
    build_scenario,
    run_simulation,
    summary,
)
from .sensors import SensorConfig, SensorEvents, generate_events

__all__ = [
    "Household",
    "HouseholdConfig",
    "Metrics",
    "RunResult",
    "SensorConfig",
    "SensorEvents",
    "Session",
    "UserConfig",
    "build_scenario",
    "generate_events",
    "generate_household",
    "run_simulation",
    "summary",
]
