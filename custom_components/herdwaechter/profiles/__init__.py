"""Device profile registry.

Version 0.1 ships the radiant cooktop profile only. Further profiles (oven,
induction, ...) implement :class:`DeviceProfile` and register here.
"""

from __future__ import annotations

from .base import DeviceProfile
from .ceran import CeranProfile

PROFILES: dict[str, type[DeviceProfile]] = {CeranProfile.key: CeranProfile}


def get_profile(key: str) -> DeviceProfile:
    """Return a profile instance by key."""
    try:
        return PROFILES[key]()
    except KeyError as err:
        msg = f"unknown profile {key!r}"
        raise ValueError(msg) from err


__all__ = ["PROFILES", "DeviceProfile", "get_profile"]
