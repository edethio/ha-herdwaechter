"""Shared test fixtures."""

from __future__ import annotations

import pytest

# Importing the integration package first makes Home Assistant's loader find
# this repository's ``custom_components`` instead of the test harness' own.
import custom_components.herdwaechter  # noqa: F401


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(request: pytest.FixtureRequest) -> None:
    """Enable custom integrations for every test that uses Home Assistant."""
    if "hass" in request.fixturenames:
        request.getfixturevalue("enable_custom_integrations")
