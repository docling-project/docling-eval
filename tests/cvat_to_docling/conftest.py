"""pytest configuration for CVAT to DoclingDocument regression tests."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "regression: CVAT to DoclingDocument regression tests"
    )
