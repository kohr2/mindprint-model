"""
Pytest configuration for mindprint-model tests.
"""

import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests by default unless --run-slow is passed."""
    if config.getoption("--run-slow"):
        # --run-slow given, don't skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
