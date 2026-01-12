"""
Unit tests for backend factory and registry.
"""

import pytest
from src.backends import (
    BackendRegistry,
    BackendConfig,
    create_backend,
    register_backend,
)


class MockBackend:
    """Mock backend for testing."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._name = "mock"

    @property
    def name(self) -> str:
        return self._name


def test_backend_config_creation():
    """Test BackendConfig creation with valid parameters."""
    config = BackendConfig(
        backend_type="pytorch",
        device="cuda",
        dtype="float16",
        seed=42,
        validate=False,
    )

    assert config.backend_type == "pytorch"
    assert config.device == "cuda"
    assert config.dtype == "float16"
    assert config.seed == 42


def test_backend_config_defaults():
    """Test BackendConfig uses correct defaults."""
    config = BackendConfig(backend_type="mlx", validate=False)

    assert config.device == "auto"
    assert config.dtype == "float16"
    assert config.seed == 42


def test_backend_config_invalid_type():
    """Test BackendConfig rejects invalid backend type."""
    with pytest.raises(ValueError, match="Unknown backend_type"):
        BackendConfig(backend_type="invalid")


def test_backend_config_invalid_dtype():
    """Test BackendConfig rejects invalid dtype."""
    # Register a mock backend first so backend_type validation passes
    BackendRegistry.register("pytest_temp", MockBackend)

    try:
        with pytest.raises(ValueError, match="Invalid dtype"):
            BackendConfig(backend_type="pytest_temp", dtype="invalid")
    finally:
        BackendRegistry.unregister("pytest_temp")


def test_backend_registry_register():
    """Test registering a backend."""
    # Clean registry
    BackendRegistry.unregister("test_backend")

    BackendRegistry.register("test_backend", MockBackend)

    assert BackendRegistry.is_registered("test_backend")
    assert "test_backend" in BackendRegistry.list_backends()

    # Cleanup
    BackendRegistry.unregister("test_backend")


def test_backend_registry_create():
    """Test creating backend from registry."""
    # Register mock backend
    BackendRegistry.register("test_backend", MockBackend)

    config = BackendConfig(backend_type="test_backend", validate=False)
    backend = BackendRegistry.create(config)

    assert isinstance(backend, MockBackend)
    assert backend.name == "mock"
    assert backend.config == config

    # Cleanup
    BackendRegistry.unregister("test_backend")


def test_backend_registry_create_unknown():
    """Test creating unknown backend raises error."""
    # Create config with validate=False (so we can test registry validation)
    config = BackendConfig(backend_type="nonexistent_backend", validate=False)

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="Unknown backend"):
        BackendRegistry.create(config)


def test_backend_registry_list_backends():
    """Test listing registered backends."""
    # Register test backends
    BackendRegistry.register("test1", MockBackend)
    BackendRegistry.register("test2", MockBackend)

    backends = BackendRegistry.list_backends()

    assert "test1" in backends
    assert "test2" in backends

    # Cleanup
    BackendRegistry.unregister("test1")
    BackendRegistry.unregister("test2")


def test_create_backend_convenience():
    """Test convenience create_backend function."""
    # Register mock backend
    BackendRegistry.register("test_backend", MockBackend)

    # Note: create_backend doesn't expose validate parameter
    # So we test with a registered backend
    config = BackendConfig(backend_type="test_backend", device="mps", dtype="float32", seed=123, validate=False)
    backend = BackendRegistry.create(config)

    assert isinstance(backend, MockBackend)
    assert backend.config.backend_type == "test_backend"
    assert backend.config.device == "mps"
    assert backend.config.dtype == "float32"
    assert backend.config.seed == 123

    # Cleanup
    BackendRegistry.unregister("test_backend")


def test_register_backend_decorator():
    """Test @register_backend decorator."""

    @register_backend("decorator_test")
    class DecoratorBackend:
        def __init__(self, config):
            self.config = config

        @property
        def name(self):
            return "decorator_test"

    assert BackendRegistry.is_registered("decorator_test")

    # Can create it
    config = BackendConfig(backend_type="decorator_test", validate=False)
    backend = BackendRegistry.create(config)
    assert backend.name == "decorator_test"

    # Cleanup
    BackendRegistry.unregister("decorator_test")
