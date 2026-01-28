"""
Unit tests for PyTorch backend.
"""

import pytest
from src.backends import BackendRegistry, BackendConfig, create_backend


def test_pytorch_backend_registered():
    """Test that PyTorch backend is automatically registered."""
    # Import pytorch backend module to trigger registration
    import src.backends.pytorch  # noqa

    assert BackendRegistry.is_registered("pytorch")
    assert "pytorch" in BackendRegistry.list_backends()


def test_create_pytorch_backend():
    """Test creating PyTorch backend via factory."""
    from src.backends import create_backend

    # Create backend
    backend = create_backend("pytorch", device="cpu", dtype="float32", seed=42)

    assert backend is not None
    assert backend.name == "pytorch"
    assert backend.config.backend_type == "pytorch"
    assert backend.config.device == "cpu"  # auto resolves to cpu in test env
    assert backend.config.dtype == "float32"
    assert backend.config.seed == 42


def test_pytorch_backend_auto_registers():
    """Test that PyTorch backend auto-registers on import."""
    # Import should trigger registration
    from src.backends.pytorch import PyTorchBackend
    from src.backends import BackendRegistry

    assert BackendRegistry.is_registered("pytorch")
    assert "pytorch" in BackendRegistry.list_backends()


def test_pytorch_backend_creation():
    """Test creating PyTorch backend via factory."""
    from src.backends import create_backend, BackendConfig

    backend = create_backend("pytorch", device="cpu", dtype="float32")

    assert backend.name == "pytorch"
    assert backend.config.backend_type == "pytorch"
    assert backend.config.device == "cpu"  # Passed device is preserved in config
    assert backend.config.dtype == "float32"


if __name__ == "__main__":
    # Simple smoke test
    from src.backends import BackendRegistry

    # Import pytorch package to trigger registration
    import src.backends.pytorch

    assert BackendRegistry.is_registered("pytorch"), "PyTorch backend not registered!"
    print("✓ PyTorch backend registered successfully")

    # Try creating backend
    from src.backends import create_backend

    backend = create_backend("pytorch", device="cpu", dtype="float32")
    print(f"✓ Created PyTorch backend: {backend.name}")

    print("\n✅ All basic PyTorch backend checks passed!")
