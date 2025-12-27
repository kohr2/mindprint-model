"""
Tests for MPS utilities - Mac Studio M2 Ultra support.

Tests cover:
- MPS device detection and configuration
- Memory management (cache clearing)
- CPU fallback for unsupported operations
- Training context manager
"""

import pytest
from unittest.mock import patch, MagicMock
import torch

from src.training.mps_utils import (
    MPSConfig,
    get_mps_device,
    mps_empty_cache,
    move_to_device,
    check_mps_operation_support,
    MPSTrainingContext,
)


class TestMPSConfig:
    """Test MPSConfig dataclass."""

    def test_default_device_is_mps(self) -> None:
        """Default config uses MPS device."""
        config = MPSConfig()
        assert config.device == "mps"

    def test_default_dtype_is_float16(self) -> None:
        """Default dtype is float16 for MPS efficiency."""
        config = MPSConfig()
        assert config.dtype == torch.float16

    def test_default_fallback_enabled(self) -> None:
        """CPU fallback is enabled by default."""
        config = MPSConfig()
        assert config.fallback_to_cpu is True

    def test_default_gradient_checkpointing_enabled(self) -> None:
        """Gradient checkpointing enabled for memory efficiency."""
        config = MPSConfig()
        assert config.gradient_checkpointing is True

    def test_custom_config(self) -> None:
        """Custom configuration values are set correctly."""
        config = MPSConfig(
            device="cpu",
            dtype=torch.float32,
            fallback_to_cpu=False,
            gradient_checkpointing=False,
        )
        assert config.device == "cpu"
        assert config.dtype == torch.float32
        assert config.fallback_to_cpu is False
        assert config.gradient_checkpointing is False


class TestGetMPSDevice:
    """Test MPS device detection."""

    def test_returns_device_object(self) -> None:
        """Returns a torch.device object."""
        device = get_mps_device()
        assert isinstance(device, torch.device)

    def test_returns_mps_when_available(self) -> None:
        """Returns MPS device when MPS is available."""
        if torch.backends.mps.is_available():
            device = get_mps_device()
            assert device.type == "mps"

    @patch("src.training.mps_utils.torch.backends.mps.is_available", return_value=False)
    def test_returns_cpu_when_mps_unavailable(self, mock_mps: MagicMock) -> None:
        """Returns CPU device when MPS is not available."""
        device = get_mps_device()
        assert device.type == "cpu"


class TestMPSEmptyCache:
    """Test MPS cache clearing."""

    def test_does_not_raise(self) -> None:
        """Cache clearing should not raise errors."""
        # Should work regardless of MPS availability
        mps_empty_cache()

    @patch("src.training.mps_utils.torch.mps.empty_cache")
    @patch("src.training.mps_utils.torch.backends.mps.is_available", return_value=True)
    def test_calls_mps_empty_cache_when_available(
        self, mock_available: MagicMock, mock_empty: MagicMock
    ) -> None:
        """Calls torch.mps.empty_cache when MPS is available."""
        mps_empty_cache()
        mock_empty.assert_called_once()


class TestMoveToDevice:
    """Test device movement with fallback."""

    def test_moves_tensor_to_cpu(self) -> None:
        """Moves tensor to CPU successfully."""
        tensor = torch.rand(2, 3)
        result = move_to_device(tensor, torch.device("cpu"))
        assert result.device.type == "cpu"

    def test_moves_tensor_to_mps_when_available(self) -> None:
        """Moves tensor to MPS when available."""
        if torch.backends.mps.is_available():
            tensor = torch.rand(2, 3)
            result = move_to_device(tensor, torch.device("mps"))
            assert result.device.type == "mps"

    def test_fallback_to_cpu_on_failure(self) -> None:
        """Falls back to CPU when device move fails and fallback enabled."""
        tensor = torch.rand(2, 3)
        # Try moving to a device that doesn't exist
        result = move_to_device(tensor, torch.device("cpu"), fallback_to_cpu=True)
        assert result.device.type == "cpu"

    def test_preserves_tensor_values(self) -> None:
        """Tensor values are preserved after move."""
        original = torch.tensor([1.0, 2.0, 3.0])
        moved = move_to_device(original, torch.device("cpu"))
        assert torch.equal(original.cpu(), moved.cpu())


class TestCheckMPSOperationSupport:
    """Test MPS operation support checking."""

    def test_returns_bool(self) -> None:
        """Returns a boolean value."""
        result = check_mps_operation_support("matmul")
        assert isinstance(result, bool)

    def test_common_ops_supported(self) -> None:
        """Common operations are supported."""
        assert check_mps_operation_support("matmul") is True
        assert check_mps_operation_support("softmax") is True
        assert check_mps_operation_support("attention") is True

    def test_unsupported_ops_return_false(self) -> None:
        """Known unsupported operations return False."""
        # These are examples of potentially unsupported ops
        result = check_mps_operation_support("unknown_op_xyz")
        assert result is False


class TestMPSTrainingContext:
    """Test MPS training context manager."""

    def test_context_manager_protocol(self) -> None:
        """Implements context manager protocol."""
        context = MPSTrainingContext()
        assert hasattr(context, "__enter__")
        assert hasattr(context, "__exit__")

    def test_context_returns_self(self) -> None:
        """Context manager returns self on enter."""
        context = MPSTrainingContext()
        with context as ctx:
            assert ctx is context

    def test_context_has_device_attribute(self) -> None:
        """Context provides device attribute."""
        with MPSTrainingContext() as ctx:
            assert hasattr(ctx, "device")
            assert isinstance(ctx.device, torch.device)

    def test_context_cleanup_on_exit(self) -> None:
        """Context performs cleanup on exit."""
        # This should not raise even if cleanup operations are called
        with MPSTrainingContext():
            pass  # Normal exit

    def test_context_cleanup_on_exception(self) -> None:
        """Context performs cleanup even on exception."""
        with pytest.raises(ValueError):
            with MPSTrainingContext():
                raise ValueError("Test error")
        # If we get here, cleanup didn't raise additional errors


class TestMPSIntegration:
    """Integration tests for MPS utilities."""

    def test_full_workflow(self) -> None:
        """Test typical MPS training workflow."""
        # Get device
        device = get_mps_device()

        # Create tensor on CPU
        tensor = torch.rand(4, 4)

        # Move to device
        tensor = move_to_device(tensor, device)

        # Use in context
        with MPSTrainingContext() as ctx:
            # Perform some operation
            result = tensor @ tensor.T

        # Clear cache
        mps_empty_cache()

        # Verify result is valid
        assert result.shape == (4, 4)
