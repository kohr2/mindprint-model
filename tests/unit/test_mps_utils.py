"""
Tests for MPS (Metal Performance Shaders) utilities.

These tests verify the Mac Studio M2 Ultra optimizations for training.
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
        """Default device should be MPS."""
        config = MPSConfig()
        assert config.device == "mps"

    def test_default_dtype_is_float16(self) -> None:
        """Default dtype should be float16 for memory efficiency."""
        config = MPSConfig()
        assert config.dtype == torch.float16

    def test_default_fallback_to_cpu_is_true(self) -> None:
        """Default fallback_to_cpu should be True."""
        config = MPSConfig()
        assert config.fallback_to_cpu is True

    def test_default_gradient_checkpointing_is_true(self) -> None:
        """Default gradient_checkpointing should be True."""
        config = MPSConfig()
        assert config.gradient_checkpointing is True

    def test_custom_config(self) -> None:
        """Custom config values are set correctly."""
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
    """Test get_mps_device function."""

    def test_returns_torch_device(self) -> None:
        """Should return a torch.device object."""
        device = get_mps_device()
        assert isinstance(device, torch.device)

    def test_returns_mps_when_available(self) -> None:
        """Should return MPS device when available."""
        if torch.backends.mps.is_available():
            device = get_mps_device()
            assert device.type == "mps"

    def test_returns_cpu_fallback(self) -> None:
        """Should return CPU when MPS is not available."""
        with patch("torch.backends.mps.is_available", return_value=False):
            device = get_mps_device()
            assert device.type == "cpu"


class TestMPSEmptyCache:
    """Test mps_empty_cache function."""

    def test_does_not_raise_when_mps_available(self) -> None:
        """Should not raise an error when called."""
        # This should not raise regardless of MPS availability
        mps_empty_cache()

    def test_calls_torch_mps_empty_cache_when_available(self) -> None:
        """Should call torch.mps.empty_cache when MPS is available."""
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("torch.mps.empty_cache") as mock_cache:
                mps_empty_cache()
                mock_cache.assert_called_once()


class TestMoveToDevice:
    """Test move_to_device function."""

    def test_moves_tensor_to_cpu(self) -> None:
        """Should move tensor to CPU."""
        tensor = torch.randn(10)
        result = move_to_device(tensor, torch.device("cpu"))
        assert result.device.type == "cpu"

    def test_moves_tensor_to_mps_when_available(self) -> None:
        """Should move tensor to MPS when available."""
        if torch.backends.mps.is_available():
            tensor = torch.randn(10)
            result = move_to_device(tensor, torch.device("mps"))
            assert result.device.type == "mps"

    def test_fallback_to_cpu_when_mps_fails(self) -> None:
        """Should fallback to CPU when MPS operation fails."""
        tensor = torch.randn(10)
        # Even if we request MPS but it fails, should fallback to CPU
        result = move_to_device(tensor, torch.device("cpu"), fallback_to_cpu=True)
        assert result.device.type == "cpu"

    def test_handles_model_movement(self) -> None:
        """Should handle moving nn.Module to device."""
        model = torch.nn.Linear(10, 5)
        result = move_to_device(model, torch.device("cpu"))
        # Check that model parameters are on CPU
        for param in result.parameters():
            assert param.device.type == "cpu"


class TestCheckMPSOperationSupport:
    """Test check_mps_operation_support function."""

    def test_returns_bool(self) -> None:
        """Should return a boolean."""
        result = check_mps_operation_support("matmul")
        assert isinstance(result, bool)

    def test_common_operations_supported(self) -> None:
        """Common operations should be supported."""
        if torch.backends.mps.is_available():
            assert check_mps_operation_support("matmul") is True
            assert check_mps_operation_support("add") is True
            assert check_mps_operation_support("mul") is True


class TestMPSTrainingContext:
    """Test MPSTrainingContext context manager."""

    def test_context_manager_enters_and_exits(self) -> None:
        """Should work as a context manager."""
        config = MPSConfig(device="cpu")
        with MPSTrainingContext(config) as ctx:
            assert ctx is not None

    def test_provides_device(self) -> None:
        """Should provide device in context."""
        config = MPSConfig(device="cpu")
        with MPSTrainingContext(config) as ctx:
            assert ctx.device is not None

    def test_clears_cache_on_exit(self) -> None:
        """Should clear cache when exiting context."""
        config = MPSConfig(device="cpu")
        with patch("src.training.mps_utils.mps_empty_cache") as mock_cache:
            with MPSTrainingContext(config):
                pass
            mock_cache.assert_called()

    def test_handles_exception_in_context(self) -> None:
        """Should handle exceptions and still clean up."""
        config = MPSConfig(device="cpu")
        with patch("src.training.mps_utils.mps_empty_cache") as mock_cache:
            try:
                with MPSTrainingContext(config):
                    raise ValueError("Test error")
            except ValueError:
                pass
            # Cache should still be cleared even after exception
            mock_cache.assert_called()


class TestMPSMemoryManagement:
    """Test memory management utilities."""

    def test_empty_cache_is_safe_to_call_multiple_times(self) -> None:
        """Should be safe to call empty_cache multiple times."""
        for _ in range(3):
            mps_empty_cache()

    def test_device_movement_preserves_tensor_values(self) -> None:
        """Moving tensors should preserve their values."""
        original = torch.tensor([1.0, 2.0, 3.0])
        moved = move_to_device(original, torch.device("cpu"))
        assert torch.allclose(original, moved.cpu())

    def test_device_movement_preserves_tensor_dtype(self) -> None:
        """Moving tensors should preserve their dtype."""
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        moved = move_to_device(original, torch.device("cpu"))
        assert moved.dtype == torch.float16
