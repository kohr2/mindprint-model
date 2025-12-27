"""
Tests for reproducibility utilities.

Tests cover:
- Seed setting for deterministic behavior
- Config hashing for experiment tracking
- Environment info collection
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
import random
import numpy as np

from src.utils.reproducibility import (
    set_seed,
    hash_config,
    get_reproducibility_info,
)


class TestSetSeed:
    """Test set_seed function."""

    def test_set_seed_makes_random_deterministic(self) -> None:
        """Random produces same sequence after set_seed."""
        set_seed(42)
        first_run = [random.random() for _ in range(5)]

        set_seed(42)
        second_run = [random.random() for _ in range(5)]

        assert first_run == second_run

    def test_set_seed_makes_numpy_deterministic(self) -> None:
        """NumPy random produces same sequence after set_seed."""
        set_seed(42)
        first_run = np.random.rand(5).tolist()

        set_seed(42)
        second_run = np.random.rand(5).tolist()

        assert first_run == second_run

    def test_set_seed_with_torch(self) -> None:
        """PyTorch random produces same sequence after set_seed."""
        import torch

        set_seed(42)
        first_run = torch.rand(5).tolist()

        set_seed(42)
        second_run = torch.rand(5).tolist()

        assert first_run == second_run

    def test_set_seed_default_value(self) -> None:
        """Default seed is 42."""
        set_seed()
        first_run = random.random()

        set_seed()
        second_run = random.random()

        assert first_run == second_run

    def test_set_seed_different_seeds_differ(self) -> None:
        """Different seeds produce different sequences."""
        set_seed(42)
        first_run = random.random()

        set_seed(123)
        second_run = random.random()

        assert first_run != second_run


class TestHashConfig:
    """Test config hashing."""

    def test_hash_returns_string(self) -> None:
        """hash_config returns a string."""
        config = {"learning_rate": 0.001, "batch_size": 8}
        result = hash_config(config)
        assert isinstance(result, str)

    def test_same_config_same_hash(self) -> None:
        """Same config produces same hash."""
        config = {"learning_rate": 0.001, "batch_size": 8}
        hash1 = hash_config(config)
        hash2 = hash_config(config)
        assert hash1 == hash2

    def test_different_config_different_hash(self) -> None:
        """Different configs produce different hashes."""
        config1 = {"learning_rate": 0.001, "batch_size": 8}
        config2 = {"learning_rate": 0.001, "batch_size": 16}
        assert hash_config(config1) != hash_config(config2)

    def test_hash_handles_dataclass(self) -> None:
        """hash_config works with dataclass configs."""

        @dataclass
        class TestConfig:
            learning_rate: float = 0.001
            batch_size: int = 8

        config = TestConfig()
        result = hash_config(config)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_handles_nested_config(self) -> None:
        """hash_config works with nested dicts."""
        config = {
            "training": {"lr": 0.001},
            "model": {"name": "gemma"},
        }
        result = hash_config(config)
        assert isinstance(result, str)

    def test_hash_is_consistent_length(self) -> None:
        """Hash has consistent length (SHA256 = 64 chars)."""
        config1 = {"a": 1}
        config2 = {"a": 1, "b": 2, "c": {"d": 3}}
        assert len(hash_config(config1)) == len(hash_config(config2))


class TestGetReproducibilityInfo:
    """Test environment info collection."""

    def test_returns_dict(self) -> None:
        """get_reproducibility_info returns a dict."""
        info = get_reproducibility_info()
        assert isinstance(info, dict)

    def test_contains_python_version(self) -> None:
        """Info includes Python version."""
        info = get_reproducibility_info()
        assert "python_version" in info

    def test_contains_torch_version(self) -> None:
        """Info includes PyTorch version."""
        info = get_reproducibility_info()
        assert "torch_version" in info

    def test_contains_device_info(self) -> None:
        """Info includes device information."""
        info = get_reproducibility_info()
        assert "device" in info
        assert info["device"] in ["cpu", "cuda", "mps"]

    def test_contains_timestamp(self) -> None:
        """Info includes timestamp."""
        info = get_reproducibility_info()
        assert "timestamp" in info

    def test_contains_platform_info(self) -> None:
        """Info includes platform information."""
        info = get_reproducibility_info()
        assert "platform" in info


class TestMPSHandling:
    """Test MPS-specific seed handling."""

    @patch("src.utils.reproducibility.torch.backends.mps.is_available", return_value=True)
    @patch("src.utils.reproducibility.torch.mps.manual_seed")
    def test_sets_mps_seed_when_available(
        self,
        mock_mps_seed: MagicMock,
        mock_mps_available: MagicMock,
    ) -> None:
        """set_seed calls torch.mps.manual_seed when MPS available."""
        set_seed(42)
        mock_mps_seed.assert_called_with(42)


class TestCUDAHandling:
    """Test CUDA-specific seed handling."""

    @patch("src.utils.reproducibility.torch.cuda.is_available", return_value=True)
    @patch("src.utils.reproducibility.torch.cuda.manual_seed_all")
    def test_sets_cuda_seed_when_available(
        self,
        mock_cuda_seed: MagicMock,
        mock_cuda_available: MagicMock,
    ) -> None:
        """set_seed calls torch.cuda.manual_seed_all when CUDA available."""
        set_seed(42)
        mock_cuda_seed.assert_called_with(42)
