"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_preference_pair():
    """Sample preference pair for testing."""
    return {
        "prompt": "What is Bitcoin?",
        "chosen": "Bitcoin is a decentralized digital currency that operates on a blockchain...",
        "rejected": "Bitcoin is a cryptocurrency.",
        "source": "unit-01/chapter-01/topic-01",
    }


@pytest.fixture
def sample_training_sample():
    """Sample training sample for testing."""
    return {
        "instruction": "Explain Bitcoin cycles",
        "output": "Bitcoin follows 4-year cycles driven by halving events...",
        "input": "",
        "source": "unit-01/chapter-01/topic-01",
    }


@pytest.fixture
def sample_logps():
    """Sample log probabilities for loss testing."""
    batch_size = 4
    return {
        "policy_chosen": np.random.randn(batch_size),
        "policy_rejected": np.random.randn(batch_size) - 0.5,
        "ref_chosen": np.random.randn(batch_size),
        "ref_rejected": np.random.randn(batch_size) - 0.5,
        "chosen_lengths": np.ones(batch_size) * 100,
        "rejected_lengths": np.ones(batch_size) * 80,
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary directory for test data."""
    return tmp_path / "test_data"


@pytest.fixture(autouse=True)
def set_test_seed():
    """Set random seed for reproducible tests."""
    import random
    random.seed(42)
    np.random.seed(42)
