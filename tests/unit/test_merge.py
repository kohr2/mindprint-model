"""
Tests for LoRA merging module.

Tests cover:
- Configuration validation
- Merge result dataclass
- LoRAMerger functionality
- Platform detection integration
- MPS device handling
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import shutil

from src.training.merge import (
    MergeConfig,
    MergeResult,
    LoRAMerger,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def mock_adapter_dir(temp_dir: str) -> str:
    """Create a mock adapter directory with required files."""
    adapter_dir = Path(temp_dir) / "adapter"
    adapter_dir.mkdir()
    # Create minimal adapter files
    (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (adapter_dir / "adapter_model.safetensors").write_text("mock")
    return str(adapter_dir)


@pytest.fixture
def merge_config(temp_dir: str, mock_adapter_dir: str) -> MergeConfig:
    """Create a test merge configuration."""
    return MergeConfig(
        base_model_path="google/gemma-3-12b",
        adapter_path=mock_adapter_dir,
        output_path=str(Path(temp_dir) / "merged"),
    )


class TestMergeConfig:
    """Test MergeConfig validation."""

    def test_config_stores_all_fields(self, mock_adapter_dir: str, temp_dir: str) -> None:
        """Config stores all fields correctly."""
        config = MergeConfig(
            base_model_path="test/model",
            adapter_path=mock_adapter_dir,
            output_path=str(Path(temp_dir) / "output"),
            dtype="bfloat16",
            device="mps",
            verify_merge=False,
        )

        assert config.base_model_path == "test/model"
        assert config.adapter_path == mock_adapter_dir
        assert config.dtype == "bfloat16"
        assert config.device == "mps"
        assert config.verify_merge is False

    def test_config_default_values(self, mock_adapter_dir: str, temp_dir: str) -> None:
        """Config has sensible defaults."""
        config = MergeConfig(
            base_model_path="test/model",
            adapter_path=mock_adapter_dir,
            output_path=str(Path(temp_dir) / "output"),
        )

        assert config.dtype == "float16"
        assert config.device == "auto"
        assert config.verify_merge is True

    def test_config_validates_adapter_path_exists(self, temp_dir: str) -> None:
        """Config raises error for non-existent adapter path."""
        with pytest.raises(ValueError, match="Adapter path does not exist"):
            MergeConfig(
                base_model_path="test/model",
                adapter_path="/nonexistent/path",
                output_path=str(Path(temp_dir) / "output"),
            )

    def test_config_accepts_valid_adapter_path(
        self, mock_adapter_dir: str, temp_dir: str
    ) -> None:
        """Config accepts valid adapter path."""
        config = MergeConfig(
            base_model_path="test/model",
            adapter_path=mock_adapter_dir,
            output_path=str(Path(temp_dir) / "output"),
        )

        assert config.adapter_path == mock_adapter_dir


class TestMergeResult:
    """Test MergeResult dataclass."""

    def test_result_stores_success(self) -> None:
        """Result stores success status."""
        result = MergeResult(
            success=True,
            output_path="/path/to/merged",
            base_model="test/model",
            adapter_path="/path/to/adapter",
            merge_time_seconds=120.5,
        )

        assert result.success is True
        assert result.merge_time_seconds == 120.5

    def test_result_stores_failure(self) -> None:
        """Result stores failure with error message."""
        result = MergeResult(
            success=False,
            output_path="",
            base_model="test/model",
            adapter_path="/path/to/adapter",
            merge_time_seconds=0.0,
            error_message="Failed to load model",
        )

        assert result.success is False
        assert result.error_message == "Failed to load model"

    def test_result_default_verification_passed(self) -> None:
        """Result defaults to verification passed."""
        result = MergeResult(
            success=True,
            output_path="/path",
            base_model="model",
            adapter_path="/adapter",
            merge_time_seconds=10.0,
        )

        assert result.verification_passed is True

    def test_result_stores_verification_details(self) -> None:
        """Result stores verification details."""
        result = MergeResult(
            success=True,
            output_path="/path",
            base_model="model",
            adapter_path="/adapter",
            merge_time_seconds=10.0,
            verification_passed=True,
            verification_details={"test_output": "Hello, world!"},
        )

        assert result.verification_details == {"test_output": "Hello, world!"}


class TestLoRAMerger:
    """Test LoRAMerger functionality."""

    def test_merger_initializes_with_config(self, merge_config: MergeConfig) -> None:
        """Merger initializes with configuration."""
        merger = LoRAMerger(merge_config)

        assert merger.config == merge_config

    @patch("src.training.merge.AutoModelForCausalLM")
    @patch("src.training.merge.AutoTokenizer")
    @patch("src.training.merge.PeftModel")
    def test_merge_returns_result(
        self,
        mock_peft: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        merge_config: MergeConfig,
    ) -> None:
        """Merge returns MergeResult object."""
        # Setup mocks
        mock_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_base
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model
        mock_merged = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        merger = LoRAMerger(merge_config)
        result = merger.merge()

        assert isinstance(result, MergeResult)

    @patch("src.training.merge.AutoModelForCausalLM")
    @patch("src.training.merge.AutoTokenizer")
    @patch("src.training.merge.PeftModel")
    def test_merge_calls_merge_and_unload(
        self,
        mock_peft: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        merge_config: MergeConfig,
    ) -> None:
        """Merge calls PEFT's merge_and_unload."""
        mock_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_base
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model

        merger = LoRAMerger(merge_config)
        merger.merge()

        mock_peft_model.merge_and_unload.assert_called_once()

    @patch("src.training.merge.detect_platform")
    @patch("src.training.merge.AutoModelForCausalLM")
    @patch("src.training.merge.AutoTokenizer")
    @patch("src.training.merge.PeftModel")
    def test_merge_saves_to_output_path(
        self,
        mock_peft: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        mock_platform: MagicMock,
        merge_config: MergeConfig,
    ) -> None:
        """Merged model is saved to configured output path."""
        # Use CPU device to avoid .to("mps") chain breaking mock
        mock_platform.return_value = {
            "device": "cpu",
            "has_mps": False,
            "recommended_dtype": "float16",
        }
        mock_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_base
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model
        mock_merged = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        merger = LoRAMerger(merge_config)
        result = merger.merge()

        mock_merged.save_pretrained.assert_called_once()
        # Check output path is used
        call_args = mock_merged.save_pretrained.call_args
        assert merge_config.output_path in str(call_args)

    @patch("src.training.merge.detect_platform")
    @patch("src.training.merge.AutoModelForCausalLM")
    @patch("src.training.merge.AutoTokenizer")
    @patch("src.training.merge.PeftModel")
    def test_merge_uses_platform_detection(
        self,
        mock_peft: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        mock_platform: MagicMock,
        merge_config: MergeConfig,
    ) -> None:
        """Merge uses detect_platform for device selection."""
        mock_platform.return_value = {
            "device": "mps",
            "has_mps": True,
            "recommended_dtype": "float16",
        }
        merge_config.device = "auto"

        mock_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_base
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model

        merger = LoRAMerger(merge_config)
        merger.merge()

        mock_platform.assert_called_once()


class TestMergeVerification:
    """Test merge verification functionality."""

    @patch("src.training.merge.AutoModelForCausalLM")
    @patch("src.training.merge.AutoTokenizer")
    @patch("src.training.merge.PeftModel")
    def test_verify_merge_runs_when_enabled(
        self,
        mock_peft: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        merge_config: MergeConfig,
    ) -> None:
        """Verification runs when verify_merge is True."""
        merge_config.verify_merge = True

        mock_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_base
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model
        mock_merged = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        # Mock generate for verification
        mock_merged.generate.return_value = MagicMock()

        merger = LoRAMerger(merge_config)
        result = merger.merge()

        # Verification should have run
        assert result.verification_passed is True

    @patch("src.training.merge.AutoModelForCausalLM")
    @patch("src.training.merge.AutoTokenizer")
    @patch("src.training.merge.PeftModel")
    def test_verify_merge_skipped_when_disabled(
        self,
        mock_peft: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        merge_config: MergeConfig,
    ) -> None:
        """Verification is skipped when verify_merge is False."""
        merge_config.verify_merge = False

        mock_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_base
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model
        mock_merged = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        merger = LoRAMerger(merge_config)
        result = merger.merge()

        # Generate should not be called (no verification)
        mock_merged.generate.assert_not_called()


class TestIncrementalMerge:
    """Test incremental merging functionality."""

    @patch("src.training.merge.AutoModelForCausalLM")
    @patch("src.training.merge.AutoTokenizer")
    @patch("src.training.merge.PeftModel")
    def test_merge_incremental_processes_list(
        self,
        mock_peft: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        merge_config: MergeConfig,
        mock_adapter_dir: str,
    ) -> None:
        """Incremental merge processes list of adapters."""
        mock_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_base
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model
        mock_merged = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        merger = LoRAMerger(merge_config)
        result = merger.merge_incremental([mock_adapter_dir])

        assert isinstance(result, MergeResult)


class TestErrorHandling:
    """Test error handling in merge operations."""

    @patch("src.training.merge.AutoModelForCausalLM")
    def test_merge_handles_model_load_failure(
        self,
        mock_model: MagicMock,
        merge_config: MergeConfig,
    ) -> None:
        """Merge handles model loading failure gracefully."""
        mock_model.from_pretrained.side_effect = Exception("Model not found")

        merger = LoRAMerger(merge_config)
        result = merger.merge()

        assert result.success is False
        assert "Model not found" in result.error_message

    @patch("src.training.merge.AutoModelForCausalLM")
    @patch("src.training.merge.AutoTokenizer")
    @patch("src.training.merge.PeftModel")
    def test_merge_handles_adapter_load_failure(
        self,
        mock_peft: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        merge_config: MergeConfig,
    ) -> None:
        """Merge handles adapter loading failure gracefully."""
        mock_base = MagicMock()
        mock_model.from_pretrained.return_value = mock_base
        mock_peft.from_pretrained.side_effect = Exception("Adapter incompatible")

        merger = LoRAMerger(merge_config)
        result = merger.merge()

        assert result.success is False
        assert "Adapter" in result.error_message or "incompatible" in result.error_message
