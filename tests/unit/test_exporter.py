"""
Tests for ModelExporter - exporting models to various formats.

Tests cover:
- ExportConfig validation
- Safetensors export
- GGUF export
- Model card generation
- Export result handling
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile
import shutil
import json

from src.export.exporter import (
    ExportConfig,
    ExportResult,
    ModelExporter,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def mock_model_dir(temp_dir: str) -> str:
    """Create a mock model directory with required files."""
    model_dir = Path(temp_dir) / "model"
    model_dir.mkdir()
    # Create minimal model files
    (model_dir / "config.json").write_text('{"model_type": "gemma"}')
    (model_dir / "model.safetensors").write_text("mock")
    (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
    return str(model_dir)


@pytest.fixture
def export_config(mock_model_dir: str, temp_dir: str) -> ExportConfig:
    """Create a test export configuration."""
    return ExportConfig(
        model_path=mock_model_dir,
        output_dir=str(Path(temp_dir) / "exports"),
    )


class TestExportConfig:
    """Test ExportConfig validation."""

    def test_config_stores_all_fields(self, mock_model_dir: str, temp_dir: str) -> None:
        """Config stores all fields correctly."""
        config = ExportConfig(
            model_path=mock_model_dir,
            output_dir=str(Path(temp_dir) / "output"),
            export_safetensors=True,
            export_gguf=True,
            gguf_quantization="Q5_K_M",
        )

        assert config.model_path == mock_model_dir
        assert config.export_safetensors is True
        assert config.export_gguf is True
        assert config.gguf_quantization == "Q5_K_M"

    def test_config_default_values(self, mock_model_dir: str, temp_dir: str) -> None:
        """Config has sensible defaults."""
        config = ExportConfig(
            model_path=mock_model_dir,
            output_dir=str(Path(temp_dir) / "output"),
        )

        assert config.export_safetensors is True
        assert config.export_gguf is True
        assert config.gguf_quantization == "Q5_K_M"

    def test_config_validates_model_path_exists(self, temp_dir: str) -> None:
        """Config raises error for non-existent model path."""
        with pytest.raises(ValueError, match="Model path does not exist"):
            ExportConfig(
                model_path="/nonexistent/path",
                output_dir=str(Path(temp_dir) / "output"),
            )

    def test_config_accepts_valid_model_path(
        self, mock_model_dir: str, temp_dir: str
    ) -> None:
        """Config accepts valid model path."""
        config = ExportConfig(
            model_path=mock_model_dir,
            output_dir=str(Path(temp_dir) / "output"),
        )

        assert config.model_path == mock_model_dir

    def test_config_various_quantizations(
        self, mock_model_dir: str, temp_dir: str
    ) -> None:
        """Config accepts various GGUF quantization formats."""
        for quant in ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"]:
            config = ExportConfig(
                model_path=mock_model_dir,
                output_dir=str(Path(temp_dir) / "output"),
                gguf_quantization=quant,
            )
            assert config.gguf_quantization == quant


class TestExportResult:
    """Test ExportResult dataclass."""

    def test_result_stores_success(self) -> None:
        """Result stores success status."""
        result = ExportResult(
            success=True,
            exports={"safetensors": "/path/to/model"},
            model_path="/path/to/source",
            export_time_seconds=30.5,
        )

        assert result.success is True
        assert result.export_time_seconds == 30.5

    def test_result_stores_failure(self) -> None:
        """Result stores failure with error message."""
        result = ExportResult(
            success=False,
            exports={},
            model_path="/path/to/source",
            export_time_seconds=0.0,
            error_message="GGUF conversion failed",
        )

        assert result.success is False
        assert result.error_message == "GGUF conversion failed"

    def test_result_stores_multiple_exports(self) -> None:
        """Result stores paths to multiple export formats."""
        result = ExportResult(
            success=True,
            exports={
                "safetensors": "/path/to/model.safetensors",
                "gguf": "/path/to/model.gguf",
            },
            model_path="/path/to/source",
            export_time_seconds=120.0,
        )

        assert "safetensors" in result.exports
        assert "gguf" in result.exports


class TestModelExporter:
    """Test ModelExporter functionality."""

    def test_exporter_initializes_with_config(self, export_config: ExportConfig) -> None:
        """Exporter initializes with configuration."""
        exporter = ModelExporter(export_config)

        assert exporter.config == export_config

    def test_exporter_creates_output_directory(
        self, mock_model_dir: str, temp_dir: str
    ) -> None:
        """Exporter creates output directory if it doesn't exist."""
        new_dir = Path(temp_dir) / "new" / "nested" / "dir"
        config = ExportConfig(
            model_path=mock_model_dir,
            output_dir=str(new_dir),
        )

        exporter = ModelExporter(config)

        assert new_dir.exists()


class TestSafetensorsExport:
    """Test safetensors export functionality."""

    @patch("src.export.exporter.AutoModelForCausalLM")
    @patch("src.export.exporter.AutoTokenizer")
    def test_export_safetensors_creates_directory(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        export_config: ExportConfig,
    ) -> None:
        """Safetensors export creates output directory."""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        exporter = ModelExporter(export_config)
        output_path = exporter.export_safetensors()

        assert Path(output_path).parent.exists()

    @patch("src.export.exporter.AutoModelForCausalLM")
    @patch("src.export.exporter.AutoTokenizer")
    def test_export_safetensors_saves_model(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        export_config: ExportConfig,
    ) -> None:
        """Safetensors export saves model with safe_serialization."""
        mock_loaded_model = MagicMock()
        mock_model.from_pretrained.return_value = mock_loaded_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        exporter = ModelExporter(export_config)
        exporter.export_safetensors()

        mock_loaded_model.save_pretrained.assert_called_once()
        call_kwargs = mock_loaded_model.save_pretrained.call_args[1]
        assert call_kwargs.get("safe_serialization") is True


class TestGGUFExport:
    """Test GGUF export functionality."""

    @patch("src.export.exporter.subprocess.run")
    def test_export_gguf_calls_converter(
        self,
        mock_run: MagicMock,
        export_config: ExportConfig,
    ) -> None:
        """GGUF export calls the llama.cpp converter script."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        exporter = ModelExporter(export_config)
        exporter.export_gguf()

        mock_run.assert_called()
        # Verify converter script is called
        call_args = str(mock_run.call_args)
        assert "convert" in call_args.lower() or "gguf" in call_args.lower()

    @patch("src.export.exporter.subprocess.run")
    def test_export_gguf_uses_quantization(
        self,
        mock_run: MagicMock,
        mock_model_dir: str,
        temp_dir: str,
    ) -> None:
        """GGUF export uses configured quantization."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        config = ExportConfig(
            model_path=mock_model_dir,
            output_dir=str(Path(temp_dir) / "exports"),
            gguf_quantization="Q4_K_M",
        )

        exporter = ModelExporter(config)
        exporter.export_gguf()

        call_args = str(mock_run.call_args)
        assert "Q4_K_M" in call_args

    @patch("src.export.exporter.subprocess.run")
    def test_export_gguf_handles_failure(
        self,
        mock_run: MagicMock,
        export_config: ExportConfig,
    ) -> None:
        """GGUF export handles converter failure."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Conversion failed"
        )

        exporter = ModelExporter(export_config)

        with pytest.raises(RuntimeError, match="GGUF"):
            exporter.export_gguf()


class TestModelCard:
    """Test model card generation."""

    def test_generates_model_card(self, export_config: ExportConfig) -> None:
        """Generates MODEL_CARD.md file."""
        exporter = ModelExporter(export_config)
        card_path = exporter.generate_model_card(
            model_name="bob-loukas-v1",
            base_model="google/gemma-3-12b-it",
            approach="dpo",
        )

        assert card_path.exists()
        assert card_path.suffix == ".md"

    def test_model_card_contains_metadata(self, export_config: ExportConfig) -> None:
        """Model card contains essential metadata."""
        exporter = ModelExporter(export_config)
        card_path = exporter.generate_model_card(
            model_name="bob-loukas-v1",
            base_model="google/gemma-3-12b-it",
            approach="dpo",
        )

        content = card_path.read_text()

        assert "bob-loukas-v1" in content
        assert "gemma-3-12b" in content
        assert "dpo" in content.lower()


class TestExportAll:
    """Test export_all convenience method."""

    @patch("src.export.exporter.subprocess.run")
    @patch("src.export.exporter.AutoModelForCausalLM")
    @patch("src.export.exporter.AutoTokenizer")
    def test_export_all_returns_result(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        mock_run: MagicMock,
        export_config: ExportConfig,
    ) -> None:
        """export_all returns ExportResult."""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        exporter = ModelExporter(export_config)
        result = exporter.export_all()

        assert isinstance(result, ExportResult)

    @patch("src.export.exporter.subprocess.run")
    @patch("src.export.exporter.AutoModelForCausalLM")
    @patch("src.export.exporter.AutoTokenizer")
    def test_export_all_includes_both_formats(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        mock_run: MagicMock,
        export_config: ExportConfig,
    ) -> None:
        """export_all includes both safetensors and GGUF."""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        exporter = ModelExporter(export_config)
        result = exporter.export_all()

        assert "safetensors" in result.exports
        assert "gguf" in result.exports

    @patch("src.export.exporter.subprocess.run")
    @patch("src.export.exporter.AutoModelForCausalLM")
    @patch("src.export.exporter.AutoTokenizer")
    def test_export_all_safetensors_only(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        mock_run: MagicMock,
        mock_model_dir: str,
        temp_dir: str,
    ) -> None:
        """export_all respects export_gguf=False."""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        config = ExportConfig(
            model_path=mock_model_dir,
            output_dir=str(Path(temp_dir) / "exports"),
            export_safetensors=True,
            export_gguf=False,
        )

        exporter = ModelExporter(config)
        result = exporter.export_all()

        assert "safetensors" in result.exports
        assert "gguf" not in result.exports
        mock_run.assert_not_called()


class TestErrorHandling:
    """Test error handling in export operations."""

    @patch("src.export.exporter.AutoModelForCausalLM")
    def test_export_handles_model_load_failure(
        self,
        mock_model: MagicMock,
        export_config: ExportConfig,
    ) -> None:
        """Export handles model loading failure gracefully."""
        mock_model.from_pretrained.side_effect = Exception("Model corrupted")

        exporter = ModelExporter(export_config)
        result = exporter.export_all()

        assert result.success is False
        assert "corrupted" in result.error_message.lower()

    @patch("src.export.exporter.subprocess.run")
    @patch("src.export.exporter.AutoModelForCausalLM")
    @patch("src.export.exporter.AutoTokenizer")
    def test_export_continues_after_gguf_failure(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        mock_run: MagicMock,
        export_config: ExportConfig,
    ) -> None:
        """Export continues and reports partial success if GGUF fails."""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="GGUF failed"
        )

        exporter = ModelExporter(export_config)
        result = exporter.export_all()

        # Safetensors should still succeed
        assert "safetensors" in result.exports
        # But overall might be marked as partial success
        # The implementation can decide if this is success=True with warnings
        # or success=False with partial exports
