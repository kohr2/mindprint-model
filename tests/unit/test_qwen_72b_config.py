"""
Unit tests for Qwen2.5-72B model configuration.

Tests the configuration loading, layer zone mapping, LoRA config generation,
and prompt formatting for the Qwen2.5-72B model.
"""

import pytest
from src.models.config import get_model_config, load_model_config, ModelConfig


def test_qwen_72b_config_loads():
    """Test that qwen-72b config loads correctly."""
    config = get_model_config("qwen-72b")

    assert config.name == "qwen2.5-72b"
    assert config.hf_path == "Qwen/Qwen2.5-72B-Instruct"
    assert config.layers == 80
    assert config.hidden_size == 8192
    assert config.attention_heads == 64
    assert config.kv_heads == 8
    assert config.context_length == 131072
    assert config.prompt_format == "chatml"


def test_qwen_72b_config_aliases():
    """Test that all qwen-72b aliases work."""
    config1 = get_model_config("qwen-72b")
    config2 = get_model_config("qwen2.5-72b")

    assert config1.name == config2.name
    assert config1.hf_path == config2.hf_path
    assert config1.layers == config2.layers


def test_qwen_72b_layer_zones_exist():
    """Test that all three layer zones are defined."""
    config = get_model_config("qwen-72b")

    assert "lexicon" in config.layer_zones
    assert "reasoning" in config.layer_zones
    assert "voice" in config.layer_zones


def test_qwen_72b_layer_zone_sizes():
    """Test layer zone distributions (25% lexicon, 50% reasoning, 25% voice)."""
    config = get_model_config("qwen-72b")

    lexicon_layers = config.get_layers_for_zone("lexicon")
    reasoning_layers = config.get_layers_for_zone("reasoning")
    voice_layers = config.get_layers_for_zone("voice")

    # Check sizes
    assert len(lexicon_layers) == 20  # 25% of 80
    assert len(reasoning_layers) == 40  # 50% of 80
    assert len(voice_layers) == 20  # 25% of 80

    # Check they sum to total layers
    total_layers = len(lexicon_layers) + len(reasoning_layers) + len(voice_layers)
    assert total_layers == 80


def test_qwen_72b_layer_zone_ranges():
    """Test that layer zones have correct ranges."""
    config = get_model_config("qwen-72b")

    lexicon_layers = config.get_layers_for_zone("lexicon")
    reasoning_layers = config.get_layers_for_zone("reasoning")
    voice_layers = config.get_layers_for_zone("voice")

    # Lexicon: layers 0-19
    assert min(lexicon_layers) == 0
    assert max(lexicon_layers) == 19

    # Reasoning: layers 20-59
    assert min(reasoning_layers) == 20
    assert max(reasoning_layers) == 59

    # Voice: layers 60-79
    assert min(voice_layers) == 60
    assert max(voice_layers) == 79


def test_qwen_72b_layer_zone_no_overlap():
    """Test that layer zones don't overlap."""
    config = get_model_config("qwen-72b")

    lexicon_layers = set(config.get_layers_for_zone("lexicon"))
    reasoning_layers = set(config.get_layers_for_zone("reasoning"))
    voice_layers = set(config.get_layers_for_zone("voice"))

    # Check no overlap
    assert len(lexicon_layers & reasoning_layers) == 0
    assert len(lexicon_layers & voice_layers) == 0
    assert len(reasoning_layers & voice_layers) == 0


def test_qwen_72b_layer_zone_modules():
    """Test correct modules targeted per zone."""
    config = get_model_config("qwen-72b")

    lexicon_modules = config.get_target_modules_for_zone("lexicon")
    reasoning_modules = config.get_target_modules_for_zone("reasoning")
    voice_modules = config.get_target_modules_for_zone("voice")

    # Lexicon targets q_proj only
    assert lexicon_modules == ["q_proj"]

    # Reasoning targets v_proj, up_proj, down_proj
    assert "v_proj" in reasoning_modules
    assert "up_proj" in reasoning_modules
    assert "down_proj" in reasoning_modules

    # Voice targets o_proj, up_proj, down_proj
    assert "o_proj" in voice_modules
    assert "up_proj" in voice_modules
    assert "down_proj" in voice_modules


def test_qwen_72b_lora_config():
    """Test LoRA configuration."""
    config = get_model_config("qwen-72b")

    assert config.lora.r == 8
    assert config.lora.alpha == 16
    assert config.lora.dropout == 0.05

    # Check target modules
    target_modules = config.lora.target_modules
    assert "q_proj" in target_modules
    assert "v_proj" in target_modules
    assert "o_proj" in target_modules
    assert "up_proj" in target_modules
    assert "down_proj" in target_modules


def test_qwen_72b_peft_config_generation():
    """Test PEFT config generation for zones."""
    config = get_model_config("qwen-72b")

    # Test lexicon zone
    peft_config_lexicon = config.to_peft_config(zone="lexicon")
    assert peft_config_lexicon["r"] == 8
    assert peft_config_lexicon["lora_alpha"] == 16
    assert peft_config_lexicon["target_modules"] == ["q_proj"]
    assert "layers_pattern" in peft_config_lexicon

    # Test reasoning zone
    peft_config_reasoning = config.to_peft_config(zone="reasoning")
    assert "v_proj" in peft_config_reasoning["target_modules"]

    # Test voice zone
    peft_config_voice = config.to_peft_config(zone="voice")
    assert "o_proj" in peft_config_voice["target_modules"]

    # Test full model (no zone)
    peft_config_all = config.to_peft_config(zone=None)
    assert len(peft_config_all["target_modules"]) >= 3


def test_qwen_72b_compute_requirements():
    """Test compute requirements."""
    config = get_model_config("qwen-72b")

    assert config.compute.vram_4bit == 36  # GB
    assert config.compute.vram_bf16 == 145  # GB
    assert config.compute.recommended_batch_size == 1
    assert config.compute.estimated_hours_dpo == 100
    assert config.compute.estimated_hours_ppo == 150


def test_qwen_72b_prompt_format():
    """Test prompt formatting (ChatML format)."""
    config = get_model_config("qwen-72b")

    question = "What is a cycle low?"
    formatted = config.format_prompt(question)

    # Qwen uses ChatML format
    assert "<|im_start|>" in formatted
    assert "<|im_end|>" in formatted
    assert question in formatted


def test_qwen_72b_yaml_config_loads():
    """Test loading config from YAML file."""
    from pathlib import Path

    yaml_path = Path(__file__).parents[2] / "src" / "models" / "qwen_72b_config.yaml"

    config = load_model_config(str(yaml_path))

    assert config.name == "qwen2.5-72b"
    assert config.layers == 80
    assert config.hidden_size == 8192


def test_qwen_72b_config_consistency():
    """Test consistency between Python config and YAML config."""
    # Load from both sources
    py_config = get_model_config("qwen-72b")

    from pathlib import Path
    yaml_path = Path(__file__).parents[2] / "src" / "models" / "qwen_72b_config.yaml"
    yaml_config = load_model_config(str(yaml_path))

    # Check key attributes match
    assert py_config.name == yaml_config.name
    assert py_config.hf_path == yaml_config.hf_path
    assert py_config.layers == yaml_config.layers
    assert py_config.hidden_size == yaml_config.hidden_size
    assert py_config.prompt_format == yaml_config.prompt_format


def test_qwen_72b_invalid_zone():
    """Test that invalid zone name raises error."""
    config = get_model_config("qwen-72b")

    with pytest.raises(KeyError, match="Unknown zone"):
        config.get_layers_for_zone("invalid_zone")

    with pytest.raises(KeyError, match="Unknown zone"):
        config.get_target_modules_for_zone("invalid_zone")


def test_qwen_72b_comparison_with_smaller_models():
    """Test that qwen-72b is properly scaled from smaller models."""
    qwen_7b = get_model_config("qwen-7b")
    qwen_72b = get_model_config("qwen-72b")
    gemma_12b = get_model_config("gemma-12b")

    # Same prompt format as qwen-7b
    assert qwen_72b.prompt_format == qwen_7b.prompt_format

    # More layers than both
    assert qwen_72b.layers > qwen_7b.layers
    assert qwen_72b.layers > gemma_12b.layers

    # Same LoRA rank as others (conservative)
    assert qwen_72b.lora.r == qwen_7b.lora.r
    assert qwen_72b.lora.r == gemma_12b.lora.r

    # Layer zone philosophy matches (25% / 50% / 25%)
    qwen_72b_lexicon_pct = len(qwen_72b.get_layers_for_zone("lexicon")) / qwen_72b.layers
    qwen_72b_reasoning_pct = len(qwen_72b.get_layers_for_zone("reasoning")) / qwen_72b.layers
    qwen_72b_voice_pct = len(qwen_72b.get_layers_for_zone("voice")) / qwen_72b.layers

    assert abs(qwen_72b_lexicon_pct - 0.25) < 0.01
    assert abs(qwen_72b_reasoning_pct - 0.50) < 0.01
    assert abs(qwen_72b_voice_pct - 0.25) < 0.01
