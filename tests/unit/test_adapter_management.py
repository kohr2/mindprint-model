"""
Unit tests for adapter management in DPO pipeline.

Tests proper unloading of SFT adapters before DPO to prevent adapter stacking.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel


def test_sft_adapter_unload_before_dpo():
    """Test that SFT adapter can be unloaded before DPO."""
    # Load small model for testing
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

    # Add SFT-style adapter (rank-8)
    sft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    sft_model = get_peft_model(model, sft_config)

    # Verify it's a PeftModel with adapter
    assert isinstance(sft_model, PeftModel)
    assert hasattr(sft_model, 'peft_config')
    assert 'default' in sft_model.peft_config
    assert sft_model.peft_config['default'].r == 8

    # Unload adapter
    clean_model = sft_model.merge_and_unload()

    # Verify it's no longer a PeftModel (adapters merged into base model)
    assert not isinstance(clean_model, PeftModel)
    # Model type should be the base model type
    assert type(clean_model).__name__ == 'OPTForCausalLM'

    # Now can apply DPO adapter cleanly
    dpo_config = LoraConfig(
        r=1,
        lora_alpha=1.0,
        target_modules=["o_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    dpo_model = get_peft_model(clean_model, dpo_config)

    # Should have only DPO adapter
    assert hasattr(dpo_model, 'peft_config')
    assert dpo_model.peft_config['default'].r == 1  # rank-1, not rank-8


def test_adapter_stacking_detection():
    """Test detection of adapter stacking."""
    # Load small model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

    # Add first adapter
    config1 = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    peft_model = get_peft_model(model, config1)

    # Verify it's a PeftModel
    assert isinstance(peft_model, PeftModel)
    assert hasattr(peft_model, 'peft_config')

    # Attempting to add another adapter should be detected
    # (This is what the defensive check in dpo_trainer.py prevents)
    # Our code should prevent passing a PEFT model to DPO trainer
    # The test for this is in test_dpo_trainer_rejects_peft_model()


def test_dpo_trainer_rejects_peft_model():
    """Test that DPO trainer rejects models with existing adapters."""
    from src.training.dpo_trainer import Rank1DPOTrainer, Rank1DPOConfig

    # Load small model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

    # Add adapter
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    peft_model = get_peft_model(model, config)

    # DPO trainer should reject this
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_config = Rank1DPOConfig(
        learning_rate=5e-7,
        max_steps=10,
        per_device_batch_size=1,
        beta=0.1,
        lora_r=1,
        lora_alpha=1.0,
    )

    trainer = Rank1DPOTrainer(peft_model, tokenizer, dpo_config)

    # Should raise ValueError when trying to prepare model with existing adapter
    with pytest.raises(ValueError, match="base model is a PeftModel"):
        trainer.prepare_model()


def test_clean_model_accepted_by_dpo():
    """Test that clean model (no adapters) is accepted by DPO trainer."""
    from src.training.dpo_trainer import Rank1DPOTrainer, Rank1DPOConfig

    # Load small model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

    # Verify no adapters
    assert not hasattr(model, 'peft_config')

    # DPO trainer should accept this
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_config = Rank1DPOConfig(
        learning_rate=5e-7,
        max_steps=10,
        per_device_batch_size=1,
        beta=0.1,
        lora_r=1,
        lora_alpha=1.0,
    )

    trainer = Rank1DPOTrainer(model, tokenizer, dpo_config)

    # Should successfully prepare model
    policy_model, ref_model = trainer.prepare_model()

    # Policy model should now have DPO adapter
    assert hasattr(policy_model, 'peft_config')
    assert policy_model.peft_config['default'].r == 1


def test_merged_model_accepted_by_dpo():
    """Test that merged model (SFT adapter merged in) is accepted by DPO."""
    from src.training.dpo_trainer import Rank1DPOTrainer, Rank1DPOConfig

    # Load small model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

    # Add SFT adapter
    sft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    sft_model = get_peft_model(model, sft_config)

    # Merge and unload
    merged_model = sft_model.merge_and_unload()

    # Verify it's no longer a PeftModel (adapters merged into base model)
    assert not isinstance(merged_model, PeftModel)

    # DPO trainer should accept this
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_config = Rank1DPOConfig(
        learning_rate=5e-7,
        max_steps=10,
        per_device_batch_size=1,
        beta=0.1,
        lora_r=1,
        lora_alpha=1.0,
    )

    trainer = Rank1DPOTrainer(merged_model, tokenizer, dpo_config)

    # Should successfully prepare model
    policy_model, ref_model = trainer.prepare_model()

    # Policy model should now have DPO adapter
    assert hasattr(policy_model, 'peft_config')
    assert policy_model.peft_config['default'].r == 1
