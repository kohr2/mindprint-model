#!/usr/bin/env python3
"""
Diagnostic test for MLX training state persistence issue.

Tests whether MLX model parameters properly persist after training
and whether generation uses the updated weights.

This test investigates why trained MLX models generate poor output
(mostly <|endoftext|> tokens) despite successful training completion.
"""

import sys
sys.path.insert(0, '.')

import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, Any, List
import numpy as np

# Import backend components
import src.backends.pytorch  # Register PyTorch backend
import src.backends.mlx  # Register MLX backend
from src.backends import create_backend
from src.evaluation.voice_evaluator import QuizEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def snapshot_parameters(model) -> Dict[str, Any]:
    """
    Capture a snapshot of all model parameters.
    
    Returns:
        Dict mapping parameter names to numpy arrays
    """
    from mlx.utils import tree_flatten
    
    snapshot = {}
    mlx_model = model.get_underlying_model()
    
    # Use tree_flatten to get all parameters including nested LoRA params
    try:
        params_dict = mlx_model.parameters()
        flat_params = tree_flatten(params_dict)
        
        for name, param in flat_params:
            # Convert MLX array to numpy for comparison
            snapshot[name] = np.array(param)
    except Exception as e:
        logger.warning(f"Could not flatten parameters: {e}")
        # Fallback to top-level parameters only
        try:
            params_dict = mlx_model.parameters()
            for name, param in params_dict.items():
                snapshot[name] = np.array(param)
        except Exception:
            pass
    
    return snapshot


def compare_parameters(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare parameter snapshots before and after training.
    
    Returns:
        Dict with statistics about parameter changes
    """
    stats = {
        'total_params': 0,
        'changed_params': 0,
        'unchanged_params': 0,
        'param_diffs': {},
        'lora_params': {},
    }
    
    # Get all parameter names
    all_names = set(before.keys()) | set(after.keys())
    
    for name in all_names:
        if name not in before or name not in after:
            continue
            
        before_val = before[name]
        after_val = after[name]
        
        stats['total_params'] += 1
        
        # Ensure both are numpy arrays with same shape
        try:
            before_arr = np.asarray(before_val).flatten()
            after_arr = np.asarray(after_val).flatten()
            
            if before_arr.shape != after_arr.shape:
                stats['changed_params'] += 1
                stats['param_diffs'][name] = {
                    'mean_diff': float('inf'),
                    'max_diff': float('inf'),
                    'mean_before': float(np.mean(np.abs(before_arr))),
                    'mean_after': float(np.mean(np.abs(after_arr))),
                    'note': 'Shape mismatch'
                }
                continue
            
            # Check if parameters changed (with tolerance for floating point)
            if np.allclose(before_arr, after_arr, rtol=1e-6, atol=1e-8):
                stats['unchanged_params'] += 1
            else:
                stats['changed_params'] += 1
                diff = np.abs(after_arr - before_arr)
                stats['param_diffs'][name] = {
                    'mean_diff': float(np.mean(diff)),
                    'max_diff': float(np.max(diff)),
                    'mean_before': float(np.mean(np.abs(before_arr))),
                    'mean_after': float(np.mean(np.abs(after_arr))),
                }
        except Exception as e:
            # If comparison fails, assume changed
            stats['changed_params'] += 1
            stats['param_diffs'][name] = {
                'mean_diff': float('nan'),
                'max_diff': float('nan'),
                'mean_before': float('nan'),
                'mean_after': float('nan'),
                'error': str(e)
            }
        
        # Check for LoRA parameters
        if 'lora' in name.lower():
            stats['lora_params'][name] = {
                'shape': after_val.shape,
                'mean': float(np.mean(np.abs(after_val))),
                'std': float(np.std(after_val)),
                'min': float(np.min(after_val)),
                'max': float(np.max(after_val)),
            }
    
    return stats


def generate_answer(model, tokenizer, question: str) -> str:
    """Generate an answer using the model."""
    from mlx_lm import generate as mlx_generate
    
    # Format prompt using tokenizer's chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    
    # Generate
    generated_text = mlx_generate(
        model.get_underlying_model(),
        tokenizer,
        prompt=prompt,
        max_tokens=512,
        verbose=False
    )
    
    # Extract only the generated part (after prompt)
    if prompt.strip() in generated_text:
        prompt_end_idx = generated_text.find(prompt.strip()) + len(prompt.strip())
        answer = generated_text[prompt_end_idx:].strip()
    else:
        answer = generated_text.strip()
    
    return answer


def test_mlx_training_state():
    """
    Comprehensive diagnostic test for MLX training state persistence.
    
    Tests all phases:
    1. Baseline verification
    2. Training simulation
    3. Parameter verification
    4. Generation after training
    5. Model state inspection
    """
    print("=" * 80)
    print("MLX Training State Diagnostic Test")
    print("=" * 80)
    
    # Test question
    test_question = "What is a cycle low?"
    
    # Minimal training data
    train_data = [
        {
            "question": "What is a cycle low?",
            "answer": "A cycle low is the bottom of a 4-year market cycle, typically occurring around 4 years after the previous low."
        },
        {
            "question": "What happens during a halving?",
            "answer": "During a halving, the block reward for miners is cut in half, reducing the rate of new coin creation."
        },
    ]
    
    # ============================================================================
    # Phase 1: Baseline Verification
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Baseline Verification")
    print("=" * 80)
    
    backend = create_backend("mlx", device="auto", dtype="float16")
    model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")
    tokenizer = model.tokenizer
    
    print(f"✓ Model loaded: {model.num_parameters:,} parameters")
    
    # Capture baseline parameters
    print("Capturing baseline parameter snapshot...")
    params_before = snapshot_parameters(model)
    print(f"✓ Captured {len(params_before)} parameter groups")
    
    # Generate baseline answer
    print(f"\nGenerating baseline answer for: '{test_question}'")
    baseline_answer = generate_answer(model, tokenizer, test_question)
    baseline_length = len(baseline_answer)
    baseline_eos_count = baseline_answer.count("<|endoftext|>")
    
    print(f"✓ Baseline answer length: {baseline_length} chars")
    print(f"✓ Baseline <|endoftext|> count: {baseline_eos_count}")
    print(f"✓ Baseline answer preview: {baseline_answer[:200]}...")
    
    # Evaluate baseline
    evaluator = QuizEvaluator(model, tokenizer)
    baseline_eval = evaluator.evaluate([
        {"question": test_question, "reference_answer": "A cycle low is the bottom."}
    ])
    baseline_voice_score = baseline_eval.get("voice_score", 0.0)
    print(f"✓ Baseline voice score: {baseline_voice_score:.4f}")
    
    # ============================================================================
    # Phase 2: Training Simulation
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Training Simulation")
    print("=" * 80)
    
    # Add LoRA adapter before training
    print("\nAdding LoRA adapter before training...")
    from src.backends.adapter_interface import AdapterConfig
    
    adapter_config = AdapterConfig(
        r=8,
        alpha=16.0,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    
    backend_adapter_manager = backend.get_adapter_manager()
    model = backend_adapter_manager.add_adapter(model, adapter_config)
    print(f"✓ LoRA adapter added: has_adapter={model.has_adapter()}")
    
    # Get underlying MLX model
    mlx_model = model.get_underlying_model()
    
    # Setup training
    learning_rate = 3e-4
    num_epochs = 1
    batch_size = 2
    max_seq_length = 2048
    
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    
    # Initialize optimizer with trainable parameters (CRITICAL for MLX)
    optimizer.init(mlx_model.trainable_parameters())
    
    # Set model to training mode
    mlx_model.train()
    
    print(f"Training config: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    
    # Format training data
    batch_texts = []
    for item in train_data:
        instruction = item["question"]
        output = item["answer"]
        
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            text = f"""<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>"""
        batch_texts.append(text)
    
    # Tokenize
    encoded = tokenizer(
        batch_texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="np",
    )
    
    input_ids = mx.array(encoded["input_ids"])
    labels = input_ids
    
    # Loss function
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        shift_logits = logits[..., :-1, :]
        shift_labels = targets[..., 1:]
        vocab_size = shift_logits.shape[-1]
        shift_logits_flat = shift_logits.reshape(-1, vocab_size)
        shift_labels_flat = shift_labels.reshape(-1)
        loss = mx.mean(
            nn.losses.cross_entropy(
                shift_logits_flat,
                shift_labels_flat,
                reduction='none'
            )
        )
        return loss
    
    # Training loop
    print("\nStarting training...")
    losses = []
    
    # Create loss_value_and_grad function using nn.value_and_grad
    # This automatically computes gradients only for trainable parameters
    loss_value_and_grad = nn.value_and_grad(mlx_model, loss_fn)
    
    for epoch in range(num_epochs):
        # Compute loss and gradients (only for trainable params)
        loss, grads = loss_value_and_grad(mlx_model, input_ids, labels)
        
        # Skip if loss is NaN
        if mx.isnan(loss).item() or mx.isinf(loss).item():
            print(f"⚠ NaN/Inf loss detected, skipping")
            continue
        
        # Update parameters (grads already filtered to trainable params)
        optimizer.update(mlx_model, grads)
        
        # Force evaluation (MLX is lazy)
        mx.eval(mlx_model.parameters())
        
        loss_val = loss.item()
        losses.append(loss_val)
        print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {loss_val:.4f}")
    
    avg_loss = np.mean(losses) if losses else 0.0
    print(f"✓ Training complete: avg_loss={avg_loss:.4f}")
    
    # Try additional evaluation methods
    print("\nForcing additional MLX evaluation...")
    try:
        # Method 1: Evaluate all parameters
        mx.eval(mlx_model.parameters())
        print("  ✓ mx.eval(model.parameters())")
        
        # Method 2: Flatten and evaluate
        try:
            # Evaluate all parameters as a list
            params_dict = mlx_model.parameters()
            param_list = list(params_dict.values())
            mx.eval(param_list)
            print(f"  ✓ mx.eval(list(model.parameters().values())) - {len(param_list)} params")
        except Exception as e:
            print(f"  ⚠ Method 2 failed: {e}")
        
        # Method 3: Evaluate each parameter individually
        params_dict = mlx_model.parameters()
        for name, param in params_dict.items():
            mx.eval(param)
        print(f"  ✓ Evaluated {len(params_dict)} parameters individually")
    except Exception as e:
        print(f"  ⚠ Evaluation error: {e}")
    
    # ============================================================================
    # Phase 3: Parameter Verification
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Parameter Verification")
    print("=" * 80)
    
    # Capture parameters after training
    print("Capturing post-training parameter snapshot...")
    params_after = snapshot_parameters(model)
    print(f"✓ Captured {len(params_after)} parameter groups")
    
    # Compare parameters
    print("\nComparing parameters before/after training...")
    param_stats = compare_parameters(params_before, params_after)
    
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {param_stats['total_params']}")
    print(f"  Changed parameters: {param_stats['changed_params']}")
    print(f"  Unchanged parameters: {param_stats['unchanged_params']}")
    
    if param_stats['changed_params'] > 0:
        print(f"\n✓ Parameters DID change during training")
        print(f"\nTop 10 parameter changes (by mean diff):")
        sorted_diffs = sorted(
            param_stats['param_diffs'].items(),
            key=lambda x: x[1]['mean_diff'],
            reverse=True
        )[:10]
        for name, diff_info in sorted_diffs:
            print(f"  {name}:")
            print(f"    Mean diff: {diff_info['mean_diff']:.6f}")
            print(f"    Max diff: {diff_info['max_diff']:.6f}")
            print(f"    Mean before: {diff_info['mean_before']:.6f}")
            print(f"    Mean after: {diff_info['mean_after']:.6f}")
    else:
        print(f"\n⚠ WARNING: NO parameters changed during training!")
        print(f"  This suggests the training loop is not updating weights.")
    
    # Check for LoRA parameters
    print(f"\nLoRA Parameter Check:")
    if param_stats['lora_params']:
        print(f"  ✓ Found {len(param_stats['lora_params'])} LoRA parameter groups:")
        for name, lora_info in list(param_stats['lora_params'].items())[:5]:
            print(f"    {name}:")
            print(f"      Shape: {lora_info['shape']}")
            print(f"      Mean: {lora_info['mean']:.6f}")
            print(f"      Std: {lora_info['std']:.6f}")
    else:
        print(f"  ⚠ WARNING: No LoRA parameters found!")
        print(f"    This suggests LoRA adapter may not be properly attached.")
    
    # ============================================================================
    # Phase 4: Generation After Training
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: Generation After Training")
    print("=" * 80)
    
    # Force evaluation one more time before generation
    print("Forcing final parameter evaluation before generation...")
    mx.eval(mlx_model.parameters())
    
    # Generate answer with trained model
    print(f"\nGenerating answer with trained model for: '{test_question}'")
    trained_answer = generate_answer(model, tokenizer, test_question)
    trained_length = len(trained_answer)
    trained_eos_count = trained_answer.count("<|endoftext|>")
    
    print(f"✓ Trained answer length: {trained_length} chars")
    print(f"✓ Trained <|endoftext|> count: {trained_eos_count}")
    print(f"✓ Trained answer preview: {trained_answer[:200]}...")
    
    # Compare with baseline
    print(f"\nComparison:")
    print(f"  Length change: {trained_length - baseline_length:+d} chars")
    print(f"  <|endoftext|> change: {trained_eos_count - baseline_eos_count:+d}")
    
    if trained_eos_count > baseline_eos_count + 10:
        print(f"  ⚠ WARNING: Significant increase in <|endoftext|> tokens!")
        print(f"    This suggests the model is generating mostly EOS tokens.")
    
    # Evaluate trained model
    trained_eval = evaluator.evaluate([
        {"question": test_question, "reference_answer": "A cycle low is the bottom."}
    ])
    trained_voice_score = trained_eval.get("voice_score", 0.0)
    print(f"\n✓ Trained voice score: {trained_voice_score:.4f}")
    print(f"  Change from baseline: {trained_voice_score - baseline_voice_score:+.4f}")
    
    if trained_voice_score < baseline_voice_score * 0.5:
        print(f"  ⚠ WARNING: Voice score dropped significantly!")
        print(f"    This suggests the trained model is performing worse than baseline.")
    
    # ============================================================================
    # Phase 5: Model State Inspection
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: Model State Inspection")
    print("=" * 80)
    
    # Check model object identity
    print("\nModel Object Identity:")
    print(f"  Model type: {type(model)}")
    print(f"  Model has adapter: {model.has_adapter()}")
    print(f"  Model device: {model.device}")
    print(f"  Model dtype: {model.dtype}")
    
    # Check underlying model
    underlying = model.get_underlying_model()
    print(f"\nUnderlying Model:")
    print(f"  Type: {type(underlying)}")
    print(f"  Has named_parameters: {hasattr(underlying, 'named_parameters')}")
    
    # Test parameter access
    try:
        params_dict = underlying.parameters()
        param_count = len(params_dict)
        print(f"  Parameter groups: {param_count}")
    except Exception as e:
        print(f"  ⚠ Error accessing parameters: {e}")
    
    # Test if model needs explicit eval before generation
    print("\nTesting explicit evaluation before generation...")
    try:
        # Force eval on all parameters
        params_dict = underlying.parameters()
        for name, param in params_dict.items():
            mx.eval(param)
        
        # Generate again
        test_answer = generate_answer(model, tokenizer, test_question)
        test_eos_count = test_answer.count("<|endoftext|>")
        print(f"  After explicit eval: <|endoftext|> count = {test_eos_count}")
        
        if test_eos_count < trained_eos_count:
            print(f"  ✓ Explicit eval improved generation!")
        elif test_eos_count == trained_eos_count:
            print(f"  → Explicit eval had no effect")
        else:
            print(f"  ⚠ Explicit eval made it worse")
    except Exception as e:
        print(f"  ⚠ Error during explicit eval test: {e}")
    
    # ============================================================================
    # Summary and Diagnosis
    # ============================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    diagnosis = []
    
    if param_stats['changed_params'] == 0:
        diagnosis.append("❌ CRITICAL: No parameters changed during training")
        diagnosis.append("   → Training loop is not updating weights")
        diagnosis.append("   → Check optimizer.update() and gradient flow")
    else:
        diagnosis.append(f"✓ Parameters changed: {param_stats['changed_params']}/{param_stats['total_params']}")
    
    if not param_stats['lora_params']:
        diagnosis.append("⚠ WARNING: No LoRA parameters found")
        diagnosis.append("   → LoRA adapter may not be attached")
        diagnosis.append("   → Check adapter_manager.add_adapter()")
    else:
        diagnosis.append(f"✓ LoRA parameters found: {len(param_stats['lora_params'])} groups")
    
    if trained_eos_count > baseline_eos_count + 10:
        diagnosis.append("❌ CRITICAL: Model generating mostly <|endoftext|> tokens")
        diagnosis.append("   → Model state not being used during generation")
        diagnosis.append("   → May need explicit mx.eval() before generation")
    
    if trained_voice_score < baseline_voice_score * 0.5:
        diagnosis.append("❌ CRITICAL: Voice score dropped significantly")
        diagnosis.append("   → Trained model performing worse than baseline")
        diagnosis.append("   → Training may be corrupting the model")
    
    for item in diagnosis:
        print(item)
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
    
    return {
        'baseline_voice_score': baseline_voice_score,
        'trained_voice_score': trained_voice_score,
        'baseline_length': baseline_length,
        'trained_length': trained_length,
        'baseline_eos_count': baseline_eos_count,
        'trained_eos_count': trained_eos_count,
        'param_stats': param_stats,
        'diagnosis': diagnosis,
    }


if __name__ == "__main__":
    try:
        results = test_mlx_training_state()
        print("\n✓ Diagnostic test completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.exception("Diagnostic test failed")
        print(f"\n❌ Diagnostic test failed: {e}")
        sys.exit(1)
