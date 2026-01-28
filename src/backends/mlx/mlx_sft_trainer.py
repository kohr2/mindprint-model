"""
MLX SFT Trainer - Supervised Fine-Tuning with MLX.

Implements SFT training with a manual training loop since MLX doesn't have
a TRL equivalent.
"""

from typing import List, Dict, Any
from pathlib import Path
import logging
import time

from ..trainer_interface import SFTTrainerInterface, TrainingResult
from ..model_interface import ModelInterface
from .mlx_model import MLXModel
from .mlx_device_manager import MLXDeviceManager
from .mlx_adapter_manager import MLXAdapterManager

logger = logging.getLogger(__name__)


class MLXSFTTrainer(SFTTrainerInterface):
    """
    MLX implementation of SFTTrainerInterface.

    Implements supervised fine-tuning with a manual training loop.
    """

    def __init__(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
        device_manager: MLXDeviceManager,
        adapter_manager: MLXAdapterManager,
    ):
        """
        Initialize MLX SFT trainer.

        Args:
            model: Model to train (must be MLXModel)
            config: Training configuration dict
            device_manager: Device manager for this backend
            adapter_manager: Adapter manager for this backend

        Raises:
            TypeError: If model is not an MLXModel
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        self._mlx_model = model
        self._device_manager = device_manager
        self._adapter_manager = adapter_manager
        self._config = config

        logger.info("Initialized MLXSFTTrainer")

    def train(self, train_data: List[Dict[str, Any]]) -> TrainingResult:
        """
        Train on Q&A data.

        Args:
            train_data: List of dicts with 'question'/'answer' or 'instruction'/'output' keys

        Returns:
            TrainingResult with training outcome
        """
        start_time = time.time()

        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim

            if not train_data:
                return TrainingResult(
                    success=False,
                    final_loss=0.0,
                    training_time_seconds=0.0,
                    samples_trained=0,
                    error_message="No training data provided",
                )

            # Extract config
            learning_rate = self._config.get("learning_rate", 3e-4)
            num_epochs = self._config.get("num_epochs", 3)
            batch_size = self._config.get("per_device_batch_size", 4)
            max_seq_length = self._config.get("max_seq_length", 2048)

            # Get model and tokenizer
            mlx_model = self._mlx_model.get_underlying_model()
            tokenizer = self._mlx_model.tokenizer

            # Setup optimizer (AdamW)
            optimizer = optim.AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01,
            )

            # Training loop
            logger.info(
                f"Starting SFT training: {len(train_data)} samples, "
                f"{num_epochs} epochs, lr={learning_rate}"
            )

            total_loss = 0.0
            num_steps = 0

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                valid_batches = 0

                # Process data in batches
                for i in range(0, len(train_data), batch_size):
                    batch = train_data[i:i + batch_size]

                    # Format and tokenize batch
                    batch_texts = []
                    for item in batch:
                        # Support both formats
                        if "question" in item and "answer" in item:
                            instruction = item["question"]
                            output = item["answer"]
                        elif "instruction" in item and "output" in item:
                            instruction = item["instruction"]
                            input_text = item.get("input", "")
                            output = item["output"]
                            if input_text:
                                instruction = f"{instruction}\n{input_text}"
                        else:
                            continue

                        # Format using tokenizer's chat template if available (Qwen, Llama, etc.)
                        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                            try:
                                messages = [
                                    {"role": "user", "content": instruction},
                                    {"role": "assistant", "content": output}
                                ]
                                text = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=False  # We have the full conversation
                                )
                            except Exception as e:
                                logger.warning(f"Failed to use chat template: {e}, falling back to default format")
                                # Fallback to Gemma-3 format
                                text = f"""<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>"""
                        else:
                            # Fallback to Gemma-3 format for models without chat template
                            text = f"""<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>"""
                        batch_texts.append(text)

                    if not batch_texts:
                        continue

                    # Tokenize
                    encoded = tokenizer(
                        batch_texts,
                        truncation=True,
                        max_length=max_seq_length,
                        padding="max_length",
                        return_tensors="np",  # NumPy arrays
                    )

                    # Convert to MLX arrays
                    input_ids = mx.array(encoded["input_ids"])
                    attention_mask = mx.array(encoded["attention_mask"])
                    labels = input_ids  # Causal LM: predict next token

                    # Forward pass
                    def loss_fn(model, inputs, targets):
                        logits = model(inputs)

                        # Shift for causal LM
                        shift_logits = logits[..., :-1, :]
                        shift_labels = targets[..., 1:]

                        # Flatten
                        vocab_size = shift_logits.shape[-1]
                        shift_logits_flat = shift_logits.reshape(-1, vocab_size)
                        shift_labels_flat = shift_labels.reshape(-1)

                        # Cross-entropy loss
                        loss = mx.mean(
                            nn.losses.cross_entropy(
                                shift_logits_flat,
                                shift_labels_flat,
                                reduction='none'
                            )
                        )
                        return loss

                    # Compute loss and gradients
                    # Use nn.value_and_grad which automatically handles trainable parameters
                    # This computes gradients only for trainable params (LoRA when adapter present)
                    import mlx.nn as nn
                    loss_value_and_grad = nn.value_and_grad(mlx_model, loss_fn)
                    (loss, toks), grads = loss_value_and_grad(mlx_model, input_ids, labels)

                    # Skip if loss is NaN
                    if mx.isnan(loss).item() or mx.isinf(loss).item():
                        logger.warning("NaN/Inf loss detected, skipping batch")
                        continue

                    # nn.value_and_grad already filters to trainable parameters
                    # When LoRA adapter is present, only LoRA params are trainable
                    # So grads already contains only LoRA gradients
                    optimizer.update(mlx_model, grads)

                    # Force evaluation (MLX is lazy)
                    mx.eval(mlx_model.parameters())

                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    num_steps += 1
                    valid_batches += 1

                avg_epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else float('nan')
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}"
                )

            final_loss = total_loss / num_steps if num_steps > 0 else 0.0
            training_time = time.time() - start_time

            logger.info(
                f"SFT training complete: loss={final_loss:.4f}, "
                f"time={training_time:.1f}s"
            )

            return TrainingResult(
                success=True,
                final_loss=final_loss,
                training_time_seconds=training_time,
                samples_trained=len(train_data) * num_epochs,
                metrics={},
            )

        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            return TrainingResult(
                success=False,
                final_loss=0.0,
                training_time_seconds=time.time() - start_time,
                samples_trained=0,
                error_message=str(e),
            )

    def train_on_topic(
        self,
        topic_data: List[Dict[str, Any]],
        topic_id: str,
    ) -> TrainingResult:
        """
        Train on a single topic's data.

        Args:
            topic_data: Q&A pairs for this topic
            topic_id: Unique identifier for the topic

        Returns:
            TrainingResult with training outcome
        """
        logger.info(f"MLX SFT training on topic: {topic_id}")
        result = self.train(topic_data)

        if result.success:
            logger.info(
                f"Topic {topic_id} SFT complete: loss={result.final_loss:.4f}"
            )

        return result

    def save_adapter(self, path: Path) -> Path:
        """
        Save the LoRA adapter.

        Args:
            path: Directory to save adapter

        Returns:
            Path to saved adapter
        """
        logger.info(f"Saving SFT adapter to {path}")
        self._mlx_model.save_adapter(path)
        return path

    def get_model(self) -> ModelInterface:
        """
        Get the trained model.

        Returns:
            MLXModel with trained adapter
        """
        return self._mlx_model

    def get_config(self) -> Dict[str, Any]:
        """
        Get training configuration.

        Returns:
            Training configuration dictionary
        """
        return self._config

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.

        Returns:
            Dictionary with training stats (losses, learning rate, etc.)
        """
        # Return empty dict for now - could be populated during training
        return {}
