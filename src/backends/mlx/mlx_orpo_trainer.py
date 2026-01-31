"""
MLX ORPO Trainer - Odds Ratio Preference Optimization with MLX.

Implements ORPO training with manual ORPO loss computation.
ORPO combines SFT and preference alignment in a single stage,
eliminating the need for a reference model.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time
import random

from ..trainer_interface import ORPOTrainerInterface, TrainingResult
from ..model_interface import ModelInterface
from .mlx_model import MLXModel
from .mlx_device_manager import MLXDeviceManager
from .mlx_adapter_manager import MLXAdapterManager

logger = logging.getLogger(__name__)


class MLXORPOTrainer(ORPOTrainerInterface):
    """
    MLX implementation of ORPOTrainerInterface.

    Implements ORPO training with manual odds ratio loss computation.
    No reference model needed - ORPO combines SFT and preference alignment.
    """

    def __init__(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
        device_manager: MLXDeviceManager,
        adapter_manager: MLXAdapterManager,
    ):
        """
        Initialize MLX ORPO trainer.

        Args:
            model: Policy model to train (must be MLXModel)
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

        # Track training stats
        self._training_stats: Dict[str, Any] = {}

        logger.info("Initialized MLXORPOTrainer")

    def _compute_log_probs(
        self,
        model: Any,
        input_ids: Any,
        labels: Any,
    ) -> Any:
        """
        Compute log probabilities of labels given inputs.

        Args:
            model: MLX model
            input_ids: Input token IDs
            labels: Target token IDs

        Returns:
            Log probabilities
        """
        import mlx.core as mx
        import mlx.nn as nn

        # Forward pass
        logits = model(input_ids)

        # Get log probabilities
        log_probs = nn.log_softmax(logits, axis=-1)

        # Gather log probs for labels
        # Shift for causal LM
        shift_log_probs = log_probs[..., :-1, :]
        shift_labels = labels[..., 1:]

        # Gather log probs at label indices
        batch_size, seq_len, vocab_size = shift_log_probs.shape
        flat_log_probs = shift_log_probs.reshape(-1, vocab_size)
        flat_labels = shift_labels.reshape(-1)

        # Use fancy indexing
        indices = mx.arange(flat_labels.shape[0])
        selected_log_probs = flat_log_probs[indices, flat_labels]

        # Reshape and sum
        selected_log_probs = selected_log_probs.reshape(batch_size, seq_len)
        log_probs_sum = mx.sum(selected_log_probs, axis=-1)

        return log_probs_sum

    def train(self, train_data: List[Dict[str, Any]]) -> TrainingResult:
        """
        Train with ORPO on preference pairs.

        Args:
            train_data: List of dicts with "prompt", "chosen", "rejected" keys

        Returns:
            TrainingResult with training outcome
        """
        start_time = time.time()

        try:
            import mlx.core as mx
            import mlx.optimizers as optim
            from src.core.losses import ORPOLoss, ORPOConfig

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
            max_steps = self._config.get("max_steps", 100)
            batch_size = self._config.get("per_device_batch_size", 4)
            lambda_orpo = self._config.get("lambda_orpo", 0.1)
            max_length = self._config.get("max_length", 1024)

            # Get model and tokenizer
            policy_model = self._mlx_model.get_underlying_model()
            tokenizer = self._mlx_model.tokenizer

            # Setup optimizer
            optimizer = optim.AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01,
            )

            # Initialize ORPO loss
            orpo_loss_fn = ORPOLoss(ORPOConfig(lambda_orpo=lambda_orpo))

            # Training loop
            logger.info(
                f"Starting ORPO training: {len(train_data)} pairs, "
                f"{max_steps} steps, lambda_orpo={lambda_orpo}, lr={learning_rate}"
            )

            total_loss = 0.0
            steps_completed = 0
            nll_losses = []
            or_losses = []
            accuracies = []

            for step in range(max_steps):
                # Sample batch randomly
                batch_size_actual = min(batch_size, len(train_data))
                batch_indices = random.sample(range(len(train_data)), batch_size_actual)
                batch = [train_data[i] for i in batch_indices]

                # Tokenize chosen and rejected responses
                chosen_texts = [
                    f"{item['prompt']}{item['chosen']}" for item in batch
                ]
                rejected_texts = [
                    f"{item['prompt']}{item['rejected']}" for item in batch
                ]

                # Encode
                chosen_encoded = tokenizer(
                    chosen_texts,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="np",
                )
                rejected_encoded = tokenizer(
                    rejected_texts,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="np",
                )

                # Convert to MLX arrays
                chosen_ids = mx.array(chosen_encoded["input_ids"])
                rejected_ids = mx.array(rejected_encoded["input_ids"])

                # Define loss function for policy model
                def loss_fn(model):
                    # Forward pass for chosen and rejected
                    chosen_logits = model(chosen_ids)
                    rejected_logits = model(rejected_ids)

                    # Compute ORPO loss
                    loss_output = orpo_loss_fn.compute(
                        logits=chosen_logits,  # For chosen responses
                        chosen_ids=chosen_ids,
                        rejected_ids=rejected_ids,
                        rejected_logits=rejected_logits,  # For rejected responses
                    )

                    return loss_output.loss

                # Compute loss and gradients
                loss, grads = mx.value_and_grad(loss_fn)(policy_model)

                # Skip if loss is NaN
                if mx.isnan(loss).item() or mx.isinf(loss).item():
                    logger.warning("NaN/Inf loss detected, skipping step")
                    continue

                # Update parameters
                optimizer.update(policy_model, grads)

                # Force evaluation
                mx.eval(policy_model.parameters())

                # Compute metrics for logging
                chosen_logits_eval = policy_model(chosen_ids)
                rejected_logits_eval = policy_model(rejected_ids)
                loss_output = orpo_loss_fn.compute(
                    logits=chosen_logits_eval,
                    chosen_ids=chosen_ids,
                    rejected_ids=rejected_ids,
                    rejected_logits=rejected_logits_eval,
                )
                mx.eval(loss_output.loss)

                total_loss += loss.item()
                steps_completed += 1

                # Store metrics
                metrics = loss_output.metrics
                nll_losses.append(metrics.get("nll_loss", 0.0))
                or_losses.append(metrics.get("or_loss", 0.0))
                accuracies.append(metrics.get("accuracy", 0.0))

                if (step + 1) % 10 == 0:
                    avg_loss = total_loss / steps_completed
                    avg_accuracy = sum(accuracies[-10:]) / min(10, len(accuracies))
                    logger.info(
                        f"Step {step + 1}/{max_steps}, Loss: {avg_loss:.4f}, "
                        f"Accuracy: {avg_accuracy:.4f}"
                    )

            final_loss = total_loss / steps_completed if steps_completed > 0 else 0.0
            training_time = time.time() - start_time

            # Store training stats
            self._training_stats = {
                "final_loss": final_loss,
                "nll_loss": sum(nll_losses) / len(nll_losses) if nll_losses else 0.0,
                "or_loss": sum(or_losses) / len(or_losses) if or_losses else 0.0,
                "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
                "steps_completed": steps_completed,
            }

            logger.info(
                f"ORPO training complete: loss={final_loss:.4f}, "
                f"time={training_time:.1f}s"
            )

            return TrainingResult(
                success=True,
                final_loss=final_loss,
                training_time_seconds=training_time,
                samples_trained=steps_completed,
                metrics=self._training_stats,
            )

        except Exception as e:
            logger.error(f"ORPO training failed: {e}", exc_info=True)
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
        Train on a single topic's preference pairs.

        Args:
            topic_data: Preference pairs for this topic
            topic_id: Unique identifier for the topic

        Returns:
            TrainingResult with training outcome
        """
        logger.info(f"MLX ORPO training on topic: {topic_id}")
        result = self.train(topic_data)

        if result.success:
            logger.info(
                f"Topic {topic_id} ORPO complete: loss={result.final_loss:.4f}"
            )

        return result

    def save_adapter(self, path: Path) -> Path:
        """
        Save the ORPO LoRA adapter.

        Args:
            path: Directory to save adapter

        Returns:
            Path to saved adapter
        """
        logger.info(f"Saving ORPO adapter to {path}")
        self._mlx_model.save_adapter(path)
        return path

    def get_model(self) -> ModelInterface:
        """
        Get the trained policy model.

        Returns:
            MLXModel with trained ORPO adapter
        """
        return self._mlx_model

    def get_config(self) -> Dict[str, Any]:
        """
        Get training configuration.

        Returns:
            Training configuration dictionary
        """
        return self._config

    def get_orpo_stats(self) -> Dict[str, Any]:
        """
        Get ORPO-specific statistics.

        Returns:
            Dictionary with ORPO loss components (NLL loss, odds ratio loss),
            accuracy, odds margins, etc.
        """
        return self._training_stats.copy()
