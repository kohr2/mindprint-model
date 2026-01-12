"""
MLX DPO Trainer - Direct Preference Optimization with MLX.

Implements DPO training with manual DPO loss computation (Bradley-Terry model)
since MLX doesn't have a TRL equivalent.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time

from ..trainer_interface import DPOTrainerInterface, TrainingResult
from ..model_interface import ModelInterface
from .mlx_model import MLXModel
from .mlx_device_manager import MLXDeviceManager
from .mlx_adapter_manager import MLXAdapterManager

logger = logging.getLogger(__name__)


class MLXDPOTrainer(DPOTrainerInterface):
    """
    MLX implementation of DPOTrainerInterface.

    Implements DPO training with manual Bradley-Terry loss computation.
    """

    def __init__(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
        device_manager: MLXDeviceManager,
        adapter_manager: MLXAdapterManager,
        ref_model: Optional[ModelInterface] = None,
    ):
        """
        Initialize MLX DPO trainer.

        Args:
            model: Policy model to train (must be MLXModel)
            config: Training configuration dict
            device_manager: Device manager for this backend
            adapter_manager: Adapter manager for this backend
            ref_model: Optional reference model (must be MLXModel if provided)

        Raises:
            TypeError: If model or ref_model is not an MLXModel
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        if ref_model is not None and not isinstance(ref_model, MLXModel):
            raise TypeError(f"Expected MLXModel for ref_model, got {type(ref_model)}")

        self._mlx_model = model
        self._mlx_ref_model = ref_model
        self._device_manager = device_manager
        self._adapter_manager = adapter_manager
        self._config = config

        logger.info("Initialized MLXDPOTrainer")

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
        # For each position, get log_prob of the true label
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

    def _dpo_loss(
        self,
        policy_chosen_logps: Any,
        policy_rejected_logps: Any,
        ref_chosen_logps: Any,
        ref_rejected_logps: Any,
        beta: float = 0.1,
    ) -> Any:
        """
        Compute DPO loss using Bradley-Terry model.

        Args:
            policy_chosen_logps: Log probs of chosen responses from policy model
            policy_rejected_logps: Log probs of rejected responses from policy model
            ref_chosen_logps: Log probs of chosen responses from reference model
            ref_rejected_logps: Log probs of rejected responses from reference model
            beta: KL penalty coefficient

        Returns:
            DPO loss
        """
        import mlx.core as mx

        # Compute log ratios
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        # DPO loss: -log(sigmoid(beta * (policy_logratios - ref_logratios)))
        logits = beta * (policy_logratios - ref_logratios)

        # Use log-sigmoid for numerical stability
        loss = -mx.mean(mx.logaddexp(0, -logits))  # -log(sigmoid(x))

        return loss

    def train(self, train_data: List[Dict[str, Any]]) -> TrainingResult:
        """
        Train with DPO on preference pairs.

        Args:
            train_data: List of dicts with "prompt", "chosen", "rejected" keys

        Returns:
            TrainingResult with training outcome
        """
        start_time = time.time()

        try:
            import mlx.core as mx
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
            learning_rate = self._config.get("learning_rate", 5e-7)
            max_steps = self._config.get("max_steps", 100)
            batch_size = self._config.get("per_device_batch_size", 2)
            beta = self._config.get("beta", 0.1)
            max_length = self._config.get("max_length", 1024)

            # Get models
            policy_model = self._mlx_model.get_underlying_model()
            ref_model = (
                self._mlx_ref_model.get_underlying_model()
                if self._mlx_ref_model
                else policy_model  # Use same model as reference
            )
            tokenizer = self._mlx_model.tokenizer

            # Setup optimizer
            optimizer = optim.AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01,
            )

            # Training loop
            logger.info(
                f"Starting DPO training: {len(train_data)} pairs, "
                f"{max_steps} steps, beta={beta}, lr={learning_rate}"
            )

            total_loss = 0.0
            steps_completed = 0
            chosen_rewards_sum = 0.0
            rejected_rewards_sum = 0.0

            for step in range(max_steps):
                # Sample batch
                batch_indices = mx.random.choice(
                    len(train_data), size=min(batch_size, len(train_data))
                )
                batch = [train_data[int(i)] for i in batch_indices.tolist()]

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

                # Define loss function
                def loss_fn(model):
                    # Policy model log probs
                    policy_chosen_logps = self._compute_log_probs(
                        model, chosen_ids, chosen_ids
                    )
                    policy_rejected_logps = self._compute_log_probs(
                        model, rejected_ids, rejected_ids
                    )

                    # Reference model log probs (frozen)
                    with mx.no_grad():
                        ref_chosen_logps = self._compute_log_probs(
                            ref_model, chosen_ids, chosen_ids
                        )
                        ref_rejected_logps = self._compute_log_probs(
                            ref_model, rejected_ids, rejected_ids
                        )

                    # Compute DPO loss
                    loss = self._dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                        beta=beta,
                    )

                    return loss

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

                total_loss += loss.item()
                steps_completed += 1

                if (step + 1) % 10 == 0:
                    avg_loss = total_loss / steps_completed
                    logger.info(f"Step {step + 1}/{max_steps}, Loss: {avg_loss:.4f}")

            final_loss = total_loss / steps_completed if steps_completed > 0 else 0.0
            training_time = time.time() - start_time

            logger.info(
                f"DPO training complete: loss={final_loss:.4f}, "
                f"time={training_time:.1f}s"
            )

            return TrainingResult(
                success=True,
                final_loss=final_loss,
                training_time_seconds=training_time,
                samples_trained=steps_completed,
                metrics={
                    "steps_completed": steps_completed,
                },
            )

        except Exception as e:
            logger.error(f"DPO training failed: {e}")
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
        logger.info(f"MLX DPO training on topic: {topic_id}")
        result = self.train(topic_data)

        if result.success:
            logger.info(
                f"Topic {topic_id} DPO complete: loss={result.final_loss:.4f}"
            )

        return result

    def save_adapter(self, path: Path) -> Path:
        """
        Save the DPO LoRA adapter.

        Args:
            path: Directory to save adapter

        Returns:
            Path to saved adapter
        """
        logger.info(f"Saving DPO adapter to {path}")
        self._mlx_model.save_adapter(path)
        return path

    def get_model(self) -> ModelInterface:
        """
        Get the trained policy model.

        Returns:
            MLXModel with trained DPO adapter
        """
        return self._mlx_model

    def get_ref_model(self) -> Optional[ModelInterface]:
        """
        Get the reference model.

        Returns:
            MLXModel reference model, or None if not provided
        """
        return self._mlx_ref_model
