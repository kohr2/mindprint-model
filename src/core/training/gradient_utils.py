"""
Gradient accumulation and clipping utilities.

Enables effective larger batch sizes and stable training.
"""

from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass
class GradientConfig:
    """Gradient management configuration."""
    accumulation_steps: int = 8  # Steps to accumulate before update
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    gradient_checkpointing: bool = False  # Enable gradient checkpointing


class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes.
    
    Why: DPO/SimPO are sensitive to batch size. Accumulation enables
    effective batch sizes of 16-32 even with limited GPU memory.
    
    Usage:
        accumulator = GradientAccumulator(config)
        for batch in dataloader:
            loss, grads = compute_gradients(batch)
            scaled_grads = accumulator.accumulate(grads)
            if scaled_grads is not None:
                optimizer.update(model, scaled_grads)
    """
    
    def __init__(self, config: GradientConfig):
        """
        Initialize gradient accumulator.
        
        Args:
            config: Gradient configuration
        """
        self.config = config
        self._accumulated_grads: Optional[Dict[str, Any]] = None
        self._step_count = 0
    
    def accumulate(self, grads: Any) -> Optional[Any]:
        """
        Accumulate gradients. Returns scaled gradients when ready.
        
        Args:
            grads: Gradients from current step
        
        Returns:
            Scaled gradients if accumulation complete, None otherwise
        """
        if self._accumulated_grads is None:
            self._accumulated_grads = grads
        else:
            # Add gradients (framework-specific)
            self._accumulated_grads = self._add_grads(
                self._accumulated_grads, grads
            )
        
        self._step_count += 1
        
        if self._step_count >= self.config.accumulation_steps:
            # Scale and return
            scaled = self._scale_grads(
                self._accumulated_grads,
                1.0 / self.config.accumulation_steps
            )
            self._accumulated_grads = None
            self._step_count = 0
            return scaled
        
        return None
    
    def clip_gradients(self, grads: Any) -> Any:
        """
        Clip gradients by global norm.
        
        Args:
            grads: Gradients to clip
        
        Returns:
            Clipped gradients
        """
        total_norm = self._compute_global_norm(grads)
        clip_coef = self.config.max_grad_norm / (total_norm + 1e-6)
        
        if clip_coef < 1.0:
            grads = self._scale_grads(grads, clip_coef)
        
        return grads
    
    def _add_grads(self, a: Any, b: Any) -> Any:
        """Add two gradient structures."""
        if hasattr(a, '__iter__') and not isinstance(a, (str, bytes)):
            if isinstance(a, dict):
                return {k: self._add_grads(a[k], b[k]) for k in a}
            elif isinstance(a, (list, tuple)):
                return type(a)(self._add_grads(ai, bi) for ai, bi in zip(a, b))
            else:
                # MLX or PyTorch tensor
                return a + b
        return a + b
    
    def _scale_grads(self, grads: Any, scale: float) -> Any:
        """Scale gradients by a factor."""
        if hasattr(grads, '__iter__') and not isinstance(grads, (str, bytes)):
            if isinstance(grads, dict):
                return {k: self._scale_grads(v, scale) for k, v in grads.items()}
            elif isinstance(grads, (list, tuple)):
                return type(grads)(self._scale_grads(g, scale) for g in grads)
            else:
                # MLX or PyTorch tensor
                return grads * scale
        return grads * scale
    
    def _compute_global_norm(self, grads: Any) -> float:
        """Compute global norm of gradients."""
        import numpy as np
        
        def _get_norm(g):
            if hasattr(g, 'norm'):
                return float(g.norm().item() if hasattr(g.norm(), 'item') else g.norm())
            elif hasattr(g, '__array__'):
                return float(np.linalg.norm(g))
            return 0.0
        
        def _collect_norms(g):
            if isinstance(g, dict):
                return [_collect_norms(v) for v in g.values()]
            elif isinstance(g, (list, tuple)):
                return [_collect_norms(item) for item in g]
            else:
                return _get_norm(g)
        
        norms = _collect_norms(grads)
        flat_norms = []
        for n in norms:
            if isinstance(n, list):
                flat_norms.extend(n)
            else:
                flat_norms.append(n)
        
        return float(np.sqrt(sum(n**2 for n in flat_norms if n > 0)))
