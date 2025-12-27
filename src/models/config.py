"""
Model configuration for Bob Loukas mindprint training.

Defines model architectures, layer targeting strategies, and training parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class LayerZone:
    """Configuration for a layer zone (lexicon, reasoning, voice)."""

    name: str
    layers: List[int]
    modules: List[str]
    content: str  # Description of what this zone learns


@dataclass
class LoRAConfig:
    """LoRA configuration for fine-tuning."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "o_proj"])


@dataclass
class ComputeRequirements:
    """Compute requirements for training."""

    vram_4bit: int  # GB for 4-bit quantization
    vram_bf16: int  # GB for bfloat16
    recommended_batch_size: int
    estimated_hours_dpo: float
    estimated_hours_ppo: float


@dataclass
class ModelConfig:
    """Complete model configuration."""

    name: str
    hf_path: str
    layers: int
    hidden_size: int
    attention_heads: int
    kv_heads: int
    context_length: int

    # Layer targeting
    layer_zones: Dict[str, LayerZone]

    # LoRA configuration
    lora: LoRAConfig

    # Compute requirements
    compute: ComputeRequirements

    # Prompt format
    prompt_format: str  # "gemma", "chatml", etc.

    def get_target_modules_for_zone(self, zone_name: str) -> List[str]:
        """Get target modules for a specific layer zone."""
        if zone_name not in self.layer_zones:
            raise KeyError(f"Unknown zone: {zone_name}")
        return self.layer_zones[zone_name].modules

    def get_layers_for_zone(self, zone_name: str) -> List[int]:
        """Get layer indices for a specific zone."""
        if zone_name not in self.layer_zones:
            raise KeyError(f"Unknown zone: {zone_name}")
        return self.layer_zones[zone_name].layers

    def to_peft_config(self, zone: Optional[str] = None) -> dict:
        """
        Generate PEFT LoRA configuration.

        Args:
            zone: Specific zone to target (or all if None)

        Returns:
            Dict compatible with LoraConfig
        """
        if zone:
            target_modules = self.get_target_modules_for_zone(zone)
            layers_pattern = self._create_layers_pattern(self.get_layers_for_zone(zone))
        else:
            # All zones
            target_modules = list(
                set(
                    module
                    for z in self.layer_zones.values()
                    for module in z.modules
                )
            )
            layers_pattern = None

        config = {
            "r": self.lora.r,
            "lora_alpha": self.lora.alpha,
            "lora_dropout": self.lora.dropout,
            "target_modules": target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        if layers_pattern:
            config["layers_pattern"] = layers_pattern

        return config

    def _create_layers_pattern(self, layers: List[int]) -> str:
        """Create a layers pattern string for PEFT."""
        return "|".join(f"layers\\.{i}\\." for i in layers)

    def format_prompt(self, question: str) -> str:
        """Format a question according to model's prompt format."""
        if self.prompt_format == "gemma":
            return f"""<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""
        elif self.prompt_format == "chatml":
            return f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"Question: {question}\n\nAnswer:"


def load_model_config(config_path: str) -> ModelConfig:
    """
    Load model configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ModelConfig instance
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse layer zones
    layer_zones = {}
    for zone_name, zone_data in data.get("layer_zones", {}).items():
        # Handle range notation for layers
        layers = zone_data.get("layers", [])
        if isinstance(layers, str):
            # Parse range like "12-35"
            start, end = map(int, layers.split("-"))
            layers = list(range(start, end + 1))

        layer_zones[zone_name] = LayerZone(
            name=zone_name,
            layers=layers,
            modules=zone_data.get("modules", []),
            content=zone_data.get("content", ""),
        )

    # Parse LoRA config
    lora_data = data.get("lora", {})
    lora_config = LoRAConfig(
        r=lora_data.get("r", 8),
        alpha=lora_data.get("alpha", 16),
        dropout=lora_data.get("dropout", 0.05),
        target_modules=lora_data.get("target_modules", ["q_proj", "v_proj", "o_proj"]),
    )

    # Parse compute requirements
    compute_data = data.get("compute", {})
    compute = ComputeRequirements(
        vram_4bit=compute_data.get("vram_4bit", 24),
        vram_bf16=compute_data.get("vram_bf16", 48),
        recommended_batch_size=compute_data.get("batch_size", 4),
        estimated_hours_dpo=compute_data.get("hours_dpo", 34),
        estimated_hours_ppo=compute_data.get("hours_ppo", 48),
    )

    return ModelConfig(
        name=data.get("name", "unknown"),
        hf_path=data.get("hf_path", ""),
        layers=data.get("layers", 48),
        hidden_size=data.get("hidden_size", 3840),
        attention_heads=data.get("attention_heads", 16),
        kv_heads=data.get("kv_heads", 8),
        context_length=data.get("context_length", 128000),
        layer_zones=layer_zones,
        lora=lora_config,
        compute=compute,
        prompt_format=data.get("prompt_format", "gemma"),
    )


# Pre-defined configurations
GEMMA_12B_CONFIG = ModelConfig(
    name="gemma-3-12b",
    hf_path="google/gemma-3-12b-it",
    layers=48,
    hidden_size=3840,
    attention_heads=16,
    kv_heads=8,
    context_length=128000,
    layer_zones={
        "lexicon": LayerZone(
            name="lexicon",
            layers=list(range(0, 12)),
            modules=["q_proj"],
            content="Bob's terminology (4-year cycle, accumulation, distribution)",
        ),
        "reasoning": LayerZone(
            name="reasoning",
            layers=list(range(12, 36)),
            modules=["v_proj", "up_proj", "down_proj"],
            content="Cycle theory, pattern recognition, market analysis",
        ),
        "voice": LayerZone(
            name="voice",
            layers=list(range(36, 48)),
            modules=["o_proj", "up_proj", "down_proj"],
            content="Confidence markers, teaching style, engagement patterns",
        ),
    },
    lora=LoRAConfig(r=8, alpha=16, dropout=0.05),
    compute=ComputeRequirements(
        vram_4bit=24,
        vram_bf16=48,
        recommended_batch_size=4,
        estimated_hours_dpo=34,
        estimated_hours_ppo=48,
    ),
    prompt_format="gemma",
)

QWEN_7B_CONFIG = ModelConfig(
    name="qwen2.5-7b",
    hf_path="Qwen/Qwen2.5-7B-Instruct",
    layers=28,
    hidden_size=3584,
    attention_heads=28,
    kv_heads=4,
    context_length=131072,
    layer_zones={
        "lexicon": LayerZone(
            name="lexicon",
            layers=list(range(0, 7)),
            modules=["q_proj"],
            content="Bob's terminology",
        ),
        "reasoning": LayerZone(
            name="reasoning",
            layers=list(range(7, 21)),
            modules=["v_proj", "up_proj", "down_proj"],
            content="Cycle theory, pattern recognition",
        ),
        "voice": LayerZone(
            name="voice",
            layers=list(range(21, 28)),
            modules=["o_proj", "up_proj", "down_proj"],
            content="Confidence markers, teaching style",
        ),
    },
    lora=LoRAConfig(r=8, alpha=16, dropout=0.05),
    compute=ComputeRequirements(
        vram_4bit=16,
        vram_bf16=32,
        recommended_batch_size=4,
        estimated_hours_dpo=20,
        estimated_hours_ppo=30,
    ),
    prompt_format="chatml",
)


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get a pre-defined model configuration.

    Args:
        model_name: "gemma" or "qwen"

    Returns:
        ModelConfig instance
    """
    configs = {
        "gemma": GEMMA_12B_CONFIG,
        "gemma-12b": GEMMA_12B_CONFIG,
        "gemma-3-12b": GEMMA_12B_CONFIG,
        "qwen": QWEN_7B_CONFIG,
        "qwen-7b": QWEN_7B_CONFIG,
        "qwen2.5-7b": QWEN_7B_CONFIG,
    }

    if model_name.lower() not in configs:
        raise KeyError(
            f"Unknown model: {model_name}. Available: {list(configs.keys())}"
        )

    return configs[model_name.lower()]
