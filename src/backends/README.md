# Backends System

Multi-framework ML backend abstraction for mindprint-model training.

## Overview

The backends system provides a unified interface for training LLMs with different ML frameworks:

- **PyTorch** (transformers, PEFT, TRL) - For cloud CUDA GPUs
- **MLX** (mlx-lm) - For Apple Silicon M-series chips

This architecture solves the fundamental PyTorch MPS corruption bugs (non-contiguous tensor issues in `merge_and_unload`) by providing a stable MLX alternative for Mac Studio while maintaining PyTorch support for cloud GPU training.

## Quick Start

### 1. Choose Your Backend

```python
from src.backends import create_backend

# For cloud GPU (PyTorch)
backend = create_backend("pytorch", device="cuda", dtype="float16")

# For Mac Studio (MLX)
backend = create_backend("mlx", device="auto", dtype="float16")
```

### 2. Load a Model

```python
model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")
```

### 3. Create a Trainer

```python
# SFT Training
sft_config = {
    "learning_rate": 3e-4,
    "num_epochs": 3,
    "per_device_batch_size": 4,
    "lora_r": 8,
    "lora_alpha": 16,
}
sft_trainer = backend.create_sft_trainer(model, sft_config)

# Train
result = sft_trainer.train(train_data)
```

### 4. DPO Training

```python
dpo_config = {
    "learning_rate": 5e-7,
    "max_steps": 100,
    "per_device_batch_size": 2,
    "beta": 0.1,
}
dpo_trainer = backend.create_dpo_trainer(model, dpo_config)

# Train with preference pairs
result = dpo_trainer.train(preference_pairs)
```

## Configuration

### Via Config File

```yaml
# configs/training_pipeline.yaml
backend:
  type: pytorch  # "pytorch", "mlx", or null for legacy mode
  device: auto   # "auto", "mps", "cuda", "cpu", "gpu"
  dtype: float16 # "float16", "float32", "bfloat16"
```

### Via Code

```python
from src.backends import BackendConfig, create_backend

config = BackendConfig(
    backend_type="mlx",
    device="auto",
    dtype="float16",
    seed=42,
)

backend = BackendRegistry.create(config)
```

## Architecture

### Core Abstraction Layer

```
src/backends/
├── protocol.py           # BackendProtocol interface
├── model_interface.py    # ModelInterface abstraction
├── trainer_interface.py  # TrainerInterface, TrainingResult
├── adapter_interface.py  # AdapterManager, AdapterConfig
└── factory.py           # BackendRegistry, create_backend()
```

**Key Interfaces:**

- `BackendProtocol`: Defines what every backend must implement
- `ModelInterface`: Unified model API (forward, generate, adapter ops)
- `TrainerInterface`: Unified trainer API (train, save_adapter)
- `AdapterManager`: LoRA adapter operations across frameworks

### Backend Implementations

#### PyTorch Backend

```
src/backends/pytorch/
├── pytorch_backend.py         # Main backend
├── pytorch_model.py           # transformers model wrapper
├── pytorch_device_manager.py  # MPS/CUDA/CPU management
├── pytorch_adapter_manager.py # PEFT adapter operations
├── pytorch_sft_trainer.py     # Wraps SFTTrainer
└── pytorch_dpo_trainer.py     # Wraps Rank1DPOTrainer
```

**Features:**
- Wraps existing PyTorch trainers (SFTTrainer, Rank1DPOTrainer)
- Uses TRL's mature DPOTrainer
- PEFT for LoRA adapters
- Supports MPS, CUDA, and CPU

**Known Issues:**
- ⚠️ PyTorch MPS has non-contiguous tensor bugs
- ⚠️ `merge_and_unload()` corrupts models on MPS
- ⚠️ `save_pretrained()` also has corruption issues on MPS

#### MLX Backend

```
src/backends/mlx/
├── mlx_backend.py         # Main backend
├── mlx_model.py           # mlx-lm model wrapper
├── mlx_device_manager.py  # Unified memory management
├── mlx_adapter_manager.py # mlx-lm LoRA operations
├── mlx_sft_trainer.py     # Manual SFT training loop
└── mlx_dpo_trainer.py     # Manual DPO loss implementation
```

**Features:**
- Manual training loops (no TRL equivalent)
- Manual DPO loss (Bradley-Terry model)
- mlx-lm native LoRA support
- Unified memory (no explicit device placement)
- ✅ No corruption issues

**Advantages:**
- Stable on Apple Silicon
- No adapter corruption bugs
- Native M-series optimization
- Simple memory management

## PyTorch vs MLX

| Feature | PyTorch | MLX |
|---------|---------|-----|
| **Target** | Cloud CUDA GPU | Mac Studio M-series |
| **Device Management** | Explicit (MPS/CUDA/CPU) | Unified memory (auto) |
| **Training Framework** | TRL (DPOTrainer) | Manual loops |
| **LoRA Library** | PEFT | mlx-lm |
| **Adapter Stability** | ❌ Corruption on MPS | ✅ Stable |
| **Maturity** | Very mature | Newer |
| **Community Support** | Extensive | Growing |

**When to Use PyTorch:**
- Cloud GPU training with CUDA
- Need TRL's advanced features
- Extensive model zoo access
- Well-tested production workflows

**When to Use MLX:**
- Mac Studio M-series training
- Avoiding MPS corruption bugs
- Apple Silicon optimization
- Stable adapter operations

## Usage Patterns

### Pattern 1: Direct Backend Usage

```python
from src.backends import create_backend

# Create backend
backend = create_backend("mlx", device="auto")

# Load model
model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")

# Create trainer
trainer = backend.create_sft_trainer(model, config)

# Train
result = trainer.train(train_data)

# Save adapter
trainer.save_adapter(Path("./adapters/sft"))
```

### Pattern 2: Pipeline Integration

```python
from src.training.dpo_pipeline import DPOPipeline, PipelineConfig
from src.backends import create_backend

# Create backend
backend = create_backend("mlx", device="auto")

# Create pipeline with backend
config = PipelineConfig(backend_type="mlx")
pipeline = DPOPipeline(
    model=None,  # Loaded via backend
    tokenizer=None,
    config=config,
    backend=backend,
)

# Train full curriculum
result = pipeline.train_curriculum()
```

### Pattern 3: Legacy Mode (Backward Compatible)

```python
from src.training.dpo_pipeline import DPOPipeline, PipelineConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model directly
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

# Create pipeline without backend
config = PipelineConfig(backend_type=None)  # Legacy mode
pipeline = DPOPipeline(model, tokenizer, config)

# Uses direct PyTorch trainers
result = pipeline.train_curriculum()
```

## MLX DPO Implementation

The MLX backend implements DPO loss from scratch using the Bradley-Terry model:

```python
def _dpo_loss(
    policy_chosen_logps,    # Log P(chosen | policy)
    policy_rejected_logps,   # Log P(rejected | policy)
    ref_chosen_logps,       # Log P(chosen | reference)
    ref_rejected_logps,     # Log P(rejected | reference)
    beta=0.1,               # KL penalty coefficient
):
    # Compute log probability ratios
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # Bradley-Terry loss: -E[log sigmoid(β(r_θ - r_ref))]
    # where r = log(p_chosen / p_rejected)
    logits = beta * (policy_logratios - ref_logratios)
    loss = -mx.mean(mx.logaddexp(0, -logits))  # -log(sigmoid(logits))

    return loss
```

This matches TRL's DPO implementation but uses pure MLX operations.

## Testing

```bash
# Run all backend tests
pytest tests/unit/backends/ -v

# Test specific backend
pytest tests/unit/backends/pytorch/ -v
pytest tests/unit/backends/mlx/ -v

# Integration tests
pytest tests/integration/test_backend_pipeline.py -v
```

**Test Coverage:**
- ✅ Backend factory: 10/10 tests
- ✅ PyTorch backend: 4/4 tests
- ✅ MLX backend: 5/5 tests
- ✅ Pipeline integration: 3/3 tests
- **Total: 22/22 tests passing**

## Installation

### PyTorch Backend

```bash
pip install torch transformers peft trl datasets
```

### MLX Backend

```bash
pip install mlx mlx-lm transformers datasets
```

### Both (Recommended)

```bash
pip install torch transformers peft trl datasets mlx mlx-lm
```

## Migration from Legacy Code

### Before (Direct PyTorch)

```python
from src.training.sft_trainer import SFTTrainer, SFTConfig

config = SFTConfig(learning_rate=3e-4, num_epochs=3)
trainer = SFTTrainer(model, tokenizer, config)
result = trainer.train(train_data)
```

### After (Backend Interface)

```python
from src.backends import create_backend

backend = create_backend("pytorch", device="cuda")
model_interface = backend.load_model("Qwen/Qwen2.5-7B-Instruct")

config = {"learning_rate": 3e-4, "num_epochs": 3}
trainer = backend.create_sft_trainer(model_interface, config)
result = trainer.train(train_data)
```

**Benefits:**
- ✅ Can switch to MLX by changing one line
- ✅ Unified interface across frameworks
- ✅ No adapter corruption on MLX
- ✅ Legacy code still works (backward compatible)

## Troubleshooting

### Issue: "Unknown backend: pytorch"

**Cause:** Backend not registered.

**Solution:** Import the backend package to trigger registration:
```python
import src.backends.pytorch  # Registers automatically
```

### Issue: "mlx-lm not installed"

**Cause:** MLX dependencies not installed.

**Solution:**
```bash
pip install mlx mlx-lm
```

### Issue: PyTorch MPS corruption

**Symptoms:**
- Model outputs degrade after `merge_and_unload()`
- Voice scores drop to 0.00
- Save/reload produces corrupted model

**Solution:** Switch to MLX backend for Mac Studio training:
```yaml
backend:
  type: mlx  # Instead of pytorch
  device: auto
```

### Issue: "Expected MLXModel, got PyTorchModel"

**Cause:** Mixing backends (creating model with PyTorch, trainer with MLX).

**Solution:** Use the same backend for both:
```python
backend = create_backend("mlx")
model = backend.load_model(...)  # Creates MLXModel
trainer = backend.create_sft_trainer(model, config)  # Expects MLXModel
```

## API Reference

### BackendProtocol

```python
class BackendProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def load_model(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ) -> ModelInterface: ...

    def create_sft_trainer(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
    ) -> TrainerInterface: ...

    def create_dpo_trainer(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
        ref_model: Optional[ModelInterface] = None,
    ) -> TrainerInterface: ...
```

### ModelInterface

```python
class ModelInterface(ABC):
    @abstractmethod
    def generate(self, input_ids, ...) -> Any: ...

    @abstractmethod
    def forward(self, input_ids, labels=None, ...) -> Dict[str, Any]: ...

    @abstractmethod
    def save_pretrained(self, path: Path) -> None: ...

    @abstractmethod
    def load_adapter(self, adapter_path: Path) -> None: ...

    @abstractmethod
    def save_adapter(self, adapter_path: Path) -> None: ...

    @property
    def device(self) -> Any: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def tokenizer(self) -> Any: ...
```

### TrainerInterface

```python
class TrainerInterface(ABC):
    @abstractmethod
    def train(self, train_data: List[Dict]) -> TrainingResult: ...

    @abstractmethod
    def train_on_topic(
        self,
        topic_data: List[Dict],
        topic_id: str,
    ) -> TrainingResult: ...

    @abstractmethod
    def save_adapter(self, path: Path) -> Path: ...

    @abstractmethod
    def get_model(self) -> ModelInterface: ...
```

### TrainingResult

```python
@dataclass
class TrainingResult:
    success: bool
    final_loss: float
    training_time_seconds: float
    samples_trained: int
    adapter_path: str = ""
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
```

## Contributing

When adding a new backend:

1. Implement all methods in `BackendProtocol`
2. Create backend-specific Model, DeviceManager, AdapterManager
3. Implement SFTTrainer and DPOTrainer
4. Register backend in `__init__.py`:
   ```python
   BackendRegistry.register("my_backend", MyBackend)
   ```
5. Add tests in `tests/unit/backends/my_backend/`
6. Update this README

## References

- [PyTorch MPS Issue #78043](https://github.com/pytorch/pytorch/issues/78043)
- [PEFT Issue #2502](https://github.com/huggingface/peft/issues/2502)
- [PEFT Issue #2764](https://github.com/huggingface/peft/issues/2764)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [mlx-lm Documentation](https://github.com/ml-explore/mlx-examples/tree/main/llms)

## License

Part of the mindprint-model project.
