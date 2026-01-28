# Documentation Index

Complete index of mindprint-model documentation organized by topic.

**Last Updated**: January 28, 2026

---

## Getting Started

- **[README.md](../README.md)** - Project overview, quick start, installation
- **[Quick Start Guide](../README.md#quick-start)** - Basic training examples

---

## Backend System

### Core Documentation

- **[Backend System Overview](../src/backends/README.md)** - Complete backend API documentation
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Backend implementation timeline and details
- **[Migration Guide](MIGRATION.md)** - Migrating from legacy code to backend system
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions

### MLX Backend (Apple Silicon)

- **[MLX LoRA Training Issue](mlx/MLX_LORA_TRAINING_ISSUE.md)** ⭐ - Investigation and resolution of LoRA adapter bug
- **[MLX LoRA Architecture](mlx/MLX_LORA_ARCHITECTURE.md)** - Technical architecture and implementation details
- **[MLX Backend Troubleshooting](mlx/MLX_BACKEND_TROUBLESHOOTING.md)** - Common issues and solutions
- **[MLX Real-World Testing](mlx/MLX_REAL_WORLD_TESTING.md)** - Testing guide for Mac Studio

### Issue Investigation

- **[Adapter Stacking Debug](../ADAPTER_STACKING_DEBUG.md)** - PyTorch MPS adapter corruption investigation
- **[Prompt Format Fix](PROMPT_FORMAT_FIX.md)** - Qwen2.5 prompt format investigation

---

## Testing

### Test Documentation

- **[Testing Ready](TESTING_READY.md)** - Overview of testing infrastructure
- **[Debug Tests README](../tests/debug/README.md)** - Diagnostic test documentation

### Test Scripts

- **Diagnostic Test**: `tests/debug/test_mlx_training_state.py` - MLX training state verification
- **Single Topic Test**: `scripts/test_single_topic_mlx.sh` - Quick validation
- **Monitoring**: `scripts/monitor_training.sh` - Training progress monitoring

---

## Models

### Model Configurations

- **[Qwen2.5-72B Guide](models/QWEN_72B.md)** - 72B model configuration and usage
- **Model Configs**: `src/models/*.yaml` - YAML configuration files

### Supported Models

| Model | Documentation | Config File |
|-------|---------------|-------------|
| Qwen2.5-7B | README.md | `src/models/qwen_config.yaml` |
| Qwen2.5-72B | `docs/models/QWEN_72B.md` | `src/models/qwen_72b_config.yaml` |
| Gemma-3-12B | README.md | `src/models/gemma_config.yaml` |

---

## Planning Documents

Historical planning documents (archived):

- **[00-overview.md](planning/00-overview.md)** - Project overview
- **[01-data-preparation.md](planning/01-data-preparation.md)** - Data preparation
- **[02-voice-evaluator.md](planning/02-voice-evaluator.md)** - Voice evaluation design
- **[02b-evaluation-pipeline.md](planning/02b-evaluation-pipeline.md)** - Evaluation pipeline
- **[03-model-selection.md](planning/03-model-selection.md)** - Model selection criteria
- **[04-dpo-trainer.md](planning/04-dpo-trainer.md)** - DPO trainer design
- **[05-dpo-pipeline.md](planning/05-dpo-pipeline.md)** - Pipeline architecture
- **[06-dpo-training.md](planning/06-dpo-training.md)** - Training procedures

---

## Documentation by Topic

### Training Issues and Resolutions

1. **MLX LoRA Training** (Jan 28, 2026)
   - Investigation: `docs/mlx/MLX_LORA_TRAINING_ISSUE.md`
   - Architecture: `docs/mlx/MLX_LORA_ARCHITECTURE.md`
   - Troubleshooting: `docs/mlx/MLX_BACKEND_TROUBLESHOOTING.md`
   - Test: `tests/debug/test_mlx_training_state.py`

2. **PyTorch MPS Adapter Corruption** (Jan 10, 2026)
   - Investigation: `ADAPTER_STACKING_DEBUG.md`
   - Solution: Use MLX backend instead

3. **Prompt Format Mismatch** (Jan 28, 2026)
   - Investigation: `docs/PROMPT_FORMAT_FIX.md`
   - Solution: Use `tokenizer.apply_chat_template()`

### Backend Implementation

1. **Architecture**
   - Overview: `src/backends/README.md`
   - Implementation: `docs/IMPLEMENTATION_SUMMARY.md`
   - Migration: `docs/MIGRATION.md`

2. **Deployment**
   - Guide: `docs/DEPLOYMENT.md`
   - Testing: `docs/MLX_REAL_WORLD_TESTING.md`
   - Scripts: `scripts/train_on_mac_studio.sh`

### Testing and Debugging

1. **Diagnostic Tests**
   - MLX Training State: `tests/debug/test_mlx_training_state.py`
   - Documentation: `tests/debug/README.md`

2. **Integration Tests**
   - Backend Equivalence: `tests/integration/test_backend_equivalence.py`
   - Pipeline Integration: `tests/integration/test_backend_pipeline.py`

---

## Quick Reference

### Common Tasks

| Task | Documentation | Command |
|------|---------------|---------|
| Train with MLX | [MLX Testing](mlx/MLX_REAL_WORLD_TESTING.md) | `python scripts/run_dpo_training.py --backend mlx` |
| Train with PyTorch | [README](../README.md) | `python scripts/run_dpo_training.py --backend pytorch` |
| Troubleshoot MLX | [Troubleshooting](mlx/MLX_BACKEND_TROUBLESHOOTING.md) | `python tests/debug/test_mlx_training_state.py` |
| Verify LoRA | [Architecture](mlx/MLX_LORA_ARCHITECTURE.md) | Check `model.num_trainable_parameters` |
| Monitor training | [Testing](mlx/MLX_REAL_WORLD_TESTING.md) | `scripts/monitor_training.sh` |

### Issue Resolution

| Symptom | Documentation | Solution |
|---------|---------------|----------|
| Voice scores 0.00 | [MLX LoRA Issue](mlx/MLX_LORA_TRAINING_ISSUE.md) | Verify LoRA adapters attached |
| `<|endoftext|>` spam | [Troubleshooting](mlx/MLX_BACKEND_TROUBLESHOOTING.md) | Run diagnostic test |
| Prompt format errors | [Prompt Fix](PROMPT_FORMAT_FIX.md) | Use `apply_chat_template()` |
| Adapter corruption | [Adapter Debug](../ADAPTER_STACKING_DEBUG.md) | Use MLX backend |
| Training too slow | [Troubleshooting](mlx/MLX_BACKEND_TROUBLESHOOTING.md) | Increase batch size |

---

## Documentation Standards

### File Naming

- `UPPERCASE_WITH_UNDERSCORES.md` - Major investigation/issue documents
- `lowercase_with_underscores.md` - Planning documents
- `PascalCase.md` - Guide documents (rare)

### Document Structure

All major documents should include:
1. Executive summary
2. Problem description
3. Investigation/analysis
4. Solution/resolution
5. Verification steps
6. References
7. Last updated date

### Cross-References

Documents should cross-reference related docs:
- Use relative paths: `docs/OTHER_DOC.md`
- Link to specific sections: `docs/OTHER_DOC.md#section-name`
- Reference code: `` `src/path/file.py` ``

---

## Contributing to Documentation

### Adding New Documentation

1. Create document in appropriate directory
2. Follow naming conventions
3. Include all standard sections
4. Add entry to this index
5. Cross-reference from related docs

### Updating Existing Documentation

1. Add "Last Updated" date
2. Note what changed in commit message
3. Update cross-references if needed
4. Verify links still work

---

**Document Version**: 1.0  
**Last Updated**: January 28, 2026  
**Maintainer**: Project Team
