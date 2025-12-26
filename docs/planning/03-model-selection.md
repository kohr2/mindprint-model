# Phase 3: Model Selection

## Objective

Select the optimal open-source model for the Bob Loukas mindprint based on knowledge injection, voice fidelity, and reasoning capability requirements.

## Requirements

### Bob Loukas Mindprint Needs

| Requirement | Importance | Notes |
|-------------|------------|-------|
| Domain Knowledge | High | Bitcoin cycles, market psychology, technical analysis |
| Voice/Personality | Critical | Confident, educational, pattern-focused |
| Reasoning | High | Explain causation, distinguish correlation |
| Instruction Following | High | Answer questions in Bob's style |

### Technical Constraints

- **VRAM**: 24-40GB available
- **Training Time**: <1 week for full curriculum
- **Inference**: Must be deployable (quantized)

## Candidate Models

### 1. Gemma-3-12B (Primary Recommendation)

**Architecture:**
- 48 layers, 3840 hidden size
- GQA: 16 Q heads, 8 KV heads
- 128k context with RoPE

**Pros:**
- Layer targeting already documented in cookbook
- Good balance of size and capability
- Strong instruction following
- Apache 2.0 license

**Layer Targeting (from cookbook):**
```
Zone        | Layers  | Target          | Bob's Content
------------|---------|-----------------|------------------
Lexicon     | 0-11    | q_proj          | "4-year cycle", terminology
Reasoning   | 12-35   | v_proj + MLP    | Cycle theory, patterns
Voice       | 36-47   | o_proj + MLP    | Confidence, teaching style
```

### 2. Qwen2.5-7B

**Pros:**
- Smaller footprint (~16GB 4-bit)
- Strong reasoning benchmarks
- Fast inference

**Cons:**
- Fewer layers = less targeting granularity
- Would need layer mapping

**When to Choose:** Limited GPU memory, faster iteration.

### 3. Qwen2.5-14B

**Pros:**
- Better reasoning than 7B
- More knowledge capacity

**Cons:**
- Requires ~32GB VRAM (4-bit)
- Slower training

**When to Choose:** Have A100 80GB, need maximum quality.

## Comparison Matrix

| Model | Params | VRAM (4-bit) | Layers | Voice Potential | License |
|-------|--------|--------------|--------|-----------------|---------|
| **Gemma-3-12B** | 12B | 24GB | 48 | Excellent | Apache 2.0 |
| Qwen2.5-7B | 7B | 16GB | 28 | Good | Apache 2.0 |
| Qwen2.5-14B | 14B | 32GB | 40 | Excellent | Apache 2.0 |

## Recommendation

### Primary: Gemma-3-12B on Mac Studio M2 Ultra

**Rationale:**
1. Layer anatomy already documented in cookbook
2. 48 layers = maximum targeting granularity
3. Proven for mindprinting approach
4. 64GB unified memory allows fp16 without quantization
5. Larger batch sizes (8) than CUDA alternatives

### Alternative: Qwen2.5-7B

**When to use:**
- Limited memory (< 32GB)
- Need faster iteration
- Consumer-grade deployment

## Layer Targeting Configuration

```yaml
# For Gemma-3-12B

lexicon_zone:
  layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  modules: ["q_proj"]
  content: Bob's terminology

reasoning_zone:
  layers: [12-35]
  modules: ["v_proj", "up_proj", "down_proj"]
  content: Cycle theory, pattern recognition

voice_zone:
  layers: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
  modules: ["o_proj", "up_proj", "down_proj"]
  content: Confidence markers, teaching style
```

## Compute Requirements

### Gemma-3-12B Training

| Setup | Precision | Batch Size | Time Estimate |
|-------|-----------|------------|---------------|
| **Mac Studio M2 Ultra (64GB)** | fp16 | 8 | ~40-50h (DPO) |
| A100 40GB | 4-bit | 4 | ~34h (DPO) / ~48h (PPO) |
| 2x RTX 4090 | 4-bit | 2/GPU | ~50h (DPO) / ~70h (PPO) |
| 1x RTX 4090 | Use Qwen-7B | - | - |

### Apple Silicon Notes

- **No quantization needed**: 64GB unified memory loads Gemma-3-12B in fp16 (~24GB) with headroom
- **Unified memory advantage**: Larger batch sizes possible (8 vs 4)
- **MPS backend**: Uses Metal Performance Shaders instead of CUDA
- **Flash Attention**: Not available on MPS; uses standard attention

---

*Phase 3 - Bob Loukas Mindprint RLHF LoRA*
*Branch: shared*

