#!/bin/bash
# Dry-run test script for MLX backend configuration
# Usage: ./test_dry_run.sh [config-file]

set -e

CONFIG_FILE="${1:-configs/training_pipeline.yaml}"
MODEL="${MAC_STUDIO_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "=== MLX Backend Dry-Run Test ==="
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

# Check MLX installation
echo "Checking MLX installation..."
if ! python3 -c "import mlx; import mlx_lm" 2>/dev/null; then
    echo "ERROR: MLX not installed. Install with: pip3 install mlx mlx-lm"
    exit 1
fi
echo "✓ MLX installed"

# Run dry-run
echo ""
echo "Running dry-run..."
python3 scripts/run_dpo_training.py \
    --config "$CONFIG_FILE" \
    --backend mlx \
    --model "$MODEL" \
    --dry-run

echo ""
echo "=== Dry-run complete ==="
echo "If no errors above, configuration is valid and ready for training."
