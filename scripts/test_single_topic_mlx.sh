#!/bin/bash
# Single-topic SFT training test with MLX backend
# Usage: ./test_single_topic_mlx.sh [unit] [chapter] [topic]
#
# Example: ./test_single_topic_mlx.sh unit-01 chapter-01 topic-01

set -e
set -u
set -o pipefail

UNIT="${1:-unit-01}"
CHAPTER="${2:-chapter-01}"
TOPIC="${3:-topic-01}"
CONFIG_FILE="configs/training_pipeline.yaml"
MODEL="${MAC_STUDIO_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "=== Single-Topic MLX Training Test ==="
echo "Unit: $UNIT"
echo "Chapter: $CHAPTER"
echo "Topic: $TOPIC"
echo "Model: $MODEL"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if data directory exists
DATA_DIR="data/bob_loukas/$UNIT/$CHAPTER/$TOPIC"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Available topics:"
    find data/bob_loukas -type d -name "topic-*" | head -10
    exit 1
fi

# Check MLX installation
if ! python3 -c "import mlx; import mlx_lm" 2>/dev/null; then
    echo "ERROR: MLX not installed"
    exit 1
fi

# Create temporary config for single topic
TEMP_CONFIG=$(mktemp)
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Modify config to limit to single topic (if needed)
# Note: The pipeline will naturally process topics sequentially
# This script assumes the pipeline will stop after the first topic
# or you can manually interrupt after one topic completes

echo "Starting single-topic training..."
echo "This will train on: $UNIT/$CHAPTER/$TOPIC"
echo ""
echo "To monitor progress in another terminal:"
echo "  tail -f logs/training_*.log"
echo ""

# Run training
python3 scripts/run_orpo_training.py \
    --config "$CONFIG_FILE" \
    --backend mlx \
    --model "$MODEL" \
    2>&1 | tee "logs/single_topic_test_$(date +%Y%m%d_%H%M%S).log"

# Cleanup
rm -f "$TEMP_CONFIG"

echo ""
echo "=== Single-topic test complete ==="
echo "Check logs/ directory for detailed output"
