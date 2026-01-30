#!/bin/bash
# Start training on Mac Studio (run this script directly on Mac Studio)
# Usage: ./scripts/local_train.sh [--dataset transcripts|textbook|combined] [--model qwen|gemma]

set -e
set -u
set -o pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Parse arguments
DATASET=""
MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset transcripts|textbook|combined] [--model qwen|gemma]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Starting Training on Mac Studio"
echo "=========================================="
echo "Project: $PROJECT_ROOT"
[ -n "$DATASET" ] && echo "Dataset: $DATASET"
[ -n "$MODEL" ] && echo "Model: $MODEL"
echo ""

# Pull latest code
echo "📥 Pulling latest code from git..."
git pull origin main || {
    echo "⚠️  Warning: git pull failed. Continuing with current code..."
}
echo ""

# Create logs directory
mkdir -p logs

# Check if config exists
if [ ! -f "configs/training_pipeline.yaml" ]; then
    echo "❌ Error: Config file not found: configs/training_pipeline.yaml"
    exit 1
fi

# Check if Python script exists
if [ ! -f "scripts/run_dpo_training.py" ]; then
    echo "❌ Error: Training script not found: scripts/run_dpo_training.py"
    exit 1
fi

# Determine data directory based on dataset
DATA_DIR=""
if [ -n "$DATASET" ]; then
    case "$DATASET" in
        transcripts|textbook|combined)
            DATA_DIR="./data/bob_loukas/$DATASET"
            ;;
        *)
            echo "❌ Error: Invalid dataset '$DATASET'. Must be: transcripts, textbook, or combined"
            exit 1
            ;;
    esac
fi

# Determine model name based on model argument
MODEL_NAME=""
if [ -n "$MODEL" ]; then
    case "$MODEL" in
        qwen)
            MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
            ;;
        gemma)
            MODEL_NAME="google/gemma-2-9b-it"
            ;;
        *)
            echo "❌ Error: Invalid model '$MODEL'. Must be: qwen or gemma"
            exit 1
            ;;
    esac
fi

# Kill any existing training processes
echo "🛑 Stopping any existing training processes..."
pkill -f run_dpo_training || true
sleep 2

# Start training with MLX backend
echo "🚀 Starting training with MLX backend..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Build command with optional arguments
TRAIN_CMD="python3 scripts/run_dpo_training.py --config configs/training_pipeline.yaml --backend mlx"
[ -n "$DATA_DIR" ] && TRAIN_CMD="$TRAIN_CMD --data-dir $DATA_DIR"
[ -n "$MODEL_NAME" ] && TRAIN_CMD="$TRAIN_CMD --model $MODEL_NAME"

nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo "$TRAINING_PID" > logs/training.pid

echo ""
echo "✅ Training started!"
echo "   PID: $TRAINING_PID"
echo "   Log: $LOG_FILE"
echo ""

# Show initial output
sleep 5
if [ -f "$LOG_FILE" ]; then
    echo "📋 Initial output:"
    echo "=========================================="
    tail -30 "$LOG_FILE"
    echo ""
fi

echo ""
echo "=========================================="
echo "Monitor with: ./scripts/local_monitor.sh"
echo "Or tail logs: tail -f $LOG_FILE"
echo "=========================================="
echo ""
echo "Usage examples:"
echo "  ./scripts/local_train.sh --dataset transcripts --model qwen"
echo "  ./scripts/local_train.sh --dataset combined --model gemma"
echo "  ./scripts/local_train.sh --dataset textbook"
echo ""
