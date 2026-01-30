#!/bin/bash
# Start training on Mac Studio (run this script directly on Mac Studio)
# Usage: ./scripts/local_train.sh

set -e
set -u
set -o pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Starting Training on Mac Studio"
echo "=========================================="
echo "Project: $PROJECT_ROOT"
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

# Kill any existing training processes
echo "🛑 Stopping any existing training processes..."
pkill -f run_dpo_training || true
sleep 2

# Start training with MLX backend
echo "🚀 Starting training with MLX backend..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

nohup python3 scripts/run_dpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    > "$LOG_FILE" 2>&1 &

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

echo "=========================================="
echo "Monitor with: ./scripts/local_monitor.sh"
echo "Or tail logs: tail -f $LOG_FILE"
echo "=========================================="
