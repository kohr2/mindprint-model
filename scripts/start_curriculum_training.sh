#!/bin/bash
# Start curriculum training on Mac Studio
# Usage: ./scripts/start_curriculum_training.sh [--background]

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Starting Curriculum Training"
echo "=========================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check config file
if [ ! -f "configs/training_pipeline_curriculum.yaml" ]; then
    echo "❌ Error: Config file not found: configs/training_pipeline_curriculum.yaml"
    exit 1
fi
echo "  ✓ Config file found"

# Check data directory
if [ ! -d "data/bob_loukas/textbook" ]; then
    echo "❌ Error: Data directory not found: data/bob_loukas/textbook"
    exit 1
fi
echo "  ✓ Data directory found"

# Check data files
if [ ! -f "data/bob_loukas/textbook/preference_data.jsonl" ]; then
    echo "❌ Error: Preference data not found"
    exit 1
fi
if [ ! -f "data/bob_loukas/textbook/sft_data.jsonl" ]; then
    echo "❌ Error: SFT data not found"
    exit 1
fi
echo "  ✓ Data files found"

# Create directories
mkdir -p output/curriculum
mkdir -p checkpoints/curriculum
mkdir -p logs
echo "  ✓ Output directories ready"

# Check for existing training
EXISTING_PID=$(pgrep -f "run_orpo_training.*curriculum" || echo "")
if [ -n "$EXISTING_PID" ]; then
    echo ""
    echo "⚠️  Warning: Training process already running (PID: $EXISTING_PID)"
    read -p "Kill existing process and start new training? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping existing training..."
        kill "$EXISTING_PID" || true
        sleep 2
    else
        echo "Aborted. Use ./scripts/monitor_curriculum_training.sh to monitor existing training."
        exit 0
    fi
fi

echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo ""
echo "  Config: configs/training_pipeline_curriculum.yaml"
echo "  Backend: MLX"
echo "  Dataset: Curriculum (10 topics)"
echo "  Output: ./output/curriculum/"
echo "  Checkpoints: ./checkpoints/curriculum/"
echo "  Log: ./logs/training_curriculum.log"
echo ""

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_curriculum_${TIMESTAMP}.log"

# Build command
TRAIN_CMD="python3 scripts/run_orpo_training.py --config configs/training_pipeline_curriculum.yaml --backend mlx"

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""

if [ "$1" == "--background" ]; then
    echo "🚀 Starting training in background..."
    nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &
    TRAINING_PID=$!
    echo "$TRAINING_PID" > logs/training_curriculum.pid
    
    echo ""
    echo "✅ Training started in background!"
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
    echo "Monitor Training"
    echo "=========================================="
    echo ""
    echo "  Monitor: ./scripts/monitor_curriculum_training.sh"
    echo "  Watch log: tail -f $LOG_FILE"
    echo "  Stop: pkill -f run_orpo_training"
    echo ""
else
    echo "🚀 Starting training (foreground mode)..."
    echo "   Press Ctrl+C to stop"
    echo ""
    echo "=========================================="
    echo ""
    
    # Run in foreground
    $TRAIN_CMD 2>&1 | tee "$LOG_FILE"
fi
