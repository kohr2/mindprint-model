#!/bin/bash
# Run evaluation on Mac Studio (run this script directly on Mac Studio)
# Usage: ./scripts/local_evaluate.sh [adapter_path]

set -e
set -u
set -o pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Default adapter path (will auto-discover if not provided)
ADAPTER_PATH="${1:-}"

echo "=========================================="
echo "Transcripts Training Evaluation"
echo "=========================================="
echo ""

# Pull latest code
echo "📥 Pulling latest code..."
git pull origin main || {
    echo "⚠️  Warning: git pull failed. Continuing with current code..."
}
echo ""

# Auto-discover adapter if not provided
if [ -z "$ADAPTER_PATH" ]; then
    echo "🔍 Auto-discovering adapter..."
    
    # Try to find the most recent adapter
    ADAPTER_PATH=$(python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path
from datetime import datetime

# Check checkpoint for adapter info
checkpoint_file = Path('checkpoints/latest.json')
if checkpoint_file.exists():
    try:
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        result = checkpoint.get('result', {})
        # Check if checkpoint has adapter path info
    except:
        pass

# Search for adapters in output directory
adapter_paths = []
output_dir = Path('output')
if output_dir.exists():
    for item in output_dir.rglob('*adapter*'):
        if item.is_dir() and (item / 'adapter_config.json').exists():
            adapter_paths.append(str(item))

# Sort by modification time, most recent first
if adapter_paths:
    adapter_paths.sort(key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0, reverse=True)
    print(adapter_paths[0])
PYTHON_SCRIPT
)
    
    if [ -z "$ADAPTER_PATH" ] || [ "$ADAPTER_PATH" = "" ]; then
        echo "❌ Error: No adapter found. Please specify adapter path:"
        echo "   ./scripts/local_evaluate.sh /path/to/adapter"
        echo ""
        echo "Or check output directory:"
        echo "   find output -name adapter_config.json"
        exit 1
    fi
    
    echo "✅ Found adapter: $ADAPTER_PATH"
    echo ""
fi

# Verify adapter exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ Error: Adapter directory not found: $ADAPTER_PATH"
    exit 1
fi

# Verify quiz data exists
QUIZ_DATA_DIR="./data/bob_loukas/transcripts"
if [ ! -d "$QUIZ_DATA_DIR" ]; then
    echo "❌ Error: Quiz data directory not found: $QUIZ_DATA_DIR"
    exit 1
fi

# Check if quiz files exist
if [ ! -f "$QUIZ_DATA_DIR/quiz_data.json" ]; then
    echo "⚠️  Warning: quiz_data.json not found or empty"
    echo "   Evaluation may be limited"
fi

echo "📊 Configuration:"
echo "   Base Model: Qwen/Qwen2.5-7B-Instruct"
echo "   Adapter: $ADAPTER_PATH"
echo "   Quiz Data: $QUIZ_DATA_DIR"
echo "   Output: ./eval_results"
echo ""

# Create output directory
mkdir -p eval_results

# Run evaluation
echo "🚀 Starting evaluation..."
echo ""

python3 scripts/run_evaluation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter "$ADAPTER_PATH" \
    --quiz-data "$QUIZ_DATA_DIR" \
    --approach dpo \
    --device mps \
    --trust-remote-code \
    --output ./eval_results

EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Evaluation Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: ./eval_results/"
    echo ""
    echo "View report:"
    echo "  cat eval_results/report.md"
    echo ""
    echo "Or check JSON:"
    echo "  cat eval_results/report.json | jq"
else
    echo ""
    echo "❌ Evaluation failed (exit code: $EVAL_EXIT)"
    exit $EVAL_EXIT
fi
