#!/bin/bash
# Monitor training progress on Mac Studio (run this script directly on Mac Studio)
# Usage: ./scripts/local_monitor.sh [--follow]

set -e
set -u

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

FOLLOW="${1:-}"

echo "=========================================="
echo "Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
echo "📊 Process Status:"
if ps aux | grep -E '[r]un_dpo_training' > /dev/null; then
    ps aux | grep -E '[r]un_dpo_training' | grep -v grep
    echo ""
    
    # Show PID if available
    if [ -f logs/training.pid ]; then
        PID=$(cat logs/training.pid)
        echo "   PID from file: $PID"
        echo ""
    fi
else
    echo "   ⚠️  No training process found"
    echo ""
fi

# Show recent log files
echo "📁 Recent Log Files:"
if [ -d logs ]; then
    ls -lht logs/training_*.log 2>/dev/null | head -5 || echo "   No log files found"
else
    echo "   Logs directory not found"
fi
echo ""

# Show training progress
echo "📈 Training Progress (last 50 lines):"
LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    tail -50 "$LATEST_LOG"
else
    echo "   No log file found"
fi
echo ""

# Show training statistics
echo "📊 Training Statistics:"
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    grep -E 'SFT complete|DPO complete|Topic.*complete|epoch|step' "$LATEST_LOG" 2>/dev/null | tail -10 || echo "   No statistics found"
else
    echo "   No log file available"
fi
echo ""

# Show system resources
echo "💾 System Resources:"
if command -v top > /dev/null; then
    top -l 1 | grep -E 'CPU usage|PhysMem' || true
elif command -v vm_stat > /dev/null; then
    vm_stat | head -5
fi
echo ""

# If --follow flag, tail the log
if [ "$FOLLOW" = "--follow" ] || [ "$FOLLOW" = "-f" ]; then
    if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
        echo "=========================================="
        echo "Following log (Ctrl+C to exit):"
        echo "=========================================="
        tail -f "$LATEST_LOG"
    else
        echo "❌ No log file to follow"
        exit 1
    fi
else
    echo "=========================================="
    echo "To follow logs live, run:"
    echo "  ./scripts/local_monitor.sh --follow"
    echo ""
    echo "Or directly:"
    if [ -n "$LATEST_LOG" ]; then
        echo "  tail -f $LATEST_LOG"
    else
        echo "  tail -f logs/training_*.log"
    fi
    echo "=========================================="
fi
