#!/bin/bash
# Stop running training process
# Usage: ./scripts/stop_training.sh

set -e
set -u

echo "🛑 Stopping training processes..."

# Find and kill training processes
if pkill -f run_orpo_training; then
    echo "✅ Stopped training process(es)"
    sleep 2
    
    # Verify they're stopped
    if ps aux | grep -E '[r]un_dpo_training' > /dev/null; then
        echo "⚠️  Warning: Some training processes may still be running"
        ps aux | grep -E '[r]un_dpo_training' | grep -v grep
    else
        echo "✅ All training processes stopped"
    fi
else
    echo "ℹ️  No training processes found to stop"
fi

# Remove PID file if it exists
if [ -f logs/training.pid ]; then
    rm logs/training.pid
    echo "✅ Removed PID file"
fi

echo ""
echo "Done!"
