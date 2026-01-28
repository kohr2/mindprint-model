#!/bin/bash
# Monitor training progress and collect metrics
# Usage: ./monitor_training.sh [log-file]

set -e

LOG_FILE="${1:-logs/training_*.log}"

echo "=== Training Monitor ==="
echo "Log file: $LOG_FILE"
echo ""

# Find latest log file if pattern used
if [[ "$LOG_FILE" == *"*"* ]]; then
    LATEST_LOG=$(ls -t $LOG_FILE 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "ERROR: No log files found matching: $LOG_FILE"
        exit 1
    fi
    LOG_FILE="$LATEST_LOG"
    echo "Using latest log: $LOG_FILE"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    exit 1
fi

# Check if training is running
if pgrep -f "run_dpo_training.py" > /dev/null; then
    echo "✓ Training process is running"
    PID=$(pgrep -f "run_dpo_training.py" | head -1)
    echo "  PID: $PID"
else
    echo "⚠ Training process not found (may have completed)"
fi

echo ""
echo "=== Recent Log Output (last 50 lines) ==="
tail -50 "$LOG_FILE"

echo ""
echo "=== Training Metrics ==="

# Extract key metrics from log
echo ""
echo "Voice Scores:"
grep -E "voice.*=|voice_score" "$LOG_FILE" | tail -10 || echo "  No voice scores found yet"

echo ""
echo "Accuracy Scores:"
grep -E "accuracy.*=|accuracy_score" "$LOG_FILE" | tail -10 || echo "  No accuracy scores found yet"

echo ""
echo "Training Progress:"
grep -E "Topic|topic|Unit|unit|Chapter|chapter" "$LOG_FILE" | tail -10 || echo "  No progress markers found"

echo ""
echo "Errors/Warnings:"
grep -iE "error|warning|exception|failed" "$LOG_FILE" | tail -10 || echo "  No errors/warnings found"

echo ""
echo "=== Memory Usage ==="
if command -v vm_stat &> /dev/null; then
    vm_stat | head -10
else
    echo "  Memory stats not available on this system"
fi

echo ""
echo "=== System Load ==="
uptime

echo ""
echo "=== To follow logs in real-time: ==="
echo "  tail -f $LOG_FILE"
