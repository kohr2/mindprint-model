#!/bin/bash
# Monitor curriculum training progress
# Usage: ./scripts/monitor_curriculum_training.sh [--watch]

set -e

LOG_FILE="./logs/training_curriculum.log"
PID_FILE="./logs/training_curriculum.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Curriculum Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Training is running (PID: $PID)${NC}"
    else
        echo -e "${RED}✗ Training process not found (PID: $PID)${NC}"
    fi
else
    # Try to find process
    PID=$(pgrep -f "run_orpo_training.*curriculum" || echo "")
    if [ -n "$PID" ]; then
        echo -e "${GREEN}✓ Training is running (PID: $PID)${NC}"
    else
        echo -e "${YELLOW}⚠ No training process found${NC}"
    fi
fi

echo ""

# Check log file
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}⚠ Log file not found: $LOG_FILE${NC}"
    echo "Looking for recent log files..."
    ls -lt logs/training_curriculum*.log 2>/dev/null | head -1 || echo "No log files found"
    echo ""
    exit 1
fi

echo "=========================================="
echo "Training Progress Summary"
echo "=========================================="
echo ""

# Extract key metrics
echo -e "${BLUE}Topics Completed:${NC}"
TOPICS_COMPLETED=$(grep -c "Topic completed\|Training topic:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "  $TOPICS_COMPLETED topics processed"

echo ""
echo -e "${BLUE}Current Status:${NC}"
LAST_TOPIC=$(grep "Training topic:" "$LOG_FILE" | tail -1 | sed 's/.*Training topic: //' || echo "Unknown")
echo "  Last topic: $LAST_TOPIC"

echo ""
echo -e "${BLUE}Loss Values (Last 5):${NC}"
grep -E "orpo_loss|Loss:" "$LOG_FILE" | tail -5 | sed 's/^/  /' || echo "  No loss data found"

echo ""
echo -e "${BLUE}Early Stopping Status:${NC}"
EARLY_STOP=$(grep -c "Early stopping triggered" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$EARLY_STOP" -gt 0 ]; then
    echo -e "  ${YELLOW}⚠ Early stopping was triggered${NC}"
    grep "Early stopping triggered" "$LOG_FILE" | tail -1 | sed 's/^/  /'
else
    echo "  Training continuing normally"
fi

echo ""
echo -e "${BLUE}Errors/Warnings:${NC}"
ERROR_COUNT=$(grep -ic "error\|exception\|failed" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo -e "  ${RED}✗ $ERROR_COUNT errors/warnings found${NC}"
    grep -i "error\|exception\|failed" "$LOG_FILE" | tail -3 | sed 's/^/  /'
else
    echo -e "  ${GREEN}✓ No errors found${NC}"
fi

echo ""
echo "=========================================="
echo "Output Directories"
echo "=========================================="
echo ""

if [ -d "output/curriculum" ]; then
    echo -e "${GREEN}✓ Curriculum output directory exists${NC}"
    OUTPUT_SIZE=$(du -sh output/curriculum/ 2>/dev/null | cut -f1 || echo "0")
    echo "  Size: $OUTPUT_SIZE"
    echo "  Contents:"
    ls -lh output/curriculum/ 2>/dev/null | tail -5 | sed 's/^/    /' || echo "    (empty)"
else
    echo -e "${YELLOW}⚠ Curriculum output directory not found${NC}"
fi

echo ""
if [ -d "checkpoints/curriculum" ]; then
    echo -e "${GREEN}✓ Checkpoint directory exists${NC}"
    CHECKPOINT_COUNT=$(ls -1 checkpoints/curriculum/*.json 2>/dev/null | wc -l || echo "0")
    echo "  Checkpoints: $CHECKPOINT_COUNT"
else
    echo -e "${YELLOW}⚠ Checkpoint directory not found${NC}"
fi

echo ""
echo "=========================================="
echo "System Resources"
echo "=========================================="
echo ""

if [ -n "$PID" ]; then
    MEM_USAGE=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}' || echo "N/A")
    CPU_USAGE=$(ps -o %cpu= -p "$PID" 2>/dev/null | awk '{printf "%.1f%%", $1}' || echo "N/A")
    echo "  Memory: $MEM_USAGE"
    echo "  CPU: $CPU_USAGE"
fi

echo ""
echo "=========================================="
echo "Recent Log Activity"
echo "=========================================="
echo ""
tail -20 "$LOG_FILE" | sed 's/^/  /'

echo ""
echo "=========================================="
echo "Monitoring Commands"
echo "=========================================="
echo ""
echo "  Watch log:     tail -f $LOG_FILE"
echo "  Filter metrics: tail -f $LOG_FILE | grep -E '(Topic|Loss|ETA|CV)'"
echo "  Check process: ps aux | grep run_orpo_training"
echo "  View outputs:  ls -lh output/curriculum/"
echo ""

# If --watch flag, continuously monitor
if [ "$1" == "--watch" ]; then
    echo "Starting watch mode (Ctrl+C to exit)..."
    echo ""
    watch -n 30 "$0"
fi
