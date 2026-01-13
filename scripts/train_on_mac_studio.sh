#!/bin/bash
# Deploy and run training on Mac Studio via SSH
# Usage: ./train_on_mac_studio.sh <mac-studio-host>

set -e

MAC_STUDIO_HOST="${1:-mac-studio.local}"
MAC_STUDIO_USER="${2:-benoit}"
MAC_STUDIO_PASSWORD="memetica"
REMOTE_DIR="~/mindprint-model"

echo "=== Deploying to Mac Studio ==="
echo "Host: $MAC_STUDIO_HOST"
echo "User: $MAC_STUDIO_USER"
echo "Remote dir: $REMOTE_DIR"
echo ""

# Create remote directory
echo "Creating remote directory..."
sshpass -p "$MAC_STUDIO_PASSWORD" ssh "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" \
    "mkdir -p $REMOTE_DIR"

# Sync code to Mac Studio (exclude large files)
echo "Syncing code to Mac Studio..."
sshpass -p "$MAC_STUDIO_PASSWORD" rsync -avz \
    --exclude='.git' \
    --exclude='output/' \
    --exclude='checkpoints/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='training_log.txt' \
    ./ "$MAC_STUDIO_USER@$MAC_STUDIO_HOST:$REMOTE_DIR/"

echo ""
echo "=== Setting up environment on Mac Studio ==="

# Install dependencies if needed
sshpass -p "$MAC_STUDIO_PASSWORD" ssh "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" << 'ENDSSH'
cd ~/mindprint-model

# Check if MLX is installed
if ! python3 -c "import mlx" 2>/dev/null; then
    echo "Installing MLX dependencies..."
    pip3 install --break-system-packages mlx mlx-lm || pip3 install --user mlx mlx-lm
fi

# Check if other dependencies are installed
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "Installing other dependencies..."
    pip3 install --break-system-packages -r requirements.txt || pip3 install --user -r requirements.txt
fi

echo "Environment ready!"
ENDSSH

echo ""
echo "=== Starting training on Mac Studio ==="

# Start training in background on Mac Studio
sshpass -p "$MAC_STUDIO_PASSWORD" ssh "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" << 'ENDSSH'
cd ~/mindprint-model

# Create logs directory
mkdir -p logs

# Start training
nohup python3 scripts/run_dpo_training.py \
    --config configs/training_pipeline.yaml \
    > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAINING_PID=$!
echo "Training started with PID: $TRAINING_PID"
echo $TRAINING_PID > logs/training.pid

# Show initial output
sleep 5
tail -50 logs/training_*.log | tail -50
ENDSSH

echo ""
echo "=== Training started on Mac Studio ==="
echo "To monitor progress:"
echo "  sshpass -p memetica ssh $MAC_STUDIO_USER@$MAC_STUDIO_HOST 'tail -f ~/mindprint-model/logs/training_*.log'"
echo ""
echo "To check status:"
echo "  sshpass -p memetica ssh $MAC_STUDIO_USER@$MAC_STUDIO_HOST 'ps aux | grep run_dpo_training'"
