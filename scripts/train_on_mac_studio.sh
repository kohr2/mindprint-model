#!/bin/bash
# Deploy and run training on Mac Studio via SSH
# Usage: ./train_on_mac_studio.sh [mac-studio-host] [user]
#
# Environment variables:
#   MAC_STUDIO_PASSWORD - SSH password (if not using SSH keys)
#   MAC_STUDIO_MODEL - Model to use (default: Qwen/Qwen2.5-7B-Instruct)

set -e

MAC_STUDIO_HOST="${1:-mac-studio.local}"
MAC_STUDIO_USER="${2:-benoit}"
MAC_STUDIO_PASSWORD="${MAC_STUDIO_PASSWORD:-}"
REMOTE_DIR="~/mindprint-model"
MODEL="${MAC_STUDIO_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "=== Deploying to Mac Studio ==="
echo "Host: $MAC_STUDIO_HOST"
echo "User: $MAC_STUDIO_USER"
echo "Remote dir: $REMOTE_DIR"
echo "Model: $MODEL"
echo ""

# Determine SSH command (use sshpass if password provided, otherwise use SSH keys)
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    SSH_CMD="sshpass -p \"$MAC_STUDIO_PASSWORD\" ssh"
    RSYNC_CMD="sshpass -p \"$MAC_STUDIO_PASSWORD\" rsync"
else
    SSH_CMD="ssh"
    RSYNC_CMD="rsync"
    echo "Using SSH keys for authentication"
fi

# Create remote directory
echo "Creating remote directory..."
$SSH_CMD "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" \
    "mkdir -p $REMOTE_DIR"

# Sync code to Mac Studio (exclude large files)
echo "Syncing code to Mac Studio..."
$RSYNC_CMD -avz \
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

# Install dependencies and validate environment
$SSH_CMD "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" << ENDSSH
cd ~/mindprint-model

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "ERROR: Python3 not found"; exit 1; }

# Check if MLX is installed
echo "Checking MLX installation..."
if ! python3 -c "import mlx" 2>/dev/null; then
    echo "Installing MLX dependencies..."
    pip3 install --break-system-packages mlx mlx-lm || pip3 install --user mlx mlx-lm
else
    echo "MLX already installed"
fi

# Verify MLX installation
if ! python3 -c "import mlx; import mlx_lm" 2>/dev/null; then
    echo "ERROR: MLX installation failed"
    exit 1
fi

# Check if other dependencies are installed
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "Installing other dependencies..."
    pip3 install --break-system-packages -r requirements.txt || pip3 install --user -r requirements.txt
fi

# Check available memory (Mac Studio should have 64GB+)
echo "Checking system memory..."
sysctl hw.memsize | awk '{printf "Total memory: %.2f GB\n", \$2/1024/1024/1024}'

echo "Environment ready!"
ENDSSH

echo ""
echo "=== Starting training on Mac Studio ==="

# Start training in background on Mac Studio
$SSH_CMD "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" << ENDSSH
cd ~/mindprint-model

# Create logs directory
mkdir -p logs

# Verify config file exists
if [ ! -f "configs/training_pipeline.yaml" ]; then
    echo "ERROR: Config file not found: configs/training_pipeline.yaml"
    exit 1
fi

# Start training with MLX backend
echo "Starting training with MLX backend..."
nohup python3 scripts/run_dpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --model "$MODEL" \
    > logs/training_\$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAINING_PID=\$!
echo "Training started with PID: \$TRAINING_PID"
echo \$TRAINING_PID > logs/training.pid

# Show initial output
sleep 5
if [ -f logs/training_*.log ]; then
    tail -50 logs/training_*.log | tail -50
else
    echo "Log file not yet created, check again in a moment"
fi
ENDSSH

echo ""
echo "=== Training started on Mac Studio ==="
echo ""
echo "To monitor progress:"
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    echo "  sshpass -p \"\$MAC_STUDIO_PASSWORD\" ssh $MAC_STUDIO_USER@$MAC_STUDIO_HOST 'tail -f ~/mindprint-model/logs/training_*.log'"
else
    echo "  ssh $MAC_STUDIO_USER@$MAC_STUDIO_HOST 'tail -f ~/mindprint-model/logs/training_*.log'"
fi
echo ""
echo "To check status:"
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    echo "  sshpass -p \"\$MAC_STUDIO_PASSWORD\" ssh $MAC_STUDIO_USER@$MAC_STUDIO_HOST 'ps aux | grep run_dpo_training'"
else
    echo "  ssh $MAC_STUDIO_USER@$MAC_STUDIO_HOST 'ps aux | grep run_dpo_training'"
fi
echo ""
echo "To check training PID:"
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    echo "  sshpass -p \"\$MAC_STUDIO_PASSWORD\" ssh $MAC_STUDIO_USER@$MAC_STUDIO_HOST 'cat ~/mindprint-model/logs/training.pid'"
else
    echo "  ssh $MAC_STUDIO_USER@$MAC_STUDIO_HOST 'cat ~/mindprint-model/logs/training.pid'"
fi
