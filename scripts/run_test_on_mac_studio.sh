#!/bin/bash
# Run MLX diagnostic test on Mac Studio via SSH
# Usage: ./run_test_on_mac_studio.sh [mac-studio-host] [user]
#
# Environment variables:
#   MAC_STUDIO_PASSWORD - SSH password (if not using SSH keys)
#   MAC_STUDIO_HOST - Mac Studio hostname/IP (default: mac-studio.local)
#   MAC_STUDIO_USER - Mac Studio username (default: benoit)

set -e

MAC_STUDIO_HOST="${1:-${MAC_STUDIO_HOST:-memeticas-mac-studio}}"
MAC_STUDIO_USER="${2:-${MAC_STUDIO_USER:-memetica-studio}}"
MAC_STUDIO_PASSWORD="${MAC_STUDIO_PASSWORD:-}"
REMOTE_DIR="~/mindprint-model"
TEST_FILE="tests/debug/test_mlx_training_state.py"

echo "=== Running MLX Diagnostic Test on Mac Studio ==="
echo "Host: $MAC_STUDIO_HOST"
echo "User: $MAC_STUDIO_USER"
echo "Remote dir: $REMOTE_DIR"
echo "Test: $TEST_FILE"
echo ""

# Determine SSH command (use sshpass if password provided, otherwise use SSH keys)
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    SSH_CMD="sshpass -p '$MAC_STUDIO_PASSWORD' ssh"
    RSYNC_CMD="sshpass -p '$MAC_STUDIO_PASSWORD' rsync"
    echo "Using password authentication"
else
    SSH_CMD="ssh"
    RSYNC_CMD="rsync"
    echo "Using SSH keys for authentication"
fi

# Create remote directory
echo "Creating remote directory..."
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    if ! sshpass -p "$MAC_STUDIO_PASSWORD" ssh -o StrictHostKeyChecking=no "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" \
        "mkdir -p $REMOTE_DIR" 2>&1; then
        echo "ERROR: Cannot connect to Mac Studio"
        echo "Please check:"
        echo "  1. Hostname/IP: $MAC_STUDIO_HOST"
        echo "  2. Username: $MAC_STUDIO_USER"
        echo "  3. Password correct"
        exit 1
    fi
else
    if ! ssh -o StrictHostKeyChecking=no "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" \
        "mkdir -p $REMOTE_DIR" 2>&1; then
        echo "ERROR: Cannot connect to Mac Studio"
        echo "Please check:"
        echo "  1. Hostname/IP: $MAC_STUDIO_HOST"
        echo "  2. Username: $MAC_STUDIO_USER"
        echo "  3. SSH keys configured"
        exit 1
    fi
fi

# Sync code to Mac Studio (exclude large files)
echo "Syncing code to Mac Studio..."
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    sshpass -p "$MAC_STUDIO_PASSWORD" rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
        --exclude='.git' \
        --exclude='output/' \
        --exclude='checkpoints/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache/' \
        --exclude='training_log.txt' \
        --exclude='logs/' \
        ./ "$MAC_STUDIO_USER@$MAC_STUDIO_HOST:$REMOTE_DIR/"
else
    rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
        --exclude='.git' \
        --exclude='output/' \
        --exclude='checkpoints/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache/' \
        --exclude='training_log.txt' \
        --exclude='logs/' \
        ./ "$MAC_STUDIO_USER@$MAC_STUDIO_HOST:$REMOTE_DIR/"
fi

echo ""
echo "=== Setting up environment on Mac Studio ==="

# Setup environment and run test
# Pass TEST_FILE as environment variable
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    sshpass -p "$MAC_STUDIO_PASSWORD" ssh -o StrictHostKeyChecking=no "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" "TEST_FILE='$TEST_FILE' bash -s" << 'ENDSSH'
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

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    exit 1
fi

# Check available memory
echo "Checking system memory..."
sysctl hw.memsize | awk '{printf "Total memory: %.2f GB\n", $2/1024/1024/1024}'

echo ""
echo "=== Running MLX Diagnostic Test ==="
echo "This test verifies LoRA adapter implementation..."
echo ""

# Run the diagnostic test
python3 "$TEST_FILE" 2>&1

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "=== Test completed successfully ==="
else
    echo "=== Test failed with exit code: $TEST_EXIT_CODE ==="
fi

exit $TEST_EXIT_CODE
ENDSSH
else
    ssh -o StrictHostKeyChecking=no "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" "TEST_FILE='$TEST_FILE' bash -s" << 'ENDSSH'
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

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    exit 1
fi

# Check available memory
echo "Checking system memory..."
sysctl hw.memsize | awk '{printf "Total memory: %.2f GB\n", $2/1024/1024/1024}'

echo ""
echo "=== Running MLX Diagnostic Test ==="
echo "This test verifies LoRA adapter implementation..."
echo ""

# Run the diagnostic test
python3 "$TEST_FILE" 2>&1

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "=== Test completed successfully ==="
else
    echo "=== Test failed with exit code: $TEST_EXIT_CODE ==="
fi

exit $TEST_EXIT_CODE
ENDSSH
fi

echo ""
echo "=== Test execution complete ==="
