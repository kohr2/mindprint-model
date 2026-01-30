#!/bin/bash
# Quick deploy: SSH to Mac Studio and start training
# Usage: source .env.local && ./scripts/quick_deploy.sh
#
# This is a convenience script for when you don't want to SSH manually.
# For normal workflow, just SSH to Mac Studio and run scripts/local_train.sh directly.

set -e
set -u
set -o pipefail

# Load environment variables
if [ -f .env.local ]; then
    source .env.local
fi

MAC_STUDIO_HOST="${MAC_STUDIO_HOST:-100.87.103.70}"
MAC_STUDIO_USER="${MAC_STUDIO_USER:-memetica-studio}"
MAC_STUDIO_PASSWORD="${MAC_STUDIO_PASSWORD:-}"

REMOTE_HOST="${MAC_STUDIO_USER}@${MAC_STUDIO_HOST}"
REMOTE_DIR="${MAC_STUDIO_BASE_DIR:-~/mindprint-model}"

echo "=========================================="
echo "Quick Deploy to Mac Studio"
echo "=========================================="
echo "Remote: ${REMOTE_HOST}"
echo "Dir: ${REMOTE_DIR}"
echo ""
echo "Note: This script will SSH to Mac Studio and run local_train.sh"
echo "For normal workflow, SSH directly and run scripts locally."
echo "=========================================="
echo ""

# Determine SSH command
if [ -n "$MAC_STUDIO_PASSWORD" ]; then
    SSH_CMD="sshpass -p '$MAC_STUDIO_PASSWORD' ssh -o StrictHostKeyChecking=no"
    echo "Using password authentication"
else
    SSH_CMD="ssh -o StrictHostKeyChecking=no"
    echo "Using SSH keys for authentication"
fi

# Check SSH connection
echo "Checking SSH connection..."
if ! eval "$SSH_CMD" "${REMOTE_HOST}" "echo 'Connected'" >/dev/null 2>&1; then
    echo "❌ Error: Cannot connect to ${REMOTE_HOST}"
    echo ""
    echo "Please check:"
    echo "  1. Host: ${MAC_STUDIO_HOST}"
    echo "  2. User: ${MAC_STUDIO_USER}"
    echo "  3. SSH access configured"
    exit 1
fi
echo "✅ SSH connection successful"
echo ""

# Run local_train.sh on Mac Studio
echo "🚀 Starting training on Mac Studio..."
eval "$SSH_CMD" "${REMOTE_HOST}" "cd ${REMOTE_DIR} && bash scripts/local_train.sh"

echo ""
echo "=========================================="
echo "✅ Deployment complete!"
echo "=========================================="
echo ""
echo "To monitor training, SSH to Mac Studio and run:"
echo "  ssh ${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  ./scripts/local_monitor.sh"
echo ""
