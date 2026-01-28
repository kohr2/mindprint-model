#!/bin/bash
# Monitor training progress on Mac Studio

MAC_STUDIO_HOST="100.87.103.70"
MAC_STUDIO_USER="memetica-studio"
MAC_STUDIO_PASSWORD="memetica"

echo "🖥️  Monitoring Qwen training on Mac Studio M3 Ultra"
echo "=================================================="
echo

# Check if training is running
echo "📡 Checking training status..."
sshpass -p "$MAC_STUDIO_PASSWORD" ssh -o StrictHostKeyChecking=no "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" "ps aux | grep run_dpo_training | grep -v grep || echo 'No training process found'"

echo
echo "📊 Latest training log (updating every 10 seconds, Ctrl+C to exit):"
echo "=================================================="
echo

# Tail the latest log file
sshpass -p "$MAC_STUDIO_PASSWORD" ssh -o StrictHostKeyChecking=no "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" "tail -f \$(ls -t ~/mindprint-model/logs/training_*.log | head -1)"
