#!/bin/bash
# Restart training on Mac Studio

MAC_STUDIO_HOST="100.87.103.70"
MAC_STUDIO_USER="memetica-studio"
MAC_STUDIO_PASSWORD="memetica"

echo "🔄 Restarting training on Mac Studio..."

sshpass -p "$MAC_STUDIO_PASSWORD" ssh -o StrictHostKeyChecking=no "$MAC_STUDIO_USER@$MAC_STUDIO_HOST" << 'ENDSSH'
cd ~/mindprint-model

# Kill existing training processes
pkill -f run_dpo_training || true

# Wait a moment
sleep 2

# Create logs directory
mkdir -p logs

# Start training
nohup /opt/homebrew/bin/python3.13 scripts/run_dpo_training.py \
    --config configs/training_pipeline.yaml \
    > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAINING_PID=$!
echo "Training restarted with PID: $TRAINING_PID"
echo $TRAINING_PID > logs/training.pid

# Show initial output
sleep 5
tail -30 $(ls -t logs/training_*.log | head -1)
ENDSSH

echo ""
echo "✅ Training restarted on Mac Studio"
