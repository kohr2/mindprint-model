#!/bin/bash
# Start PPO training on Mac Studio
# Run this script on the Mac Studio machine

cd /Users/benoit/Documents/Memetica/Code/mindprint-model

echo "Pulling latest code from git..."
git pull origin ppo

echo "Starting PPO training with Qwen2.5-7B..."
nohup python3 scripts/run_ppo_training.py \
  --config configs/training_pipeline.yaml \
  --model Qwen/Qwen2.5-7B-Instruct \
  > logs/ppo_training_mac_studio.log 2>&1 &

TRAINING_PID=$!
echo "Training started with PID: $TRAINING_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/ppo_training_mac_studio.log"
echo ""
echo "Check process status:"
echo "  ps -p $TRAINING_PID"
