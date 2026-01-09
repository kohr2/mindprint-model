#!/bin/bash
# Test single topic on Mac Studio to validate adapter stacking fix
#
# Usage: ./scripts/test_single_topic.sh

set -e

echo "=== Testing Adapter Stacking Fix on Single Topic ==="
echo ""

# Pull latest changes
echo "Step 1: Pulling latest changes from dpo branch..."
git pull origin dpo
echo "✓ Changes pulled"
echo ""

# Check we're on the right branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "dpo" ]; then
    echo "❌ ERROR: Not on dpo branch (currently on $BRANCH)"
    echo "Run: git checkout dpo"
    exit 1
fi
echo "✓ On dpo branch"
echo ""

# Show the commit
echo "Latest commit:"
git log -1 --oneline
echo ""

# Clean previous test outputs
echo "Step 2: Cleaning previous test outputs..."
rm -rf output_test checkpoint_test
mkdir -p logs
echo "✓ Cleaned"
echo ""

# Run single topic test
echo "Step 3: Running training on single topic (unit-01/chapter-01)..."
echo "This will take ~3-5 minutes..."
echo ""

python scripts/run_dpo_training.py \
  --config configs/training_pipeline.yaml \
  --data-dir data/bob_loukas/unit-01/chapter-01 \
  --output-dir ./output_test \
  --checkpoint-dir ./checkpoint_test

EXIT_CODE=$?

echo ""
echo "=== Training Complete ==="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training succeeded!"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
fi

# Show latest log
LATEST_LOG=$(ls -t logs/dpo_training_*.log | head -1)
echo ""
echo "=== Latest Log: $LATEST_LOG ==="
echo ""

# Check for key success indicators
echo "Checking for adapter unloading..."
if grep -q "Unloading SFT adapter before DPO training" "$LATEST_LOG"; then
    echo "✅ Found: 'Unloading SFT adapter before DPO training'"
else
    echo "❌ Missing: 'Unloading SFT adapter before DPO training'"
fi

if grep -q "SFT adapter merged and unloaded successfully" "$LATEST_LOG"; then
    echo "✅ Found: 'SFT adapter merged and unloaded successfully'"
else
    echo "❌ Missing: 'SFT adapter merged and unloaded successfully'"
fi

echo ""
echo "Checking for adapter stacking warnings..."
if grep -q "trying to modify a model with PEFT for a second time" "$LATEST_LOG"; then
    echo "❌ Found PEFT stacking warning (BAD!)"
else
    echo "✅ No PEFT stacking warnings (GOOD!)"
fi

echo ""
echo "Checking post-DPO evaluation scores..."
# Get the last DPO evaluation result
LAST_EVAL=$(grep -A 2 "Re-evaluate after DPO" "$LATEST_LOG" | tail -3)
echo "$LAST_EVAL"

if echo "$LAST_EVAL" | grep -q "accuracy=0.0"; then
    echo "❌ Accuracy dropped to 0 after DPO (BAD!)"
elif echo "$LAST_EVAL" | grep -q "voice=0.0"; then
    echo "❌ Voice score is 0 after DPO (BAD!)"
else
    echo "✅ Non-zero scores after DPO (GOOD!)"
fi

echo ""
echo "=== Full Log ==="
echo "View full log with: tail -100 $LATEST_LOG"
echo "Or monitor live: tail -f $LATEST_LOG"
echo ""

exit $EXIT_CODE
