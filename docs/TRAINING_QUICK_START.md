# Curriculum Training - Quick Start Guide

## 🚀 Start Training

**Option 1: Quick Start Script (Recommended)**
```bash
cd ~/mindprint-model
./scripts/start_curriculum_training.sh --background
```

**Option 2: Direct Command**
```bash
cd ~/mindprint-model
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline_curriculum.yaml \
    --backend mlx
```

## 📊 Monitor Training

**Quick Status Check:**
```bash
./scripts/monitor_curriculum_training.sh
```

**Watch Log in Real-Time:**
```bash
tail -f logs/training_curriculum.log | grep -E "(Topic|Loss|ETA|CV)"
```

**Check Process:**
```bash
ps aux | grep run_orpo_training
```

## 🛑 Stop Training

```bash
pkill -f run_orpo_training
```

## 📁 Key Locations

- **Config**: `configs/training_pipeline_curriculum.yaml`
- **Output**: `output/curriculum/`
- **Checkpoints**: `checkpoints/curriculum/`
- **Logs**: `logs/training_curriculum.log`

## ⏱️ Expected Duration

- **Per Topic**: ~130 minutes
- **Total (10 topics)**: ~21.7 hours
- **Early Stop (if triggered)**: ~10-15 hours

## ✅ Pre-Flight Checklist

- [ ] Code pulled: `git pull origin main`
- [ ] Data exists: `ls data/bob_loukas/textbook/*.jsonl`
- [ ] No conflicts: `ps aux | grep run_orpo_training`
- [ ] Disk space: `df -h` (need ~50GB free)

## 📈 Key Metrics

**Watch for:**
- Topics completed (1-10)
- Loss values (should decrease)
- Early stopping triggers (CV < 15%)
- Errors/warnings

**Success Indicators:**
- Loss decreases from ~2500 → ~200
- All 10 topics complete
- Output directory grows
- No critical errors

## 🔍 Troubleshooting

**Training won't start:**
- Check MLX backend: `python3 -c "import mlx.core as mx; print(mx.metal.is_available())"`
- Verify config: `python3 -c "from scripts.run_orpo_training import load_config; load_config('configs/training_pipeline_curriculum.yaml')"`

**Out of memory:**
- Close other applications
- Reduce batch size in config

**Early stopping too early:**
- Increase `early_stopping_min_topics` in config
- Increase `early_stopping_cv_threshold`

## 📚 Full Documentation

See `docs/TRAINING_PLAN.md` for complete details.

---

**Quick Commands Reference:**

```bash
# Start
./scripts/start_curriculum_training.sh --background

# Monitor
./scripts/monitor_curriculum_training.sh

# Watch log
tail -f logs/training_curriculum.log

# Stop
pkill -f run_orpo_training
```
