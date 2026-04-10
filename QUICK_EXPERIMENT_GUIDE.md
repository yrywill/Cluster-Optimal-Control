# Quick Experiment Guide

## 🚀 Getting Started (30 seconds)

```bash
# Start default training run on 8 GPUs
deepspeed --num_gpus=8 train.py --config configs/default.yaml

# Check it's running
tail -f outputs/run/train.log
```

---

## 📋 Pre-configured Experiments

### 1️⃣ Baseline: Default Configuration
```bash
deepspeed --num_gpus=8 train.py --config configs/default.yaml
```
**What it does:** Standard minibatch KMeans clustering with intermediate layer features, PMP weight updates every 100 steps.

### 2️⃣ Fastest: Early-Exit Clustering  
```bash
deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    clustering.kmeans.feature=intermediate \
    clustering.kmeans.embed_layer=16
```
**Speed:** ~50% faster feature extraction
**Best for:** Large datasets, memory-constrained systems

### 3️⃣ Most Accurate: Full KMeans
```bash
deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    clustering.method=kmeans \
    clustering.kmeans.n_init=10 \
    clustering.kmeans.max_iter=500
```
**Quality:** Best clustering quality
**Trade-off:** Slower initialization (~30 min vs 5 min)

### 4️⃣ Massive Scale: FAISS Clustering
```bash
deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    clustering.method=faiss
```
**Speed:** 2x faster than KMeans
**For:** 1M+ samples

### 5️⃣ Memory-Limited: CPU Offload
```bash
deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    deepspeed.config_file=configs/ds_zero3_offload.json \
    training.batch_size=4
```
**Memory:** Save 30-50% GPU memory
**Trade-off:** Slightly slower (10-15%)

### 6️⃣ Baseline Comparison: Random Clustering
```bash
deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    clustering.method=random
```
**Purpose:** Compare against random data selection

---

## 🔬 Hyperparameter Sweeps

### Temperature Sensitivity
```bash
for temp in 0.5 1.0 2.0 4.0 8.0; do
  echo "Running with temperature=$temp"
  deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    pmp.temperature=$temp \
    training.save_dir="outputs/run_temp_${temp}"
done
```

### Learning Rate Ablation
```bash
for lr in 1e-5 3e-5 1e-4 3e-4; do
  echo "Running with lr=$lr"
  deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    training.lr=$lr \
    training.save_dir="outputs/run_lr_${lr}"
done
```

### Cluster Size Sensitivity
```bash
for size in 50 100 200 500 1000; do
  echo "Running with cluster_size=$size"
  deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    clustering.cluster_size=$size \
    training.save_dir="outputs/run_clusters_${size}"
done
```

### Feature Extraction Methods
```bash
for feature in intermediate projection embedding ghost; do
  echo "Running with feature=$feature"
  deepspeed --num_gpus=8 train.py --config configs/default.yaml \
    clustering.kmeans.feature=$feature \
    training.save_dir="outputs/run_feature_${feature}"
done
```

---

## 🧪 Testing

### Run Full Test Suite
```bash
pytest tests/test_early_exit.py -v          # Early-exit tests (28+)
pytest tests/test_ghost.py -v               # Ghost projection tests
python tests/run_manual_tests.py            # Manual tests
```

### Quick Smoke Test
```bash
python -c "
from clustering import KMeansClusterer
from utils.config import load_config
cfg = load_config('configs/default.yaml')
print('✅ Config loads OK')
print('✅ All imports work')
"
```

---

## 📊 Monitoring

### Real-time Log Monitoring
```bash
# Watch training in real-time
watch -n 5 'tail -20 outputs/run/train.log'

# Count training steps completed
grep "step=" outputs/run/train.log | tail -5

# Get final evaluation metrics
grep "\[eval\]" outputs/run/train.log | tail -1
```

### Parse Results
```bash
# Extract all loss values
grep "step=" outputs/run/train.log | awk '{print $NF}' | grep loss

# Find best validation accuracy
grep "\[eval\]" outputs/run/train.log | awk '{print $NF}' | sort -rn | head -1

# Get checkpoint locations
ls -lh outputs/run/checkpoint-*/
```

---

## 🎯 Common Configurations

| Use Case | Command |
|----------|---------|
| **Quick test** | `python train.py --config configs/default.yaml training.total_iters=100` |
| **Single GPU** | `python train.py --config configs/default.yaml` |
| **Fast iteration** | Add `clustering.kmeans.feature=intermediate` |
| **Reproducible** | Add `training.seed=42` (set in config) |
| **Resume training** | Add `--resume_from=outputs/run/checkpoint-5000` |
| **Different dataset** | Add `data.train_dir=<your_path>` |
| **Skip eval at start** | Add `training.no_eval_at_start=true` |

---

## 🐛 Debugging

### Check Model Loads
```bash
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('/apdcephfs_jn4/share_304380933/rongyiyu/code/qwen3-8B')
print(f'Model loaded: {type(model).__name__}')
print(f'Config: {model.config}')
"
```

### Test Data Loading
```bash
python -c "
from data import JsonFolderDataset
ds = JsonFolderDataset('dataset-100k', num_samples=10)
print(f'Loaded {len(ds)} samples')
print(f'Sample: {ds[0][:100]}...')
"
```

### Check GPU Setup
```bash
python -c "
import torch
print(f'GPUs available: {torch.cuda.device_count()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
"
```

### Verify DeepSpeed Config
```bash
python -c "
import json
with open('configs/ds_zero3.json') as f:
    cfg = json.load(f)
print(json.dumps(cfg, indent=2))
"
```

---

## 📈 Key Metrics to Track

| Metric | Good Range | Warning Signs |
|--------|------------|----------------|
| **Training Loss** | Decreasing | Increasing or stuck |
| **Few-shot Accuracy** | 70%+ | Below 60% |
| **Learning Rate** | Decreasing to min_lr | Stuck at high value |
| **Ring Buffer Size** | 20 (max) | Below 20 early on |
| **Active Clusters** | Most clusters | Many clusters at min weight |
| **Validation Loss** | Decreasing | Increasing (overfitting) |

---

## 💡 Pro Tips

1. **Start small for debugging:**
   ```bash
   python train.py --config configs/default.yaml training.total_iters=10
   ```

2. **Save multiple runs:**
   ```bash
   deepspeed --num_gpus=8 train.py --config configs/default.yaml \
       training.save_dir="outputs/run_$(date +%s)"
   ```

3. **Compare configurations side-by-side:**
   ```bash
   # Run baseline
   deepspeed --num_gpus=8 train.py --config configs/default.yaml \
       training.save_dir="outputs/baseline"
   
   # Run experiment
   deepspeed --num_gpus=8 train.py --config configs/default.yaml \
       clustering.kmeans.feature=intermediate \
       training.save_dir="outputs/experiment"
   
   # Compare: diff outputs/baseline/train.log outputs/experiment/train.log
   ```

4. **Use CPU offload for memory issues:**
   ```bash
   deepspeed --num_gpus=8 train.py --config configs/default.yaml \
       deepspeed.config_file=configs/ds_zero3_offload.json
   ```

5. **Reduce evaluation frequency for faster training:**
   ```bash
   deepspeed --num_gpus=8 train.py --config configs/default.yaml \
       training.eval_interval=1000
   ```

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of Memory** | Reduce batch_size, enable CPU offload, use intermediate features |
| **Slow clustering** | Use `feature=intermediate`, enable FAISS backend |
| **Training won't start** | Check paths with `ls -la dataset-100k/ valid/` |
| **Metrics all zeros** | Check validation set path and format |
| **Low accuracy** | Try higher learning rate, increase training.total_iters |
| **DeepSpeed errors** | Run without DeepSpeed: `deepspeed.enabled=false` |

---

## 📝 Example: Complete Hyperparameter Study

```bash
#!/bin/bash
# Run complete ablation study

METHODS=("minibatch" "kmeans" "random")
FEATURES=("intermediate" "projection" "embedding")
TEMPS=("0.5" "1.0" "2.0")

for method in "${METHODS[@]}"; do
  for feature in "${FEATURES[@]}"; do
    for temp in "${TEMPS[@]}"; do
      name="method_${method}_feat_${feature}_temp_${temp}"
      echo "Starting: $name"
      
      deepspeed --num_gpus=8 train.py --config configs/default.yaml \
        clustering.method=$method \
        clustering.kmeans.feature=$feature \
        pmp.temperature=$temp \
        training.save_dir="outputs/$name" \
        training.total_iters=1000
      
      echo "Completed: $name"
    done
  done
done
```

---

## 📚 For More Details

- **Full reference:** See `CODEBASE_EXPLORATION.md`
- **Configuration:** See `configs/default.yaml` with inline comments
- **Implementation:** See `EARLY_EXIT_IMPLEMENTATION.md`
- **Tests:** See `tests/test_early_exit.py` for usage examples

