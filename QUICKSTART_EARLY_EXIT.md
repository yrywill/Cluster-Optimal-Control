# Quick Start: Early-Exit Implementation

**Time to first test:** 5 minutes  
**Time to full integration:** 30 minutes

## Step 1: Verify Installation (1 minute)

```bash
cd /apdcephfs_jn4/share_304380933/rongyiyu/code/cluster_data_selection

# Check syntax
python3 -m py_compile utils/layer_access.py clustering/early_exit_kmeans.py

# Check imports
python3 << 'PYEOF'
from utils.layer_access import get_layer_count, extract_single_layer_features
from clustering.early_exit_kmeans import EarlyExitKMeansClusterer
print("✓ All imports successful")
PYEOF
```

## Step 2: Run Unit Tests (5 minutes)

```bash
# Install pytest if needed
pip install pytest

# Run all early-exit tests (uses mocks, no GPU needed)
pytest tests/test_early_exit.py -v

# Expected output: 28+ tests passing
```

## Step 3: Test with Real Model (10 minutes)

```bash
python3 << 'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.layer_access import get_layer_count, get_intermediate_hidden_states

# Load Qwen3-8B
print("Loading Qwen3-8B model...")
model_path = "/apdcephfs_jn4/share_304380933/rongyiyu/code/qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = model.eval().to('cuda')

# Check layer count
num_layers = get_layer_count(model)
print(f"✓ Model has {num_layers} layers")

# Test extraction from layer 18
print(f"Testing layer 18 extraction...")
test_texts = ["This is a test.", "Another sample."]
inputs = tokenizer(test_texts, return_tensors='pt', padding=True).to('cuda')
model.eval()

with torch.no_grad():
    hidden_states, mask = get_intermediate_hidden_states(
        model, 
        inputs['input_ids'], 
        inputs['attention_mask'],
        layer_idx=18
    )
print(f"✓ Layer 18 hidden states shape: {hidden_states.shape}")
print(f"✓ Expected shape: [batch_size, seq_len, 4096] = [{inputs['input_ids'].shape[0]}, {inputs['input_ids'].shape[1]}, 4096]")
PYEOF
```

## Step 4: Use in Training (20 minutes)

### Option A: Direct API Usage

```python
from clustering import EarlyExitKMeansClusterer
from utils.layer_access import get_layer_count

clusterer = EarlyExitKMeansClusterer()

# Cluster using layer 18
cluster_ids = clusterer.fit_with_intermediate_layer(
    dataset=train_dataset,
    model=model,
    tokenizer=tokenizer,
    device=torch.device('cuda'),
    cfg=cfg,
    layer_idx=18,  # intermediate layer
    rank=0
)

print(f"Cluster assignments: {cluster_ids[:10]}")
```

### Option B: Configuration-Based (Recommended)

Update `configs/default.yaml`:

```yaml
clustering:
  method: "minibatch"
  cluster_size: 100
  kmeans:
    feature: "embedding"
    feature_batch_size: 64
```

Then run:
```bash
python train.py --config configs/default.yaml
```

## Layer Selection Quick Reference

For Qwen3-8B (36 layers):

Layer Range | Use Case
----------|----------
0-5       | Very fast, low quality (testing only)
6-12      | Fast, moderate quality (baseline)
12-18     | **Balanced (RECOMMENDED)**
18-24     | Slower, higher quality
24-30     | Much slower
35        | Final layer (baseline, slowest)

**Start with layer 18** for Qwen3-8B (50% depth, good quality/speed tradeoff).

## Common Commands

Get model layer info:
```bash
python3 -c "
from transformers import AutoModelForCausalLM
from utils.layer_access import get_layer_count
model = AutoModelForCausalLM.from_pretrained('/path/to/model')
print(f'Layers: {get_layer_count(model)}')
"
```

Extract features from layer 12:
```python
from utils.layer_access import extract_single_layer_features
features = extract_single_layer_features(model, dataset, device, layer_idx=12)
print(f"Features shape: {features.shape}")
```

Run clustering on specific layer:
```python
from clustering import EarlyExitKMeansClusterer
clusterer = EarlyExitKMeansClusterer()
cluster_ids = clusterer.fit_with_intermediate_layer(
    dataset, model, tokenizer, device, cfg, layer_idx=18
)
```

## Troubleshooting

Issue: ModuleNotFoundError: No module named 'utils.layer_access'
- Solution: Run from project root directory

Issue: ValueError: layer_idx out of bounds
- Solution: Check with get_layer_count(model) for valid range

Issue: CUDA out of memory
- Solution: Reduce feature_batch_size in config or use earlier layer

Issue: Tests fail
- Solution: Ensure pytest installed: `pip install pytest`

Issue: Import succeeds but extraction fails
- Solution: Model must be on correct device (model.to('cuda'))

## Performance Expectations

Operation | Time (Qwen3-8B) | Memory
-----------|-----------------|--------
get_layer_count() | <1ms | <1MB
extract_single_layer_features() (100 samples) | 5-10s | Variable
fit_with_intermediate_layer() (100k samples) | 5-10 min | 16GB GPU
Clustering only (after features) | 30s-2min | <1GB

## Next Steps After Quick Start

1. Verify imports and run unit tests
2. Test with real model to confirm layer access works
3. Run clustering on different layers (12, 18, 24) to compare quality
4. Benchmark speed and memory for your use case
5. Update production config with best-performing layer
6. Monitor clustering metrics during training

## Support

- **API Reference:** See EARLY_EXIT_IMPLEMENTATION.md
- **Configuration Guide:** See EARLY_EXIT_CONFIG_PATCH.md
- **Test Suite:** tests/test_early_exit.py
- **Full Checkpoint:** See IMPLEMENTATION_CHECKPOINT.md

**Status: Ready to use! Start with Step 1.**
