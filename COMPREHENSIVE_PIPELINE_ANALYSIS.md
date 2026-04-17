# Comprehensive PMP Training Pipeline Analysis
**Complete Exploration Report | April 2026**

---

## Executive Summary

This codebase implements a **Proxy Model Policy (PMP) backward pass** for dynamic cluster-based data selection during continual pre-training. The system:

1. **Clusters training data** into K groups (~200 clusters for 100k samples)
2. **Samples from clusters** proportional to learned weights
3. **Periodically runs PMP backward** to compute validation gradient contribution of each cluster
4. **Updates weights** to prioritize high-contribution clusters
5. **Repeats** to concentrate training on most valuable data

### Key Innovation
Instead of static data selection, this system **dynamically reweights clusters based on their gradient contribution to validation loss**, enabling efficient selective pre-training.

---

## Directory Structure & File Count

Total Python Files: 30+ across 7 directories

```
trainer/              - Main training loop and PMP backward
  ├─ integrated_trainer.py  (1460 lines) - Core trainer class
  └─ ring_buffer.py         (100 lines)  - Stores recent training steps

pmp/                  - PMP gradient computation and projection
  ├─ grad_utils.py          (500 lines) - Standard ring-buffer JVP
  ├─ grad_utils_sketch.py   (174 lines) - CountSketch fast path
  ├─ model_wrapper.py       (135 lines) - Functional differentiation
  ├─ count_sketch.py        (173 lines) - Hash-based sketching
  └─ projection.py          - Random projection utilities

data/                 - Data loading and sampling
  ├─ cluster_dataset.py     (310 lines) - ClusterDataset + ClusterWeightedSampler
  ├─ json_dataset.py        - JSON/JSONL file loading
  └─ eval_dataset.py        - Evaluation data utilities

clustering/          - Initial data clustering
  ├─ kmeans_clusterer.py    - Full KMeans
  ├─ base_clusterer.py      - Base interface
  └─ random_clusterer.py    - Baseline method

utils/              - Config, logging, layer access
  ├─ config.py       - OmegaConf configuration loader
  └─ layer_access.py - Model layer indexing

tests/              - Test suite
  ├─ test_ghost.py
  ├─ test_early_exit.py
  └─ run_manual_tests.py

configs/            - YAML configuration files
  └─ default.yaml   - Main configuration with all hyperparameters
```

---

## Training Pipeline: Five Phases

### Phase 1: Initialization (Lines 257-571 in IntegratedClusterTrainer.__init__)

```
Input:  Configuration + CLI overrides
        Training corpus (JSON/JSONL files)
        Validation set (MMLU or custom domains)

Process:
  1. Load config via OmegaConf
  2. Initialize distributed training (DDP or DeepSpeed)
  3. Run clustering on all training samples
  4. Load main model + tokenizer
  5. Create ClusterDataset and ClusterWeightedSampler
  6. Cache validation batches on CPU
  7. Initialize ring buffer (capacity = window_size)
  8. Initialize CountSketch projector (optional)

Output: Ready-to-train IntegratedClusterTrainer instance
        with uniform cluster weights (1/K each)
```

### Phase 2: Training Loop (Lines 697-870)

```
FOR each global_step in range(total_steps):
  
  ┌─ Micro-batch Processing (gradient accumulation) ─┐
  │                                                  │
  │  1. Sample batch from ClusterWeightedSampler    │
  │     (proportional to current weights)           │
  │                                                  │
  │  2. Move batch to GPU                           │
  │     Extract cluster_ids from batch.__indices__  │
  │                                                  │
  │  3. Forward pass + loss computation             │
  │     loss = CrossEntropy(logits, labels)         │
  │     (per-token, masked, then averaged)          │
  │                                                  │
  │  4. Backward pass                               │
  │     loss_scaled.backward()                      │
  │     (accumulate across micro-batches)           │
  │                                                  │
  │  5. Store batch on CPU                          │
  │     for later PMP backward pass                 │
  │                                                  │
  └──────────────────────────────────────────────────┘
  
  ┌─ Optimizer Step (every gradient_accumulation_steps) ─┐
  │                                                      │
  │  1. Clip gradient norms                             │
  │  2. optimizer.step()                                │
  │  3. lr_scheduler.step()                             │
  │  4. optimizer.zero_grad()                           │
  │  5. Push accumulated batch to ring buffer           │
  │  6. global_step += 1                                │
  │                                                      │
  └──────────────────────────────────────────────────────┘
  
  IF global_step % update_interval == 0:
    ┌─ PMP BACKWARD: Update Cluster Weights ──────┐
    │                                             │
    │  1. Compute ∇L_dev(θ)                      │
    │     (gradient of validation loss)          │
    │                                             │
    │  2. For each step in ring buffer (oldest→new):
    │     Compute per-cluster JVP contribution   │
    │     ct_k = pmp_lr · ⟨∇L_dev, ∇L_k⟩        │
    │     grad_gamma_delta[k] += ct_k            │
    │                                             │
    │  3. Accumulate: grad_gamma += grad_gamma_delta
    │                                             │
    │  4. Convert to weights:                    │
    │     w_k = softmax(-grad_gamma / temp)      │
    │                                             │
    │  5. Update sampler with new weights        │
    │                                             │
    │  Result: HIGH clusters get higher weights  │
    │          LOW clusters get lower weights    │
    │                                             │
    └─────────────────────────────────────────────┘
  
  IF global_step % eval_interval == 0:
    ┌─ Periodic Evaluation ──────────┐
    │  1. Compute dev loss           │
    │  2. Compute perplexity         │
    │  3. Log metrics                │
    │  4. Optional: Fewshot accuracy │
    └────────────────────────────────┘
  
  IF global_step % save_interval == 0:
    └─ Save checkpoint ─────────────────┐
       1. Model weights                 │
       2. Optimizer state               │
       3. Cluster weights               │
       4. grad_gamma                    │
       5. Training logs                 │
       └────────────────────────────────┘
```

### Phase 3: PMP Backward Pass (Lines 893-1135)

**Three Execution Paths** (configured via `pmp.ghost_ip`):

#### Path A: CountSketch Fast Path (Recommended)
**Config**: `pmp.ghost_ip.enabled=true` + `pmp.ghost_ip.proj_type="count_sketch"`
**Memory**: ~60 MB (hash tables)
**Distributed**: YES (ZeRO-3 compatible)

```
1. Initialize sketch dimension m = 8192
2. For each parameter, precompute:
   h_p = hash(param_name) ∈ [0, m-1]
   σ_p ∈ {-1, +1}^{d_p}

3. Sketch dev gradient:
   q = zeros(m)
   FOR each dev_batch:
     grads = torch.autograd.grad(loss, params)
     FOR each param:
       q.scatter_add_(h_p, grads * σ_p)
   q = q / n_dev_batches

4. Per-cluster contributions:
   FOR each cluster k in training batch:
     v_k = sketch(∇L_k) using same procedure
     ct_k = <q, v_k>  (dot product)
     grad_gamma_delta[k] += pmp_lr * ct_k

5. All-reduce sketch vectors if distributed
```

#### Path B: Ghost Inner Product Path (Legacy)
**Config**: `pmp.ghost_ip.enabled=true` + `pmp.ghost_ip.proj_type="rademacher|gaussian"`
**Memory**: ~16 GB (explicit matrix)
**Distributed**: Requires GatheredParameters context

Similar to CountSketch but uses explicit projection matrix P instead of hash-based sketch.

#### Path C: Standard Ring-Buffer JVP Path (Exact)
**Config**: `pmp.ghost_ip.enabled=false` (default)
**Memory**: Variable (depends on window size)
**Distributed**: YES

```
Initialize: lam = None
FOR t = T-1 DOWN TO 0 (newest to oldest):
  1. Restore model to θ_t
  2. Compute g_dev_t = ∇L_dev(θ_t)
  3. Update lambda:
     if lam is None:
       lam = g_dev_t
     else:
       lam = g_dev_t + lam  (Hessian=0 assumption)
  4. For each cluster k in batch at time t:
     ct_k = pmp_lr · ⟨∇loss_k(θ_t), lam⟩
     (via vmap of per-sample JVP)
     grad_gamma_delta[k] += ct_k
```

### Phase 4: Weight Update & Sampling

```python
# Accumulate or reset
if pmp.accumulate_grad_gamma:
    grad_gamma = grad_gamma + grad_gamma_delta  # cumulative
else:
    grad_gamma = grad_gamma_delta  # reset each window

# Convert to weights
weights = softmax(-grad_gamma / temperature)
weights = clamp(weights, min=min_weight)
weights = weights / weights.sum()

# Optional: drop bad clusters
if drop_bad_clusters:
    FOR each cluster k:
        if consecutive_negative_updates[k] >= drop_patience:
            weights[k] = 0  # permanently drop
            
# Update sampler for next epoch
sampler.update_weights(weights)
```

### Phase 5: Data Sampling Strategy

```python
# ClusterWeightedSampler.__iter__():

FOR each batch in epoch:
  1. Sample clusters proportional to weights w_k
     cluster_draws = rng.choice(K, size=batch_size, p=weights)
  
  2. For each cluster draw:
     Pick random sample from cluster
     indices[i] = cluster_samples[k][random_idx]
  
  3. Shard across ranks for distributed training
     indices_for_rank = indices[rank::world_size]
  
  Result: Batch with samples from high-weight clusters
```

---

## Key Data Structures

### RingBuffer (trainer/ring_buffer.py)

```python
class RingBuffer:
    capacity = pmp.window_size  # e.g., 20
    
    entries: Deque[Tuple[
        params_vec: Tensor[param_dim],
        batch_cpu: Dict[str, Tensor],
        cluster_ids: Tensor[B]
    ]]
    
    def push(params, batch, cluster_ids):
        # Add entry (auto-drop oldest if full)
    
    def get_all_ordered() -> List:
        # Return entries oldest→newest for backward pass
    
    def get_latest() -> Tuple:
        # Return most recent entry for Ghost IP
```

**Usage**: Stores recent training steps needed for PMP backward pass.

### ClusterWeightedSampler (data/cluster_dataset.py)

```python
class ClusterWeightedSampler:
    _weights: Tensor[K]           # cluster sampling weights
    _grad_gamma: Tensor[K]        # accumulated PMP contributions
    temperature: float            # softmax temperature
    min_weight: float             # weight floor
    _dead_clusters: Tensor[K, bool]  # permanently dropped
    
    def update_weights(grad_gamma, grad_gamma_delta=None):
        # Convert grad_gamma to weights
        # Track bad clusters if enabled
        # All-reduce if distributed
    
    def __iter__():
        # Generate non-overlapping indices per rank
        # Sample clusters proportional to weights
```

**Key feature**: Dynamically reweights as PMP updates grad_gamma.

### TransformerWrapper (pmp/model_wrapper.py)

```python
class TransformerWrapper(nn.Module):
    base_model: nn.Module
    
    def compute_loss_func(params, buffers, model, batch):
        # Functional version for torch.func.grad
        # Returns scalar loss
    
    def compute_loss_func_single(params, buffers, model, single_sample):
        # Per-sample version for torch.func.vmap
        # Returns scalar loss per sample
    
    def get_params_vec() -> Tensor[param_dim]:
        # Flatten all parameters
    
    def set_params_vec(vec: Tensor[param_dim]):
        # Restore parameters from flat vector
    
    def params_to_vector(params_dict):
        # Convert Dict[name, Tensor] → flat vector
    
    def vector_to_params(vec):
        # Convert flat vector → Dict[name, Tensor]
```

**Purpose**: Enable torch.func operations (grad, vmap) on model parameters.

### CountSketchProjector (pmp/count_sketch.py)

```python
class CountSketchProjector:
    m: int  # sketch dimension (e.g., 8192)
    seed: int
    _cache: Dict[(param_name, device) → (hash_table, sign_table)]
    
    def _get_hash_sign(name, numel, device):
        # Lazy compute hash and sign tables
        # Return h ∈ [0, m-1]^numel and σ ∈ {-1,+1}^numel
    
    def sketch_grad(model, named_grads):
        # For each parameter:
        #   s.scatter_add_(h, grad * sigma)
        # Return sketch vector s ∈ R^m
    
    def sketch_vector(named_tensor_dict):
        # Same operation on arbitrary named tensors
```

**Memory**: O(m × log(n_params)) bits vs O(d × m) for explicit matrix.

---

## Loss Computation Details

### Standard LM Loss (Lines 876-887)

```python
def _compute_lm_loss(self, model_batch, no_model_batch):
    # Forward pass
    outputs = self.model(**model_batch, use_cache=False)
    logits = outputs.logits  # [B, L, V]
    
    # Cross-entropy loss per token
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    losses = loss_fn(
        logits.view(-1, logits.size(-1)),
        no_model_batch["label"].view(-1)
    ).view(no_model_batch["label"].shape)  # [B, L]
    
    # Apply loss mask (padding, special tokens)
    lm_loss_per_sample = (
        (losses * no_model_batch["loss_mask"]).sum(dim=-1) /
        no_model_batch["loss_mask"].sum(dim=-1).clamp(min=1)
    )  # [B]
    
    return lm_loss_per_sample.mean()  # scalar
```

### Per-Sample JVP (grad_utils.py lines 37-59)

```python
def _jvp_single(input_ids, attention_mask, label, loss_mask,
                 model, lam_param, params, buffers):
    """
    Compute: ⟨∇loss_n(θ), λ⟩ = JVP of loss w.r.t. params in direction λ
    Used for per-sample contribution to cluster JVP
    """
    def loss_func(p):
        return compute_loss_func_single(
            p, buffers, model,
            input_ids, attention_mask, label, loss_mask
        )
    
    _, tangent = jvp(loss_func, (params,), (lam_param,))
    return tangent  # scalar ⟨∇loss, λ⟩
```

### Cluster-Level JVP via vmap (grad_utils.py lines 66-105)

```python
def cluster_jvp_batch(model, cluster_batch, lam_param, params, buffers, chunk_size):
    """
    Compute: ct_k = mean_{n ∈ C_k} [ ⟨∇loss_n(θ), λ⟩ ]
    Uses vmap to parallelize over batch samples
    """
    per_sample_jvps = vmap(
        _jvp_single,
        in_dims=(0, 0, 0, 0, None, None, None, None),
        chunk_size=chunk_size
    )(
        cluster_batch["input_ids"],
        cluster_batch["attention_mask"],
        cluster_batch["label"],
        cluster_batch["loss_mask"],
        model,
        lam_param,
        params,
        buffers
    )  # Shape: [cluster_batch_size]
    
    return per_sample_jvps.mean()  # scalar ct_k
```

---

## Distributed Training Support

### DDP (Data Parallel)

```
Setup: torchrun --nproc_per_node=4 train.py ...

Behavior:
  - Each GPU has full model copy + optimizer state
  - Backward pass accumulates gradients
  - all_reduce synchronizes gradients across GPUs
  - Optimizer step applied identically on all GPUs
  
PMP Integration:
  - Wrap model with DDP
  - PMP uses raw_model (module attribute)
  - All-reduce dev gradients and cluster contributions
  - Barrier after PMP for safety
```

### DeepSpeed ZeRO-3

```
Setup: deepspeed --num_gpus=4 train.py --config configs/default.yaml deepspeed.enabled=true

Behavior:
  - Parameters sharded across GPUs (each GPU has N/4 params)
  - Gradients also sharded
  - Training backward automatically collects shards for computation
  
PMP Integration (CRITICAL):
  - MUST use CountSketch (linear operation)
  - All-reduce works on sketch vectors
  - Sketch computation happens per-rank locally
  - Never materializes full gradient vector
  
Why CountSketch works:
  - sketch(a + b) = sketch(a) + sketch(b)  ✓ (linearity)
  - all_reduce(sketch_1) + all_reduce(sketch_2) = all_reduce(sketch_combined)
  - Ghost Inner Product with explicit matrix FAILS on ZeRO-3
```

---

## Configuration Schema (default.yaml)

### Model Configuration

```yaml
model:
  path: "llama-3.2-3B"              # HuggingFace model or local path
  type: "auto"                      # auto | mistral | llama | gpt2
  max_length: 1024                  # max tokens per sample
  dtype: "bfloat16"                 # float16 | bfloat16 | float32
  attn_impl: "flash_attention_2"    # eager | sdpa | flash_attention_2
  gradient_checkpointing: true      # save memory during backward
```

### Clustering Configuration

```yaml
clustering:
  method: "minibatch"               # minibatch | kmeans | faiss | random
  cluster_size: 500                 # target samples per cluster
  recluster_interval: -1            # -1 = once at start
  kmeans:
    feature: "intermediate"         # projection | embedding | ghost | intermediate
    embed_layer: -1                 # layer for feature extraction (-1 = middle)
  embedding_model:
    enabled: true                   # use small model for features
    path: "qwen2.5-0.5B"
```

### PMP Configuration

```yaml
pmp:
  window_size: 20                   # ring buffer capacity
  update_interval: 20               # PMP backward frequency
  lr: 1.0                          # scaling for cluster contributions
  temperature: 0.1                  # softmax temperature (lower = sharper)
  accumulate_grad_gamma: true       # cumulative vs reset
  drop_bad_clusters: true           # drop persistently negative clusters
  drop_patience: 5                  # streak threshold
  
  ghost_ip:                         # fast path config
    enabled: true                   # use fast path
    proj_dim: 8192                  # sketch/projection dimension
    proj_type: "count_sketch"       # count_sketch | rademacher | gaussian
```

### Training Configuration

```yaml
training:
  total_iters: 500                  # total gradient steps
  batch_size: 4                     # per-GPU micro-batch
  gradient_accumulation_steps: 2    # micro-steps before optimizer
  lr: 3.0e-5                       # learning rate
  optimizer: "adamw"                # adamw | adam | sgd
  scheduler: "cosine"               # cosine | constant | noam
  warmup_iters: 200                 # learning rate warmup steps
  weight_decay: 0.01
  clip_grad: 1.0                   # gradient clipping threshold
  eval_interval: 500                # validation frequency
  save_interval: 1000               # checkpoint frequency
```

---

## Common Error Messages & Solutions

### OOM (Out of Memory)

```
Error: "CUDA out of memory"

Solutions (in order of priority):
1. Reduce batch_size (4 → 2 or 1)
2. Enable gradient_checkpointing: true
3. Reduce pmp.window_size (20 → 10)
4. Use smaller proj_dim for CountSketch (8192 → 4096)
5. Switch to CountSketch if using Ghost IP
6. Use DeepSpeed ZeRO-3 instead of DDP
```

### Ring Buffer Memory Issues

```
Problem: Ring buffer growing uncontrollably

Likely cause: Storing full parameter vectors on CPU

Solution:
  - Current code stores params_vec (necessary for JVP)
  - For very large models, consider:
    a) Compression: store as bfloat16 instead of float32
    b) Delta compression: store params_before and updates
    c) Checkpoint saving: save to disk between PMP passes
```

### Validation Loss Not Decreasing

```
Problem: PMP weights not helping validation

Diagnostics:
  1. Check if rings buffer is filling:
     if global_step < window_size:
         ring buffer has < window_size entries
         PMP backward may be incomplete
  
  2. Check cluster weight distribution:
     Log: weights.min(), weights.max(), weights.entropy()
     If max weight ~1.0, single cluster dominates (too sharp)
     Increase temperature (0.1 → 1.0)
  
  3. Check grad_gamma values:
     If all zeros: validation gradient not being computed
     If all negative: all clusters hurting validation (data mismatch)
  
  4. Disable PMP temporarily:
     Set pmp.update_interval = infinity
     Train with uniform sampling
     Compare to PMP results
```

---

## Performance Optimization Strategies

### 1. Reduce PMP Backward Time

```
Current: ~30% of training time spent on PMP

Options:
a) Reduce window_size: 20 → 10 (halves backward time)
b) Reduce update_interval: 20 → 50 (less frequent updates)
c) Switch to CountSketch: ~2x faster than ring-buffer JVP
d) Increase eval_batch_size in PMP: less memory pressure
```

### 2. Reduce Clustering Time

```
Clustering runs once at init (1-5 min typically)

Options:
a) Use embedding_model=false (extract from main model)
b) Reduce feature_batch_size: 256 → 512
c) Use random clustering for testing
d) Cache clustering results and load next time
```

### 3. Reduce Memory Pressure

```
Solutions (priority order):
a) Use CountSketch (60 MB vs 16 GB)
b) Reduce batch_size (micro-batch on GPU)
c) Enable gradient_checkpointing
d) Use bfloat16 instead of float32
e) Use DeepSpeed ZeRO-3 (parameter sharding)
```

---

## Testing & Debugging

### Quick Sanity Checks

```bash
# 1. Configuration loads correctly
python -c "from utils.config import load_config; cfg = load_config('configs/default.yaml'); print(cfg)"

# 2. Data loads correctly
python -c "from data.json_dataset import JsonFolderDataset; ds = JsonFolderDataset('data/train'); print(len(ds))"

# 3. Clustering runs
python -c "python train.py --config configs/default.yaml --dry-run"

# 4. Single training step
torchrun --nproc_per_node=1 train.py \
  --config configs/default.yaml \
  training.total_iters=1 \
  training.log_interval=1
```

### Debug Scripts

```
debug_sketch.py     - Test CountSketch hash/sign correctness
debug_e2e.py        - End-to-end training sanity check
debug_ddp.py        - Multi-GPU communication test
debug_ds_sketch.py  - DeepSpeed + CountSketch integration
```

### Test Suite

```
tests/test_ghost.py       - Ghost projection correctness
tests/test_early_exit.py  - Early exit layer selection
tests/run_manual_tests.py - Manual test runner
```

---

## Quick Start Examples

### Single GPU, Small Model

```bash
python train.py \
  --config configs/default.yaml \
  model.path=gpt2 \
  model.max_length=512 \
  training.batch_size=8 \
  training.total_iters=100 \
  data.train_dir=data/train \
  data.dev_dir=data/dev
```

### Multi-GPU DDP

```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/default.yaml \
  model.path=llama-3.2-3B \
  data.train_dir=data/train \
  data.dev_dir=data/dev
```

### DeepSpeed ZeRO-3

```bash
deepspeed --num_gpus=4 train.py \
  --config configs/default.yaml \
  deepspeed.enabled=true \
  model.path=llama-3.2-3B \
  pmp.ghost_ip.proj_type=count_sketch \
  data.train_dir=data/train \
  data.dev_dir=data/dev
```

---

## Summary: System Design Principles

1. **Dynamic Data Selection**: Cluster weights determined by validation gradients, not fixed
2. **Efficient Gradients**: Hessian=0 simplification + ring buffer
3. **Memory Efficient**: CountSketch instead of explicit projections
4. **Distributed Ready**: Works with DDP and DeepSpeed ZeRO-3
5. **Modular**: Three PMP backward execution paths
6. **Fault Tolerant**: Bad clusters auto-dropped
7. **Multi-Domain**: Support weighted combination of validation objectives

The innovation is in **precisely targeting training on data clusters that matter most for validation**, based on continuous gradient feedback.

