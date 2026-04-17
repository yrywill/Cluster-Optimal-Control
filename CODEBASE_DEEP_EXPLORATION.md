# Comprehensive Codebase Exploration: Cluster Data Selection

## Overview
This is a cluster-based continual pre-training system that uses **PMP (Per-Parameter Memory) backward pass** with **Hessian=0 simplification** to perform on-the-fly cluster-level data selection. The system optimizes which training data clusters are most valuable by monitoring their gradient contribution to validation loss.

---

## Architecture Summary

### High-Level Flow
```
1. CLUSTERING PHASE (Init)
   ├─ Load small embedding model (Qwen 0.5B) or use main model
   ├─ Extract features for all ~100k training samples
   └─ Run KMeans → cluster_ids for each sample

2. TRAINING PHASE (Main Loop)
   ├─ Normal LM training with cluster-weighted sampling
   ├─ Every update_interval steps → PMP BACKWARD
   │  ├─ Compute ∇L_dev(θ) = gradient of validation loss
   │  ├─ Traverse ring buffer of recent training steps
   │  ├─ Compute per-cluster JVP contribution
   │  └─ Update grad_gamma (accumulated cluster scores)
   ├─ Convert grad_gamma → cluster sampling weights
   └─ Next training batch drawn from high-weight clusters

3. EVALUATION (Periodic)
   ├─ Compute validation loss on dev domains
   ├─ Multi-domain support with configurable weights
   └─ Fewshot accuracy optional (MCQ evaluation)
```

---

## 1. VALIDATION AND LOSS COMPUTATION

### Where Validation Happens

#### 1.1 Multi-Domain Validation (`_evaluate_multi_domain`)
**File**: `trainer/integrated_trainer.py` lines 1267-1314

```python
def _evaluate_multi_domain(self) -> Dict[str, float]:
    # For each registered domain (math, code, general, etc.)
    #   - Compute LM loss on up to 50 batches
    #   - Collect per-domain results
    # Return weighted average + individual losses
    
    # Multi-domain setup supports dynamic weighting:
    # weighted = Σ_d (w_d / Σw) · loss_d
```

**Key characteristics**:
- **Runs with `torch.no_grad()`** → No gradients kept
- **Batches pre-cached on CPU** and moved to device per evaluation
- **Capped at 50 batches per domain** for speed
- **Distributed all_reduce** to sync across ranks
- **Perplexity computed** as exp(loss)

#### 1.2 Loss Computation (`_compute_lm_loss`)
**File**: `trainer/integrated_trainer.py` lines 876-887

```python
def _compute_lm_loss(self, model_batch, no_model_batch):
    outputs = self.model(**model_batch, use_cache=False)  # use_cache=False
    logits = outputs.logits  # [B, L, V]
    
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    losses = loss_fn(logits.view(-1, V), labels.view(-1))
    
    # Normalize by loss_mask (valid tokens only)
    lm_loss = (losses * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
    return lm_loss.mean()
```

**Key design**:
- Uses **per-token masking** for padding/special tokens
- Computes **token-level cross-entropy** then averages within sample
- Finally averages across batch

#### 1.3 Fewshot Evaluation (`_evaluate_fewshot`)
**File**: `trainer/integrated_trainer.py` lines 1316-1374

```python
# For MCQ (A/B/C/D) questions:
# 1. Encode prompt + "Answer:"
# 2. Look at logits at last non-padding position
# 3. Extract logits for tokens A, B, C, D
# 4. Compare: argmax → predicted label vs target
# 5. Accuracy = correct / total
```

**Optional feature** when `eval_format == "fewshot"` in config.

---

### Validation Data Management

#### 1.4 Dev Data Caching (`_cache_dev_batches`)
**File**: `trainer/integrated_trainer.py` lines ~647-694 (from pattern matching)

The trainer caches dev data **on CPU** in pre-tokenized form:
- **Each dev batch**: Tuple(model_batch, no_model_batch) on CPU
- **Kept as list** → moved to GPU only when needed for PMP or evaluation
- **Why?** To avoid pinning GPU memory during training

#### 1.5 Multi-Domain Manager (`DevDomainManager`)
**File**: `trainer/integrated_trainer.py` lines 143-251

```python
class DevDomainManager:
    def __init__(self):
        self._domains: Dict[str, Dict] = {}  # {name: {weight, batches}}
    
    def add_domain(self, name, weight, batches_cpu):
        """Register validation domain with independent weight."""
    
    def get_domain_batches_for_pmp(self):
        """Return [(name, weight, batches)] for PMP."""
    
    def get_domain_batches_on_device(self, device):
        """Move all domain batches to device."""
```

**Key feature**: Allows weighting different validation sets dynamically:
- Example: `dev_domains: [{name: "math", weight: 0.5}, {name: "code", weight: 0.3}]`
- During training, can call `trainer.dev_domain_manager.update_weight("math", 0.8)` to shift focus

---

### Evaluation Schedule

**Config parameters**:
- `training.eval_interval`: Run evaluation every N steps (default: 500)
- `training.no_eval_at_start`: Skip eval before training starts
- `training.eval_batch_size`: Batch size for evaluation (default: 16)

**When called**:
```python
# Line 851 in train loop
if global_step % cfg.training.eval_interval == 0:
    torch.cuda.empty_cache()          # Free training memory
    self.model.eval()
    eval_results = self._evaluate_multi_domain()
    self.model.train()
    torch.cuda.empty_cache()          # Free eval memory
```

**Logged metrics**:
- Per-domain: loss, ppl
- Weighted: loss, ppl
- Optional: fewshot_acc

---

## 2. MEMORY MANAGEMENT

### 2.1 Memory Hierarchy

The system has **three separate GPU memory regions**:

```
GPU Memory Layout:
┌──────────────────────────────────────────┐
│ Model Parameters (fp16/bf16)             │ ~8-16 GB (model size)
├──────────────────────────────────────────┤
│ Optimizer States (if DDP) or sharded     │ ~8-24 GB (ZeRO-3 divides by world_size)
├──────────────────────────────────────────┤
│ Activations & Gradients (training)       │ 4-16 GB (during forward/backward)
├──────────────────────────────────────────┤
│ Projector matrices or sketches           │ 16 GB (explicit) or 60 MB (CountSketch)
├──────────────────────────────────────────┤
│ Temporary tensors (JVP, batch)           │ 1-4 GB
└──────────────────────────────────────────┘
Total: 40-80 GB on typical 40GB GPU with ZeRO-3
```

### 2.2 Gradient Checkpointing

**Config**:
```yaml
model:
  gradient_checkpointing: true  # Enable to save memory
```

**Implementation** (`trainer/integrated_trainer.py` lines 349-358):
```python
if cfg.model.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()

# DeepSpeed may override, so re-enable:
if self.use_deepspeed:
    if cfg.model.gradient_checkpointing:
        self._raw_model.gradient_checkpointing_enable()
```

**Effect**: Trade compute for memory by recomputing activations instead of storing them.

### 2.3 Dtype and Precision

**Config** (`configs/default.yaml` lines 14-15):
```yaml
model:
  dtype: "bfloat16"              # float16 | bfloat16 | float32
  attn_impl: "flash_attention_2" # eager | sdpa | flash_attention_2
```

**Implementation** (`trainer/integrated_trainer.py` lines 284-285):
```python
dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
self.dtype = dtype_map[cfg.model.dtype if not cfg.training.fp32 else "float32"]
```

**Model loading** (line 344):
```python
self.model = AutoModelForCausalLM.from_pretrained(
    cfg.model.path,
    torch_dtype=self.dtype,  # ← Applied here
    attn_implementation=cfg.model.attn_impl,
    trust_remote_code=True,
)
```

**Affects**:
- Model parameters: 50% GPU memory reduction (fp32 → fp16/bf16)
- Activations: 50% reduction
- BUT: Potential numerical instability with fp16 (bfloat16 preferred)

### 2.4 DeepSpeed ZeRO Optimization

**Config file**: `configs/ds_zero3.json`

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,  // Stage 3 = parameters + optimizer states sharded
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto"
  }
}
```

**Key memory benefit**:
- ZeRO-3 divides model parameters by world_size
- 8.2B model on 8 GPUs → ~1GB per GPU (vs 8GB without ZeRO)
- Optimizer states similarly divided

**Used in PMP** (`trainer/integrated_trainer.py` lines 966-972):
```python
if self.use_deepspeed:
    pmp_context = deepspeed.zero.GatheredParameters(
        list(self._raw_model.parameters()), 
        modifier_rank=None  # Gather on all ranks for PMP
    )
```

When computing PMP gradients, parameters must be **gathered** (assembled) on each GPU temporarily.

### 2.5 Explicit Memory Cleanup

**Three strategic `torch.cuda.empty_cache()` calls** (lines 720, 852, 856, 922):

```python
# Before training
torch.cuda.empty_cache()  # Free eval activations

# Training loop
# ... training steps ...

# Before PMP
torch.cuda.empty_cache()  # Free training activations/gradients

# Before evaluation
torch.cuda.empty_cache()  # Free training activations

# Before resuming training
torch.cuda.empty_cache()  # Free eval activations
```

**Why needed?** PyTorch's memory allocator caches freed blocks. Explicit cache clearing helps when:
- Transitioning between training and evaluation (different batch sizes)
- Running PMP backward (needs space for dev gradients + JVP)
- Running on near-full GPUs

### 2.6 Ring Buffer (Minimal GPU Impact)

**File**: `trainer/ring_buffer.py` lines 19-100

```python
class RingBuffer:
    def __init__(self, capacity: int, param_dim: int):
        self._buffer: Deque[...] = deque(maxlen=capacity)
        # maxlen=capacity → auto-removes oldest when full
    
    def push(self, params_vec, batch, cluster_ids):
        # Store on CPU to avoid GPU memory pressure
        entry = (
            params_vec.detach().cpu().clone(),           # CPU tensor
            {k: v.detach().cpu() for k, v in batch.items()},  # CPU dict
            cluster_ids_batch.detach().cpu().clone(),    # CPU tensor
        )
        self._buffer.append(entry)
```

**Memory usage**:
- Stores last N (default 20) training steps
- Each step: model params + batch + cluster ids
- **On CPU** → doesn't consume GPU memory
- Example: 8.2B params = ~32GB on CPU (bfloat16)

### 2.7 Projection Matrices (Fast Path Saves Huge Memory)

**Legacy explicit projection** (GhostGradProjector):
- Projection matrix P ∈ R^{d×proj_dim}
- d = 8.2B params, proj_dim = 8192
- Size: 8.2B × 8192 × 4 bytes = **256 GB** (!)
- **Must fit on GPU** → not feasible

**CountSketch fast path** (recommended):
- Hash tables: h ∈ {0,...,m-1}^d
- Sign tables: σ ∈ {-1,+1}^d
- Per parameter: 8 bytes (int64) + 4 bytes (float32) = 12 bytes per param
- Total: 8.2B × 12 bytes = **~100 MB** (!)
- **Never materialized fully** → only cached per-parameter

**Implementation** (`pmp/count_sketch.py` lines 56-79):
```python
def _get_hash_sign(self, name: str, numel: int, device):
    # Lazily build hash/sign on first use, cache by parameter name
    cache_key = (name, str(device))
    if cache_key not in self._cache:
        name_hash = hash(name) & 0xFFFFFFFF
        g = torch.Generator(device="cpu").manual_seed(self.seed + name_hash)
        
        h = torch.randint(0, self.m, (numel,), generator=g, dtype=torch.int64)
        sign = torch.randint(0, 2, (numel,), generator=g, dtype=torch.float32) * 2 - 1
        
        self._cache[cache_key] = (h.to(device), sign.to(device))
    return self._cache[cache_key]
```

---

## 3. PMP BACKWARD AND CLUSTER SCORING

### 3.1 Overview: How Validation Feeds Back into Selection

**Goal**: Given validation loss L_dev and training loss L_train, determine which training clusters are most helpful.

**Key insight**: For each cluster k, we compute:
```
ct_k = mean_{n ∈ C_k} ⟨∇_θ loss_n(θ), λ⟩
```

where:
- ∇_θ loss_n = gradient of training sample n
- λ = accumulated "importance vector" from validation loss
- ⟨·,·⟩ = dot product (JVP = Jacobian-Vector Product)

**Interpretation**: ct_k > 0 means cluster k's gradients are **aligned** with validation loss reduction.

### 3.2 Three Execution Paths

**File**: `trainer/integrated_trainer.py` lines 893-1135

#### Path 1: CountSketch Fast Path (RECOMMENDED)
**Lines 927-962**

```python
if self.count_sketch_projector is not None:
    # Use only the LATEST training step (current params)
    latest = self.ring_buffer.get_latest()  # → (params, batch, cluster_ids)
    
    # Compute ∇L_dev (validation gradient)
    # Compute ∇loss_train per cluster (training gradient)
    # Inner product via sketch: ct_k ≈ ⟨sketch(∇L_dev), sketch(∇loss_k)⟩
    
    grad_gamma_delta = compute_cluster_contributions_sketch(
        model, dev_batches, batch, cluster_ids,
        n_clusters, pmp_lr=0.1, sketcher=count_sketch_projector
    )
```

**Speed**: ~seconds per update (vs minutes for full backward)
**Memory**: ~60MB for sketches (vs 256GB for explicit matrix)

#### Path 2: Ghost Inner Product Legacy Path
**Lines 985-1027**

```python
if ghost_ip_projector is not None:
    # Still uses ring buffer (only latest entry)
    latest = self.ring_buffer.get_latest()
    
    # Ghost project ∇L_dev
    q = ghost_projector.ghost_project_vector(g_dev)  # [proj_dim]
    
    # For each cluster, ghost project ∇loss_k and dot product
    v_k = ghost_projector.ghost_project_vector(g_k)
    ct_k = dot(q, v_k)
```

**Speed**: Similar to CountSketch but uses explicit matrix
**Memory**: 16GB projection matrix (feasible but large)

#### Path 3: Standard Ring Buffer JVP Path (EXACT but SLOW)
**Lines 1032-1092**

```python
else:
    # Full historical backward pass
    history = self.ring_buffer.get_all_ordered()  # [T steps]
    
    for i in range(T-1, -1, -1):  # Newest → Oldest
        # Restore θ_i from ring buffer
        self.model_wrapper.set_params_vec(params_vec_i)
        
        # 1. Compute ∇L_dev(θ_i)
        g_dev = compute_dev_grad_multi_domain(...)
        
        # 2. Init λ at terminal step
        if lam is None:
            lam = g_dev
            continue  # No contribution at last step
        
        # 3. Accumulate: λ_i = ∇L_dev(θ_i) + λ_{i+1}  [Hessian=0!]
        lam = g_dev + lam
        
        # 4. Compute cluster JVP
        delta = compute_cluster_contributions(
            model, batch, cluster_ids, lam_param,
            params_i, buffers_i, n_clusters
        )
        grad_gamma_delta += delta
```

**Key simplification** (Hessian=0):
- Standard PMP: λ_i = ∇L_dev(θ_i) + ∇²L_dev HVP(λ_{i+1}, Δθ_i)
- This code: λ_i = ∇L_dev(θ_i) + λ_{i+1}
- **Removes Hessian term** → 10-20x faster but less accurate

---

### 3.3 Multi-Domain Weighted Gradient

**File**: `pmp/grad_utils.py` lines 427-500

```python
def compute_dev_grad_multi_domain(domain_batches, params, buffers):
    """
    ∇L_dev_weighted = Σ_d (w_d / Σw) · ∇L_dev_d(θ)
    
    Each domain d has weight w_d (configurable at runtime).
    """
    total_weight = sum(w for _, w, _ in domain_batches)
    g_weighted = None
    
    for domain_name, weight, batches in domain_batches:
        normalized_w = weight / total_weight
        g_d = compute_dev_grad(model, batches, params, buffers)
        
        if g_weighted is None:
            g_weighted = normalized_w * g_d
        else:
            g_weighted = g_weighted + normalized_w * g_d
    
    return g_weighted
```

**Effect**: Can dynamically reweight domains during training:
- Start with balanced domains (math: 0.5, code: 0.3, general: 0.2)
- After 5K steps, boost math to 0.8
- Clusters beneficial for math will get higher scores

---

### 3.4 Per-Sample JVP Computation

**File**: `pmp/grad_utils.py` lines 37-105

```python
def cluster_jvp_batch(model, cluster_batch, lam_param, params, buffers):
    """
    Compute ct_k = mean_{n ∈ C_k} ⟨∇loss_n, λ⟩
    
    Uses torch.func.vmap to compute JVP for all samples in cluster in parallel.
    """
    # For each sample n in cluster:
    #   loss_n(θ) → scalar
    #   ∇loss_n = grad(loss_n, θ)
    #   JVP = dot(∇loss_n, λ)
    
    per_sample_ct = vmap(
        _jvp_single,
        in_dims=(0, 0, 0, 0, None, None, None, None),
        chunk_size=cfg.pmp.jvp_chunk_size,  # Reduce if OOM
    )(input_ids, attn_mask, labels, loss_mask, model, lam, params, buffers)
    
    return per_sample_ct.mean()  # ← Cluster-level score
```

**Optimization**:
- `chunk_size`: Process samples in chunks (batch process vmap) to fit in memory
- If OOM: Reduce chunk_size or reduce batch_size

---

### 3.5 Cluster Weight Update

**File**: `data/cluster_dataset.py` lines 150-217

```python
def update_weights(self, grad_gamma, grad_gamma_delta=None):
    """
    Convert accumulated scores → sampling weights.
    
    w_k = softmax(-grad_gamma / temperature)  [clamp min_weight, renorm]
    """
    gg = grad_gamma.float()  # [n_clusters]
    logits = -gg / self.temperature  # Negative: higher scores → lower logits
    logits = logits - logits.max()   # Numerical stability
    weights = torch.exp(logits)      # Softmax
    weights = weights.clamp(min=self.min_weight)  # Prevent starvation
    
    # ---- Optional: Bad cluster auto-drop ----
    if self.drop_bad_clusters and grad_gamma_delta is not None:
        delta = grad_gamma_delta.float().cpu()
        
        for k in range(n_clusters):
            if delta[k] < 0:
                self._negative_streak[k] += 1
            elif delta[k] > 0:
                self._negative_streak[k] = 0
            
            if self._negative_streak[k] >= self.drop_patience:
                self._dead_clusters[k] = True  # Permanently drop
    
    weights[self._dead_clusters] = 0.0
    total = weights.sum()
    weights = weights / total  # Renormalize
    
    self._weights = weights
```

**Parameters** (`configs/default.yaml` lines 85-95):
- `pmp.temperature`: 0.1 (sharp) to 1.0 (diffuse)
  - 0.1 → High-scoring clusters get ~80% probability
  - 1.0 → More uniform distribution
- `pmp.min_weight`: 0.01 (minimum per-cluster sampling rate, prevents starvation)
- `pmp.drop_bad_clusters`: Enable auto-drop of persistently negative clusters
- `pmp.drop_patience`: 5 consecutive negative updates before drop

---

### 3.6 Sampling with Updated Weights

**File**: `data/cluster_dataset.py` lines 241-271

```python
def __iter__(self):
    """
    For each training batch:
      1. Draw cluster assignments ~ Categorical(weights)
      2. For each cluster, uniformly draw a sample
    """
    K = dataset.n_clusters
    weights_np = self._weights.numpy()  # Convert to numpy for numpy.choice
    total_needed = self._num_samples_per_rank * self.world_size
    
    # Sample clusters
    cluster_draws = rng.choice(K, size=total_needed, replace=True, p=weights_np)
    
    # For each cluster draw, pick a random sample within cluster
    indices = np.zeros(total_needed, dtype=np.int64)
    for i, k in enumerate(cluster_draws):
        cluster_idxs = dataset.get_cluster_indices(k)  # All indices in cluster k
        if len(cluster_idxs) > 0:
            indices[i] = cluster_idxs[rng.integers(len(cluster_idxs))]
        else:
            indices[i] = rng.integers(len(dataset))  # Fallback
    
    # Shard for distributed training
    indices_for_rank = indices[rank::world_size]
    return iter(indices_for_rank.tolist())
```

**Effect**:
- If cluster 0 has weight 0.6, cluster 1 has weight 0.4
- Next epoch, ~60% of samples come from cluster 0
- **Feedback loop**: High-scoring clusters are oversampled, directly improving next batch quality

---

## 4. DATA SELECTION LOOP (Full Cycle)

```
Training Step t:
  ├─ Sample batch B ~ ClusterWeightedSampler (current weights)
  ├─ Forward + backward on L_train(B)
  ├─ Push (params_t, B, cluster_ids_B) to ring buffer
  └─ global_step += 1

Every pmp.update_interval steps (default 20):
  ├─ Compute ∇L_dev(θ_t)
  │  └─ If multi-domain: weighted sum of domain gradients
  ├─ [CountSketch path] OR [Ghost IP path] OR [Full JVP path]
  │  └─ For each cluster: ct_k = ⟨∇L_dev, mean(∇loss_k)⟩
  ├─ grad_gamma_delta += ct
  ├─ grad_gamma (persistent) += grad_gamma_delta
  ├─ Compute new weights: w_k = softmax(-grad_gamma / temp)
  ├─ ClusterWeightedSampler.update_weights(grad_gamma, grad_gamma_delta)
  └─ Next batch will favor high-weight clusters

Evaluation (every eval_interval steps, default 500):
  ├─ Compute loss on each dev domain
  ├─ Log per-domain loss + weighted avg
  └─ Metrics feed PMP on next backward pass
```

---

## 5. KEY CONFIGURATION PARAMETERS

### Training Config

```yaml
training:
  total_iters: 500              # Total gradient steps
  batch_size: 4                 # Per-GPU
  gradient_accumulation_steps: 2 # Effective batch = 4*2=8 per GPU
  eval_batch_size: 16           # For evaluation
  eval_interval: 500            # Evaluate every 500 steps
  no_eval_at_start: false       # Run eval before training
  fp32: false                   # Force fp32 (else use dtype from model)
  
  lr: 3.0e-5
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_iters: 200
  weight_decay: 0.01
  clip_grad: 1.0
```

### Data Config

```yaml
data:
  train_dir: "dataset-100k"     # Training data
  dev_dir: "valid"              # Validation data (single domain)
  dev_num: 300                  # Max samples for PMP backward
  eval_batch_size: 16           # When evaluating
  eval_format: "fewshot"        # fewshot | text
  n_shot: 3                     # Few-shot demonstrations
  
  dev_domains: []               # Optional: multi-domain validation
  # - name: "math"
  #   dir: "data/dev/math"
  #   weight: 0.5
```

### Clustering Config

```yaml
clustering:
  method: "minibatch"           # minibatch | kmeans | random
  cluster_size: 500             # Samples per cluster (N → n_clusters = N / cluster_size)
  recluster_interval: -1        # -1 = once at init only
  kmeans:
    feature: "intermediate"     # intermediate | projection | embedding | ghost
    embed_layer: -1             # Which layer for intermediate mode
  embedding_model:
    enabled: true               # Use small model for clustering
    path: "qwen2.5-0.5B"        # Small model
    dtype: "bfloat16"
    attn_impl: "sdpa"
```

### PMP Config

```yaml
pmp:
  window_size: 20               # Ring buffer size (steps)
  update_interval: 20           # Update weights every N steps
  lr: 1.0                       # PMP update rate (ct_k scaled by this)
  temperature: 0.1              # Softmax temp (lower = sharper weights)
  min_weight: 0.01              # Minimum per-cluster probability
  dev_batch_size: 4             # When computing ∇L_dev
  jvp_chunk_size: null          # Vmap chunk (null = full)
  
  ghost_ip:
    enabled: true               # Use CountSketch/Ghost IP fast path
    proj_type: "count_sketch"   # count_sketch | rademacher
    proj_dim: 8192              # Sketch dimension (for CountSketch)
    seed: 42
```

### Model Config

```yaml
model:
  path: "llama-3.2-3B"          # HuggingFace or local path
  dtype: "bfloat16"             # float16 | bfloat16 | float32
  attn_impl: "flash_attention_2" # eager | sdpa | flash_attention_2
  max_length: 1024              # Max tokens per sample
  gradient_checkpointing: true  # Trade compute for memory
```

### DeepSpeed Config (`configs/ds_zero3.json`)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

---

## 6. VRAM MANAGEMENT STRATEGIES

### Memory Budgeting (Typical 40GB GPU with ZeRO-3, 8 GPUs)

```
Per-GPU memory budget: 40 GB / 8 = ~5 GB

Allocation:
- Model params (sharded): ~1 GB (8.2B model / 8 GPUs)
- Optimizer states (sharded): ~2 GB
- Activations (training): 1 GB
- Batch tensor: 0.5 GB
- Misc / safety margin: 0.5 GB
────────────────────────
Total: ~5 GB
```

### If OOM During Training

1. **Reduce batch_size** (default 4)
   ```yaml
   training:
     batch_size: 2              # Try 2 or 1
   ```

2. **Enable gradient_checkpointing** (if not already)
   ```yaml
   model:
     gradient_checkpointing: true
   ```

3. **Reduce max_length** (default 1024)
   ```yaml
   model:
     max_length: 512            # Fewer tokens = fewer activations
   ```

4. **Use CountSketch** (default, already enabled)
   ```yaml
   pmp:
     ghost_ip:
       enabled: true
       proj_type: "count_sketch"  # NOT "rademacher"
   ```

### If OOM During PMP

1. **Reduce jvp_chunk_size**
   ```yaml
   pmp:
     jvp_chunk_size: 32         # Chunk vmap to 32 samples at a time
   ```

2. **Reduce dev_batch_size** (default 4)
   ```yaml
   pmp:
     dev_batch_size: 2          # Fewer samples when computing ∇L_dev
   ```

3. **Reduce window_size** (default 20)
   ```yaml
   pmp:
     window_size: 5             # Shorter ring buffer = fewer steps to traverse
   ```

---

## 7. KEY FILES AND THEIR ROLES

| File | Lines | Purpose |
|------|-------|---------|
| `train.py` | 1-152 | Entry point, DDP + DeepSpeed setup |
| `trainer/integrated_trainer.py` | 1-1440+ | Main training loop, evaluation, PMP backward |
| `pmp/grad_utils.py` | 1-500 | ∇L_dev computation, cluster contributions, multi-domain |
| `pmp/grad_utils_sketch.py` | - | CountSketch variant |
| `pmp/count_sketch.py` | 1-173 | Hash-based projection matrix (~60MB) |
| `pmp/projection.py` | - | Legacy explicit projection (16GB) |
| `pmp/model_wrapper.py` | - | Parameter vectorization, JVP |
| `data/cluster_dataset.py` | 1-330+ | Cluster-aware dataset & weighted sampler |
| `data/eval_dataset.py` | - | Fewshot MCQ evaluation |
| `data/json_dataset.py` | - | Load & tokenize training data |
| `clustering/` | - | KMeans, embedding, feature extraction |
| `trainer/ring_buffer.py` | 1-100 | CPU storage of training step history |

---

## 8. VALIDATION LOSS → CLUSTER SELECTION FEEDBACK

### Information Flow

```
Dev Batch → Loss L_dev(θ_t)
             ↓
           ∇L_dev(θ_t)  [computed via grad_and_value]
             ↓
         Validation gradient tells us:
         "Which parameters are important for validation task?"
             ↓
      For each training cluster k:
      ├─ Compute mean ∇loss_k (training gradient)
      ├─ Inner product: ct_k = ⟨∇L_dev, ∇loss_k⟩
      │  (positive = aligned, negative = misaligned)
      └─ ct_k > 0 → Increase cluster weight
         ct_k < 0 → Decrease cluster weight
             ↓
      Next epoch:
      High-weight clusters sampled more often
      → Focus on data that helps validation task
```

### Example Scenario

```
Cluster A: (High math content)
  ct_A = 0.8  → w_A increases to 0.15
  
Cluster B: (Low-quality general content)
  ct_B = -0.3 → w_B decreases to 0.02
  
Cluster C: (Balanced)
  ct_C = 0.1  → w_C stays around 0.05

Next 20 training steps:
- 75% of samples from cluster A
- 5% from cluster B
- 20% from cluster C
```

---

## Summary

**Validation** is computed periodically on multiple dev domains with independent weights, enabling dynamic weighting during training.

**Memory** is managed through:
- Gradient checkpointing
- bfloat16 precision
- DeepSpeed ZeRO-3 parameter sharding
- CPU-based ring buffer
- CountSketch projection (~60MB vs 256GB)
- Explicit `torch.cuda.empty_cache()` calls

**Data Selection** works by:
1. Computing weighted validation gradient
2. Scoring each training cluster via JVP inner product
3. Converting scores to sampling weights
4. Oversampling high-scoring clusters

The system achieves **on-the-fly adaptive data selection** with minimal memory overhead.

