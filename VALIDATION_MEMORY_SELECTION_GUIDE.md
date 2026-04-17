# Validation, Memory Management, and Data Selection - Quick Reference

## 1. VALIDATION LOSS COMPUTATION FLOW

```
┌─────────────────────────────────────────────────────────────┐
│ EVALUATION CYCLE (every eval_interval=500 steps)            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  trainer._evaluate_multi_domain()                            │
│  ├─ torch.cuda.empty_cache()          [FREE GPU]            │
│  ├─ model.eval()                                            │
│  │                                                           │
│  ├─ FOR each domain (math, code, general):                  │
│  │  ├─ FOR each cached batch (up to 50):                    │
│  │  │  ├─ Move batch CPU → GPU device                       │
│  │  │  ├─ outputs = model(**batch, use_cache=False)         │
│  │  │  ├─ loss = CrossEntropyLoss(logits, labels)           │
│  │  │  │           [normalized by per-token mask]           │
│  │  │  ├─ domain_losses.append(loss.item())                 │
│  │  │  └─ dist.all_reduce(loss) [sync across ranks]         │
│  │  └─ results[domain] = mean(domain_losses)                │
│  │  └─ results[f"{domain}_ppl"] = exp(loss)                 │
│  │                                                           │
│  ├─ results["weighted"] = Σ_d (w_d / Σw) · loss_d           │
│  ├─ results["fewshot_acc"] = MCQ accuracy (optional)         │
│  ├─ model.train()                                           │
│  └─ torch.cuda.empty_cache()          [FREE GPU]            │
│                                                               │
│  LOGGED: loss_math=X, loss_code=Y, weighted=Z, fewshot=W    │
└─────────────────────────────────────────────────────────────┘
```

### Key Methods

**Loss Computation** (`_compute_lm_loss`):
```python
# For each batch:
logits = model(input_ids, attention_mask)         # [B, L, V]
losses = CE(logits.view(-1,V), labels.view(-1))   # Per token
lm_loss = (losses * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(1)
return lm_loss.mean()                             # Average
```

**Multi-Domain Evaluation** (`_evaluate_multi_domain`):
- Each domain cached as list of (model_batch, no_model_batch) on CPU
- Moved to device batch-by-batch during eval
- Up to 50 batches per domain for speed
- Distributed all_reduce for DDP synchronization
- Returns dict: `{"math": 2.3, "code": 2.5, "weighted": 2.4, "ppl": 11.0}`

**Fewshot Evaluation** (`_evaluate_fewshot`, optional):
- For MCQ (A/B/C/D) questions
- Extract logits at last non-padding position
- Argmax over [logit_A, logit_B, logit_C, logit_D]
- Compare to ground truth label

---

## 2. MEMORY MANAGEMENT ARCHITECTURE

### 2.1 Memory Hierarchy (Typical 40GB GPU, 8-GPU setup with ZeRO-3)

```
GPU 0 Total: ~40 GB
├─ Model Parameters (sharded by ZeRO-3):        1 GB  (8.2B / 8)
├─ Optimizer States (sharded):                  2 GB  (Adam has 2 states)
├─ Activations (during forward):                1 GB  (depends on seq_len, batch)
├─ Gradients (during backward):                 1 GB  (same as activations)
├─ Ring Buffer Data (on CPU, not GPU):          0 GB  (stored on CPU)
├─ Projection matrices/sketches:               60 MB (CountSketch, minimal!)
├─ Development batch (cached, on GPU):         0.5 GB (moved on-demand)
└─ Misc / PyTorch overhead / Safety:            1.5 GB
════════════════════════════════════════════════
TOTAL: ~7-8 GB (leaving ~32-33 GB headroom)
```

### 2.2 Key Memory Optimization Techniques

#### Technique 1: Gradient Checkpointing
```python
# Config: gradient_checkpointing: true
model.gradient_checkpointing_enable()

# Effect:
#   - Don't store all activations during forward pass
#   - Recompute them during backward (trade CPU time for GPU memory)
#   - Save ~40-50% of activation memory
```

#### Technique 2: bfloat16 Precision
```python
# Config: dtype: "bfloat16"
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, ...
)

# Effect:
#   - Parameters: 50% memory (32-bit → 16-bit)
#   - Activations: 50% memory
#   - Numerically stable (vs fp16)
```

#### Technique 3: DeepSpeed ZeRO-3
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"}
  }
}
```

```
Without ZeRO: GPU 0 holds full 8.2B model params (32 GB)
With ZeRO-3:  GPU 0 holds 8.2B / 8 params (4 GB)
              → 8x memory saving!
```

#### Technique 4: CountSketch Projection (~60 MB)
```python
# Instead of explicit projection matrix:
#   Old: P ∈ R^{8.2B × 8192} = 256 GB!
#   New: Hash table + sign table = 60 MB

# How it works:
#   sketch(grad) = scatter_add(hash_buckets, grad * signs)
#   ⟨sketch(g1), sketch(g2)⟩ ≈ ⟨g1, g2⟩  (Johnson-Lindenstrauss)
```

#### Technique 5: CPU-Based Ring Buffer
```python
# Ring buffer stores LAST 20 training steps:
#   - Model params: on CPU (32 GB on CPU, 0 GB on GPU)
#   - Training batch: on CPU (small, ~100 MB)
#   - Cluster IDs: on CPU (tiny)

# Only moved to GPU during PMP backward (which is rare)
```

#### Technique 6: Explicit GPU Cache Clearing
```python
# 4 strategic torch.cuda.empty_cache() calls:
torch.cuda.empty_cache()  # After evaluation
torch.cuda.empty_cache()  # Before PMP
torch.cuda.empty_cache()  # After PMP
torch.cuda.empty_cache()  # Before resuming training

# Why: PyTorch's allocator caches freed memory blocks.
#      Explicit clearing helps when transitioning between phases.
```

---

## 3. PMP BACKWARD AND CLUSTER SCORING

### 3.1 Overall Process

```
┌──────────────────────────────────────────────────────────────┐
│ EVERY pmp.update_interval STEPS (default 20)                 │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  _run_pmp_backward_and_update(global_step)                    │
│  ├─ torch.cuda.empty_cache()                                  │
│  ├─ model.zero_grad()                                         │
│  │                                                            │
│  ├─ PATH 1: CountSketch Fast Path [RECOMMENDED]              │
│  │  ├─ latest = ring_buffer.get_latest()                     │
│  │  ├─ Compute ∇L_dev (validation gradient)                  │
│  │  ├─ Compute ∇loss per cluster (training gradient)         │
│  │  ├─ Inner product via hash sketch (~60MB, O(1))          │
│  │  └─ grad_gamma_delta = per-cluster contributions           │
│  │                                                            │
│  ├─ PATH 2: Ghost Inner Product [LEGACY]                    │
│  │  ├─ Similar to Path 1 but explicit projection matrix      │
│  │  └─ Memory: 16GB projection matrix                        │
│  │                                                            │
│  ├─ PATH 3: Full JVP Path [EXACT but SLOW]                   │
│  │  ├─ FOR i in [T-1...0]:  (oldest to newest in ring buffer)│
│  │  │  ├─ Restore θ_i from ring_buffer[i]                   │
│  │  │  ├─ g_dev_i = ∇L_dev(θ_i)                             │
│  │  │  ├─ λ_i = g_dev_i + λ_{i+1}  [Hessian=0!]            │
│  │  │  ├─ FOR each cluster k in batch_i:                     │
│  │  │  │  └─ ct_k = mean JVP(loss_n, λ_i) for n in C_k      │
│  │  │  └─ grad_gamma_delta += ct_k                          │
│  │  └─ grad_gamma_delta /= world_size                        │
│  │                                                            │
│  ├─ grad_gamma += grad_gamma_delta                           │
│  ├─ weights = softmax(-grad_gamma / temperature)             │
│  ├─ sampler.update_weights(grad_gamma, grad_gamma_delta)     │
│  │  └─ [Optional] Drop bad clusters (drop_patience)         │
│  │                                                            │
│  └─ dist.barrier()  [Sync all ranks]                        │
│                                                                │
│  LOGGED: grad_gamma_delta norm, weights entropy, alive count  │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Understanding Cluster Scores (ct_k)

**Formula**:
```
ct_k = mean_{n ∈ C_k} ⟨∇_θ loss_n(θ), λ⟩

where:
  C_k       = samples in cluster k
  ∇loss_n   = gradient of training loss for sample n
  λ         = "importance vector" from validation loss
  ⟨·,·⟩     = dot product (inner product)
  ct_k > 0  = cluster k's gradients aligned with validation task
  ct_k < 0  = cluster k's gradients misaligned (harmful)
```

**Interpretation**:
- If cluster A (math) has ct_A = 0.8 (high positive)
  → Math samples help reduce validation loss → increase weight → oversample math
  
- If cluster B (noise) has ct_B = -0.3 (negative)
  → Noise samples increase validation loss → decrease weight → undersample noise

### 3.3 Multi-Domain Weighted Gradients

```python
# Config example:
dev_domains:
  - name: "math"
    dir: "data/dev/math"
    weight: 0.5
  - name: "code"
    dir: "data/dev/code"
    weight: 0.3
  - name: "general"
    dir: "data/dev/general"
    weight: 0.2

# During PMP backward:
g_dev_weighted = (0.5/(0.5+0.3+0.2)) * ∇L_math
               + (0.3/(0.5+0.3+0.2)) * ∇L_code
               + (0.2/(0.5+0.3+0.2)) * ∇L_general

# Effect: Cluster scores biased toward math (50%) > code (30%) > general (20%)
#         Can dynamically call: trainer.dev_domain_manager.update_weight("math", 0.8)
```

### 3.4 Weight Conversion (grad_gamma → sampling weights)

```python
def update_weights(self, grad_gamma, grad_gamma_delta):
    # grad_gamma: accumulated scores [n_clusters]
    
    # 1. Invert and scale
    logits = -grad_gamma / temperature
    logits = logits - logits.max()  # Numerical stability
    
    # 2. Softmax
    weights = exp(logits)  # Higher ct_k → higher weight
    
    # 3. Clamp to prevent starvation
    weights = weights.clamp(min=min_weight)  # min_weight=0.01
    
    # 4. Optional: Drop bad clusters
    if drop_bad_clusters:
        for k in range(n_clusters):
            if grad_gamma_delta[k] < 0:
                negative_streak[k] += 1
            else:
                negative_streak[k] = 0
            
            if negative_streak[k] >= drop_patience:  # drop_patience=5
                weights[k] = 0.0  # Permanently drop
    
    # 5. Renormalize
    weights = weights / weights.sum()
    
    return weights  # → Used by sampler for next epoch
```

**Effect of temperature**:
```
temperature=0.1 (sharp):
  ct=[1.0, 0.5, -0.2] → w=[0.75, 0.20, 0.05]  (80-20 split)

temperature=1.0 (diffuse):
  ct=[1.0, 0.5, -0.2] → w=[0.45, 0.35, 0.20]  (more uniform)
```

### 3.5 Cluster Sampling (weights → batches)

```python
def __iter__(self):
    # Each epoch, generate all indices for all ranks combined
    
    # 1. Draw clusters ~ Categorical(weights)
    cluster_draws = rng.choice(K, size=total_needed, replace=True, p=weights)
    
    # 2. For each cluster draw, sample one random sample
    indices = np.zeros(total_needed)
    for i, k in enumerate(cluster_draws):
        cluster_samples = dataset.get_cluster_indices(k)
        indices[i] = cluster_samples[rng.integers(len(cluster_samples))]
    
    # 3. Shard for distributed training
    indices_for_this_rank = indices[rank::world_size]
    
    return iter(indices_for_this_rank.tolist())
```

**Result**:
- If w = [0.6, 0.3, 0.1], next epoch samples 60%, 30%, 10% from clusters 0, 1, 2
- Cluster 0 is "good" (helped validation) → oversampled
- Cluster 1 is "okay" → normal sampling
- Cluster 2 is "bad" (hurt validation) → undersampled

---

## 4. COMPLETE DATA SELECTION CYCLE

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Training (every step)                                   │
├─────────────────────────────────────────────────────────────────┤
│  sampler = ClusterWeightedSampler(weights_current)              │
│  batch_indices = sampler.sample()                               │
│  batch = dataset[batch_indices]                                 │
│  loss = model(batch)                                            │
│  loss.backward()                                                │
│  optimizer.step()                                               │
│  ring_buffer.push(params_before_update, batch, cluster_ids)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Evaluation (every eval_interval=500 steps)              │
├─────────────────────────────────────────────────────────────────┤
│  dev_losses = {}                                                │
│  FOR domain IN [math, code, general]:                           │
│    FOR batch IN dev_domain_batches:                             │
│      loss_domain = model.forward(batch)                         │
│      dev_losses[domain] = mean(losses)                          │
│  dev_losses["weighted"] = weighted average                      │
│  # Logged: "math=2.3, code=2.5, weighted=2.4, ppl=11.0"        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: PMP Backward (every update_interval=20 steps)           │
├─────────────────────────────────────────────────────────────────┤
│  # Dev gradient tells us "which parameters matter for val task" │
│  g_dev = ∇L_dev(θ_current)                                      │
│                                                                 │
│  # For each training cluster, score via inner product           │
│  FOR cluster k IN range(n_clusters):                            │
│    samples_k = ring_buffer.latest_batch[cluster==k]            │
│    g_k = mean(∇L_train(sample) for sample in samples_k)        │
│    ct_k = dot(g_dev, g_k)                                       │
│    grad_gamma_delta[k] = ct_k                                   │
│                                                                 │
│  # Accumulate and convert to weights                            │
│  grad_gamma += grad_gamma_delta                                 │
│  weights_new = softmax(-grad_gamma / temperature)              │
│  # Logged: "grad_gamma: min/max/norm, weights: entropy/alive"   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Weight Update (same as Step 3)                          │
├─────────────────────────────────────────────────────────────────┤
│  sampler.update_weights(grad_gamma, grad_gamma_delta)           │
│  # sampler._weights = weights_new                               │
│  # Next epoch will use updated weights!                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Repeat (feedback loop)                                  │
├─────────────────────────────────────────────────────────────────┤
│  # High-weight clusters sampled more → directly improves batch  │
│  # quality by focusing on data that helps validation task       │
│  # Loop continues: train → eval → score → resample → train ...  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. TROUBLESHOOTING MEMORY ISSUES

### Issue: OOM During Training

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions** (in order of ease):

1. **Reduce batch size**
   ```yaml
   training:
     batch_size: 2  # Reduce from 4
   ```

2. **Enable gradient checkpointing**
   ```yaml
   model:
     gradient_checkpointing: true  # Trade compute for memory
   ```

3. **Reduce sequence length**
   ```yaml
   model:
     max_length: 512  # Reduce from 1024
   ```

4. **Reduce gradient accumulation steps**
   ```yaml
   training:
     gradient_accumulation_steps: 1  # Reduce from 2
   ```

### Issue: OOM During PMP Backward

**Symptom**: `RuntimeError: CUDA out of memory` after "PMP: step=X" log

**Solutions**:

1. **Reduce jvp_chunk_size** (process fewer samples per vmap iteration)
   ```yaml
   pmp:
     jvp_chunk_size: 32  # Reduce from null (full)
   ```

2. **Reduce dev_batch_size** (fewer samples when computing ∇L_dev)
   ```yaml
   pmp:
     dev_batch_size: 2  # Reduce from 4
   ```

3. **Reduce window_size** (shorter ring buffer = fewer backward steps)
   ```yaml
   pmp:
     window_size: 10  # Reduce from 20
   ```

4. **Use CountSketch** (already default, verify it's enabled)
   ```yaml
   pmp:
     ghost_ip:
       enabled: true
       proj_type: "count_sketch"  # NOT "rademacher"
   ```

### Issue: Evaluation is Slow

**Symptom**: Eval takes 5+ minutes

**Solutions**:

1. **Increase eval_interval** (evaluate less frequently)
   ```yaml
   training:
     eval_interval: 1000  # From 500
   ```

2. **Reduce eval_batch_size** (process fewer samples)
   ```yaml
   training:
     eval_batch_size: 8  # From 16
   ```

3. **Reduce dev_num** (fewer validation samples)
   ```yaml
   data:
     dev_num: 200  # From 300
   ```

---

## 6. QUICK REFERENCE TABLE

| Component | Config Key | Default | Effect |
|-----------|-----------|---------|--------|
| **VALIDATION** | | | |
| Eval frequency | `training.eval_interval` | 500 | Eval every 500 training steps |
| Eval batch size | `training.eval_batch_size` | 16 | Process 16 samples per eval batch |
| Dev samples | `data.dev_num` | 300 | Max 300 samples for PMP |
| Eval format | `data.eval_format` | "fewshot" | MCQ accuracy (or "text" for LM loss) |
| **PMP / SELECTION** | | | |
| Update interval | `pmp.update_interval` | 20 | Update weights every 20 steps |
| Window size | `pmp.window_size` | 20 | Keep last 20 steps in ring buffer |
| Temperature | `pmp.temperature` | 0.1 | Lower = sharper weight distribution |
| Min weight | `pmp.min_weight` | 0.01 | Prevent clusters from being starved |
| PMP LR | `pmp.lr` | 1.0 | ct_k × lr → grad_gamma_delta |
| Drop bad | `pmp.drop_bad_clusters` | False | Auto-drop persistently harmful clusters |
| **MEMORY** | | | |
| Grad ckpt | `model.gradient_checkpointing` | True | Save memory, trade compute |
| Dtype | `model.dtype` | "bfloat16" | fp16/bf16 (50% memory) vs fp32 |
| Attention | `model.attn_impl` | "flash_attention_2" | Faster, lower memory attention |
| Batch size | `training.batch_size` | 4 | Per-GPU batch size |
| Max length | `model.max_length` | 1024 | Max tokens (↓ = less memory) |
| Grad accum | `training.gradient_accumulation_steps` | 2 | Accumulate N steps (↑ = less memory) |

---

## 7. Key Equations

```
Validation Loss:
  L_dev(θ) = mean loss over dev batches

Dev Gradient (weighted multi-domain):
  ∇L_dev_weighted = Σ_d (w_d / Σw) · ∇L_dev_d(θ)

Cluster Contribution:
  ct_k = mean_{n ∈ C_k} ⟨∇loss_n(θ), λ⟩

Lambda Accumulation (Hessian=0):
  λ_i = ∇L_dev(θ_i) + λ_{i+1}  [for i=T-1...0]

Grad Gamma:
  grad_gamma_new = grad_gamma_old + (pmp_lr × grad_gamma_delta)

Cluster Weights (softmax):
  w_k = softmax(-grad_gamma / temperature)[k]
      = exp(-grad_gamma[k] / temp) / Σ_j exp(-grad_gamma[j] / temp)

Weighted Dev Loss:
  L_weighted = Σ_d (w_d / Σw) · L_dev_d
```

