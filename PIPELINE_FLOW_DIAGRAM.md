# TRAINING PIPELINE FLOW DIAGRAM

## High-Level Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│ INITIALIZATION (Lines 305-500)                                  │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load Llama-3.2-3B (bfloat16)                                 │
│ 2. Enable Gradient Checkpointing (Line 306)                     │
│ 3. Wrap with DeepSpeed ZeRO-3 (Line 310)                        │
│ 4. RE-enable GradCP on raw_model (Line 315)                     │
│ 5. Initialize RingBuffer(capacity=20, param_dim=0)  [CountSketch]
│ 6. Initialize CountSketchProjector(m=8192, seed=42)             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING LOOP (per gradient step, Lines 717-777)                │
├─────────────────────────────────────────────────────────────────┤
│ For each micro-batch (6 samples):                               │
│                                                                  │
│   Line 712-720: Extract cluster_ids from batch indices          │
│                                                                  │
│   Line 725-729: [CountSketch Path Only]                         │
│                 params_before_update = torch.tensor([0.0])      │
│                 (Dummy: no param history needed!)               │
│                                                                  │
│   Line 733: Forward pass → compute_lm_loss()                    │
│   Line 738/740: Backward on loss_scaled                         │
│                                                                  │
│   Lines 744-754: Accumulate batch (keep on CPU)                │
│                  Accumulate cluster_ids                        │
│                                                                  │
│   IF gradient accumulation complete (line 758):                │
│     Line 764/768: Optimizer step (DeepSpeed/PyTorch)           │
│     Line 771: global_step += 1                                 │
│     Line 777: ring_buffer.push(params_dummy, batch, cluster_id)
│                                                                  │
│   IF global_step % 20 == 0:  → PMP Update                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ PMP UPDATE (Lines 880-930, every 20 steps)                       │
├─────────────────────────────────────────────────────────────────┤
│ [CountSketch Fast Path] (Line 896: if count_sketch_projector)   │
│                                                                  │
│ Call: compute_cluster_contributions_sketch(                     │
│       model=self._raw_model,        # Raw model (not engine!)   │
│       dev_batches=...,              # All dev domains on device │
│       batch=batch_device,           # Latest training batch     │
│       batch_cluster_ids=...,        # Cluster assignments       │
│       sketcher=count_sketch_projector,  # 8192-dim sketcher     │
│       distributed=True,             # All-reduce for ZeRO-3     │
│ )                                                                │
│ ↓ Returns: grad_gamma_delta ∈ R^K  (K = n_clusters)           │
│                                                                  │
│ self.grad_gamma += grad_gamma_delta  # Accumulate updates       │
│ self.sampler.update_weights(grad_gamma)  # Update sampler       │
└─────────────────────────────────────────────────────────────────┘
```

## CountSketch PMP Update (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│ compute_cluster_contributions_sketch()                           │
│ (pmp/grad_utils_sketch.py, Lines 83-185)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ STEP 1: Sketch Dev Gradient                                     │
│ ────────────────────────────────                                │
│ q = torch.zeros(8192)  [on GPU]                                │
│                                                                  │
│ For each dev_batch in dev_batches:                             │
│   ├─ model.zero_grad()                                         │
│   ├─ logits = model(input_ids, attention_mask)                │
│   ├─ loss.backward()                                           │
│   ├─ q += sketcher.sketch_grad(inner_model)                   │
│   │   └─ For each parameter p:                                │
│   │       (h_p, σ_p) = _get_hash_sign(name, p.numel(), device)
│   │       s.scatter_add_(0, h_p, p.grad * σ_p)               │
│   │   └─ O(d) total, never materializes full d-dim vector!   │
│   └─ model.zero_grad()                                         │
│                                                                  │
│ q = q / n_dev_batches                                           │
│ if distributed:                                                 │
│   dist.all_reduce(q)  [ZeRO-3: linearity preserves correctness]
│   q = q / world_size                                            │
│                                                                  │
│ STEP 2: Per-Cluster Contributions                               │
│ ───────────────────────────────────                             │
│ grad_gamma_delta = torch.zeros(K)  [K clusters]                │
│                                                                  │
│ For each unique cluster k in batch_cluster_ids:                │
│   ├─ Extract k-th cluster samples                              │
│   ├─ loss_k.backward()                                         │
│   ├─ v_k = sketcher.sketch_grad(inner_model)  [8192-dim]      │
│   ├─ if distributed:                                           │
│   │   dist.all_reduce(v_k)                                     │
│   │   v_k = v_k / world_size                                   │
│   ├─ ct_k = torch.dot(q, v_k)  [scalar inner product]          │
│   └─ grad_gamma_delta[k] += pmp_lr * ct_k                      │
│                                                                  │
│ Return: grad_gamma_delta ∈ R^K                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Memory Savings: CountSketch vs Explicit Projection

```
┌──────────────────┬──────────────────┬────────────────────┐
│ Component        │ Explicit Matrix  │ CountSketch        │
├──────────────────┼──────────────────┼────────────────────┤
│ Projection Matrix│ P ∈ R^{d×m}      │ None!              │
│ Size             │ 8.2B × 8192      │ ~60MB (hash+sign)  │
│ Memory           │ ~16GB @ fp32      │ ~60MB              │
│ Materialization  │ Full gradient d  │ Never! Scatter-add │
│ Max mem during PMP│ ~30-50GB         │ <5GB               │
└──────────────────┴──────────────────┴────────────────────┘
```

## CountSketch Hash Function (sketch_grad, Lines 109-114)

```python
# For parameter named "layer_0.weight" with 1M elements:

for name, p in model.named_parameters():
    if p.grad is None:
        continue
    
    # Lazy cache key per param
    h, sign = self._get_hash_sign(name, p.numel(), p.device)
    
    # h:    [1000000]  dtype=int64  — hash buckets {0..8191}
    # sign: [1000000]  dtype=float32 — random {-1, +1}
    # p.grad: [1000000]
    
    # Scatter-add: s[h_i] += p.grad_i * sign_i
    s.scatter_add_(0, h, p.grad.view(-1) * sign)
    
    # Result: s ∈ R^8192, computed in O(1M) time
    # No [1M × 8192] matrix materialized!
```

## Integration Points

```
┌─────────────────────────────────────────┐
│ trainer/integrated_trainer.py           │
├─────────────────────────────────────────┤
│                                         │
│ Line 463-465: Detect CountSketch mode  │
│ Line 476-478: Set param_dim = 0        │
│ Line 480: RingBuffer(capacity=20, 0)   │
│ Line 489-494: Init CountSketchProjector│
│                                         │
│ Line 727: Use dummy params             │
│                                         │
│ Line 896-926: Call sketch path         │
│   ├─ _do_pmp_update()                  │
│   ├─ compute_cluster_contributions_sketch()  [import line 897]
│   └─ Returns grad_gamma_delta          │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ pmp/grad_utils_sketch.py                │
├─────────────────────────────────────────┤
│ compute_cluster_contributions_sketch()  │
│  └─ For each batch/cluster:            │
│      └─ sketcher.sketch_grad(model)    │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ pmp/count_sketch.py                     │
├─────────────────────────────────────────┤
│ CountSketchProjector                    │
│  ├─ sketch_grad(model) [Lines 85-116]  │
│  └─ _get_hash_sign(...) [Lines 56-79]  │
└─────────────────────────────────────────┘
```

## Config Cascade

```
configs/default.yaml (Line 150-151)
  ├─ deepspeed.enabled = true
  ├─ deepspeed.config_file = "configs/ds_zero3.json"
  │   └─ ZeRO-3 with optimizer+param offload
  │
  ├─ pmp.ghost_ip.enabled = true
  ├─ pmp.ghost_ip.proj_type = "count_sketch"  ← CRITICAL
  ├─ pmp.ghost_ip.proj_dim = 8192
  ├─ pmp.ghost_ip.seed = 42
  │
  ├─ pmp.window_size = 20     ← RingBuffer capacity
  ├─ pmp.update_interval = 20 ← PMP update frequency
  │
  ├─ training.batch_size = 6
  ├─ training.gradient_accumulation_steps = 2
  │
  └─ model.gradient_checkpointing = true
     (re-enabled after DeepSpeed wrapping)
```

## Execution Timeline (Example: 50 steps)

```
Step  1-2: Train (no PMP yet, not enough data in ring buffer)
Step  3-19: Train (accumulate batches in ring buffer)
Step  20: [TRAIN] + [PMP UPDATE #1]
           ├─ Compute dev sketch q  (all 300 dev samples)
           ├─ Per-cluster sketch v_k (from latest training batch)
           ├─ Update grad_gamma from ct_k = ⟨q, v_k⟩
           └─ Update ClusterWeightedSampler weights
Step  21-39: Train (new batches in ring buffer, grad_gamma influences sampling)
Step  40: [TRAIN] + [PMP UPDATE #2]  (same process)
...
```
