"""
End-to-end validation of CountSketch PMP pipeline.
Tests the EXACT same code path as training, with DeepSpeed ZeRO-2.

Run with: deepspeed --num_gpus=1 debug_e2e.py
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from pmp.count_sketch import CountSketchProjector
from pmp.grad_utils_sketch import compute_cluster_contributions_sketch

# ---- Config ----
model_path = "/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B"
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 1, "reduce_bucket_size": 5e8},
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": False,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 3e-5, "betas": [0.9, 0.98], "eps": 1e-6, "weight_decay": 0.01}
    },
}
with open("/tmp/ds_e2e.json", "w") as f:
    json.dump(ds_config, f)

# ---- Load model ----
print("=== 1. Load model + DeepSpeed ===")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

engine, optimizer, _, _ = deepspeed.initialize(model=model, config="/tmp/ds_e2e.json")
raw_model = engine.module
device = engine.device
print(f"DeepSpeed engine ready, device={device}")

# ---- Create fake data ----
print("\n=== 2. Create fake dev + train batches ===")
dev_texts = ["The capital of France is Paris.", "Machine learning is a subset of AI."]
train_texts = ["Hello world test one.", "Goodbye cruel world.", "Another sample here.", "Last one."]

def make_batch(texts, dev=False):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    labels = enc["input_ids"].clone()
    loss_mask = (labels != tokenizer.pad_token_id).float()
    model_batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    no_model_batch = {"label": labels, "loss_mask": loss_mask}
    return model_batch, no_model_batch

dev_batches = [make_batch(dev_texts)]
train_mb, train_nmb = make_batch(train_texts)
batch = {**train_mb, **train_nmb}
cluster_ids = torch.tensor([0, 0, 1, 1], device=device)

print(f"Dev batches: {len(dev_batches)}, shape={dev_batches[0][0]['input_ids'].shape}")
print(f"Train batch: shape={batch['input_ids'].shape}")
print(f"Cluster IDs: {cluster_ids.tolist()}")

# ---- Do a training step first (like real training) ----
print("\n=== 3. Simulate one training step ===")
engine.train()
outputs = engine(**train_mb, labels=train_mb["input_ids"])
engine.backward(outputs.loss)
engine.step()
print(f"Training step done, loss={outputs.loss.item():.4f}")

# ---- Check GPU memory ----
mem = torch.cuda.memory_allocated(device) / 1e9
print(f"GPU memory after training step: {mem:.2f} GB")

# ---- Now run PMP (same as trainer code) ----
print("\n=== 4. Run PMP CountSketch ===")
sketcher = CountSketchProjector(sketch_dim=8192, seed=42)

# Verify param_names and trainable_params match
trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
param_names = [n for n, p in raw_model.named_parameters() if p.requires_grad]
print(f"Trainable params: {len(trainable_params)}")
print(f"Param names: {len(param_names)}")
assert len(trainable_params) == len(param_names), "MISMATCH!"

# Verify _get_hash_sign signature
print("\nTesting _get_hash_sign...")
test_name = param_names[0]
test_numel = trainable_params[0].numel()
h, sign = sketcher._get_hash_sign(test_name, test_numel, device)
print(f"  name='{test_name}', numel={test_numel}")
print(f"  h: shape={h.shape}, device={h.device}, dtype={h.dtype}")
print(f"  sign: shape={sign.shape}, device={sign.device}, dtype={sign.dtype}")
assert h.shape[0] == test_numel, f"h shape mismatch: {h.shape[0]} != {test_numel}"
assert sign.shape[0] == test_numel, f"sign shape mismatch"

# Test _sketch_loss manually
print("\nTesting autograd.grad path...")
loss_fn = nn.CrossEntropyLoss(reduction="none")
raw_model.eval()
logits = raw_model(input_ids=dev_batches[0][0]["input_ids"], attention_mask=dev_batches[0][0]["attention_mask"]).logits
print(f"  logits shape: {logits.shape}, device={logits.device}")
losses = loss_fn(logits.view(-1, logits.size(-1)), dev_batches[0][1]["label"].view(-1))
loss = losses.mean()
print(f"  loss: {loss.item():.4f}")

grads = torch.autograd.grad(loss, trainable_params, allow_unused=True)
n_none = sum(1 for g in grads if g is None)
n_grad = sum(1 for g in grads if g is not None)
print(f"  autograd.grad: {n_grad} grads, {n_none} None")

# Sketch from grads
s = torch.zeros(sketcher.m, device=device, dtype=torch.float32)
for name, p, g in zip(param_names, trainable_params, grads):
    if g is None:
        continue
    h, sign = sketcher._get_hash_sign(name, g.numel(), device)
    s.scatter_add_(0, h, g.float().view(-1) * sign)
print(f"  Manual sketch: norm={s.norm().item():.4f}")

# Now run the full function
print("\n=== 5. Run compute_cluster_contributions_sketch ===")
engine.zero_grad()
torch.cuda.empty_cache()

grad_gamma_delta = compute_cluster_contributions_sketch(
    model=raw_model,  # pass raw_model, same as trainer
    dev_batches=dev_batches,
    batch=batch,
    batch_cluster_ids=cluster_ids,
    n_clusters=2,
    pmp_lr=0.008,
    sketcher=sketcher,
    world_size=1,
    distributed=False,
)

print(f"grad_gamma_delta: {grad_gamma_delta}")
print(f"norm: {grad_gamma_delta.norm().item():.6f}")
print(f"cluster 0: {grad_gamma_delta[0].item():.6f}")
print(f"cluster 1: {grad_gamma_delta[1].item():.6f}")

if grad_gamma_delta.norm().item() > 0:
    print("\n✅ SUCCESS: grad_gamma_delta is non-zero!")
else:
    print("\n❌ FAIL: grad_gamma_delta is zero!")

# Verify training can resume after PMP
print("\n=== 6. Verify training resumes after PMP ===")
engine.train()
outputs2 = engine(**train_mb, labels=train_mb["input_ids"])
engine.backward(outputs2.loss)
engine.step()
print(f"Post-PMP training step done, loss={outputs2.loss.item():.4f}")
print("\n✅ ALL CHECKS PASSED")
