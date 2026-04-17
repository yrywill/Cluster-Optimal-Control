"""
End-to-end validation of CountSketch PMP pipeline.
Pure DDP, no DeepSpeed.

Run with: torchrun --nproc_per_node=1 debug_ddp.py
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer
from pmp.count_sketch import CountSketchProjector
from pmp.grad_utils_sketch import compute_cluster_contributions_sketch

model_path = "/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B"

print("=== 1. Load model (pure PyTorch, no DeepSpeed) ===")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda:0")
mem = torch.cuda.memory_allocated(device) / 1e9
print(f"Model loaded, GPU mem: {mem:.2f} GB")

# ---- Fake data ----
print("\n=== 2. Create batches ===")
def make_batch(texts):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    labels = enc["input_ids"].clone()
    loss_mask = (labels != tokenizer.pad_token_id).float()
    mb = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    nmb = {"label": labels, "loss_mask": loss_mask}
    return mb, nmb

dev_mb, dev_nmb = make_batch(["The capital of France is Paris.", "ML is a subset of AI."])
dev_batches = [(dev_mb, dev_nmb)]

train_mb, train_nmb = make_batch(["Hello world.", "Goodbye world.", "Test three.", "Test four."])
batch = {**train_mb, **train_nmb}
cluster_ids = torch.tensor([0, 0, 1, 1], device=device)

print(f"Dev: {dev_mb['input_ids'].shape}, Train: {batch['input_ids'].shape}")

# ---- Training step ----
print("\n=== 3. Training step ===")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
optimizer.zero_grad()
out = model(**train_mb, labels=train_mb["input_ids"])
out.loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Training loss: {out.loss.item():.4f}")
mem = torch.cuda.memory_allocated(device) / 1e9
print(f"GPU mem after train step: {mem:.2f} GB")

# ---- PMP ----
print("\n=== 4. PMP CountSketch ===")
sketcher = CountSketchProjector(sketch_dim=8192, seed=42)
torch.cuda.empty_cache()
mem = torch.cuda.memory_allocated(device) / 1e9
print(f"GPU mem after empty_cache: {mem:.2f} GB")

grad_gamma_delta = compute_cluster_contributions_sketch(
    model=model,
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
    print("\n✅ PMP SUCCESS: non-zero grad_gamma_delta!")
else:
    print("\n❌ PMP FAIL: zero grad_gamma_delta!")

# ---- Resume training ----
print("\n=== 5. Resume training after PMP ===")
model.train()
optimizer.zero_grad()
out2 = model(**train_mb, labels=train_mb["input_ids"])
out2.loss.backward()
optimizer.step()
print(f"Post-PMP training loss: {out2.loss.item():.4f}")

print("\n✅ ALL CHECKS PASSED")
