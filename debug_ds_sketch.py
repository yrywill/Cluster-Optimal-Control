"""
Debug: test gradient flow under DeepSpeed ZeRO-2.
Run with: deepspeed --num_gpus=1 debug_ds_sketch.py
"""
import torch
import torch.nn as nn
import deepspeed
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Minimal DS config
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 2},
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

with open("/tmp/ds_debug.json", "w") as f:
    json.dump(ds_config, f)

# Load
print("=== Load model ===")
model_path = "/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# DeepSpeed init
engine, optimizer, _, _ = deepspeed.initialize(
    model=model, config="/tmp/ds_debug.json"
)
inner = engine.module
print(f"DeepSpeed engine ready, ZeRO stage={engine.zero_optimization_stage()}")

# Dummy batch
texts = ["Hello world this is a test.", "The quick brown fox jumps over."]
enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(engine.device)
labels = enc["input_ids"].clone()

import sys
sys.path.insert(0, ".")
from pmp.count_sketch import CountSketchProjector
sketcher = CountSketchProjector(sketch_dim=8192, seed=42)

# Test 1: engine forward + engine.backward
print("\n=== Test 1: engine.forward + engine.backward ===")
engine.zero_grad()
outputs = engine(**enc, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")
engine.backward(loss)

n_grads_engine = sum(1 for p in inner.parameters() if p.requires_grad and p.grad is not None)
n_total = sum(1 for p in inner.parameters() if p.requires_grad)
if n_grads_engine > 0:
    grad_norm = sum(p.grad.norm().item() for p in inner.parameters() if p.requires_grad and p.grad is not None)
else:
    grad_norm = 0
print(f"inner.parameters() .grad: {n_grads_engine}/{n_total}, norm={grad_norm:.4f}")

s1 = sketcher.sketch_grad(inner)
print(f"Sketch(inner): norm={s1.norm().item():.6f}")

# Also check engine.parameters()
n_grads_eng_params = sum(1 for p in engine.parameters() if p.requires_grad and p.grad is not None)
print(f"engine.parameters() .grad: {n_grads_eng_params}/{n_total}")
s1e = sketcher.sketch_grad(engine)
print(f"Sketch(engine): norm={s1e.norm().item():.6f}")

# Test 2: engine forward + loss.backward (standard PyTorch)
print("\n=== Test 2: engine.forward + loss.backward (standard) ===")
engine.zero_grad()
outputs2 = engine(**enc, labels=labels)
loss2 = outputs2.loss
loss2.backward()

n_grads2 = sum(1 for p in inner.parameters() if p.requires_grad and p.grad is not None)
if n_grads2 > 0:
    grad_norm2 = sum(p.grad.norm().item() for p in inner.parameters() if p.requires_grad and p.grad is not None)
else:
    grad_norm2 = 0
print(f"inner .grad after loss.backward(): {n_grads2}/{n_total}, norm={grad_norm2:.4f}")

s2 = sketcher.sketch_grad(inner)
print(f"Sketch(inner) after loss.backward(): norm={s2.norm().item():.6f}")

# Test 3: eval mode
print("\n=== Test 3: eval mode + loss.backward ===")
engine.eval()
engine.zero_grad()
loss_fn = nn.CrossEntropyLoss()
logits = engine(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
loss3 = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
loss3.backward()

n_grads3 = sum(1 for p in inner.parameters() if p.requires_grad and p.grad is not None)
if n_grads3 > 0:
    grad_norm3 = sum(p.grad.norm().item() for p in inner.parameters() if p.requires_grad and p.grad is not None)
else:
    grad_norm3 = 0
print(f"inner .grad (eval + loss.backward): {n_grads3}/{n_total}, norm={grad_norm3:.4f}")

s3 = sketcher.sketch_grad(inner)
print(f"Sketch(inner, eval): norm={s3.norm().item():.6f}")

# Test 4: inner model directly (bypass DeepSpeed)
print("\n=== Test 4: inner model directly (bypass DS) ===")
inner.zero_grad()
logits4 = inner(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
loss4 = loss_fn(logits4.view(-1, logits4.size(-1)), labels.view(-1))
loss4.backward()

n_grads4 = sum(1 for p in inner.parameters() if p.requires_grad and p.grad is not None)
if n_grads4 > 0:
    grad_norm4 = sum(p.grad.norm().item() for p in inner.parameters() if p.requires_grad and p.grad is not None)
else:
    grad_norm4 = 0
print(f"inner .grad (direct): {n_grads4}/{n_total}, norm={grad_norm4:.4f}")

s4 = sketcher.sketch_grad(inner)
print(f"Sketch(inner, direct): norm={s4.norm().item():.6f}")

print("\n=== DONE ===")
