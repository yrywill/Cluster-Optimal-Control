"""
Debug script: single-GPU test of CountSketch PMP gradient computation.
Tests whether model.backward() properly populates .grad for sketch_grad().
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model
print("=== Step 1: Load model ===")
model_path = "/apdcephfs_jn4/share_304380933/rongyiyu/code/llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# 2. Create dummy batch
print("\n=== Step 2: Create dummy batch ===")
texts = ["Hello world this is a test.", "The quick brown fox jumps over the lazy dog."]
enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
labels = enc["input_ids"].clone()
print(f"Batch: {enc['input_ids'].shape}")

# 3. Forward + backward (standard PyTorch)
print("\n=== Step 3: Standard forward + backward ===")
model.train()
model.zero_grad()
outputs = model(**enc, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")
loss.backward()

n_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
n_total = sum(1 for p in model.parameters() if p.requires_grad)
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
print(f"Gradients: {n_grads}/{n_total} params have .grad")
print(f"Total grad norm: {grad_norm:.6f}")

# 4. Test CountSketch
print("\n=== Step 4: CountSketch ===")
import sys
sys.path.insert(0, ".")
from pmp.count_sketch import CountSketchProjector

sketcher = CountSketchProjector(sketch_dim=8192, seed=42)
s = sketcher.sketch_grad(model)
print(f"Sketch: shape={s.shape}, norm={s.norm().item():.6f}, min={s.min().item():.6f}, max={s.max().item():.6f}")

# 5. Test in eval mode
print("\n=== Step 5: Eval mode forward + backward ===")
model.eval()
model.zero_grad()
outputs = model(**enc, labels=labels)
loss = outputs.loss
print(f"Loss (eval): {loss.item():.4f}")
loss.backward()

n_grads_eval = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
grad_norm_eval = sum(p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
print(f"Gradients (eval): {n_grads_eval}/{n_total} params have .grad")
print(f"Total grad norm (eval): {grad_norm_eval:.6f}")

s_eval = sketcher.sketch_grad(model)
print(f"Sketch (eval): norm={s_eval.norm().item():.6f}")

# 6. Test with manual loss (like our PMP code does)
print("\n=== Step 6: Manual loss computation (like PMP) ===")
model.eval()
model.zero_grad()
loss_fn = nn.CrossEntropyLoss(reduction="none")
logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
losses = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.shape)
loss_manual = losses.mean()
print(f"Loss (manual): {loss_manual.item():.4f}")
loss_manual.backward()

n_grads_manual = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
grad_norm_manual = sum(p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
print(f"Gradients (manual): {n_grads_manual}/{n_total} params have .grad")
print(f"Total grad norm (manual): {grad_norm_manual:.6f}")

s_manual = sketcher.sketch_grad(model)
print(f"Sketch (manual): norm={s_manual.norm().item():.6f}")

# 7. Test inner product
print("\n=== Step 7: Inner product between two sketches ===")
model.zero_grad()
logits1 = model(input_ids=enc["input_ids"][:1], attention_mask=enc["attention_mask"][:1]).logits
loss1 = loss_fn(logits1.view(-1, logits1.size(-1)), labels[:1].view(-1)).mean()
loss1.backward()
s1 = sketcher.sketch_grad(model)

model.zero_grad()
logits2 = model(input_ids=enc["input_ids"][1:], attention_mask=enc["attention_mask"][1:]).logits
loss2 = loss_fn(logits2.view(-1, logits2.size(-1)), labels[1:].view(-1)).mean()
loss2.backward()
s2 = sketcher.sketch_grad(model)

ip = torch.dot(s1, s2)
print(f"Sketch 1 norm: {s1.norm().item():.6f}")
print(f"Sketch 2 norm: {s2.norm().item():.6f}")
print(f"Inner product: {ip.item():.6f}")

print("\n=== ALL TESTS PASSED ===")
