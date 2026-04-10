"""
CountSketch-based cluster contribution computation for PMP backward.

Replaces compute_cluster_contributions_ghost_ip() with a version that:
  1. Never materialises the full d-dimensional gradient vector.
  2. Uses CountSketch (hash + sign) instead of explicit projection matrices.
  3. Works natively with ZeRO-3 sharded gradients (sketch is linear → all_reduce).

Flow:
  1. Forward + backward on dev set → sketch dev gradient  q = sketch(∇L_dev)
  2. For each cluster k in the training batch:
       forward + backward → sketch train gradient  v_k = sketch(∇L_k)
       ct_k = pmp_lr * <q, v_k>
  3. Return grad_gamma_delta[k] += ct_k
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from .count_sketch import CountSketchProjector

logger = logging.getLogger(__name__)


def compute_cluster_contributions_sketch(
    model: nn.Module,
    dev_batches: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    batch: Dict[str, torch.Tensor],
    batch_cluster_ids: torch.Tensor,
    n_clusters: int,
    pmp_lr: float,
    sketcher: CountSketchProjector,
    world_size: int = 1,
    distributed: bool = False,
) -> torch.Tensor:
    """
    CountSketch fast path for PMP cluster contribution update.

    Approximates:
        ct_k ≈ pmp_lr · ⟨sketch(∇L_dev(θ)), sketch(mean_{n∈C_k} ∇loss_n(θ))⟩

    Unlike the Ghost IP path, this function:
      - Does NOT require GatheredParameters (ZeRO-3 compatible via all_reduce).
      - Does NOT build a [param_dim, proj_dim] matrix.
      - Does NOT flatten all gradients into a single vector.

    Args:
        model:             Any nn.Module (DeepSpeed engine, DDP, or raw).
        dev_batches:       List of (model_batch, no_model_batch) on device.
                           model_batch has "input_ids", "attention_mask".
                           no_model_batch has "label", "loss_mask".
        batch:             Training batch dict with "input_ids", "attention_mask",
                           "label", "loss_mask".
        batch_cluster_ids: Tensor [B] mapping each sample to its cluster id.
        n_clusters:        Total number of clusters K.
        pmp_lr:            PMP learning rate factor.
        sketcher:          CountSketchProjector instance.
        world_size:        Number of distributed ranks.
        distributed:       Whether to all_reduce sketch vectors across ranks.

    Returns:
        grad_gamma_delta: Tensor [n_clusters] on the same device as batch.
    """
    device = batch["input_ids"].device
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    was_training = model.training
    model.eval()  # BN / dropout → eval mode for gradient computation

    # Free training activations before PMP
    torch.cuda.empty_cache()

    # ==================================================================
    # Step 1: Sketch dev gradient  q = sketch(∇L_dev)
    # Accumulate sketch across dev batches (NOT computational graph).
    # Each batch: forward → backward → sketch → zero_grad → next batch.
    # This keeps memory flat (only 1 batch of activations at a time).
    # ==================================================================
    q = torch.zeros(sketcher.m, device=device, dtype=torch.float32)
    n_dev = 0

    for model_batch, no_model_batch in dev_batches:
        model.zero_grad()

        logits = model(
            input_ids=model_batch["input_ids"],
            attention_mask=model_batch["attention_mask"],
        ).logits

        losses = loss_fn(
            logits.view(-1, logits.size(-1)),
            no_model_batch["label"].view(-1),
        ).view(no_model_batch["label"].shape)

        loss = (
            (losses * no_model_batch["loss_mask"]).sum()
            / no_model_batch["loss_mask"].sum().clamp(min=1)
        )
        loss.backward()

        # Sketch this batch's gradient and accumulate into q
        q += sketcher.sketch_grad(model)
        n_dev += 1

    model.zero_grad()

    # Average over dev batches
    if n_dev > 1:
        q = q / n_dev

    # Under ZeRO-3, each rank has sketched its shard's gradient.
    # all_reduce(SUM) recovers the full sketch thanks to linearity.
    if distributed:
        dist.all_reduce(q, op=dist.ReduceOp.SUM)
        q = q / world_size

    logger.debug(f"[Sketch] dev sketch: norm={q.norm():.4f}, n_dev_batches={n_dev}")

    # ==================================================================
    # Step 2: Per-cluster sketch + inner product
    # ==================================================================
    grad_gamma_delta = torch.zeros(n_clusters, device=device, dtype=torch.float32)
    unique_clusters = batch_cluster_ids.unique().tolist()

    for k in unique_clusters:
        k = int(k)
        mask = batch_cluster_ids == k
        n_samples = mask.sum().item()
        if n_samples == 0:
            continue

        model.zero_grad()

        c_input_ids = batch["input_ids"][mask]
        c_attn = batch["attention_mask"][mask]
        c_labels = batch["label"][mask]
        c_loss_mask = batch["loss_mask"][mask]

        logits = model(
            input_ids=c_input_ids,
            attention_mask=c_attn,
        ).logits

        losses = loss_fn(
            logits.view(-1, logits.size(-1)),
            c_labels.view(-1),
        ).view(c_labels.shape)

        loss_k = (
            (losses * c_loss_mask).sum()
            / c_loss_mask.sum().clamp(min=1)
        )
        loss_k.backward()

        v_k = sketcher.sketch_grad(model)  # [m]

        # all_reduce for ZeRO-3
        if distributed:
            dist.all_reduce(v_k, op=dist.ReduceOp.SUM)
            v_k = v_k / world_size

        ct_k = torch.dot(q, v_k)  # scalar
        grad_gamma_delta[k] = grad_gamma_delta[k] + pmp_lr * ct_k

    if was_training:
        model.train()

    model.zero_grad()  # clean up leftover grads

    logger.debug(
        f"[Sketch] grad_gamma_delta: norm={grad_gamma_delta.norm():.4f}, "
        f"clusters_updated={len(unique_clusters)}"
    )
    return grad_gamma_delta
