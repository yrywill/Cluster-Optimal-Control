"""
CountSketch-based cluster contribution computation for PMP backward.

Uses torch.autograd.grad (not loss.backward) to bypass DeepSpeed/DDP gradient hooks.
Cluster samples are randomly drawn from the full dataset (not from ring buffer).

Flow:
  1. Forward + autograd.grad on dev set → sketch dev gradient  q = sketch(∇L_dev)
  2. For each cluster k: randomly sample min(10, cluster_size) from train_dataset
       forward + autograd.grad → sketch train gradient  v_k = sketch(∇L_k)
       ct_k = pmp_lr * <q, v_k>
  3. Return grad_gamma_delta[k] += ct_k
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from .count_sketch import CountSketchProjector

logger = logging.getLogger(__name__)


def compute_cluster_contributions_sketch(
    model: nn.Module,
    dev_batches: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    n_clusters: int,
    pmp_lr: float,
    sketcher: CountSketchProjector,
    train_dataset=None,
    cluster_ids_to_eval: List[int] = None,
    n_samples_per_cluster: int = 10,
    world_size: int = 1,
    distributed: bool = False,
) -> torch.Tensor:
    """
    CountSketch fast path for PMP cluster contribution update.

    Args:
        model:                  Raw nn.Module (NOT DeepSpeed engine).
        dev_batches:            List of (model_batch, no_model_batch) on device.
        n_clusters:             Total number of clusters K.
        pmp_lr:                 PMP learning rate factor.
        sketcher:               CountSketchProjector instance.
        train_dataset:          ClusterDataset with get_cluster_indices().
        cluster_ids_to_eval:    Which clusters to evaluate. If None, eval all clusters.
        n_samples_per_cluster:  Max samples per cluster (default 10).
        world_size:             Number of distributed ranks.
        distributed:            Whether to all_reduce sketch vectors.

    Returns:
        grad_gamma_delta: Tensor [n_clusters].
    """
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    was_training = model.training
    model.eval()
    torch.cuda.empty_cache()

    # Inner model (in case model is DDP-wrapped)
    inner_model = getattr(model, 'module', model)

    # Trainable params + names (for autograd.grad + sketch)
    trainable_params = [p for p in inner_model.parameters() if p.requires_grad]
    param_names = [n for n, p in inner_model.named_parameters() if p.requires_grad]

    def _sketch_loss(m, input_ids, attention_mask, labels, loss_mask):
        """Forward + autograd.grad → sketch. No .backward(), no parameter hooks.
        All intermediate tensors (logits, losses, grads) are explicitly deleted
        to free the computation graph and prevent memory buildup."""
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        losses = loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ).view(labels.shape)
        loss = (losses * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        # Compute grads — this is the only thing we need
        grads = torch.autograd.grad(loss, trainable_params, allow_unused=True)

        # Immediately delete forward tensors to release computation graph
        del logits, losses, loss

        # Sketch from grads, then delete grads
        s = torch.zeros(sketcher.m, device=device, dtype=torch.float32)
        for name, p, g in zip(param_names, trainable_params, grads):
            if g is None:
                continue
            h, sign = sketcher._get_hash_sign(name, g.numel(), device)
            s.scatter_add_(0, h, g.float().view(-1) * sign)

        # Delete grads tuple to release all autograd graph references
        del grads

        return s  # s is a plain tensor, no graph attached

    # Current rank for sharding (used both in dev and cluster loop)
    if distributed and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # ==================================================================
    # Step 1: Sketch dev gradient  q = sketch(∇L_dev)
    # Each rank processes a shard of dev_batches; all_reduce at end.
    # ==================================================================
    q = torch.zeros(sketcher.m, device=device, dtype=torch.float32)
    n_dev_local = 0
    total_dev = len(dev_batches)

    for i, (model_batch, no_model_batch) in enumerate(dev_batches):
        if distributed and (i % world_size) != rank:
            continue
        q += _sketch_loss(
            inner_model,
            model_batch["input_ids"], model_batch["attention_mask"],
            no_model_batch["label"], no_model_batch["loss_mask"],
        )
        n_dev_local += 1

    if distributed:
        dist.all_reduce(q, op=dist.ReduceOp.SUM)

    if total_dev > 1:
        q = q / max(total_dev, 1)

    logger.info(
        f"[Sketch] dev sketch done: norm={q.norm():.4f}, "
        f"n_dev_total={total_dev}, n_dev_local(rank{rank})={n_dev_local}"
    )

    # ==================================================================
    # Step 2: Per-cluster sketch from random samples
    # Each rank owns a shard of clusters: my_ids = ids[rank::world_size].
    # Final all_reduce(SUM) combines partial results (no double counting
    # because each k is only computed on one rank).
    # ==================================================================
    grad_gamma_delta = torch.zeros(n_clusters, device=device, dtype=torch.float32)

    if cluster_ids_to_eval is None:
        cluster_ids_to_eval = list(range(n_clusters))

    # Rank-sharded cluster iteration
    if distributed:
        my_cluster_ids = cluster_ids_to_eval[rank::world_size]
    else:
        my_cluster_ids = cluster_ids_to_eval

    n_local = 0
    for k in my_cluster_ids:
        # Get cluster sample indices
        cluster_indices = train_dataset.get_cluster_indices(k)
        if len(cluster_indices) == 0:
            continue

        # Random sample min(n_samples_per_cluster, cluster_size)
        n_take = min(n_samples_per_cluster, len(cluster_indices))
        sampled_indices = random.sample(list(cluster_indices), n_take)

        # Collect samples into a batch
        samples = [train_dataset[i] for i in sampled_indices]
        collated = train_dataset.collate(samples)
        if collated[0] is None:
            continue
        model_batch, no_model_batch = collated
        train_dataset.move_to_device(model_batch, no_model_batch, device)

        v_k = _sketch_loss(
            inner_model,
            model_batch["input_ids"], model_batch["attention_mask"],
            no_model_batch["label"], no_model_batch["loss_mask"],
        )

        ct_k = torch.dot(q, v_k)
        grad_gamma_delta[k] = grad_gamma_delta[k] + pmp_lr * ct_k
        n_local += 1

    # Combine per-rank contributions (each cluster k is computed on exactly
    # one rank, so SUM reconstructs the full grad_gamma_delta).
    if distributed:
        dist.all_reduce(grad_gamma_delta, op=dist.ReduceOp.SUM)

    if was_training:
        model.train()

    n_evaluated = sum(1 for k in cluster_ids_to_eval
                      if len(train_dataset.get_cluster_indices(k)) > 0)
    logger.info(
        f"[Sketch] PMP done: clusters_total={n_evaluated}, "
        f"local(rank{rank})={n_local}, "
        f"grad_gamma_delta norm={grad_gamma_delta.norm():.4f}"
    )
    return grad_gamma_delta
