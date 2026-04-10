"""
Gradient utilities for cluster-based PMP backward pass.

Key differences from the original LMOps implementation:
  1. cluster_jvp_batch(): returns a scalar ct_k = mean over samples in a cluster
     (instead of per-sample). Used to accumulate grad_gamma per cluster.
  2. compute_dev_grad(): computes ∇L_dev(θ) as a flat vector.
     Hessian term is intentionally omitted (zeroed out).
  3. No hvp_fwdrev() — the Hessian-vector product is set to zero,
     simplifying the λ update to: λ_t = ∇L_dev(θ_t) + λ_{t+1}

New utilities:
  4. compute_cluster_contributions_ghost_ip(): fast approximation using ghost
     inner product — replaces ring-buffer JVP with a single-step dot product.
  5. compute_dev_grad_multi_domain(): weighted dev gradient over multiple
     domain-specific validation sets.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.func import grad, grad_and_value, jvp, vmap

from .model_wrapper import TransformerWrapper

logger = logging.getLogger(__name__)


# ======================================================================
# Per-sample JVP (identical to original jvp_single)
# ======================================================================

def _jvp_single(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label: torch.Tensor,
    loss_mask: torch.Tensor,
    model: TransformerWrapper,
    lam_param: Dict[str, torch.Tensor],
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the JVP of loss_n w.r.t. params in direction lam_param.

    Returns a scalar: ⟨∇_θ loss_n(θ), λ⟩
    """
    def loss_func(p):
        return TransformerWrapper.compute_loss_func_single(
            p, buffers, model,
            input_ids, attention_mask, label, loss_mask,
        )

    _, ct = jvp(loss_func, (params,), (lam_param,))
    return ct


# ======================================================================
# Cluster-level JVP  (new: aggregated over all samples in a cluster)
# ======================================================================

def cluster_jvp_batch(
    model: TransformerWrapper,
    cluster_batch: Dict[str, torch.Tensor],
    lam_param: Dict[str, torch.Tensor],
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the cluster-level contribution:
        ct_k = mean_{n ∈ C_k} [ ⟨∇_θ loss_n(θ), λ⟩ ]

    Args:
        model:         TransformerWrapper.
        cluster_batch: Batch dict for ONE cluster
                       {"input_ids", "attention_mask", "label", "loss_mask"}
                       all of shape [cluster_bs, seq_len].
        lam_param:     λ as a named parameter dict (tangent direction).
        params:        Current model parameters as named dict.
        buffers:       Model buffers.
        chunk_size:    vmap chunk size (None = full batch; reduce if OOM).

    Returns:
        Scalar tensor: mean JVP value across all samples in the cluster.
    """
    per_sample_ct = vmap(
        _jvp_single,
        in_dims=(0, 0, 0, 0, None, None, None, None),
        chunk_size=chunk_size,
    )(
        cluster_batch["input_ids"],
        cluster_batch["attention_mask"],
        cluster_batch["label"],
        cluster_batch["loss_mask"],
        model,
        lam_param,
        params,
        buffers,
    )
    return per_sample_ct.mean()


# ======================================================================
# Dev-set gradient  (∇L_dev(θ))
# ======================================================================

def compute_dev_grad(
    model: TransformerWrapper,
    dev_batches: List[Tuple[Dict, Dict]],
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    world_size: int = 1,
    distributed: bool = False,
) -> torch.Tensor:
    """
    Compute the gradient of the dev loss w.r.t. θ as a flat parameter vector.
    Average is taken over all dev batches.

    Args:
        model:       TransformerWrapper.
        dev_batches: List of (model_batch, no_model_batch) tuples already on device.
        params:      Current model params as named dict (from vector_to_params).
        buffers:     Model buffers.
        world_size:  Number of distributed ranks (for all_reduce normalisation).
        distributed: Whether to all_reduce across ranks.

    Returns:
        g_dev_vec: Flat tensor of shape [param_dim].
    """
    g_dev_total = None
    n_batches = 0

    for model_batch, no_model_batch in dev_batches:
        g, loss_val = grad_and_value(TransformerWrapper.compute_loss_func)(
            params, buffers, model,
            **model_batch,
            **no_model_batch,
        )
        g_vec = model.params_to_vector(g)

        if distributed:
            dist.all_reduce(g_vec, op=dist.ReduceOp.SUM)
            g_vec = g_vec / world_size

        if g_dev_total is None:
            g_dev_total = g_vec
        else:
            g_dev_total = g_dev_total + g_vec

        n_batches += 1

    if g_dev_total is None:
        # No dev batches — return zero gradient
        param_dim = sum(p.numel() for _, p in model.named_parameters())
        return torch.zeros(param_dim, device=next(model.parameters()).device)

    return g_dev_total / n_batches


# ======================================================================
# Cluster contribution accumulator
# ======================================================================

def compute_cluster_contributions(
    model: TransformerWrapper,
    batch: Dict[str, torch.Tensor],
    batch_cluster_ids: torch.Tensor,
    lam_param: Dict[str, torch.Tensor],
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    n_clusters: int,
    pmp_lr: float,
    chunk_size: Optional[int] = None,
    distributed: bool = False,
    world_size: int = 1,
) -> torch.Tensor:
    """
    For all clusters represented in `batch`, compute their PMP contribution
    and return a grad_gamma_delta vector of shape [n_clusters].

    This is called once per step in the PMP backward loop.

    Args:
        batch:             Combined dict with "input_ids","attention_mask","label","loss_mask".
        batch_cluster_ids: Tensor [B] mapping each sample in batch to its cluster id.
        lam_param:         λ as named parameter dict.
        n_clusters:        Total number of clusters K.
        pmp_lr:            lr factor (scales the contribution).
        chunk_size:        vmap chunk size.
        distributed:       Whether to all_reduce contributions.
        world_size:        Number of distributed ranks.

    Returns:
        grad_gamma_delta: Tensor [n_clusters], contribution to grad_gamma.
    """
    grad_gamma_delta = torch.zeros(
        n_clusters,
        device=batch["input_ids"].device,
        dtype=torch.float32,
    )

    unique_clusters = batch_cluster_ids.unique().tolist()

    for k in unique_clusters:
        k = int(k)
        mask = (batch_cluster_ids == k)
        if mask.sum() == 0:
            continue

        cluster_batch = {
            key: batch[key][mask] for key in ("input_ids", "attention_mask", "label", "loss_mask")
        }

        ct_k = cluster_jvp_batch(
            model, cluster_batch, lam_param, params, buffers, chunk_size=chunk_size
        )

        if distributed:
            dist.all_reduce(ct_k, op=dist.ReduceOp.SUM)
            ct_k = ct_k / world_size

        grad_gamma_delta[k] += pmp_lr * ct_k

    return grad_gamma_delta


# ======================================================================
# Dev-set gradient with Ghost projection (Method 2)
# ======================================================================

def compute_dev_grad_with_ghost(
    model: TransformerWrapper,
    dev_batches: List[Tuple[Dict, Dict]],
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    ghost_projector: Optional[object] = None,
    world_size: int = 1,
    distributed: bool = False,
) -> torch.Tensor:
    """
    Compute the gradient of the dev loss w.r.t. θ as a flat parameter vector,
    optionally with ghost projection masking applied.
    
    When ghost_projector is provided, applies masking: g_ghost = mask ⊙ g_dev
    This is Method 2 integration: apply ghost at the λ update phase.
    
    Args:
        model:           TransformerWrapper.
        dev_batches:     List of (model_batch, no_model_batch) tuples already on device.
        params:          Current model params as named dict.
        buffers:         Model buffers.
        ghost_projector: Optional GhostGradProjector for masking. If None, returns standard gradient.
        world_size:      Number of distributed ranks.
        distributed:     Whether to all_reduce across ranks.
    
    Returns:
        g_dev_vec: Flat tensor of shape [param_dim], with ghost masking applied if provided.
    """
    g_dev_total = None
    n_batches = 0
    
    for model_batch, no_model_batch in dev_batches:
        g, loss_val = grad_and_value(TransformerWrapper.compute_loss_func)(
            params, buffers, model,
            **model_batch,
            **no_model_batch,
        )
        g_vec = model.params_to_vector(g)
        
        # Apply ghost masking if provided
        if ghost_projector is not None:
            mask = ghost_projector.build_mask()
            target_device = g_vec.device
            mask = mask.to(target_device)
            g_vec = g_vec * mask  # Element-wise masking
        
        if distributed:
            dist.all_reduce(g_vec, op=dist.ReduceOp.SUM)
            g_vec = g_vec / world_size
        
        if g_dev_total is None:
            g_dev_total = g_vec
        else:
            g_dev_total = g_dev_total + g_vec
        
        n_batches += 1
    
    if g_dev_total is None:
        # No dev batches — return zero gradient
        param_dim = sum(p.numel() for _, p in model.named_parameters())
        return torch.zeros(param_dim, device=next(model.parameters()).device)

    return g_dev_total / n_batches


# ======================================================================
# Ghost Inner Product: fast cluster contribution (no ring buffer needed)
# ======================================================================

def compute_cluster_contributions_ghost_ip(
    model: TransformerWrapper,
    dev_batches: List[Tuple[Dict, Dict]],
    batch: Dict[str, torch.Tensor],
    batch_cluster_ids: torch.Tensor,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    n_clusters: int,
    pmp_lr: float,
    ghost_projector,
    world_size: int = 1,
    distributed: bool = False,
) -> torch.Tensor:
    """
    Ghost Inner Product fast path for cluster contribution update.

    Instead of traversing the ring buffer with JVP, approximates:
        ct_k ≈ pmp_lr · <ghost_proj(∇L_dev(θ)), ghost_proj(mean_{n∈C_k} ∇loss_n(θ))>

    This requires only two gradient computations at the current step plus a
    proj_dim-dimensional dot product per cluster, avoiding the full backward
    loop over the ring buffer window.

    Mathematical justification: by the Johnson-Lindenstrauss lemma, random
    projections approximately preserve inner products, so:
        <∇L_dev, ∑_{n∈C_k} ∇loss_n> ≈ <P^T ∇L_dev, P^T ∑_{n∈C_k} ∇loss_n>
    for a random projection matrix P with appropriate scaling.

    Args:
        model:            TransformerWrapper (used in eval mode, no grad).
        dev_batches:      List of (model_batch, no_model_batch) on device.
        batch:            Combined training batch on device (all clusters).
        batch_cluster_ids:Tensor [B] mapping samples to cluster indices.
        params:           Current model params as named dict.
        buffers:          Model buffers.
        n_clusters:       Total number of clusters K.
        pmp_lr:           PMP learning rate factor.
        ghost_projector:  GhostGradProjector instance (shared mask for
                          both dev and train gradients ensures same subspace).
        world_size:       Number of distributed ranks.
        distributed:      Whether to all_reduce contributions.

    Returns:
        grad_gamma_delta: Tensor [n_clusters].
    """
    device = batch["input_ids"].device

    # ---- Step 1: dev gradient, projected via ghost ----
    g_dev = compute_dev_grad(
        model, dev_batches, params, buffers,
        world_size=world_size, distributed=distributed,
    )
    # Build mask once and reuse for both dev and train projections.
    # Using a fixed mask guarantees both vectors lie in the same subspace.
    ghost_projector.build_mask()
    q = ghost_projector.ghost_project_vector(g_dev.to(device))  # [proj_dim]

    # ---- Step 2: per-cluster mean train gradient, projected via ghost ----
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    grad_gamma_delta = torch.zeros(n_clusters, device=device, dtype=torch.float32)
    unique_clusters = batch_cluster_ids.unique().tolist()

    # Temporarily set model to eval for gradient computation
    was_training = model.training
    model.eval()

    for k in unique_clusters:
        k = int(k)
        mask = batch_cluster_ids == k
        if mask.sum() == 0:
            continue

        cluster_input_ids = batch["input_ids"][mask]
        cluster_attn = batch["attention_mask"][mask]
        cluster_labels = batch["label"][mask]
        cluster_loss_mask = batch["loss_mask"][mask]

        # Compute mean gradient over cluster samples via a single forward-backward
        model.zero_grad()
        logits = model(
            input_ids=cluster_input_ids,
            attention_mask=cluster_attn,
        ).logits
        losses = loss_fn(
            logits.view(-1, logits.size(-1)),
            cluster_labels.view(-1),
        ).view(cluster_labels.shape)
        # Normalise by number of valid tokens (matching _extract_gradient_features)
        loss_k = (losses * cluster_loss_mask).sum() / cluster_loss_mask.sum().clamp(min=1)
        loss_k.backward()

        g_parts = [
            p.grad.detach().view(-1).float()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        model.zero_grad()

        if not g_parts:
            continue

        g_k = torch.cat(g_parts)  # [param_dim]
        v_k = ghost_projector.ghost_project_vector(g_k)  # [proj_dim]

        ct_k = torch.dot(q.to(v_k.device), v_k)  # scalar

        if distributed:
            dist.all_reduce(ct_k, op=dist.ReduceOp.SUM)
            ct_k = ct_k / world_size

        grad_gamma_delta[k] = grad_gamma_delta[k] + pmp_lr * ct_k

    if was_training:
        model.train()

    return grad_gamma_delta


# ======================================================================
# Multi-domain weighted dev gradient
# ======================================================================

def compute_dev_grad_multi_domain(
    model: TransformerWrapper,
    domain_batches: List[Tuple[str, float, List[Tuple[Dict, Dict]]]],
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    world_size: int = 1,
    distributed: bool = False,
) -> torch.Tensor:
    """
    Compute a weighted combination of dev gradients over multiple domains:

        ∇L_dev_weighted(θ) = Σ_d  (w_d / Σ_d w_d) · ∇L_dev_d(θ)

    This allows the PMP backward pass to be steered by multiple heterogeneous
    validation sets (e.g., math, code, general) each with independent weights.

    Args:
        model:         TransformerWrapper.
        domain_batches:List of (domain_name, weight, batches) where batches is
                       a list of (model_batch, no_model_batch) already on device.
        params:        Current model params as named dict.
        buffers:       Model buffers.
        world_size:    Number of distributed ranks.
        distributed:   Whether to all_reduce across ranks.

    Returns:
        g_weighted: Flat tensor [param_dim], weighted combination of domain gradients.
    """
    if not domain_batches:
        param_dim = sum(p.numel() for _, p in model.named_parameters())
        return torch.zeros(
            param_dim, device=next(model.parameters()).device, dtype=torch.float32
        )

    total_weight = sum(float(w) for _, w, _ in domain_batches)
    if total_weight <= 0:
        raise ValueError(
            "compute_dev_grad_multi_domain: total domain weight must be > 0, "
            f"got {total_weight}"
        )

    g_weighted: Optional[torch.Tensor] = None

    for domain_name, weight, batches in domain_batches:
        if not batches:
            logger.warning(f"compute_dev_grad_multi_domain: domain '{domain_name}' has no batches, skipping.")
            continue

        normalized_w = float(weight) / total_weight

        g_d = compute_dev_grad(
            model, batches, params, buffers,
            world_size=world_size,
            distributed=distributed,
        )

        if g_weighted is None:
            g_weighted = normalized_w * g_d
        else:
            g_weighted = g_weighted + normalized_w * g_d

        logger.debug(
            f"compute_dev_grad_multi_domain: domain='{domain_name}' "
            f"weight={normalized_w:.4f} grad_norm={g_d.norm():.4f}"
        )

    if g_weighted is None:
        param_dim = sum(p.numel() for _, p in model.named_parameters())
        return torch.zeros(
            param_dim, device=next(model.parameters()).device, dtype=torch.float32
        )

    return g_weighted
