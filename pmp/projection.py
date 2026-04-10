"""
Random projection utilities for gradient dimensionality reduction.

Two projection types:
  - "rademacher": P[i,j] ∈ {+1, -1} / sqrt(proj_dim)   (JL-style)
  - "gaussian":   P[i,j] ~ N(0, 1/proj_dim)
  - "identity":   no projection (proj_dim == param_dim)

The projection matrix is stored in fp16 on CPU to minimise memory.
It is lazily moved to the target device on first use.

Usage:
    projector = GradProjector(param_dim=1e8, proj_dim=1024, ...)
    projected = projector.project_vector(grad_vec)   # [proj_dim]
"""
from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradProjector:
    """
    Projects a flat gradient vector from param_dim → proj_dim.

    Args:
        param_dim: Total number of trainable parameters.
        proj_dim:  Target projection dimension.
        proj_type: 'rademacher' | 'gaussian' | 'identity'.
        seed:      Random seed for reproducible matrix.
        device:    Computation device.
    """

    def __init__(
        self,
        param_dim: int,
        proj_dim: int,
        proj_type: str,
        seed: int,
        device: torch.device,
    ):
        self.param_dim = int(param_dim)
        self.proj_dim = int(proj_dim)
        self.proj_type = proj_type.lower()
        self.seed = seed
        self.device = device

        self._P: torch.Tensor | None = None  # lazy init

        if self.proj_type == "identity":
            logger.info("GradProjector: identity mode (no projection)")
        else:
            logger.info(
                f"GradProjector: type={proj_type}, "
                f"param_dim={param_dim}, proj_dim={proj_dim}"
            )

    def _build_matrix(self):
        """Build and cache projection matrix on CPU (fp16 to save RAM)."""
        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.proj_type == "rademacher":
            # {-1, +1} / sqrt(proj_dim)
            P = (
                torch.randint(0, 2, (self.param_dim, self.proj_dim), generator=g)
                .float()
                .mul_(2)
                .add_(-1)
                .div_(self.proj_dim ** 0.5)
            )
        elif self.proj_type == "gaussian":
            P = torch.randn(
                self.param_dim, self.proj_dim, generator=g
            ).div_(self.proj_dim ** 0.5)
        elif self.proj_type == "identity":
            assert self.param_dim == self.proj_dim, (
                "Identity projection requires param_dim == proj_dim"
            )
            P = torch.eye(self.param_dim)
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        # Store in fp16 on CPU to save ~2x memory vs fp32
        self._P = P.half().cpu()
        logger.info(
            f"Projection matrix built: {self._P.shape}, "
            f"dtype={self._P.dtype}, "
            f"mem={self._P.element_size() * self._P.numel() / 1e6:.1f} MB"
        )

    def project_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Project a flat gradient vector.

        Args:
            vec: Tensor of shape [param_dim], any dtype/device.
        Returns:
            Tensor of shape [proj_dim], float32, on the same device as vec.
        """
        if self.proj_type == "identity":
            return vec.float()

        if self._P is None:
            self._build_matrix()

        target_device = vec.device
        P = self._P.to(target_device).float()  # [param_dim, proj_dim]
        v = vec.float().view(-1)

        if v.shape[0] != self.param_dim:
            # If model has fewer trainable params than expected (e.g. frozen layers),
            # pad or truncate
            if v.shape[0] < self.param_dim:
                pad = torch.zeros(
                    self.param_dim - v.shape[0], device=target_device, dtype=torch.float32
                )
                v = torch.cat([v, pad])
            else:
                v = v[: self.param_dim]

        return torch.mv(P.T, v)  # [proj_dim]

    def project_grad_dict(
        self, grad_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Project a gradient dict (param_name → grad tensor) to proj_dim.

        Args:
            grad_dict: Dict of gradient tensors.
        Returns:
            Tensor of shape [proj_dim].
        """
        parts = [g.detach().view(-1).float() for g in grad_dict.values() if g is not None]
        if not parts:
            return torch.zeros(self.proj_dim)
        vec = torch.cat(parts)
        return self.project_vector(vec)

class GhostGradProjector(GradProjector):
    """
    Ghost Projection variant: selectively zero-masks parameters before projection.
    
    Instead of projecting the full gradient g, projects a masked version:
    g_ghost = mask ⊙ g  (element-wise product)
    
    Three masking strategies:
    1. "layerwise":  Zero-mask entire layers (e.g., attention layers)
                     Useful when you know which layer types matter most.
    2. "random":     Randomly mask a fraction of parameters
                     Quick baseline, useful for ablation studies.
    3. "frequency":  Mask by parameter update frequency during training
                     Self-adaptive: keeps "active" parameters.
    
    Args:
        param_dim:        Total number of parameters.
        proj_dim:         Target projection dimension.
        proj_type:        'rademacher' | 'gaussian' | 'identity'.
        seed:             Random seed.
        device:           Computation device.
        ghost_strategy:   'layerwise' | 'random' | 'frequency'.
        ghost_fraction:   For 'random' and 'frequency': fraction of params to KEEP (1.0 - mask_fraction).
                         Default 0.5 (keep 50%, mask 50%).
        layer_indices:    For 'layerwise': list of layer indices to KEEP.
                         Other layers will be masked. Example: [0, 2, 4] keeps layers 0, 2, 4.
    """

    def __init__(
        self,
        param_dim: int,
        proj_dim: int,
        proj_type: str,
        seed: int,
        device: torch.device,
        ghost_strategy: str = "layerwise",
        ghost_fraction: float = 0.5,
        layer_indices: list | None = None,
        num_layers: int | None = None,
    ):
        super().__init__(param_dim, proj_dim, proj_type, seed, device)
        
        self.ghost_strategy = ghost_strategy.lower()
        self.ghost_fraction = ghost_fraction
        self.layer_indices = layer_indices or []
        self.num_layers = num_layers
        
        if self.ghost_strategy not in ["layerwise", "random", "frequency"]:
            raise ValueError(f"Unknown ghost_strategy: {self.ghost_strategy}")
        
        self._mask: torch.Tensor | None = None
        self._update_frequency: torch.Tensor | None = None
        
        logger.info(
            f"GhostGradProjector: strategy={ghost_strategy}, "
            f"fraction={ghost_fraction}, "
            f"param_dim={param_dim}, proj_dim={proj_dim}"
        )

    def _build_mask_layerwise(self) -> torch.Tensor:
        """
        Build layerwise mask: keep specified layers, zero others.
        
        Assumes parameters are organized into contiguous layer blocks.
        layer_size = param_dim / num_layers
        
        Returns:
            mask ∈ {0, 1}^[param_dim]
        """
        if self.num_layers is None:
            raise ValueError(
                "num_layers required for layerwise ghost strategy"
            )
        
        mask = torch.zeros(self.param_dim, dtype=torch.float32)
        layer_size = self.param_dim // self.num_layers
        remainder = self.param_dim % self.num_layers
        
        for layer_idx in self.layer_indices:
            if layer_idx >= self.num_layers:
                logger.warning(f"Layer index {layer_idx} >= num_layers {self.num_layers}, skipping")
                continue
            
            start = layer_idx * layer_size
            end = start + layer_size
            if layer_idx < remainder:
                end += 1
            
            mask[start:end] = 1.0
        
        logger.debug(
            f"Layerwise mask: keeping {mask.sum():.0f}/{self.param_dim} params "
            f"from layers {self.layer_indices}"
        )
        return mask

    def _build_mask_random(self, seed_offset: int = 0) -> torch.Tensor:
        """
        Build random mask: keep ghost_fraction of parameters randomly.
        
        Args:
            seed_offset: Additional seed offset (for per-sample variations).
        
        Returns:
            mask ∈ {0, 1}^[param_dim]
        """
        g = torch.Generator()
        g.manual_seed(self.seed + seed_offset)
        
        keep_count = int(self.param_dim * self.ghost_fraction)
        mask = torch.zeros(self.param_dim, dtype=torch.float32)
        
        # Random permutation of indices
        perm = torch.randperm(self.param_dim, generator=g)
        mask[perm[:keep_count]] = 1.0
        
        logger.debug(
            f"Random mask: keeping {keep_count}/{self.param_dim} params "
            f"(fraction={self.ghost_fraction})"
        )
        return mask

    def _build_mask_frequency(self) -> torch.Tensor:
        """
        Build frequency-based mask: keep parameters updated most frequently.
        
        Uses _update_frequency if available (set via update_frequency()).
        Falls back to random mask if frequency data not available.
        
        Returns:
            mask ∈ {0, 1}^[param_dim]
        """
        if self._update_frequency is None:
            logger.warning(
                "Frequency mask requested but no update frequency data. "
                "Falling back to random mask."
            )
            return self._build_mask_random()
        
        # Threshold: keep top ghost_fraction of parameters by frequency
        keep_count = int(self.param_dim * self.ghost_fraction)
        _, top_indices = torch.topk(self._update_frequency, keep_count)
        
        mask = torch.zeros(self.param_dim, dtype=torch.float32)
        mask[top_indices] = 1.0
        
        logger.debug(
            f"Frequency mask: keeping {keep_count}/{self.param_dim} params "
            f"(fraction={self.ghost_fraction})"
        )
        return mask

    def build_mask(self, seed_offset: int = 0) -> torch.Tensor:
        """
        Build and cache the ghost mask based on strategy.
        
        Args:
            seed_offset: For 'random' strategy, allows different masks per sample.
        
        Returns:
            mask ∈ {0, 1}^[param_dim], on CPU fp32.
        """
        if self.ghost_strategy == "layerwise":
            self._mask = self._build_mask_layerwise()
        elif self.ghost_strategy == "random":
            self._mask = self._build_mask_random(seed_offset=seed_offset)
        elif self.ghost_strategy == "frequency":
            self._mask = self._build_mask_frequency()
        
        return self._mask.clone()

    def update_frequency(self, param_grads: Dict[str, torch.Tensor]):
        """
        Update parameter frequency statistics from gradient dict.
        
        Called during training to accumulate which parameters are most active.
        Used by 'frequency' strategy to adaptively mask inactive parameters.
        
        Args:
            param_grads: Dict of gradient tensors {name: grad_tensor}.
        """
        # Flatten gradients into single vector
        parts = [
            (g.detach().view(-1).float() != 0).float()
            for g in param_grads.values()
            if g is not None
        ]
        if not parts:
            return
        
        grad_active = torch.cat(parts)
        
        # Truncate/pad to param_dim
        if grad_active.shape[0] < self.param_dim:
            pad = torch.zeros(
                self.param_dim - grad_active.shape[0], dtype=torch.float32
            )
            grad_active = torch.cat([grad_active, pad])
        else:
            grad_active = grad_active[: self.param_dim]
        
        # Accumulate: keep count of how many times each param was non-zero
        if self._update_frequency is None:
            self._update_frequency = grad_active.clone()
        else:
            self._update_frequency = self._update_frequency + grad_active

    def ghost_project_vector(
        self, vec: torch.Tensor, seed_offset: int = 0
    ) -> torch.Tensor:
        """
        Project a gradient vector with ghost masking.
        
        Computes: proj(mask ⊙ vec)
        
        Args:
            vec:           Gradient vector [param_dim].
            seed_offset:   For 'random' strategy, allows per-sample different masks.
        
        Returns:
            Projected and masked gradient [proj_dim].
        """
        # Build/rebuild mask if needed (for random strategy, rebuild each time)
        if self.ghost_strategy == "random":
            mask = self._build_mask_random(seed_offset=seed_offset)
        else:
            if self._mask is None:
                self.build_mask(seed_offset=seed_offset)
            mask = self._mask
        
        # Apply mask: zero out non-selected parameters
        target_device = vec.device
        mask = mask.to(target_device)
        vec_masked = vec * mask
        
        # Project masked gradient
        return self.project_vector(vec_masked)

    def ghost_project_grad_dict(
        self, grad_dict: Dict[str, torch.Tensor], seed_offset: int = 0
    ) -> torch.Tensor:
        """
        Project a gradient dict with ghost masking.
        
        Args:
            grad_dict:  Dict of gradient tensors.
            seed_offset: For 'random' strategy, allows per-sample different masks.
        
        Returns:
            Projected and masked gradient [proj_dim].
        """
        parts = [g.detach().view(-1).float() for g in grad_dict.values() if g is not None]
        if not parts:
            return torch.zeros(self.proj_dim)
        vec = torch.cat(parts)
        return self.ghost_project_vector(vec, seed_offset=seed_offset)
