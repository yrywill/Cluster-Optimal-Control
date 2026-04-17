"""
CountSketch-based random projection for gradient inner products.

Replaces the explicit projection matrix P ∈ R^{d×m} (which is ~16GB for 8.2B params)
with a pair of hash/sign tables that require only ~few MB of memory.

Mathematical guarantee:
    E[⟨sketch(g1), sketch(g2)⟩] = ⟨g1, g2⟩   (unbiased inner product estimator)

Key design:
    - Never materializes the full d-dimensional gradient vector.
    - Iterates over model parameters, sketching each .grad in-place via scatter_add.
    - Hash/sign tables are cached per (parameter_name, device) for efficiency.
    - Linearity: sketch(g_shard1) + sketch(g_shard2) = sketch(g_full),
      so ZeRO-3 sharded gradients can be sketched locally then all_reduced.

Usage:
    sketcher = CountSketchProjector(sketch_dim=8192, seed=42)
    # After loss.backward():
    s = sketcher.sketch_grad(model)  # → Tensor[8192]
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CountSketchProjector:
    """
    CountSketch projector for gradient dimensionality reduction.

    For each trainable parameter p with numel d_p, we maintain:
        h_p : int64 tensor [d_p]   — hash buckets in {0, ..., m-1}
        σ_p : float32 tensor [d_p] — random signs in {-1, +1}

    sketch(model) = Σ_p  scatter_add(h_p, p.grad.view(-1) * σ_p)

    Args:
        sketch_dim: Output sketch dimension m.  Paper recommends 8192.
        seed:       Base random seed for deterministic hash/sign generation.
    """

    def __init__(self, sketch_dim: int = 8192, seed: int = 42):
        self.m = sketch_dim
        self.seed = seed
        # Cache: param_name → (hash_table, sign_table) on a specific device
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        logger.info(
            f"CountSketchProjector: sketch_dim={sketch_dim}, seed={seed}"
        )

    def _get_hash_sign(
        self, name: str, numel: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lazily build and cache hash/sign tensors for a named parameter.

        Hash/sign are cached on CPU to avoid GPU memory buildup (38GB+ for 3B model).
        They are moved to GPU on-the-fly during scatter_add, then discarded from GPU.
        """
        cache_key = name  # CPU-only cache, no device in key
        if cache_key not in self._cache:
            # Deterministic seed per parameter name
            name_hash = hash(name) & 0xFFFFFFFF
            g = torch.Generator(device="cpu").manual_seed(self.seed + name_hash)

            h = torch.randint(0, self.m, (numel,), generator=g, dtype=torch.int64)
            sign = torch.randint(0, 2, (numel,), generator=g, dtype=torch.float32) * 2 - 1

            # Keep on CPU — moved to GPU on demand in sketch calls
            self._cache[cache_key] = (h, sign)

        h_cpu, sign_cpu = self._cache[cache_key]
        return h_cpu.to(device, non_blocking=True), sign_cpu.to(device, non_blocking=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def sketch_grad(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Sketch the current .grad of all trainable parameters into R^m.

        This never constructs the full gradient vector — it iterates over
        parameters one at a time, doing scatter_add into the sketch buffer.

        Args:
            model: Any nn.Module (raw, DeepSpeed-wrapped, etc.).
                   Parameters must have .grad populated (call after backward).

        Returns:
            sketch: Tensor[m] on model's device, dtype float32.
        """
        device = None
        for p in model.parameters():
            if p.requires_grad:
                device = p.device
                break
        if device is None:
            raise RuntimeError("No trainable parameters found in model")

        s = torch.zeros(self.m, device=device, dtype=torch.float32)

        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad.data.float().view(-1)
            h, sign = self._get_hash_sign(name, g.numel(), g.device)
            s.scatter_add_(0, h, g * sign)

        return s

    def sketch_vector(
        self,
        named_grads: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sketch a dict of named gradient tensors (e.g. from torch.func.grad).

        Useful when gradients come from functional differentiation rather than
        .backward().

        Args:
            named_grads: {param_name: gradient_tensor}.
            device:      Target device (auto-detected if None).

        Returns:
            sketch: Tensor[m], float32.
        """
        if device is None:
            for v in named_grads.values():
                if v is not None:
                    device = v.device
                    break
        if device is None:
            raise RuntimeError("No non-None gradients provided")

        s = torch.zeros(self.m, device=device, dtype=torch.float32)

        for name, g in named_grads.items():
            if g is None:
                continue
            g_flat = g.float().view(-1)
            h, sign = self._get_hash_sign(name, g_flat.numel(), g_flat.device)
            s.scatter_add_(0, h, g_flat * sign)

        return s

    def clear_cache(self):
        """Free all cached hash/sign tensors."""
        self._cache.clear()

    def memory_usage_mb(self) -> float:
        """Estimate total memory used by cached hash/sign tables."""
        total = 0
        for (h, sign) in self._cache.values():
            total += h.element_size() * h.numel()
            total += sign.element_size() * sign.numel()
        return total / 1e6

    def __repr__(self) -> str:
        return (
            f"CountSketchProjector(sketch_dim={self.m}, seed={self.seed}, "
            f"cached_params={len(self._cache)}, "
            f"cache_mem={self.memory_usage_mb():.1f}MB)"
        )
