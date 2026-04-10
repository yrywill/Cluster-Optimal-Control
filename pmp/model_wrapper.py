"""
TransformerWrapper: wraps a HuggingFace causal LM for functional differentiation.
Adapted from microsoft/LMOps/data_selection/pmp_solver/model_wrapper.py
with minor additions for cluster-based selection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call


class TransformerWrapper(nn.Module):
    """
    Thin wrapper around a HuggingFace causal language model.

    Provides:
      - compute_loss(): standard forward + masked CE loss
      - compute_loss_func(): static functional version (for torch.func grad)
      - compute_loss_func_single(): per-sample functional version (for vmap)
      - vector ↔ params conversion utilities
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        """
        Standard (non-functional) forward + loss.
        Returns (mean_loss, per_sample_losses).
        """
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = self.forward(input_ids, attention_mask).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
        losses = losses.view(label.size(0), -1)
        per_sample = (losses * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
        loss = per_sample.mean()
        return loss, per_sample

    @staticmethod
    def compute_loss_func(
        params: dict,
        buffers: dict,
        model: "TransformerWrapper",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Functional (stateless) batch loss — used by torch.func.grad and jvp.
        """
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = functional_call(
            model, (params, buffers), (input_ids, attention_mask)
        ).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
        losses = losses.view(label.size(0), -1)
        per_sample = (losses * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
        return per_sample.mean()

    @staticmethod
    def compute_loss_func_single(
        params: dict,
        buffers: dict,
        model: "TransformerWrapper",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Functional per-sample loss — used by vmap inside jvp_batch.
        Input tensors are unbatched (no batch dimension).
        """
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        label = label.unsqueeze(0)
        loss_mask = loss_mask.unsqueeze(0)

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        logits = functional_call(
            model, (params, buffers), (input_ids, attention_mask)
        ).logits
        losses = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
        losses = losses.view(1, -1)
        per_sample = (losses * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
        return per_sample.mean()

    # ------------------------------------------------------------------
    # Parameter ↔ vector utilities
    # ------------------------------------------------------------------

    def vector_to_params(self, vec: torch.Tensor) -> dict:
        """Convert a flat vector back to a named parameter dict."""
        pointer = 0
        d = {}
        for n, p in self.named_parameters():
            numel = p.numel()
            d[n] = nn.Parameter(
                vec[pointer : pointer + numel].view(p.size()),
                requires_grad=False,
            )
            pointer += numel
        assert pointer == vec.numel(), f"vector size mismatch: {pointer} != {vec.numel()}"
        return d

    def params_to_vector(self, params: dict) -> torch.Tensor:
        """Flatten a named parameter dict to a 1-D vector."""
        return torch.cat([params[n].view(-1) for n, _ in self.named_parameters()])

    def get_params_vec(self) -> torch.Tensor:
        """Return current parameters as a flat vector (detached)."""
        return torch.cat([p.detach().view(-1) for p in self.parameters()])

    def set_params_vec(self, vec: torch.Tensor):
        """Load a flat vector into model parameters in-place."""
        params = self.vector_to_params(vec)
        for n, p in self.named_parameters():
            p.data.copy_(params[n].data)
