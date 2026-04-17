"""
Cluster-aware Dataset and Sampler.

ClusterDataset: wraps JsonFolderDataset and holds cluster_ids.
ClusterWeightedSampler: samples batches proportionally to cluster weights,
    which are updated periodically via PMP backward.
"""
from __future__ import annotations

import math
import logging
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .json_dataset import JsonFolderDataset

logger = logging.getLogger(__name__)


class ClusterDataset(Dataset):
    """
    Thin wrapper around JsonFolderDataset that exposes cluster membership.

    Args:
        base_dataset: The underlying JsonFolderDataset.
        cluster_ids:  np.ndarray of shape [N] mapping each sample to its cluster.
    """

    def __init__(self, base_dataset: JsonFolderDataset, cluster_ids: np.ndarray):
        super().__init__()
        self.base = base_dataset
        self.cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
        assert len(self.cluster_ids) == len(self.base), (
            f"cluster_ids length {len(self.cluster_ids)} != dataset length {len(self.base)}"
        )
        self.n_clusters = int(self.cluster_ids.max()) + 1

        # Pre-build per-cluster index lists for fast sampling
        self._cluster_to_indices: List[List[int]] = [[] for _ in range(self.n_clusters)]
        for idx, cid in enumerate(self.cluster_ids):
            self._cluster_to_indices[cid].append(idx)

        for k, idxs in enumerate(self._cluster_to_indices):
            if len(idxs) == 0:
                logger.warning(f"Cluster {k} is empty!")

        logger.info(
            f"ClusterDataset: {len(self.base)} samples, "
            f"{self.n_clusters} clusters, "
            f"avg {len(self.base)/self.n_clusters:.1f} samples/cluster"
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        return self.base[index]

    def collate(self, samples):
        return self.base.collate(samples)

    def move_to_device(self, model_batch, no_model_batch, device):
        return self.base.move_to_device(model_batch, no_model_batch, device)

    def get_cluster_indices(self, cluster_id: int) -> List[int]:
        """Return global sample indices belonging to `cluster_id`."""
        return self._cluster_to_indices[cluster_id]

    def update_cluster_ids(self, new_cluster_ids: np.ndarray):
        """Replace cluster assignments (called on re-clustering)."""
        self.cluster_ids = np.asarray(new_cluster_ids, dtype=np.int32)
        self.n_clusters = int(self.cluster_ids.max()) + 1
        self._cluster_to_indices = [[] for _ in range(self.n_clusters)]
        for idx, cid in enumerate(self.cluster_ids):
            self._cluster_to_indices[cid].append(idx)


class ClusterWeightedSampler(Sampler):
    """
    Sampler that draws samples according to per-cluster importance weights.

    Strategy:
        1. For each batch, sample `batch_size` indices by:
           a. First draw clusters proportionally to their weights.
           b. Then uniformly draw one sample from each drawn cluster.
        2. Weights are updated externally via `update_weights(grad_gamma)`.

    Supports distributed training: each rank processes a non-overlapping
    shard of the total indices list.

    Args:
        dataset:       ClusterDataset.
        batch_size:    Total batch size across all ranks × grad_accum
                       (used to pre-generate a full epoch of indices).
        temperature:   Softmax temperature for weight computation.
        min_weight:    Minimum weight floor (prevents cluster starvation).
        seed:          Random seed.
        rank:          Distributed rank (default 0).
        world_size:    Distributed world size (default 1).
        num_replicas:  Alias for world_size (compatibility).
    """

    def __init__(
        self,
        dataset: ClusterDataset,
        batch_size: int,
        temperature: float = 1.0,
        min_weight: float = 0.01,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        drop_bad_clusters: bool = False,
        drop_patience: int = 5,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size  # global batch size
        self.temperature = temperature
        self.min_weight = min_weight
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        K = dataset.n_clusters
        self._weights = torch.ones(K, dtype=torch.float32) / K  # uniform init
        self._grad_gamma = torch.zeros(K, dtype=torch.float32)

        self._epoch = 0
        self._rng = np.random.default_rng(seed)

        # ---- Bad cluster auto-drop ----
        self.drop_bad_clusters = drop_bad_clusters
        self.drop_patience = drop_patience
        self._negative_streak = torch.zeros(K, dtype=torch.int32)  # consecutive negative ct_k count
        self._dead_clusters = torch.zeros(K, dtype=torch.bool)     # True = permanently dropped
        self._n_dropped = 0

        # How many total indices to generate per "epoch" iteration
        N = len(dataset)
        # Generate enough indices to cover at least one full pass
        self._num_samples_per_rank = math.ceil(N / world_size)

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def update_weights(self, grad_gamma: torch.Tensor, grad_gamma_delta: torch.Tensor = None):
        """
        Recompute cluster sampling weights from grad_gamma.

        w_k = softmax(-grad_gamma / temperature)  [clamp min_weight, renorm]

        If drop_bad_clusters is enabled and grad_gamma_delta is provided,
        tracks consecutive negative contributions per cluster. A cluster with
        `drop_patience` consecutive negative ct_k values is permanently dropped
        (weight set to 0, never sampled again).

        Args:
            grad_gamma:       Accumulated grad_gamma tensor [K].
            grad_gamma_delta: This update's delta [K] (needed for streak tracking).
                              If None, streak tracking is skipped.
        """
        gg = grad_gamma.float()
        logits = -gg / self.temperature
        # Numerically stable softmax
        logits = logits - logits.max()
        weights = torch.exp(logits)
        weights = weights.clamp(min=self.min_weight)

        # ---- Bad cluster auto-drop ----
        if self.drop_bad_clusters and grad_gamma_delta is not None:
            delta = grad_gamma_delta.float().cpu()
            K = delta.shape[0]

            for k in range(K):
                if self._dead_clusters[k]:
                    continue  # already dead, skip
                if delta[k] < 0:
                    # negative contribution → increment streak
                    self._negative_streak[k] += 1
                elif delta[k] > 0:
                    # positive contribution → reset streak
                    self._negative_streak[k] = 0
                # delta[k] == 0 means cluster not seen this round → don't change streak

                if self._negative_streak[k] >= self.drop_patience:
                    self._dead_clusters[k] = True
                    self._n_dropped += 1
                    logger.info(
                        f"Cluster {k} DROPPED: {self.drop_patience} consecutive "
                        f"negative contributions (total dropped: {self._n_dropped})"
                    )

            # Force dead cluster weights to zero
            weights[self._dead_clusters] = 0.0

        # Renormalize
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            # All clusters dead (shouldn't happen) — fall back to uniform over alive
            alive = ~self._dead_clusters
            weights = alive.float()
            weights = weights / weights.sum().clamp(min=1)
            logger.warning("All clusters dropped! Falling back to uniform over remaining.")

        self._weights = weights
        self._grad_gamma = gg.clone()
        logger.debug(
            f"Cluster weights updated: min={weights.min():.4f}, "
            f"max={weights.max():.4f}, entropy={(-weights[weights>0]*weights[weights>0].log()).sum():.3f}, "
            f"dropped={self._n_dropped}/{weights.shape[0]}"
        )

    @property
    def weights(self) -> torch.Tensor:
        return self._weights.clone()

    @property
    def n_dropped(self) -> int:
        return self._n_dropped

    @property
    def n_alive(self) -> int:
        return int((~self._dead_clusters).sum().item())

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        return self._num_samples_per_rank

    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for one epoch.

        Each index is drawn by:
          1. Sample a cluster k ~ Categorical(weights)
          2. Sample a random sample from cluster k
        """
        K = self.dataset.n_clusters
        weights_np = self._weights.numpy()
        total_needed = self._num_samples_per_rank * self.world_size

        rng = np.random.default_rng(self.seed + self._epoch)

        # Draw cluster assignments for all total_needed indices
        cluster_draws = rng.choice(K, size=total_needed, replace=True, p=weights_np)

        # For each cluster draw, pick a random sample
        indices = np.zeros(total_needed, dtype=np.int64)
        for i, k in enumerate(cluster_draws):
            cluster_idxs = self.dataset.get_cluster_indices(int(k))
            if len(cluster_idxs) == 0:
                # Fallback: random sample from entire dataset
                indices[i] = rng.integers(len(self.dataset))
            else:
                indices[i] = cluster_idxs[rng.integers(len(cluster_idxs))]

        # Shard across ranks
        indices_for_rank = indices[self.rank::self.world_size]

        return iter(indices_for_rank.tolist())

    def update_weights_with_ghost(self, grad_gamma: torch.Tensor, ghost_mask: torch.Tensor | None = None):
        """
        Recompute cluster sampling weights from grad_gamma with optional ghost masking.
        
        Method 3 integration: apply ghost masking at the weight update phase.
        
        If ghost_mask is provided, applies selective masking before softmax:
        w_k = softmax(-(ghost_mask ⊙ grad_gamma) / temperature)
        
        Args:
            grad_gamma:   Per-cluster gradient contribution vector [n_clusters].
            ghost_mask:   Optional binary mask [n_clusters] for selective masking.
                         If None, behaves like standard update_weights().
        """
        gg = grad_gamma.float()
        
        # Apply ghost masking if provided
        if ghost_mask is not None:
            ghost_mask = ghost_mask.to(gg.device).float()
            gg_masked = gg * ghost_mask  # Element-wise masking
        else:
            gg_masked = gg
        
        logits = -gg_masked / self.temperature
        # Numerically stable softmax
        logits = logits - logits.max()
        weights = torch.exp(logits)
        weights = weights.clamp(min=self.min_weight)
        weights = weights / weights.sum()
        self._weights = weights
        self._grad_gamma = gg.clone()
        
        ghost_info = " (with ghost mask)" if ghost_mask is not None else ""
        logger.debug(
            f"Cluster weights updated{ghost_info}: min={weights.min():.4f}, "
            f"max={weights.max():.4f}, entropy={(-weights*weights.log()).sum():.3f}"
        )
