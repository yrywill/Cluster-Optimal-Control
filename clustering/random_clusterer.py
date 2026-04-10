"""
Random clusterer: shuffles all samples and assigns them to clusters
in a round-robin fashion. Fast, zero compute, good baseline.
"""
from __future__ import annotations

import logging

import numpy as np

from .base_clusterer import BaseClusterer

logger = logging.getLogger(__name__)


class RandomClusterer(BaseClusterer):
    """
    Assigns samples to clusters randomly (deterministic given seed).

    cluster_size samples → cluster 0, next cluster_size → cluster 1, …
    The last cluster may be smaller than cluster_size.
    """

    def fit(self, dataset, model, tokenizer, device, cfg, rank: int = 0, world_size: int = 1) -> np.ndarray:
        N = len(dataset)
        cluster_size = cfg.clustering.cluster_size
        K = max(1, N // cluster_size)

        logger.info(
            f"RandomClusterer: N={N}, cluster_size={cluster_size}, K={K}"
        )

        rng = np.random.default_rng(cfg.training.seed)
        shuffled = rng.permutation(N)

        cluster_ids = np.zeros(N, dtype=np.int32)
        for i, idx in enumerate(shuffled):
            cluster_ids[idx] = i * K // N  # evenly distribute

        logger.info(
            f"RandomClusterer: assigned {N} samples to {cluster_ids.max()+1} clusters"
        )
        return cluster_ids
