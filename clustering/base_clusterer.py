"""Abstract base class for clustering algorithms."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClusterer(ABC):
    """
    Abstract interface for all clusterers.
    A clusterer takes a dataset and a (possibly frozen) model,
    and returns a cluster_ids array of shape [N].
    """

    @abstractmethod
    def fit(self, dataset, model, tokenizer, device, cfg, rank: int = 0, world_size: int = 1) -> np.ndarray:
        """
        Compute cluster assignments for all samples in `dataset`.

        Args:
            dataset:   JsonFolderDataset (or ClusterDataset).
            model:     The LM (may be used for feature extraction).
            tokenizer: The tokenizer.
            device:    torch.device.
            cfg:       Full OmegaConf config.
            rank:      Distributed rank. All ranks participate in feature
                       extraction (required by ZeRO-3), but only rank 0
                       runs the actual clustering algorithm.

        Returns:
            cluster_ids: np.ndarray of shape [N] with values in [0, K-1].
                         Only rank 0 returns valid ids; other ranks return zeros.
        """
        raise NotImplementedError
