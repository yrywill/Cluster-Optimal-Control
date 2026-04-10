"""
Early-Exit KMeans Clusterer Extension.

Extends KMeans-family clusterers to support intermediate layer feature extraction
for early-exit inference and layer-wise clustering.

This module provides:
  - EarlyExitKMeansClusterMixin: mixin for adding intermediate layer support
  - Concrete implementations: EarlyExitMiniBatchKMeansClusterer, EarlyExitFullKMeansClusterer, etc.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .kmeans_clusterer import MiniBatchKMeansClusterer, FullKMeansClusterer, FaissKMeansClusterer
from utils.layer_access import (
    extract_single_layer_features,
    get_layer_count,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Mixin for Early-Exit Support
# ======================================================================

class EarlyExitKMeansClusterMixin:
    """
    Mixin to add intermediate layer support to any KMeans-family clusterer.
    
    Provides the fit_with_intermediate_layer() method that can be used by
    MiniBatchKMeansClusterer, FullKMeansClusterer, FaissKMeansClusterer.
    """
    
    def _extract_intermediate_layer_features(
        self,
        dataset,
        model,
        device,
        cfg,
        batch_size: int,
        layer_idx: int = -1,
    ) -> np.ndarray:
        """
        Extract hidden state from an intermediate layer, pooled over sequence.

        Args:
            dataset: JsonFolderDataset
            model: Qwen3ForCausalLM
            device: torch.device
            cfg: OmegaConf config
            batch_size: batch size for extraction
            layer_idx: which layer (-1 = final, 0-N = specific layer)

        Returns:
            np.ndarray of shape [N, hidden_size]
        """
        # Handle layer_idx = -1 (final layer)
        if layer_idx == -1:
            num_layers = get_layer_count(model)
            layer_idx = num_layers - 1

        logger.info(f"Extracting intermediate layer features from layer {layer_idx}")

        # Use the layer_access utility
        features = extract_single_layer_features(
            model=model,
            dataset=dataset,
            device=device,
            layer_idx=layer_idx,
            batch_size=batch_size,
            pooling="mean",
        )

        return features

    def fit_with_intermediate_layer(
        self,
        dataset,
        model,
        tokenizer,
        device,
        cfg,
        layer_idx: int = -1,
        rank: int = 0,
    ) -> np.ndarray:
        """
        KMeans clustering using features from an intermediate layer.

        This is the main entry point for early-exit clustering. It extracts
        features from the specified layer and performs KMeans clustering.

        Args:
            dataset: JsonFolderDataset
            model: Qwen3ForCausalLM
            tokenizer: The tokenizer
            device: torch.device
            cfg: OmegaConf config
            layer_idx: which layer to extract from (-1 = final, 0-N = specific)
            rank: distributed rank (only rank 0 performs clustering)

        Returns:
            cluster_ids: np.ndarray of shape [N] with values in [0, K-1]
                         Only rank 0 returns valid ids; other ranks return zeros.
        """
        N = len(dataset)
        cluster_size = cfg.clustering.cluster_size
        K = max(1, N // cluster_size)
        batch_size = cfg.clustering.kmeans.feature_batch_size

        num_layers = get_layer_count(model)
        if layer_idx == -1:
            actual_layer_idx = num_layers - 1
        else:
            actual_layer_idx = layer_idx

        logger.info(
            f"{self.__class__.__name__}.fit_with_intermediate_layer: "
            f"N={N}, K={K}, layer_idx={actual_layer_idx}/{num_layers}, "
            f"batch_size={batch_size}"
        )

        # Feature extraction (all ranks participate for ZeRO-3 compatibility)
        try:
            import deepspeed
            _is_zero3 = (
                hasattr(model, "zero_optimization_stage")
                and model.zero_optimization_stage() == 3
            )
        except ImportError:
            _is_zero3 = False

        if _is_zero3:
            logger.warning(
                "ZeRO-3 detected: all ranks participate in feature extraction"
            )

        features = self._extract_intermediate_layer_features(
            dataset, model, device, cfg, batch_size, layer_idx
        )
        logger.info(f"Feature matrix shape: {features.shape}")

        # Clustering (only rank 0)
        if rank == 0:
            cluster_ids = self._run_kmeans(features, K, cfg)
            logger.info(
                f"Clustering done. Cluster sizes: "
                f"min={np.bincount(cluster_ids).min()}, "
                f"max={np.bincount(cluster_ids).max()}, "
                f"mean={np.bincount(cluster_ids).mean():.1f}"
            )
        else:
            cluster_ids = np.zeros(N, dtype=np.int32)

        return cluster_ids


# ======================================================================
# Concrete Early-Exit Clusterers
# ======================================================================

class EarlyExitMiniBatchKMeansClusterer(EarlyExitKMeansClusterMixin, MiniBatchKMeansClusterer):
    """MiniBatchKMeans with early-exit layer support."""
    pass


class EarlyExitFullKMeansClusterer(EarlyExitKMeansClusterMixin, FullKMeansClusterer):
    """Full KMeans with early-exit layer support."""
    pass


class EarlyExitFaissKMeansClusterer(EarlyExitKMeansClusterMixin, FaissKMeansClusterer):
    """Faiss KMeans with early-exit layer support."""
    pass


# Backward compatibility: default to MiniBatch
EarlyExitKMeansClusterer = EarlyExitMiniBatchKMeansClusterer
