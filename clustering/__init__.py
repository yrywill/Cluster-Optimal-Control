"""clustering package"""
from .base_clusterer import BaseClusterer
from .random_clusterer import RandomClusterer
from .kmeans_clusterer import (
    MiniBatchKMeansClusterer,
    FullKMeansClusterer,
    FaissKMeansClusterer,
    KMeansClusterer,  # backward-compat alias → MiniBatchKMeansClusterer
)
from .early_exit_kmeans import (
    EarlyExitKMeansClusterer,
    EarlyExitMiniBatchKMeansClusterer,
    EarlyExitFullKMeansClusterer,
    EarlyExitFaissKMeansClusterer,
)


def build_clusterer(cfg) -> BaseClusterer:
    """Factory: returns a clusterer based on cfg.clustering.method."""
    method = cfg.clustering.method.lower()
    if method == "random":
        return RandomClusterer()
    elif method == "minibatch":
        return MiniBatchKMeansClusterer()
    elif method == "kmeans":
        return FullKMeansClusterer()
    elif method == "faiss":
        return FaissKMeansClusterer()
    else:
        raise ValueError(
            f"Unknown clustering method: {method}. "
            "Choose 'minibatch', 'kmeans', 'faiss', or 'random'."
        )


__all__ = [
    "BaseClusterer",
    "RandomClusterer",
    "MiniBatchKMeansClusterer",
    "FullKMeansClusterer",
    "FaissKMeansClusterer",
    "KMeansClusterer",
    "EarlyExitKMeansClusterer",
    "EarlyExitMiniBatchKMeansClusterer",
    "EarlyExitFullKMeansClusterer",
    "EarlyExitFaissKMeansClusterer",
    "build_clusterer",
]
