"""
KMeans-family clusterers.

Supports three clustering backends (controlled by cfg.clustering.method):
  - "minibatch": sklearn MiniBatchKMeans (default, fast, scalable)
  - "kmeans":    sklearn full KMeans (more accurate, slower)
  - "faiss":     Faiss GPU KMeans (fastest for large datasets)

All backends share the same feature extraction pipeline:
  - "projection": per-sample LM-gradient projected to low dim
  - "embedding":  mean of last hidden states (no gradient)
  - "ghost":      selective gradient projection with masking
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

try:
    from sklearn.cluster import MiniBatchKMeans, KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .base_clusterer import BaseClusterer
from pmp.projection import GradProjector

logger = logging.getLogger(__name__)


# ======================================================================
# Shared base with feature extraction + fit() template
# ======================================================================

class _KMeansBase(BaseClusterer):
    """
    Abstract base for all KMeans-family clusterers.

    Subclasses only need to implement ``_run_kmeans(features, K, cfg)``.
    """

    def fit(self, dataset, model, tokenizer, device, cfg, rank: int = 0, world_size: int = 1) -> np.ndarray:
        N = len(dataset)
        cluster_size = cfg.clustering.cluster_size
        K = max(1, N // cluster_size)
        feature_mode = cfg.clustering.kmeans.feature
        batch_size = cfg.clustering.kmeans.feature_batch_size

        logger.info(
            f"{self.__class__.__name__}: N={N}, K={K}, "
            f"feature={feature_mode}, batch_size={batch_size}, "
            f"rank={rank}, world_size={world_size}"
        )

        # ZeRO-3 fallback
        try:
            import deepspeed as _ds
            _is_zero3 = hasattr(model, "zero_optimization_stage") and model.zero_optimization_stage() == 3
        except ImportError:
            _is_zero3 = False
        if _is_zero3 and feature_mode in ("ghost", "projection"):
            logger.warning(
                f"ZeRO-3 detected: falling back to 'intermediate' features."
            )
            feature_mode = "intermediate"

        # ---- Streaming MiniBatch KMeans (multi-GPU data-parallel) ----
        if world_size > 1 and isinstance(self, MiniBatchKMeansClusterer):
            return self._fit_streaming(
                dataset, model, device, cfg, feature_mode, batch_size,
                K, N, rank, world_size,
            )

        # ---- Single-GPU fallback: extract all features then cluster ----
        features = self._extract_features(dataset, model, device, cfg, feature_mode, batch_size)
        logger.info(f"Feature matrix shape: {features.shape}")

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

    def _extract_features(self, dataset, model, device, cfg, feature_mode, batch_size):
        """Dispatch to the right feature extraction method."""
        if feature_mode == "projection":
            return self._extract_gradient_features(dataset, model, device, cfg, batch_size)
        elif feature_mode == "embedding":
            return self._extract_embedding_features(dataset, model, device, batch_size)
        elif feature_mode == "ghost":
            return self._extract_ghost_features(dataset, model, device, cfg, batch_size)
        elif feature_mode == "intermediate":
            return self._extract_intermediate_features(dataset, model, device, cfg, batch_size)
        else:
            raise ValueError(f"Unknown feature mode: {feature_mode}")

    def _fit_streaming(
        self, dataset, model, device, cfg, feature_mode, batch_size,
        K, N, rank, world_size,
    ) -> np.ndarray:
        """
        Streaming MiniBatch KMeans with multi-GPU data-parallel embedding.

        Pass 1: Each rank streams its data shard, extracts embeddings batch by
                 batch, gathers to rank 0, rank 0 calls partial_fit().
        Pass 2: Each rank streams again, extracts embeddings, assigns to nearest
                 center (broadcast from rank 0), gathers cluster_ids.
        """
        import torch.distributed as dist
        from torch.utils.data import DataLoader, Subset, SequentialSampler

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for MiniBatchKMeans")

        # ---- Shard dataset across ranks ----
        shard_indices = list(range(rank, N, world_size))
        shard_dataset = Subset(dataset, shard_indices)
        shard_dataset.collate = dataset.collate
        shard_dataset.move_to_device = dataset.move_to_device
        shard_N = len(shard_indices)

        logger.info(
            f"[Streaming] rank={rank}, shard={shard_N}/{N}, K={K}, "
            f"feature={feature_mode}, batch_size={batch_size}"
        )

        # Build feature extractor (returns iterator of np.ndarray batches)
        def _iter_features():
            """Yield (batch_features_np, batch_size) from this rank's shard."""
            loader = DataLoader(
                shard_dataset,
                batch_size=batch_size,
                sampler=SequentialSampler(shard_dataset),
                collate_fn=dataset.collate,
                drop_last=False,
            )
            model.eval()
            # Determine extraction method
            if feature_mode == "intermediate":
                from utils.layer_access import get_intermediate_hidden_states, pool_hidden_states, get_layer_count
                num_layers = get_layer_count(model)
                embed_layer = getattr(cfg.clustering.kmeans, "embed_layer", -1)
                layer_idx = num_layers // 2 if embed_layer == -1 else int(embed_layer)

                with torch.no_grad():
                    for model_batch, no_model_batch in loader:
                        dataset.move_to_device(model_batch, no_model_batch, device)
                        hidden, mask = get_intermediate_hidden_states(
                            model, model_batch["input_ids"],
                            model_batch["attention_mask"], layer_idx,
                        )
                        pooled = pool_hidden_states(hidden, mask, pooling="mean")
                        yield pooled.cpu().float().numpy()
            elif feature_mode == "embedding":
                with torch.no_grad():
                    for model_batch, no_model_batch in loader:
                        dataset.move_to_device(model_batch, no_model_batch, device)
                        outputs = model(
                            input_ids=model_batch["input_ids"],
                            attention_mask=model_batch["attention_mask"],
                            output_hidden_states=True,
                        )
                        hidden = outputs.hidden_states[-1]
                        mask = model_batch["attention_mask"].unsqueeze(-1).float()
                        lengths = mask.sum(dim=1).clamp(min=1)
                        emb = (hidden * mask).sum(dim=1) / lengths
                        yield emb.cpu().float().numpy()
            else:
                # Fallback: extract all then yield in chunks
                features = self._extract_features(shard_dataset, model, device, cfg, feature_mode, batch_size)
                for i in range(0, len(features), batch_size):
                    yield features[i:i+batch_size]

        # ================================================================
        # Pass 1: Streaming partial_fit to learn cluster centers
        # ================================================================
        logger.info(f"[Streaming] Pass 1: learning centers (partial_fit) ...")

        km = MiniBatchKMeans(
            n_clusters=K,
            n_init=1,  # streaming only supports n_init=1
            max_iter=1,  # partial_fit handles iterations
            random_state=cfg.training.seed,
            batch_size=batch_size * world_size,  # effective batch across all ranks
        )

        n_batches = 0
        accumulator = [] if rank == 0 else None  # Buffer on rank 0
        accum_len = 0

        for batch_feats in tqdm(
            _iter_features(),
            desc=f"[rank{rank}] Pass1: extract+partial_fit",
            total=(shard_N + batch_size - 1) // batch_size,
            leave=False,
        ):
            B, D = batch_feats.shape

            if world_size > 1:
                # Gather batch features from all ranks to rank 0
                batch_t = torch.tensor(batch_feats, dtype=torch.float32, device=device)
                # Pad to same size (last batch may differ across ranks)
                max_B = torch.tensor([B], device=device)
                dist.all_reduce(max_B, op=dist.ReduceOp.MAX)
                max_B = int(max_B.item())
                if B < max_B:
                    pad = torch.zeros(max_B - B, D, dtype=torch.float32, device=device)
                    batch_t = torch.cat([batch_t, pad], dim=0)
                sizes_t = torch.tensor([B], dtype=torch.long, device=device)
                all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
                dist.all_gather(all_sizes, sizes_t)
                gathered = [torch.zeros(max_B, D, dtype=torch.float32, device=device) for _ in range(world_size)]
                dist.all_gather(gathered, batch_t)

                if rank == 0:
                    parts = [gathered[r][:int(all_sizes[r].item())].cpu().numpy() for r in range(world_size)]
                    combined = np.concatenate(parts, axis=0)
                    accumulator.append(combined)
                    accum_len += len(combined)
                    # Flush buffer when we have enough samples for partial_fit
                    if accum_len >= K:
                        chunk = np.concatenate(accumulator, axis=0)
                        km.partial_fit(chunk)
                        accumulator = []
                        accum_len = 0
                        n_batches += 1
            else:
                if rank == 0:
                    km.partial_fit(batch_feats)
                n_batches += 1

        # Flush remaining buffer on rank 0
        if rank == 0 and accumulator and accum_len > 0:
            chunk = np.concatenate(accumulator, axis=0)
            km.partial_fit(chunk)
            n_batches += 1

        logger.info(f"[Streaming] Pass 1 done. {n_batches} partial_fit calls.")

        # Broadcast centers from rank 0
        if rank == 0:
            centers_t = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        else:
            # Need to know D; get it from first batch
            centers_t = torch.zeros(K, D, dtype=torch.float32, device=device)
        dist.broadcast(centers_t, src=0)
        centers = centers_t.cpu().numpy()  # [K, D]

        # ================================================================
        # Pass 2: Assign cluster IDs using final centers
        # ================================================================
        logger.info(f"[Streaming] Pass 2: assigning cluster IDs ...")

        shard_cluster_ids = []
        for batch_feats in tqdm(
            _iter_features(),
            desc=f"[rank{rank}] Pass2: assign clusters",
            total=(shard_N + batch_size - 1) // batch_size,
            leave=False,
        ):
            # Nearest center assignment
            # dists: [B, K]
            dists = np.sum((batch_feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            ids = np.argmin(dists, axis=1).astype(np.int32)
            shard_cluster_ids.append(ids)

        shard_ids = np.concatenate(shard_cluster_ids)  # [shard_N]

        # ---- Gather cluster_ids from all ranks ----
        max_shard = (N + world_size - 1) // world_size
        shard_ids_t = torch.tensor(shard_ids, dtype=torch.int32, device=device)
        if len(shard_ids) < max_shard:
            pad = torch.zeros(max_shard - len(shard_ids), dtype=torch.int32, device=device)
            shard_ids_t = torch.cat([shard_ids_t, pad])
        gathered_ids = [torch.zeros(max_shard, dtype=torch.int32, device=device) for _ in range(world_size)]
        dist.all_gather(gathered_ids, shard_ids_t)

        # Reconstruct full cluster_ids in original order
        cluster_ids = np.zeros(N, dtype=np.int32)
        for r in range(world_size):
            r_indices = list(range(r, N, world_size))
            r_ids = gathered_ids[r].cpu().numpy()[:len(r_indices)]
            cluster_ids[r_indices] = r_ids

        if rank == 0:
            logger.info(
                f"[Streaming] Clustering done. Cluster sizes: "
                f"min={np.bincount(cluster_ids).min()}, "
                f"max={np.bincount(cluster_ids).max()}, "
                f"mean={np.bincount(cluster_ids).mean():.1f}"
            )

        return cluster_ids

    @abstractmethod
    def _run_kmeans(self, features: np.ndarray, K: int, cfg) -> np.ndarray:
        """
        Run clustering on the feature matrix.

        Args:
            features: np.ndarray of shape [N, D].
            K:        number of clusters.
            cfg:      full OmegaConf config.

        Returns:
            cluster_ids: np.ndarray of shape [N], dtype int32.
        """
        raise NotImplementedError

    @staticmethod
    def _get_full_param_dim(model) -> int:
        """Get the true total parameter count, handling ZeRO-3 sharding."""
        try:
            import deepspeed
            with deepspeed.zero.GatheredParameters(
                list(model.parameters()), modifier_rank=None
            ):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
        except (ImportError, RuntimeError, AttributeError):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Ghost entry point (backward-compatible)
    # ------------------------------------------------------------------

    def fit_with_ghost(self, dataset, model, tokenizer, device, cfg, rank: int = 0) -> np.ndarray:
        """
        KMeans clustering using Ghost-projected gradient features.

        This is an alternative to fit() that uses selective gradient projection
        for potentially better feature quality and clustering performance.
        """
        N = len(dataset)
        cluster_size = cfg.clustering.cluster_size
        K = max(1, N // cluster_size)
        batch_size = cfg.clustering.kmeans.feature_batch_size

        logger.info(
            f"{self.__class__.__name__}.fit_with_ghost: N={N}, K={K}, "
            f"batch_size={batch_size}"
        )

        features = self._extract_ghost_features(
            dataset, model, device, cfg, batch_size
        )
        logger.info(f"Ghost feature matrix shape: {features.shape}")

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

    # ------------------------------------------------------------------
    # Gradient-projection features
    # ------------------------------------------------------------------

    def _extract_gradient_features(
        self, dataset, model, device, cfg, batch_size: int
    ) -> np.ndarray:
        """
        For each sample, compute ∂L/∂θ (LM loss gradient) and project to
        proj_dim via GradProjector.

        Returns np.ndarray of shape [N, proj_dim].
        """
        # Under ZeRO-3, params are sharded; gather to get true total count.
        param_dim = self._get_full_param_dim(model)
        proj_cfg = cfg.projection
        proj_dim = proj_cfg.dim if proj_cfg.enabled and proj_cfg.dim > 0 else param_dim

        projector = GradProjector(
            param_dim=param_dim,
            proj_dim=proj_dim,
            proj_type=proj_cfg.type if proj_cfg.enabled else "identity",
            seed=proj_cfg.seed,
            device=device,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=dataset.collate,
            drop_last=False,
        )

        model.eval()
        all_features = []
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        for model_batch, no_model_batch in tqdm(
            dataloader, desc="Extracting gradient features", leave=False
        ):
            dataset.move_to_device(model_batch, no_model_batch, device)
            bs = model_batch["input_ids"].shape[0]

            for i in range(bs):
                input_ids = model_batch["input_ids"][i : i + 1]
                attn_mask = model_batch["attention_mask"][i : i + 1]
                label = no_model_batch["label"][i : i + 1]
                loss_mask = no_model_batch["loss_mask"][i : i + 1]

                model.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                losses = loss_fn(
                    logits.view(-1, logits.size(-1)), label.view(-1)
                ).view(1, -1)
                loss = (losses * loss_mask).sum() / loss_mask.sum().clamp(min=1)
                loss.backward()

                # Flatten all gradients into a single vector
                grad_parts = []
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_parts.append(p.grad.detach().view(-1))
                if not grad_parts:
                    all_features.append(np.zeros(proj_dim, dtype=np.float32))
                    continue

                grad_vec = torch.cat(grad_parts).float()
                proj = projector.project_vector(grad_vec)
                all_features.append(proj.cpu().numpy())

                model.zero_grad()

        return np.stack(all_features, axis=0)

    # ------------------------------------------------------------------
    # Embedding features
    # ------------------------------------------------------------------

    def _extract_embedding_features(
        self, dataset, model, device, batch_size: int
    ) -> np.ndarray:
        """
        For each sample, run a forward pass and compute the mean of the
        last hidden state as the feature vector.

        Returns np.ndarray of shape [N, hidden_size].
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=dataset.collate,
            drop_last=False,
        )

        model.eval()
        all_features = []

        with torch.no_grad():
            for model_batch, no_model_batch in tqdm(
                dataloader, desc="Extracting embedding features", leave=False
            ):
                dataset.move_to_device(model_batch, no_model_batch, device)
                outputs = model(
                    input_ids=model_batch["input_ids"],
                    attention_mask=model_batch["attention_mask"],
                    output_hidden_states=True,
                )
                # last hidden state: [B, L, H]
                hidden = outputs.hidden_states[-1]
                # mean over non-padding positions
                mask = model_batch["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
                lengths = mask.sum(dim=1).clamp(min=1)  # [B, 1]
                emb = (hidden * mask).sum(dim=1) / lengths  # [B, H]
                all_features.append(emb.cpu().float().numpy())

        return np.concatenate(all_features, axis=0)

    # ------------------------------------------------------------------
    # Intermediate layer features (early-exit, no output_hidden_states)
    # ------------------------------------------------------------------

    def _extract_intermediate_features(
        self, dataset, model, device, cfg, batch_size: int
    ) -> np.ndarray:
        """
        Extract hidden states from an intermediate transformer layer.

        Only runs forward through layers [0, layer_idx], then stops.
        No output_hidden_states=True needed — no extra layers in memory.

        Controlled by cfg.clustering.kmeans.embed_layer:
          -1   = middle layer (num_layers // 2)
          0..N = specific layer index

        Returns np.ndarray of shape [N, hidden_size].
        """
        from utils.layer_access import (
            extract_single_layer_features,
            get_layer_count,
        )

        num_layers = get_layer_count(model)
        embed_layer = getattr(cfg.clustering.kmeans, "embed_layer", -1)

        if embed_layer == -1:
            # Default: middle layer
            layer_idx = num_layers // 2
        else:
            layer_idx = int(embed_layer)
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(
                    f"embed_layer={layer_idx} out of range [0, {num_layers - 1}]"
                )

        logger.info(
            f"Intermediate feature extraction: layer {layer_idx}/{num_layers} "
            f"(only forward through layers 0..{layer_idx}, "
            f"skipping layers {layer_idx + 1}..{num_layers - 1})"
        )

        features = extract_single_layer_features(
            model=model,
            dataset=dataset,
            device=device,
            layer_idx=layer_idx,
            batch_size=batch_size,
            pooling="mean",
        )

        logger.info(
            f"Intermediate features: [{features.shape[0]}, {features.shape[1]}] "
            f"from layer {layer_idx}"
        )
        return features

    # ------------------------------------------------------------------
    # Ghost-projection features (selective gradient projection)
    # ------------------------------------------------------------------

    def _extract_ghost_features(
        self, dataset, model, device, cfg, batch_size: int
    ) -> np.ndarray:
        """
        For each sample, compute ∂L/∂θ and project with Ghost masking.

        Ghost masking selectively zeros-out certain parameters before projection:
        - "layerwise": mask entire layers
        - "random": randomly mask a fraction
        - "frequency": mask parameters by update frequency

        Returns np.ndarray of shape [N, proj_dim].
        """
        from pmp.projection import GhostGradProjector

        # Under ZeRO-3, params are sharded; gather to get true total count.
        param_dim = self._get_full_param_dim(model)
        proj_cfg = cfg.projection
        proj_dim = proj_cfg.dim if proj_cfg.enabled and proj_cfg.dim > 0 else param_dim

        ghost_cfg = cfg.clustering.get("ghost", {})
        ghost_strategy = ghost_cfg.get("strategy", "layerwise")
        ghost_fraction = ghost_cfg.get("fraction", 0.5)
        layer_indices = ghost_cfg.get("layer_indices", [])
        num_layers = ghost_cfg.get("num_layers", None)

        # Auto-detect num_layers from model config if not specified
        if num_layers is None:
            model_cfg = getattr(model, "config", None)
            num_layers = getattr(model_cfg, "num_hidden_layers", None)
            if num_layers is not None:
                logger.info(f"Auto-detected num_layers={num_layers} from model config")

        # Auto-generate layer_indices if empty (keep fraction of layers)
        if not layer_indices and num_layers is not None:
            n_keep = max(1, int(num_layers * ghost_fraction))
            # Evenly spaced layers
            layer_indices = list(
                np.linspace(0, num_layers - 1, n_keep, dtype=int)
            )
            logger.info(
                f"Auto-generated layer_indices: {layer_indices} "
                f"({n_keep}/{num_layers} layers)"
            )

        # Create Ghost projector
        projector = GhostGradProjector(
            param_dim=param_dim,
            proj_dim=proj_dim,
            proj_type=proj_cfg.type if proj_cfg.enabled else "identity",
            seed=proj_cfg.seed,
            device=device,
            ghost_strategy=ghost_strategy,
            ghost_fraction=ghost_fraction,
            layer_indices=layer_indices,
            num_layers=num_layers,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=dataset.collate,
            drop_last=False,
        )

        model.eval()
        all_features = []
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        for model_batch, no_model_batch in tqdm(
            dataloader, desc="Extracting ghost-projected gradient features", leave=False
        ):
            dataset.move_to_device(model_batch, no_model_batch, device)
            bs = model_batch["input_ids"].shape[0]

            for i in range(bs):
                input_ids = model_batch["input_ids"][i : i + 1]
                attn_mask = model_batch["attention_mask"][i : i + 1]
                label = no_model_batch["label"][i : i + 1]
                loss_mask = no_model_batch["loss_mask"][i : i + 1]

                model.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                losses = loss_fn(
                    logits.view(-1, logits.size(-1)), label.view(-1)
                ).view(1, -1)
                loss = (losses * loss_mask).sum() / loss_mask.sum().clamp(min=1)
                loss.backward()

                # Flatten all gradients into a single vector
                grad_parts = []
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_parts.append(p.grad.detach().view(-1))
                if not grad_parts:
                    all_features.append(np.zeros(proj_dim, dtype=np.float32))
                    continue

                grad_vec = torch.cat(grad_parts).float()
                proj = projector.ghost_project_vector(grad_vec, seed_offset=i)
                all_features.append(proj.cpu().numpy())

                model.zero_grad()

        return np.stack(all_features, axis=0)


# ======================================================================
# Concrete clusterers
# ======================================================================

class MiniBatchKMeansClusterer(_KMeansBase):
    """
    sklearn MiniBatchKMeans — fast, memory-efficient, good default.
    """

    def _run_kmeans(self, features: np.ndarray, K: int, cfg) -> np.ndarray:
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for MiniBatchKMeans clustering. "
                "Install with: pip install scikit-learn"
            )
        km = MiniBatchKMeans(
            n_clusters=K,
            n_init=cfg.clustering.kmeans.n_init,
            max_iter=cfg.clustering.kmeans.max_iter,
            random_state=cfg.training.seed,
            verbose=0,
        )
        logger.info(f"Running MiniBatchKMeans with K={K} ...")
        return km.fit_predict(features).astype(np.int32)


class FullKMeansClusterer(_KMeansBase):
    """
    sklearn full KMeans — more accurate cluster assignments, but slower.
    """

    def _run_kmeans(self, features: np.ndarray, K: int, cfg) -> np.ndarray:
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for KMeans clustering. "
                "Install with: pip install scikit-learn"
            )
        km = KMeans(
            n_clusters=K,
            n_init=cfg.clustering.kmeans.n_init,
            max_iter=cfg.clustering.kmeans.max_iter,
            random_state=cfg.training.seed,
            verbose=0,
        )
        logger.info(f"Running full KMeans with K={K} ...")
        return km.fit_predict(features).astype(np.int32)


class FaissKMeansClusterer(_KMeansBase):
    """
    Faiss GPU KMeans — fastest for large-scale datasets.

    Requires: pip install faiss-gpu  (or faiss-cpu for CPU-only)
    """

    def _run_kmeans(self, features: np.ndarray, K: int, cfg) -> np.ndarray:
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for Faiss KMeans clustering. "
                "Install with: pip install faiss-gpu  (or faiss-cpu)"
            )

        D = features.shape[1]
        niter = cfg.clustering.kmeans.max_iter
        nredo = cfg.clustering.kmeans.n_init
        seed = cfg.training.seed
        verbose = False

        # Faiss requires float32 contiguous arrays
        features = np.ascontiguousarray(features, dtype=np.float32)

        # Try GPU first, fall back to CPU
        use_gpu = faiss.get_num_gpus() > 0
        logger.info(
            f"Running Faiss KMeans with K={K}, D={D}, "
            f"niter={niter}, nredo={nredo}, gpu={use_gpu} ..."
        )

        kmeans = faiss.Kmeans(
            d=D,
            k=K,
            niter=niter,
            nredo=nredo,
            seed=seed,
            verbose=verbose,
            gpu=use_gpu,
        )
        kmeans.train(features)

        # Assign clusters
        _, cluster_ids = kmeans.index.search(features, 1)
        return cluster_ids.flatten().astype(np.int32)


# Backward compatibility alias
KMeansClusterer = MiniBatchKMeansClusterer
