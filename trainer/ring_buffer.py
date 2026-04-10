"""
RingBuffer: stores the last `capacity` (model_params_vec, batch, cluster_ids_in_batch)
tuples needed for PMP backward pass.

All data is kept on CPU to avoid GPU memory pressure during training.
When the PMP backward runs, entries are moved to device on demand.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Iterator, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class RingBuffer:
    """
    Fixed-size FIFO buffer storing training step snapshots.

    Each entry is a triple:
        (params_vec, batch_cpu, cluster_ids_in_batch)

    where:
        params_vec          : CPU float tensor [param_dim]  — model params BEFORE this step's update
        batch_cpu           : dict of CPU tensors           — training batch used in this step
        cluster_ids_in_batch: CPU int tensor [B]            — cluster id of each sample in batch

    Args:
        capacity:  Maximum number of entries to keep (= pmp.window_size).
        param_dim: Number of model parameters (used for pre-allocation hint).
    """

    def __init__(self, capacity: int, param_dim: int):
        self.capacity = capacity
        self.param_dim = param_dim
        self._buffer: Deque[
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]
        ] = deque(maxlen=capacity)

    def push(
        self,
        params_vec: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        cluster_ids_in_batch: torch.Tensor,
    ):
        """
        Add a new entry to the ring buffer.

        Args:
            params_vec:           Model parameter vector (will be cloned to CPU).
            batch:                Training batch dict (will be cloned to CPU).
            cluster_ids_in_batch: Cluster id for each sample in the batch [B].
        """
        entry = (
            params_vec.detach().cpu().clone(),
            {k: v.detach().cpu() for k, v in batch.items()},
            cluster_ids_in_batch.detach().cpu().clone(),
        )
        self._buffer.append(entry)

    def get_all_ordered(
        self,
    ) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
        """
        Return entries in chronological order (oldest first).
        This is the order needed for inner_forward in PMP.
        """
        return list(self._buffer)

    def get_latest(
        self,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
        """
        Return the most recent entry (params_vec, batch_cpu, cluster_ids), or None
        if the buffer is empty.  Used by the Ghost Inner Product fast path which
        only needs the current step's batch rather than the full history.
        """
        if not self._buffer:
            return None
        return self._buffer[-1]

    def __len__(self) -> int:
        return len(self._buffer)

    def is_full(self) -> bool:
        return len(self._buffer) == self.capacity

    def clear(self):
        self._buffer.clear()

    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self.capacity}, "
            f"current_size={len(self._buffer)}, "
            f"param_dim={self.param_dim})"
        )
