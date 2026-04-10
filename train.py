"""
Entry point for cluster-based continual pre-training.

Single GPU:
    python train.py --config configs/default.yaml model.path=gpt2 data.train_dir=data/train data.dev_dir=data/dev

Multi-GPU (torchrun / DDP):
    torchrun --nproc_per_node=4 train.py --config configs/default.yaml model.path=gpt2 ...

DeepSpeed ZeRO-3:
    deepspeed --num_gpus=4 train.py --config configs/default.yaml deepspeed.enabled=true

DeepSpeed ZeRO-3 + CPU Offload:
    deepspeed --num_gpus=4 train.py --config configs/default.yaml deepspeed.enabled=true deepspeed.config_file=configs/ds_zero3_offload.json

CLI overrides use OmegaConf dot-list syntax (key=value pairs appended after --config):
    python train.py --config configs/default.yaml training.lr=1e-5 clustering.method=random pmp.window_size=10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.dirname(__file__))

from utils.config import load_config
from trainer.integrated_trainer import IntegratedClusterTrainer


def _setup_logging(rank: int = 0):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _init_distributed(use_deepspeed: bool = False) -> tuple[int, int]:
    """
    Initialise distributed process group.

    When use_deepspeed=True, DeepSpeed handles init_process_group internally,
    so we only read LOCAL_RANK from the environment (set by `deepspeed` launcher).

    Returns (rank, world_size).
    """
    if use_deepspeed:
        import deepspeed
        deepspeed.init_distributed(dist_backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        # Not distributed
        return 0, 1

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, world_size


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster-based continual pre-training with PMP data selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML config file (e.g. configs/default.yaml)",
    )
    # DeepSpeed adds its own CLI args (--local_rank, --deepspeed, etc.)
    # We use parse_known_args to avoid conflicts.
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by deepspeed launcher (do not set manually)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="OmegaConf dot-list overrides (e.g. training.lr=1e-5 model.path=gpt2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load config (need it early to check deepspeed.enabled) ---
    cfg = load_config(args.config, overrides=args.overrides)

    # --- Check if DeepSpeed is enabled ---
    use_deepspeed = getattr(getattr(cfg, "deepspeed", None), "enabled", False)

    # --- Distributed init ---
    rank, world_size = _init_distributed(use_deepspeed=use_deepspeed)
    _setup_logging(rank)

    logger = logging.getLogger(__name__)
    logger.info(f"Rank {rank}/{world_size} initialised (deepspeed={use_deepspeed})")

    # Basic validation
    if not cfg.model.path:
        raise ValueError(
            "model.path is required. Set it in the YAML or pass model.path=<name_or_path> as override."
        )
    if not cfg.data.train_dir:
        raise ValueError(
            "data.train_dir is required. Pass data.train_dir=<path> as override."
        )
    if not cfg.data.dev_dir:
        raise ValueError(
            "data.dev_dir is required. Pass data.dev_dir=<path> as override."
        )

    if rank == 0:
        from omegaconf import OmegaConf
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Run ---
    trainer = IntegratedClusterTrainer(cfg)
    trainer.train()

    # --- Cleanup ---
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
