"""
Configuration utilities using OmegaConf.

Usage:
    cfg = load_config("configs/default.yaml")
    # Or with CLI overrides:
    cfg = load_config("configs/default.yaml", overrides=["training.lr=1e-5", "clustering.method=random"])
"""
from __future__ import annotations

import os
from typing import List, Optional

from omegaconf import OmegaConf, DictConfig


def load_config(
    yaml_path: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load a YAML config and optionally merge CLI dot-list overrides.

    Args:
        yaml_path: Path to the YAML config file.
        overrides: List of dot-list override strings, e.g.
                   ["training.lr=1e-5", "clustering.method=random"].

    Returns:
        OmegaConf DictConfig object.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    cfg = OmegaConf.load(yaml_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve any variable interpolations
    OmegaConf.resolve(cfg)

    return cfg
