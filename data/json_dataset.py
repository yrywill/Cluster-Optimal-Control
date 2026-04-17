"""
Data loading utilities for continual pre-training.
Reads all *.json / *.jsonl files in a directory and extracts text field.
"""
from __future__ import annotations

import os
import json
import glob
import logging
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _load_texts_from_file(path: str, text_field: str) -> List[str]:
    """Load text strings from a single JSON or JSONL file."""
    texts = []

    # --- Try JSONL first: read line-by-line (handles .json that are actually JSONL) ---
    jsonl_ok = True
    jsonl_texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and text_field in obj:
                    jsonl_texts.append(obj[text_field])
            except json.JSONDecodeError:
                jsonl_ok = False
                break

    if jsonl_ok and jsonl_texts:
        return jsonl_texts

    # --- JSON: could be an array, dict, or dict with a data/items key ---
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse {path}: {e}")
        return texts

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and text_field in item:
                texts.append(item[text_field])
    elif isinstance(data, dict):
        # Try common wrapper keys
        for key in ("data", "items", "records", "samples"):
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict) and text_field in item:
                        texts.append(item[text_field])
                break
        else:
            # Single record
            if text_field in data:
                texts.append(data[text_field])

    return texts


def load_texts_from_dir(data_dir: str, text_field: str) -> List[str]:
    """Recursively load all texts from *.json and *.jsonl files in data_dir."""
    patterns = [
        os.path.join(data_dir, "*.json"),
        os.path.join(data_dir, "*.jsonl"),
        os.path.join(data_dir, "**", "*.json"),
        os.path.join(data_dir, "**", "*.jsonl"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No *.json or *.jsonl files found in {data_dir}"
        )

    all_texts = []
    for f in files:
        t = _load_texts_from_file(f, text_field)
        logger.info(f"  {f}: {len(t)} samples")
        all_texts.extend(t)

    logger.info(f"Total texts loaded from {data_dir}: {len(all_texts)}")
    return all_texts


class JsonFolderDataset(Dataset):
    """
    Dataset that reads all JSON/JSONL files from a directory and tokenises
    the specified text field. Compatible with the original LMDataset collate
    interface (returns model_batch, no_model_batch).

    Args:
        data_dir:   Path to folder containing JSON/JSONL files.
        tokenizer:  HuggingFace tokenizer.
        text_field: Key to extract from each JSON record.
        max_length: Max token length (sequences are truncated to max_length+1
                    so there's room for the shifted label).
        max_samples: If > 0, cap the dataset at this many samples.
        split_name: Human-readable label for logging.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        text_field: str = "text",
        max_length: int = 1024,
        max_samples: int = -1,
        split_name: str = "data",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length
        self.split_name = split_name

        # pad_id: use eos if no explicit pad token
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        logger.info(f"[{split_name}] Loading texts from {data_dir} ...")
        texts = load_texts_from_dir(data_dir, text_field)

        if max_samples > 0 and len(texts) > max_samples:
            texts = random.sample(texts, max_samples)

        logger.info(f"[{split_name}] Tokenising {len(texts)} texts ...")
        self.data: List[np.ndarray] = self._tokenise_all(texts)
        logger.info(
            f"[{split_name}] Done. {len(self.data)} samples after tokenisation."
        )

    def _tokenise_all(self, texts: List[str]) -> List[np.ndarray]:
        """Tokenise texts; store as int32 numpy arrays."""
        tokenised = []
        for text in texts:
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length + 1,  # +1 for the shift
            )
            if len(ids) < 2:
                continue  # skip extremely short samples
            tokenised.append(np.array(ids, dtype=np.int32))
        return tokenised

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return index, self.data[index].astype(int)

    # ------------------------------------------------------------------
    # Collate (compatible with original LMDataset.collate)
    # ------------------------------------------------------------------

    def collate(self, samples):
        """
        Produces:
            model_batch:    {"input_ids": [B, L], "attention_mask": [B, L]}
            no_model_batch: {"label": [B, L], "loss_mask": [B, L]}
        """
        if not samples or samples[0] is None:
            return None, None

        bs = len(samples)
        max_length = self.max_length

        model_batch = {
            "input_ids": torch.full((bs, max_length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
        }
        no_model_batch = {
            "label": torch.full((bs, max_length), self.pad_id, dtype=torch.long),
            "loss_mask": torch.zeros(bs, max_length, dtype=torch.float),
        }

        for i, (idx, data) in enumerate(samples):
            # Truncate to max_length+1 tokens so we have input[:-1] and label[1:]
            full_ids = data[: max_length + 1]
            seq_len = len(full_ids) - 1  # actual input length

            model_batch["input_ids"][i, :seq_len] = torch.tensor(
                full_ids[:-1], dtype=torch.long
            )
            model_batch["attention_mask"][i, :seq_len] = 1
            no_model_batch["label"][i, :seq_len] = torch.tensor(
                full_ids[1:], dtype=torch.long
            )
            # loss mask: 1 where input is not padding
            no_model_batch["loss_mask"][i, :seq_len] = (
                torch.tensor(full_ids[:-1], dtype=torch.long) != self.pad_id
            ).float()

        return model_batch, no_model_batch

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)
