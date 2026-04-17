"""
Few-shot / Zero-shot evaluation dataset for multiple-choice QA.

Parses the existing MMLU-style text format:
    {"text": "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer: X. ..."}

into structured (question, choices, answer) and builds few-shot prompts.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CHOICE_LABELS = ["A", "B", "C", "D"]

# Pattern: "Question: <question>\nA. <a>\nB. <b>\nC. <c>\nD. <d>\nAnswer: <label>. <text>"
_PATTERN = re.compile(
    r"Question:\s*(?P<question>.+?)\n"
    r"A\.\s*(?P<A>.+?)\n"
    r"B\.\s*(?P<B>.+?)\n"
    r"C\.\s*(?P<C>.+?)\n"
    r"D\.\s*(?P<D>.+?)\n"
    r"Answer:\s*(?P<answer>[A-D])\.",
    re.DOTALL,
)


def parse_mcq_text(text: str) -> Optional[Dict]:
    """Parse a single MMLU-style text string into structured fields.

    Returns dict with keys: question, choices (list[str]), answer (str 'A'-'D'),
    or None if parsing fails.
    """
    m = _PATTERN.search(text)
    if m is None:
        return None
    return {
        "question": m.group("question").strip(),
        "choices": [m.group(l).strip() for l in _CHOICE_LABELS],
        "answer": m.group("answer").strip(),
    }


def load_mcq_from_dir(data_dir: str, text_field: str = "text") -> List[Dict]:
    """Load and parse all MCQ samples from JSON/JSONL files in *data_dir*."""
    import glob as _glob

    patterns = [
        os.path.join(data_dir, "*.json"),
        os.path.join(data_dir, "*.jsonl"),
        os.path.join(data_dir, "**", "*.json"),
        os.path.join(data_dir, "**", "*.jsonl"),
    ]
    files = sorted({f for p in patterns for f in _glob.glob(p, recursive=True)})
    if not files:
        raise FileNotFoundError(f"No *.json or *.jsonl files in {data_dir}")

    samples: List[Dict] = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(text_field)
                if text is None:
                    # Try structured format directly
                    if "question" in obj and "choices" in obj and "answer" in obj:
                        samples.append(obj)
                    continue
                parsed = parse_mcq_text(text)
                if parsed is not None:
                    # Carry over extra fields (e.g. subject) if present
                    for k in ("subject",):
                        if k in obj:
                            parsed[k] = obj[k]
                    samples.append(parsed)

    logger.info(f"Loaded {len(samples)} MCQ samples from {data_dir}")
    return samples


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_single_question(
    question: str,
    choices: List[str],
    include_answer: bool = False,
    answer: Optional[str] = None,
) -> str:
    """Format one MCQ as a string.

    If *include_answer* is True, the answer line is appended (for demonstrations).
    Otherwise the prompt ends with "Answer:" for the model to complete.
    """
    lines = [f"Question: {question}"]
    for label, choice in zip(_CHOICE_LABELS, choices):
        lines.append(f"{label}. {choice}")
    if include_answer and answer is not None:
        choice_idx = _CHOICE_LABELS.index(answer)
        lines.append(f"Answer: {answer}. {choices[choice_idx]}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_fewshot_prompt(
    target: Dict,
    demonstrations: List[Dict],
    subject: Optional[str] = None,
) -> str:
    """Build a complete few-shot prompt string.

    Args:
        target: The sample to evaluate (question, choices, answer).
        demonstrations: List of example dicts used as few-shot prefix.
        subject: Optional subject name for the instruction line.
    """
    parts: List[str] = []

    # Instruction header
    if subject:
        parts.append(
            f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n"
        )
    else:
        parts.append(
            "The following are multiple choice questions (with answers).\n"
        )

    # Few-shot demonstrations
    for demo in demonstrations:
        parts.append(
            _format_single_question(
                demo["question"],
                demo["choices"],
                include_answer=True,
                answer=demo["answer"],
            )
        )
        parts.append("")  # blank line separator

    # Target question (no answer)
    parts.append(
        _format_single_question(target["question"], target["choices"])
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FewShotEvalDataset(Dataset):
    """
    Evaluation dataset that constructs few-shot or zero-shot prompts for MCQ.

    Each item returns a tokenised prompt ending with "Answer:" and the target
    answer label token id (A/B/C/D).

    Args:
        data_dir:    Path to folder with JSON/JSONL files.
        tokenizer:   HuggingFace tokenizer.
        n_shot:      Number of few-shot demonstrations (0 = zero-shot).
        text_field:  Key in JSON records containing the raw text.
        max_length:  Maximum token length for the prompt.
        max_samples: Cap on number of eval samples (-1 = all).
        split_name:  Human-readable name for logging.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        n_shot: int = 5,
        text_field: str = "text",
        max_length: int = 2048,
        max_samples: int = -1,
        split_name: str = "eval",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_shot = n_shot
        self.max_length = max_length
        self.split_name = split_name

        # Pad token
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        # Precompute answer token ids for A, B, C, D
        # We try multiple tokenisation strategies to be robust across tokenizers
        self.answer_token_ids: Dict[str, int] = {}
        for label in _CHOICE_LABELS:
            # Tokenise " A", " B", etc. and take the last token
            ids = tokenizer.encode(f" {label}", add_special_tokens=False)
            self.answer_token_ids[label] = ids[-1]
        logger.info(
            f"[{split_name}] Answer token ids: "
            + ", ".join(f"{k}={v}" for k, v in self.answer_token_ids.items())
        )

        # Load & parse samples
        all_samples = load_mcq_from_dir(data_dir, text_field)
        if max_samples > 0:
            all_samples = all_samples[:max_samples]

        # Group by subject for few-shot sampling
        self._by_subject: Dict[str, List[Dict]] = defaultdict(list)
        for s in all_samples:
            subj = s.get("subject", "__all__")
            self._by_subject[subj].append(s)

        # Build prompts & tokenise
        self.prompts: List[str] = []
        self.target_labels: List[str] = []  # 'A', 'B', 'C', or 'D'
        self._token_ids: List[np.ndarray] = []

        logger.info(f"[{split_name}] Building {n_shot}-shot prompts for {len(all_samples)} samples ...")
        for idx, sample in enumerate(all_samples):
            subj = sample.get("subject", "__all__")
            group = self._by_subject[subj]
            # Select demonstrations: pick from the same group, excluding self
            demos: List[Dict] = []
            for d in group:
                if d is sample:
                    continue
                demos.append(d)
                if len(demos) >= n_shot:
                    break

            prompt = build_fewshot_prompt(
                target=sample,
                demonstrations=demos,
                subject=sample.get("subject"),
            )
            self.prompts.append(prompt)
            self.target_labels.append(sample["answer"])

            ids = tokenizer.encode(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            self._token_ids.append(np.array(ids, dtype=np.int32))

        logger.info(f"[{split_name}] Done. {len(self._token_ids)} eval prompts built.")

    def __len__(self) -> int:
        return len(self._token_ids)

    def __getitem__(self, index: int):
        return index, self._token_ids[index], self.target_labels[index]

    # ------------------------------------------------------------------
    # Collate — left-padded for generation-style evaluation
    # ------------------------------------------------------------------

    def collate(self, samples) -> Tuple:
        """
        Returns:
            model_batch:    {"input_ids": [B, L], "attention_mask": [B, L]}
            no_model_batch: {"target_label": list[str], "target_token_ids": [B]}
        Left-pads so that the last token is always at position L-1.
        """
        if not samples or samples[0] is None:
            return None, None

        indices, token_arrays, labels = zip(*samples)
        bs = len(token_arrays)
        max_len = min(max(len(t) for t in token_arrays), self.max_length)

        model_batch = {
            "input_ids": torch.full((bs, max_len), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, max_len, dtype=torch.long),
        }
        target_token_ids = torch.tensor(
            [self.answer_token_ids[l] for l in labels], dtype=torch.long
        )

        for i, ids in enumerate(token_arrays):
            ids = ids[:max_len]  # truncate if needed
            seq_len = len(ids)
            # Left-pad: place tokens at the right end
            start = max_len - seq_len
            model_batch["input_ids"][i, start:] = torch.tensor(ids, dtype=torch.long)
            model_batch["attention_mask"][i, start:] = 1

        no_model_batch = {
            "target_label": list(labels),
            "target_token_ids": target_token_ids,
        }
        return model_batch, no_model_batch

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)
        if "target_token_ids" in no_model_batch:
            no_model_batch["target_token_ids"] = no_model_batch["target_token_ids"].to(device)
