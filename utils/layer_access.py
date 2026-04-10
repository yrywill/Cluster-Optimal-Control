"""
Layer Access Utilities for Early-Exit Forward Passes.

Provides functions to extract hidden states from intermediate layers of
HuggingFace CausalLM models (specifically Qwen3ForCausalLM).

Key functions:
  - get_layer_count(model): safely get num_hidden_layers from model.config
  - get_intermediate_hidden_states(): forward pass up to layer k
  - extract_layer_features(): batch feature extraction from specific layer
  - pool_hidden_states(): average pooling over sequence dimension

Handles:
  - Both wrapped (DDP/DeepSpeed) and unwrapped models
  - Attention masks and padding
  - Device placement (GPU/CPU)
  - Batch processing efficiently
  - Optional gradient computation
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ========================================================================
# Utility: Get Model Layer Information
# ========================================================================

def get_layer_count(model: torch.nn.Module) -> int:
    """
    Safely get num_hidden_layers from model.config.
    
    Handles both wrapped (DDP/DeepSpeed) and unwrapped models.
    
    Args:
        model: HuggingFace CausalLM model (wrapped or unwrapped)
    
    Returns:
        int: Number of hidden layers in the model
    
    Raises:
        ValueError: If num_hidden_layers cannot be determined
    """
    # Try to access config from wrapped model
    if hasattr(model, "module"):
        # DDP or DeepSpeed wrapper
        actual_model = model.module
    else:
        actual_model = model
    
    # Get config
    if not hasattr(actual_model, "config"):
        raise ValueError(f"Model {type(actual_model)} has no .config attribute")
    
    config = actual_model.config
    
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError(
            f"Model config {type(config)} has no num_hidden_layers attribute. "
            f"Available: {dir(config)}"
        )
    
    num_layers = config.num_hidden_layers
    if num_layers is None or num_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers: {num_layers}")
    
    return num_layers


def validate_layer_idx(model: torch.nn.Module, layer_idx: int) -> bool:
    """
    Validate if layer_idx is within valid range [0, num_layers-1].
    
    Args:
        model: HuggingFace CausalLM model
        layer_idx: Layer index to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    num_layers = get_layer_count(model)
    
    if layer_idx < 0 or layer_idx >= num_layers:
        logger.warning(
            f"Invalid layer_idx={layer_idx}. Must be in [0, {num_layers-1}]"
        )
        return False
    
    return True


def get_hidden_size(model: torch.nn.Module) -> int:
    """
    Get hidden size from model config.
    
    Args:
        model: HuggingFace CausalLM model
    
    Returns:
        int: Hidden dimension size
    
    Raises:
        ValueError: If hidden_size cannot be determined
    """
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model
    
    config = actual_model.config
    if not hasattr(config, "hidden_size"):
        raise ValueError(f"Model config has no hidden_size attribute")
    
    return config.hidden_size


# ========================================================================
# Core: Extract Intermediate Hidden States
# ========================================================================

def get_intermediate_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass up to layer_idx and return hidden state at that layer.
    
    This performs a partial forward pass, extracting the hidden state after
    the transformer layer at index layer_idx (0-indexed).
    
    Args:
        model: HuggingFace Qwen3ForCausalLM (wrapped or unwrapped)
        input_ids: [batch_size, seq_len] token indices
        attention_mask: [batch_size, seq_len] attention mask (1 = attend, 0 = pad)
        layer_idx: layer index in [0, num_layers-1]
        requires_grad: whether to track gradients for this computation
    
    Returns:
        (hidden_states, attention_mask):
        - hidden_states: [batch_size, seq_len, hidden_size]
        - attention_mask: same as input (for pooling)
    
    Raises:
        ValueError: if layer_idx is out of bounds
    """
    # Get unwrapped model
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model
    
    # Validate layer_idx
    num_layers = get_layer_count(model)
    if not validate_layer_idx(model, layer_idx):
        raise ValueError(f"layer_idx={layer_idx} out of bounds [0, {num_layers-1}]")
    
    device = input_ids.device
    
    # Context manager for gradient computation
    if requires_grad:
        context = torch.enable_grad()
    else:
        context = torch.no_grad()
    
    with context:
        # Get the inner transformer model (e.g., Qwen2Model inside Qwen2ForCausalLM)
        inner_model = actual_model.model

        # Embed input
        hidden = inner_model.embed_tokens(input_ids)  # [B, L, H]

        # Prepare causal attention mask in the format the decoder layers expect.
        # SDPA requires attn_mask dtype to match query dtype (e.g., bfloat16).
        B, L = input_ids.shape
        mask_dtype = hidden.dtype
        mask = torch.full((L, L), torch.finfo(mask_dtype).min, device=device, dtype=mask_dtype)
        mask = torch.triu(mask, diagonal=1)  # upper triangle = large negative
        causal_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask.to(mask_dtype)).unsqueeze(1).unsqueeze(2) * torch.finfo(mask_dtype).min
            causal_mask = causal_mask + pad_mask

        # Apply RMSNorm on embedding if the model has it (some models do)
        if hasattr(inner_model, "embed_layer_norm"):
            hidden = inner_model.embed_layer_norm(hidden)

        # Forward through layers 0 to layer_idx (inclusive)
        # Only these layers are computed — later layers are skipped entirely.
        position_ids = torch.arange(hidden.shape[1], device=device).unsqueeze(0)

        # Compute rotary position embeddings (cos, sin) once — shared by all layers.
        # Qwen3 / Llama-style models store rotary_emb on the inner model.
        position_embeddings = None
        if hasattr(inner_model, "rotary_emb"):
            position_embeddings = inner_model.rotary_emb(hidden, position_ids)

        for i in range(layer_idx + 1):
            layer = inner_model.layers[i]

            layer_kwargs = dict(
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_outputs = layer(hidden, **layer_kwargs)

            # Extract hidden state (first element of tuple output)
            if isinstance(layer_outputs, tuple):
                hidden = layer_outputs[0]
            else:
                hidden = layer_outputs

    return hidden, attention_mask


# ========================================================================
# Pooling: Average over Sequence
# ========================================================================

def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    """
    Pool hidden states over sequence dimension.
    
    Applies pooling (typically mean) over non-padding positions using the
    attention mask.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        pooling: pooling strategy ("mean" or "last")
    
    Returns:
        torch.Tensor: [batch_size, hidden_size] pooled features
    """
    if pooling == "mean":
        # Expand mask to match hidden_size dimension
        mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        
        # Mask out padding
        masked_hidden = hidden_states * mask  # [batch_size, seq_len, hidden_size]
        
        # Sum over sequence and divide by lengths
        lengths = mask.sum(dim=1).clamp(min=1)  # [batch_size, 1]
        pooled = masked_hidden.sum(dim=1) / lengths  # [batch_size, hidden_size]
        
        return pooled
    
    elif pooling == "last":
        # Get last non-padding position for each sample
        lengths = attention_mask.sum(dim=1)  # [batch_size]
        batch_size = hidden_states.size(0)
        
        # Gather last valid hidden state for each sample
        last_hidden = torch.zeros(
            batch_size, hidden_states.size(-1),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for i in range(batch_size):
            last_idx = lengths[i].long() - 1
            last_idx = max(0, min(last_idx, hidden_states.size(1) - 1))
            last_hidden[i] = hidden_states[i, last_idx, :]
        
        return last_hidden
    
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")


# ========================================================================
# Feature Extraction: Single and Batch
# ========================================================================

def extract_single_layer_features(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    layer_idx: int,
    batch_size: int = 32,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Extract features from intermediate layer for all samples in dataset.
    
    Performs batch feature extraction, returning mean-pooled hidden states
    from the specified layer.
    
    Args:
        model: HuggingFace Qwen3ForCausalLM
        dataset: JsonFolderDataset or similar (must have .collate() and .move_to_device())
        device: torch.device
        layer_idx: which layer to extract from [0, num_layers-1]
        batch_size: batch size for extraction
        pooling: pooling strategy ("mean" or "last")
    
    Returns:
        np.ndarray: [num_samples, hidden_size] feature matrix
    """
    num_layers = get_layer_count(model)
    if not validate_layer_idx(model, layer_idx):
        raise ValueError(f"layer_idx={layer_idx} out of bounds [0, {num_layers-1}]")
    
    # Create dataloader
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
            dataloader,
            desc=f"Extracting features from layer {layer_idx}",
            leave=False,
        ):
            # Move to device
            dataset.move_to_device(model_batch, no_model_batch, device)
            
            # Forward pass to layer_idx
            hidden, mask = get_intermediate_hidden_states(
                model,
                input_ids=model_batch["input_ids"],
                attention_mask=model_batch["attention_mask"],
                layer_idx=layer_idx,
                requires_grad=False,
            )
            
            # Pool over sequence
            pooled = pool_hidden_states(hidden, mask, pooling=pooling)
            
            # Append to list
            all_features.append(pooled.cpu().float().numpy())
    
    # Stack and return
    return np.concatenate(all_features, axis=0)


# ========================================================================
# Feature Extraction with Gradient Support (for ghost/projection modes)
# ========================================================================

def extract_layer_features_with_grad(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    layer_idx: int,
    batch_size: int = 32,
    pooling: str = "mean",
    apply_loss_mask: bool = True,
) -> np.ndarray:
    """
    Extract features from intermediate layer with gradient computation support.
    
    This version enables gradient computation for projection/ghost modes.
    Each sample is processed individually to isolate gradients.
    
    Args:
        model: HuggingFace Qwen3ForCausalLM
        dataset: JsonFolderDataset
        device: torch.device
        layer_idx: which layer to extract from
        batch_size: batch size for extraction
        pooling: pooling strategy
        apply_loss_mask: whether to apply loss_mask from dataset
    
    Returns:
        np.ndarray: [num_samples, hidden_size] feature matrix
    """
    num_layers = get_layer_count(model)
    if not validate_layer_idx(model, layer_idx):
        raise ValueError(f"layer_idx={layer_idx} out of bounds [0, {num_layers-1}]")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=dataset.collate,
        drop_last=False,
    )
    
    model.eval()
    all_features = []
    
    for model_batch, no_model_batch in tqdm(
        dataloader,
        desc=f"Extracting features from layer {layer_idx} (with grad support)",
        leave=False,
    ):
        dataset.move_to_device(model_batch, no_model_batch, device)
        bs = model_batch["input_ids"].shape[0]
        
        for i in range(bs):
            # Extract single sample
            input_ids = model_batch["input_ids"][i : i + 1]
            attn_mask = model_batch["attention_mask"][i : i + 1]
            
            # Forward pass
            hidden, mask = get_intermediate_hidden_states(
                model,
                input_ids=input_ids,
                attention_mask=attn_mask,
                layer_idx=layer_idx,
                requires_grad=True,
            )
            
            # Pool
            pooled = pool_hidden_states(hidden, mask, pooling=pooling)
            
            # Append
            all_features.append(pooled.detach().cpu().float().numpy())
    
    return np.stack(all_features, axis=0)


# ========================================================================
# Feature Extraction: Final Layer (backward compatibility)
# ========================================================================

def extract_final_layer_features(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Extract features from the final layer (backward compatible wrapper).
    
    Equivalent to: extract_single_layer_features(..., layer_idx=num_layers-1, ...)
    
    Args:
        model: HuggingFace Qwen3ForCausalLM
        dataset: JsonFolderDataset
        device: torch.device
        batch_size: batch size for extraction
        pooling: pooling strategy
    
    Returns:
        np.ndarray: [num_samples, hidden_size] feature matrix
    """
    num_layers = get_layer_count(model)
    final_layer_idx = num_layers - 1
    
    return extract_single_layer_features(
        model=model,
        dataset=dataset,
        device=device,
        layer_idx=final_layer_idx,
        batch_size=batch_size,
        pooling=pooling,
    )
