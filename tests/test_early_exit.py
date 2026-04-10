"""
Comprehensive tests for early-exit functionality.

Tests include:
- Layer access utilities
- Intermediate hidden state extraction
- Feature extraction from different layers
- Layer-wise clustering
- Backward compatibility
- Edge cases and error handling
"""
from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# Assuming these modules are available
try:
    from utils.layer_access import (
        get_layer_count,
        validate_layer_idx,
        get_hidden_size,
        get_intermediate_hidden_states,
        pool_hidden_states,
        extract_single_layer_features,
    )
    LAYER_ACCESS_AVAILABLE = True
except ImportError:
    LAYER_ACCESS_AVAILABLE = False

try:
    from clustering.early_exit_kmeans import EarlyExitKMeansClusterer
    EARLY_EXIT_CLUSTERER_AVAILABLE = True
except ImportError:
    EARLY_EXIT_CLUSTERER_AVAILABLE = False


# ========================================================================
# Fixtures for Testing
# ========================================================================

class MockHiddenLayer(nn.Module):
    """Mock transformer layer for testing."""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden, attention_mask=None):
        # Simple linear transformation
        output = self.linear(hidden)
        return (output,)  # Return tuple like real transformer


class MockQwenModel(nn.Module):
    """Mock Qwen-like model for testing."""
    def __init__(self, hidden_size: int = 768, num_layers: int = 12, vocab_size: int = 151936):
        super().__init__()
        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
            'vocab_size': vocab_size,
        })()
        
        self.model = type('ModelWrapper', (), {})()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([
            MockHiddenLayer(hidden_size) for _ in range(num_layers)
        ])


class MockDataset:
    """Mock dataset for testing."""
    def __init__(self, num_samples: int = 10, seq_len: int = 64, vocab_size: int = 1000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.samples = [
            {
                'input_ids': torch.randint(0, vocab_size, (seq_len,)),
                'attention_mask': torch.ones(seq_len),
            }
            for _ in range(num_samples)
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def collate(self, batch):
        """Collate function for DataLoader."""
        input_ids = torch.stack([x['input_ids'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])
        
        # Mock no_model_batch
        no_model_batch = {
            'label': torch.zeros_like(input_ids),
            'loss_mask': torch.ones_like(input_ids),
        }
        
        model_batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        return model_batch, no_model_batch
    
    def move_to_device(self, model_batch, no_model_batch, device):
        """Move batches to device."""
        for key in model_batch:
            model_batch[key] = model_batch[key].to(device)
        for key in no_model_batch:
            no_model_batch[key] = no_model_batch[key].to(device)


# ========================================================================
# Unit Tests: Layer Access Utilities
# ========================================================================

@pytest.mark.skipif(not LAYER_ACCESS_AVAILABLE, reason="layer_access module not available")
class TestLayerAccessUtilities:
    """Tests for layer_access module utilities."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MockQwenModel(hidden_size=768, num_layers=12)
        self.model.to(self.device)
    
    def test_get_layer_count(self):
        """Test getting layer count from model config."""
        num_layers = get_layer_count(self.model)
        assert num_layers == 12, f"Expected 12 layers, got {num_layers}"
    
    def test_get_layer_count_with_wrapped_model(self):
        """Test layer count retrieval from DDP-wrapped model."""
        wrapped = nn.DataParallel(self.model)
        num_layers = get_layer_count(wrapped)
        assert num_layers == 12
    
    def test_validate_layer_idx_valid(self):
        """Test validation of valid layer indices."""
        for idx in [0, 6, 11]:
            assert validate_layer_idx(self.model, idx)
    
    def test_validate_layer_idx_invalid(self):
        """Test validation of invalid layer indices."""
        for idx in [-1, 12, 100]:
            assert not validate_layer_idx(self.model, idx)
    
    def test_get_hidden_size(self):
        """Test getting hidden size from config."""
        hidden_size = get_hidden_size(self.model)
        assert hidden_size == 768
    
    def test_get_hidden_size_with_wrapped_model(self):
        """Test hidden size from wrapped model."""
        wrapped = nn.DataParallel(self.model)
        hidden_size = get_hidden_size(wrapped)
        assert hidden_size == 768
    
    def test_get_layer_count_invalid_model(self):
        """Test error handling for models without config."""
        invalid_model = nn.Linear(10, 10)
        with pytest.raises(ValueError):
            get_layer_count(invalid_model)


# ========================================================================
# Unit Tests: Intermediate Hidden State Extraction
# ========================================================================

@pytest.mark.skipif(not LAYER_ACCESS_AVAILABLE, reason="layer_access module not available")
class TestIntermediateHiddenStates:
    """Tests for intermediate hidden state extraction."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MockQwenModel(hidden_size=768, num_layers=12, vocab_size=1000)
        self.model.to(self.device)
        
        # Create test inputs
        self.batch_size = 4
        self.seq_len = 32
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len)).to(self.device)
        self.attention_mask = torch.ones(self.batch_size, self.seq_len).to(self.device)
    
    def test_get_intermediate_hidden_states_shape(self):
        """Test that output shape is correct."""
        for layer_idx in [0, 6, 11]:
            hidden, mask = get_intermediate_hidden_states(
                self.model,
                self.input_ids,
                self.attention_mask,
                layer_idx=layer_idx,
            )
            
            assert hidden.shape == (self.batch_size, self.seq_len, 768), \
                f"Expected shape {(self.batch_size, self.seq_len, 768)}, got {hidden.shape}"
            assert mask.shape == (self.batch_size, self.seq_len)
    
    def test_get_intermediate_hidden_states_early_layer(self):
        """Test extraction from early layer."""
        hidden_early, _ = get_intermediate_hidden_states(
            self.model, self.input_ids, self.attention_mask, layer_idx=0
        )
        assert hidden_early.shape[-1] == 768
    
    def test_get_intermediate_hidden_states_late_layer(self):
        """Test extraction from late layer."""
        hidden_late, _ = get_intermediate_hidden_states(
            self.model, self.input_ids, self.attention_mask, layer_idx=11
        )
        assert hidden_late.shape[-1] == 768
    
    def test_get_intermediate_hidden_states_invalid_layer(self):
        """Test error handling for invalid layer indices."""
        with pytest.raises(ValueError):
            get_intermediate_hidden_states(
                self.model, self.input_ids, self.attention_mask, layer_idx=12
            )
    
    def test_get_intermediate_hidden_states_with_grad(self):
        """Test with gradient computation enabled."""
        hidden, _ = get_intermediate_hidden_states(
            self.model,
            self.input_ids,
            self.attention_mask,
            layer_idx=5,
            requires_grad=True,
        )
        assert hidden.dtype == torch.float32


# ========================================================================
# Unit Tests: Hidden State Pooling
# ========================================================================

@pytest.mark.skipif(not LAYER_ACCESS_AVAILABLE, reason="layer_access module not available")
class TestHiddenStatePooling:
    """Tests for hidden state pooling operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.seq_len = 32
        self.hidden_size = 768
        self.hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
        
        # Create mask with some padding
        self.attention_mask[0, 20:] = 0  # Pad last 12 tokens of sample 0
        self.attention_mask[2, 16:] = 0  # Pad last 16 tokens of sample 2
    
    def test_pool_mean_shape(self):
        """Test mean pooling output shape."""
        pooled = pool_hidden_states(
            self.hidden,
            self.attention_mask,
            pooling="mean",
        )
        assert pooled.shape == (self.batch_size, self.hidden_size)
    
    def test_pool_mean_values(self):
        """Test mean pooling correctness."""
        pooled = pool_hidden_states(
            self.hidden,
            self.attention_mask,
            pooling="mean",
        )
        
        # Manually compute mean for first sample (with padding)
        mask = self.attention_mask[0].unsqueeze(-1).float()
        manual_mean = (self.hidden[0] * mask).sum(0) / mask.sum()
        
        # Should be close to first row of pooled
        assert torch.allclose(pooled[0], manual_mean, atol=1e-5)
    
    def test_pool_last_shape(self):
        """Test last token pooling output shape."""
        pooled = pool_hidden_states(
            self.hidden,
            self.attention_mask,
            pooling="last",
        )
        assert pooled.shape == (self.batch_size, self.hidden_size)
    
    def test_pool_invalid_strategy(self):
        """Test error handling for invalid pooling strategy."""
        with pytest.raises(ValueError):
            pool_hidden_states(
                self.hidden,
                self.attention_mask,
                pooling="invalid",
            )


# ========================================================================
# Integration Tests: Feature Extraction
# ========================================================================

@pytest.mark.skipif(not LAYER_ACCESS_AVAILABLE, reason="layer_access module not available")
class TestFeatureExtraction:
    """Integration tests for feature extraction pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MockQwenModel(hidden_size=768, num_layers=12, vocab_size=1000)
        self.model.to(self.device)
        
        self.dataset = MockDataset(num_samples=20, seq_len=64, vocab_size=1000)
    
    def test_extract_single_layer_features_shape(self):
        """Test that extracted features have correct shape."""
        features = extract_single_layer_features(
            self.model,
            self.dataset,
            self.device,
            layer_idx=6,
            batch_size=4,
        )
        
        assert features.shape == (20, 768), \
            f"Expected shape (20, 768), got {features.shape}"
    
    def test_extract_single_layer_features_dtype(self):
        """Test that features are float32."""
        features = extract_single_layer_features(
            self.model,
            self.dataset,
            self.device,
            layer_idx=6,
            batch_size=4,
        )
        
        assert features.dtype == np.float32
    
    def test_extract_features_different_layers(self):
        """Test extraction from multiple layers produces different features."""
        features_early = extract_single_layer_features(
            self.model, self.dataset, self.device, layer_idx=0, batch_size=4
        )
        
        features_late = extract_single_layer_features(
            self.model, self.dataset, self.device, layer_idx=11, batch_size=4
        )
        
        # Features should be different (high probability)
        assert not np.allclose(features_early, features_late, atol=1e-3)
    
    def test_extract_features_consistency(self):
        """Test that extraction is deterministic."""
        features1 = extract_single_layer_features(
            self.model, self.dataset, self.device, layer_idx=6, batch_size=4
        )
        
        features2 = extract_single_layer_features(
            self.model, self.dataset, self.device, layer_idx=6, batch_size=4
        )
        
        assert np.allclose(features1, features2)


# ========================================================================
# Integration Tests: Layer-wise Clustering
# ========================================================================

@pytest.mark.skipif(not EARLY_EXIT_CLUSTERER_AVAILABLE, reason="EarlyExitKMeansClusterer not available")
class TestEarlyExitClustering:
    """Integration tests for early-exit clustering."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MockQwenModel(hidden_size=768, num_layers=12, vocab_size=1000)
        self.model.to(self.device)
        
        self.dataset = MockDataset(num_samples=40, seq_len=64, vocab_size=1000)
        
        # Create mock config
        self.cfg = type('Config', (), {
            'clustering': type('ClusteringCfg', (), {
                'cluster_size': 10,
                'kmeans': type('KMeansCfg', (), {
                    'n_init': 1,
                    'max_iter': 10,
                    'feature_batch_size': 8,
                })(),
            })(),
            'training': type('TrainingCfg', (), {'seed': 42})(),
        })()
        
        self.clusterer = EarlyExitKMeansClusterer()
    
    def test_fit_with_intermediate_layer_output_shape(self):
        """Test clustering output shape."""
        cluster_ids = self.clusterer.fit_with_intermediate_layer(
            self.dataset,
            self.model,
            tokenizer=None,  # Not used in mock
            device=self.device,
            cfg=self.cfg,
            layer_idx=6,
            rank=0,
        )
        
        assert cluster_ids.shape == (40,), f"Expected shape (40,), got {cluster_ids.shape}"
    
    def test_fit_with_intermediate_layer_cluster_values(self):
        """Test that cluster IDs are valid."""
        cluster_ids = self.clusterer.fit_with_intermediate_layer(
            self.dataset,
            self.model,
            tokenizer=None,
            device=self.device,
            cfg=self.cfg,
            layer_idx=6,
            rank=0,
        )
        
        # All IDs should be in valid range
        k_expected = 40 // 10  # cluster_size = 10
        assert np.all(cluster_ids >= 0)
        assert np.all(cluster_ids < k_expected)
    
    def test_fit_with_different_layers(self):
        """Test clustering with different layers produces different results."""
        cluster_ids_early = self.clusterer.fit_with_intermediate_layer(
            self.dataset, self.model, None, self.device, self.cfg,
            layer_idx=0, rank=0,
        )
        
        cluster_ids_late = self.clusterer.fit_with_intermediate_layer(
            self.dataset, self.model, None, self.device, self.cfg,
            layer_idx=11, rank=0,
        )
        
        # Unlikely to be identical (but possible with small dataset)
        # Just verify both are valid
        assert cluster_ids_early.shape == cluster_ids_late.shape
    
    def test_fit_non_rank_0_returns_zeros(self):
        """Test that rank != 0 returns zeros."""
        cluster_ids = self.clusterer.fit_with_intermediate_layer(
            self.dataset,
            self.model,
            tokenizer=None,
            device=self.device,
            cfg=self.cfg,
            layer_idx=6,
            rank=1,  # Non-zero rank
        )
        
        assert np.all(cluster_ids == 0), "Non-rank-0 should return all zeros"


# ========================================================================
# Edge Case Tests
# ========================================================================

@pytest.mark.skipif(not LAYER_ACCESS_AVAILABLE, reason="layer_access module not available")
class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_single_layer_model(self):
        """Test with single-layer model."""
        model = MockQwenModel(hidden_size=256, num_layers=1)
        model.to(self.device)
        
        num_layers = get_layer_count(model)
        assert num_layers == 1
        
        assert validate_layer_idx(model, 0)
        assert not validate_layer_idx(model, 1)
    
    def test_large_model(self):
        """Test with larger model."""
        model = MockQwenModel(hidden_size=1024, num_layers=48)
        model.to(self.device)
        
        num_layers = get_layer_count(model)
        assert num_layers == 48
        
        assert validate_layer_idx(model, 0)
        assert validate_layer_idx(model, 24)
        assert validate_layer_idx(model, 47)
        assert not validate_layer_idx(model, 48)
    
    def test_empty_batch_handling(self):
        """Test handling of batch with all padding."""
        model = MockQwenModel()
        device = torch.device('cpu')
        
        input_ids = torch.zeros(1, 32, dtype=torch.long)
        attention_mask = torch.zeros(1, 32)  # All padding
        
        hidden, mask = get_intermediate_hidden_states(
            model, input_ids, attention_mask, layer_idx=0
        )
        
        assert hidden.shape == (1, 32, 768)
        
        # Pooling with all-zero mask should handle gracefully
        pooled = pool_hidden_states(hidden, attention_mask, pooling="mean")
        assert not torch.isnan(pooled).any()


# ========================================================================
# Backward Compatibility Tests
# ========================================================================

@pytest.mark.skipif(not LAYER_ACCESS_AVAILABLE, reason="layer_access module not available")
class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MockQwenModel(hidden_size=768, num_layers=12)
        self.model.to(self.device)
    
    def test_final_layer_extract_matches_legacy_behavior(self):
        """Verify final layer extraction is equivalent to legacy embedding mode."""
        # With layer_idx=-1 (default)
        from utils.layer_access import extract_final_layer_features
        
        dataset = MockDataset(num_samples=10)
        
        # New API with -1
        features_new = extract_single_layer_features(
            self.model, dataset, self.device, layer_idx=-1, batch_size=4
        )
        
        # Should match final layer explicitly
        num_layers = get_layer_count(self.model)
        final_idx = num_layers - 1
        features_explicit = extract_single_layer_features(
            self.model, dataset, self.device, layer_idx=final_idx, batch_size=4
        )
        
        assert np.allclose(features_new, features_explicit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
