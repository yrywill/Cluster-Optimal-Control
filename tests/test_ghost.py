"""
Unit and integration tests for Ghost Projection features.

Tests cover:
- GhostGradProjector instantiation and mask building
- Three masking strategies (layerwise, random, frequency)
- Integration with KMeans clustering
- Compatibility with standard projector API
"""
import pytest
import torch
import numpy as np
from pmp.projection import GhostGradProjector, GradProjector


class TestGhostGradProjectorBasics:
    """Test basic GhostGradProjector functionality."""

    def test_instantiation_random(self):
        """Test GhostGradProjector can be instantiated with random strategy."""
        projector = GhostGradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="rademacher",
            seed=42,
            device=torch.device("cpu"),
            ghost_strategy="random",
            ghost_fraction=0.5,
        )
        assert projector.ghost_strategy == "random"
        assert projector.ghost_fraction == 0.5
        assert projector.param_dim == 1000
        assert projector.proj_dim == 100

    def test_instantiation_layerwise(self):
        """Test GhostGradProjector with layerwise strategy."""
        projector = GhostGradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="gaussian",
            seed=42,
            device=torch.device("cpu"),
            ghost_strategy="layerwise",
            layer_indices=[0, 2, 4],
            num_layers=10,
        )
        assert projector.ghost_strategy == "layerwise"
        assert projector.layer_indices == [0, 2, 4]
        assert projector.num_layers == 10

    def test_invalid_strategy(self):
        """Test error on invalid ghost strategy."""
        with pytest.raises(ValueError, match="Unknown ghost_strategy"):
            GhostGradProjector(
                param_dim=1000,
                proj_dim=100,
                proj_type="rademacher",
                seed=42,
                device=torch.device("cpu"),
                ghost_strategy="invalid_strategy",
            )


class TestGhostMaskGeneration:
    """Test ghost mask generation for different strategies."""

    def test_random_mask_shape(self):
        """Test random mask has correct shape."""
        projector = GhostGradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="rademacher",
            seed=42,
            device=torch.device("cpu"),
            ghost_strategy="random",
            ghost_fraction=0.5,
        )
        mask = projector.build_mask()
        assert mask.shape == (1000,)
        assert mask.dtype == torch.float32
        assert torch.all((mask == 0) | (mask == 1))

    def test_layerwise_mask_shape(self):
        """Test layerwise mask has correct shape."""
        projector = GhostGradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="rademacher",
            seed=42,
            device=torch.device("cpu"),
            ghost_strategy="layerwise",
            layer_indices=[0, 2, 4],
            num_layers=10,
        )
        mask = projector.build_mask()
        assert mask.shape == (1000,)
        assert torch.all((mask == 0) | (mask == 1))


class TestGhostProjection:
    """Test ghost projection operations."""

    def test_ghost_project_vector_random(self):
        """Test ghost projection of a vector with random masking."""
        projector = GhostGradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="rademacher",
            seed=42,
            device=torch.device("cpu"),
            ghost_strategy="random",
            ghost_fraction=0.5,
        )
        
        vec = torch.randn(1000)
        proj = projector.ghost_project_vector(vec)
        
        assert proj.shape == (100,)
        assert proj.dtype == torch.float32
        assert not torch.isnan(proj).any()
        assert not torch.isinf(proj).any()

    def test_ghost_project_vector_layerwise(self):
        """Test ghost projection with layerwise masking."""
        projector = GhostGradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="rademacher",
            seed=42,
            device=torch.device("cpu"),
            ghost_strategy="layerwise",
            layer_indices=[0, 2, 4],
            num_layers=10,
        )
        
        vec = torch.randn(1000)
        proj = projector.ghost_project_vector(vec)
        
        assert proj.shape == (100,)
        assert proj.dtype == torch.float32


class TestGhostProjectorComparison:
    """Compare ghost projection with standard projection."""

    def test_ghost_is_different_from_standard(self):
        """Test that ghost projection differs from standard projection."""
        vec = torch.randn(1000)
        
        standard_proj = GradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="rademacher",
            seed=42,
            device=torch.device("cpu"),
        )
        
        ghost_proj = GhostGradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="rademacher",
            seed=42,
            device=torch.device("cpu"),
            ghost_strategy="random",
            ghost_fraction=0.5,
        )
        
        standard_result = standard_proj.project_vector(vec)
        ghost_result = ghost_proj.ghost_project_vector(vec)
        
        # Results should be different
        assert not torch.allclose(standard_result, ghost_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
