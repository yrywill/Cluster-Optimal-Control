"""
Manual test script for Ghost Projection features (no pytest required).
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pmp.projection import GhostGradProjector, GradProjector


def test_ghost_instantiation_random():
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
    print("✅ test_ghost_instantiation_random passed")


def test_ghost_instantiation_layerwise():
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
    print("✅ test_ghost_instantiation_layerwise passed")


def test_ghost_mask_generation_random():
    """Test random mask generation."""
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
    assert torch.all((mask == 0) | (mask == 1))
    print("✅ test_ghost_mask_generation_random passed")


def test_ghost_mask_generation_layerwise():
    """Test layerwise mask generation."""
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
    print("✅ test_ghost_mask_generation_layerwise passed")


def test_ghost_project_vector():
    """Test ghost projection of a vector."""
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
    print("✅ test_ghost_project_vector passed")


def test_ghost_vs_standard():
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
    
    # Results should be different (masking changes the projection)
    assert not torch.allclose(standard_result, ghost_result)
    print("✅ test_ghost_vs_standard passed")


def test_ghost_grad_dict_projection():
    """Test ghost projection of gradient dict."""
    projector = GhostGradProjector(
        param_dim=1000,
        proj_dim=100,
        proj_type="rademacher",
        seed=42,
        device=torch.device("cpu"),
        ghost_strategy="random",
        ghost_fraction=0.5,
    )
    
    grad_dict = {
        "layer1": torch.randn(500),
        "layer2": torch.randn(500),
    }
    proj = projector.ghost_project_grad_dict(grad_dict)
    
    assert proj.shape == (100,)
    assert proj.dtype == torch.float32
    print("✅ test_ghost_grad_dict_projection passed")


def test_frequency_mask_accumulation():
    """Test frequency-based masking."""
    projector = GhostGradProjector(
        param_dim=1000,
        proj_dim=100,
        proj_type="rademacher",
        seed=42,
        device=torch.device("cpu"),
        ghost_strategy="frequency",
        ghost_fraction=0.5,
    )
    
    # Simulate gradient updates
    for i in range(3):
        grad_dict = {
            "layer1": torch.randn(500),
            "layer2": torch.randn(500),
        }
        projector.update_frequency(grad_dict)
    
    assert projector._update_frequency is not None
    assert projector._update_frequency.shape == (1000,)
    print("✅ test_frequency_mask_accumulation passed")


def test_kmeans_has_ghost_method():
    """Test that KMeansClusterer has fit_with_ghost method."""
    from clustering.kmeans_clusterer import KMeansClusterer
    clusterer = KMeansClusterer()
    assert hasattr(clusterer, 'fit_with_ghost')
    assert callable(getattr(clusterer, 'fit_with_ghost'))
    assert hasattr(clusterer, '_extract_ghost_features')
    print("✅ test_kmeans_has_ghost_method passed")


def test_config_ghost_settings():
    """Test that configuration includes ghost settings."""
    from utils.config import load_config
    cfg = load_config('configs/default.yaml')
    
    # Check clustering ghost config
    assert hasattr(cfg.clustering, 'ghost')
    assert hasattr(cfg.clustering.ghost, 'enabled')
    assert hasattr(cfg.clustering.ghost, 'strategy')
    assert hasattr(cfg.clustering.ghost, 'fraction')
    
    # Check PMP ghost config
    assert hasattr(cfg.pmp, 'ghost')
    assert hasattr(cfg.pmp.ghost, 'enabled_in_lambda')
    assert hasattr(cfg.pmp.ghost, 'enabled_in_weights')
    
    print("✅ test_config_ghost_settings passed")


def test_invalid_ghost_strategy():
    """Test error on invalid ghost strategy."""
    try:
        GhostGradProjector(
            param_dim=1000,
            proj_dim=100,
            proj_type="rademacher",
            seed=42,
            device=torch.device("cpu"),
            ghost_strategy="invalid_strategy",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown ghost_strategy" in str(e)
        print("✅ test_invalid_ghost_strategy passed")


def main():
    """Run all tests."""
    tests = [
        test_ghost_instantiation_random,
        test_ghost_instantiation_layerwise,
        test_ghost_mask_generation_random,
        test_ghost_mask_generation_layerwise,
        test_ghost_project_vector,
        test_ghost_vs_standard,
        test_ghost_grad_dict_projection,
        test_frequency_mask_accumulation,
        test_kmeans_has_ghost_method,
        test_config_ghost_settings,
        test_invalid_ghost_strategy,
    ]
    
    print("=" * 60)
    print("Running Ghost Projection Tests")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
