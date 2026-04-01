import pytest
import numpy as np
from src.portfolio import compute_portfolio_value

def test_portfolio_value_initialization():
    """Verify initial portfolio value matches investment."""
    n_assets, days, sims = 3, 10, 5
    initial = 100000.0
    weights = [0.5, 0.3, 0.2]
    
    # Mock asset paths (flat)
    asset_paths = np.ones((days, n_assets, sims)) * 100.0
    
    portfolio_paths = compute_portfolio_value(asset_paths, weights, initial)
    
    assert portfolio_paths.shape == (days, sims)
    assert np.allclose(portfolio_paths[0, :], initial)

def test_portfolio_rebalancing():
    """Verify that rebalancing preserves value logic."""
    n_assets, days, sims = 2, 2, 1
    initial = 100.0
    weights = [1.0, 0.0] # 100% in asset 1
    
    # Asset 1 doubles, Asset 2 stays flat
    asset_paths = np.array([[[1.0], [1.0]], [[2.0], [1.0]]])
    
    portfolio_paths = compute_portfolio_value(asset_paths, weights, initial)
    
    # Portfolio should double because 100% was in Asset 1
    assert portfolio_paths[1, 0] == 200.0
