import pytest
import numpy as np
from src.gbm import simulate_portfolio_paths, generate_correlated_randoms

def test_generate_correlated_randoms_shape():
    """Verify the shape of correlated shocks."""
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    days, sims = 10, 5
    shocks = generate_correlated_randoms(cov, days, sims)
    assert shocks.shape == (days, 2, sims)

def test_gbm_output_ranges():
    """Verify that simulated prices are strictly positive."""
    S0 = np.array([100.0, 50.0])
    mu = np.array([0.05, 0.03])
    cov = np.diag([0.04, 0.01])
    days, sims = 100, 10
    
    paths, regimes, crashes = simulate_portfolio_paths(
        S0, mu, cov, days, sims, dt=1/252, config=None
    )
    
    assert paths.shape == (days, 2, sims)
    assert np.all(paths > 0)
    assert paths[0, 0, 0] == 100.0
    assert paths[0, 1, 0] == 50.0

def test_cholesky_stabilization():
    """Verify stabilization logic for non-positive definite matrices."""
    # Singular matrix
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])
    days, sims = 10, 5
    shocks = generate_correlated_randoms(cov, days, sims)
    assert shocks.shape == (10, 2, 5)
