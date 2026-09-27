import pytest
import numpy as np
from src.gbm import simulate_portfolio_paths, generate_correlated_randoms
from src.portfolio import compute_portfolio_value

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

def test_zero_volatility_matches_compound_return_baseline():
    """With no volatility, all paths follow the exact deterministic growth curve."""
    initial_prices = np.array([100.0, 50.0])
    period_return = 0.04
    mu = np.full(2, np.log1p(period_return))
    covariance = np.zeros((2, 2))
    periods = 12
    initial_investment = 1000.0

    asset_paths, _, _ = simulate_portfolio_paths(
        initial_prices, mu, covariance, periods + 1, 4, dt=1.0
    )
    portfolio_paths = compute_portfolio_value(
        asset_paths, [0.4, 0.6], initial_investment
    )
    expected_terminal_value = initial_investment * (1 + period_return) ** periods

    np.testing.assert_allclose(
        portfolio_paths[-1], expected_terminal_value, rtol=1e-12, atol=1e-12
    )

def test_simulated_asset_correlations_match_input_matrix():
    """One-step log returns from 100,000 paths recover the input correlations."""
    correlation = np.array([
        [1.0, 0.65, -0.2],
        [0.65, 1.0, 0.1],
        [-0.2, 0.1, 1.0],
    ])
    np.random.seed(314159)

    asset_paths, _, _ = simulate_portfolio_paths(
        np.ones(3), np.zeros(3), correlation, 2, 100_000, dt=1.0
    )
    simulated_log_returns = np.log(asset_paths[1] / asset_paths[0])
    sample_correlation = np.corrcoef(simulated_log_returns)

    np.testing.assert_allclose(sample_correlation, correlation, atol=0.01, rtol=0.0)

def test_doubling_simulations_reduces_mean_standard_error_by_sqrt_two():
    """The standard error of terminal wealth scales inversely with sqrt(N)."""
    np.random.seed(271828)
    simulation_count = 20_000
    asset_paths, _, _ = simulate_portfolio_paths(
        np.array([100.0]), np.array([0.05]), np.array([[0.04]]),
        2, 2 * simulation_count, dt=1.0,
    )
    terminal_values = asset_paths[-1, 0]
    standard_error_n = np.std(terminal_values[:simulation_count], ddof=1) / np.sqrt(simulation_count)
    standard_error_2n = np.std(terminal_values, ddof=1) / np.sqrt(2 * simulation_count)

    assert standard_error_n / standard_error_2n == pytest.approx(np.sqrt(2), abs=0.02)
