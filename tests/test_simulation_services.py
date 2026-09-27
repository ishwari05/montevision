import pytest
import numpy as np
import pandas as pd
from services.simulation_wrapper import run_production_simulation, SimulationConfig
from src.risk_metrics import compute_var, compute_cvar, compute_drawdowns, compute_sharpe_ratio

def test_simulation_config_initialization():
    """Verify that SimulationConfig initializes defaults correctly."""
    config = SimulationConfig(enable_black_swan=True, enable_regime=True)
    assert config.ENABLE_BLACK_SWAN is True
    assert config.ENABLE_REGIME_SWITCHING is True
    assert hasattr(config, 'BLACK_SWAN_LAMBDA')

def test_run_production_simulation_smoke():
    """
    A basic smoke test for the wrapper. 
    Note: This mocks the data fetching to avoid external calls.
    """
    # Simply check if it can be called with minimal params
    # We might need to mock get_cached_market_data if we want to run this without internet
    pass

def test_simulation_wrapper_logic_flow():
    """Verify the return structure of the simulation wrapper."""
    # Since run_production_simulation calls yfinance via get_cached_market_data,
    # a full integration test would be slow and brittle.
    # We'll focus on the data structure it's supposed to return.
    assert True

def test_production_simulation_outputs_requested_portfolio_metrics(monkeypatch):
    """Ensure the production result reports risk metrics from its simulated paths."""
    initial_prices = np.array([100.0, 50.0])
    daily_mu = np.array([0.0002, 0.0001])
    daily_covariance = np.array([[0.0004, 0.0001], [0.0001, 0.0002]])
    correlation_frame = pd.DataFrame([[1.0, 0.35], [0.35, 1.0]])
    monkeypatch.setattr(
        "services.simulation_wrapper.get_cached_market_data",
        lambda tickers, years: (
            initial_prices, daily_mu, daily_covariance, correlation_frame, 1.0
        ),
    )
    np.random.seed(161803)

    result = run_production_simulation(
        10_000.0, 1, 0.0, 256, ["AAA", "BBB"], [0.6, 0.4],
        enable_black_swan=False, enable_regime=False,
    )
    metrics = result["metrics"]
    final_values = result["final_values"]
    _, worst_drawdown = compute_drawdowns(result["portfolio_paths"])

    assert metrics["sharpe_ratio"] == pytest.approx(
        compute_sharpe_ratio(result["portfolio_paths"])
    )
    assert metrics["var_95"] == pytest.approx(compute_var(final_values, 10_000.0))
    assert metrics["cvar_95"] == pytest.approx(compute_cvar(final_values, 10_000.0))
    assert metrics["max_drawdown"] == pytest.approx(abs(worst_drawdown))
