import pytest
import numpy as np
from src.risk_metrics import compute_var, compute_cvar, compute_drawdowns, compute_sharpe_ratio

def test_var_calculation():
    """Verify VaR calculation against known cases."""
    # Data: 1 to 100
    data = np.arange(1, 101)
    initial = 100.0
    # 95th VaR means 5th percentile. 5th percentile of 1-100 is 5.
    # Loss = 100 - 5 = 95.
    var_95 = compute_var(data, initial, 0.95)
    assert np.isclose(var_95, 95.0, atol=1.0)

def test_cvar_calculation():
    """Verify CVaR is always >= VaR."""
    data = np.random.normal(100, 20, 1000)
    initial = 100.0
    var_95 = compute_var(data, initial, 0.95)
    cvar_95 = compute_cvar(data, initial, 0.95)
    
    assert cvar_95 >= var_95

def test_sharpe_and_maximum_drawdown_use_all_simulated_paths():
    """Verify path-based output metrics against hand-computed returns and drawdowns."""
    portfolio_paths = np.array([
        [100.0, 100.0],
        [110.0, 90.0],
        [99.0, 108.0],
        [108.9, 86.4],
    ])
    all_period_returns = portfolio_paths[1:] / portfolio_paths[:-1] - 1.0
    expected_sharpe = (
        np.mean(all_period_returns) / np.std(all_period_returns) * np.sqrt(252)
    )
    average_drawdown, worst_drawdown = compute_drawdowns(portfolio_paths)

    assert compute_sharpe_ratio(portfolio_paths) == pytest.approx(expected_sharpe)
    assert average_drawdown == pytest.approx(-0.15)
    assert worst_drawdown == pytest.approx(-0.2)
