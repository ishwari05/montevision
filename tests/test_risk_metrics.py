import pytest
import numpy as np
from src.risk_metrics import compute_var, compute_cvar

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
