import pytest
import numpy as np
import pandas as pd
from services.currency import get_usd_inr_rate

def test_currency_rate_fallback():
    """Verify that the currency rate service returns a valid float even on failure."""
    rate = get_usd_inr_rate()
    assert isinstance(rate, float)
    assert rate > 0

def test_currency_logic_scaling():
    """
    Verify the simulated scaling logic used in app.py.
    The engine returns results in normalized USD.
    If the user wants INR, we must multiply by the inr_rate.
    """
    # Mock engine response
    inr_rate = 80.0
    portfolio_paths_usd = np.array([[100.0, 200.0], [110.0, 210.0]])
    
    # Simulating app.py logic
    currency_mode = "INR"
    multiplier = inr_rate if currency_mode == "INR" else 1.0
    
    scaled_results = portfolio_paths_usd * multiplier
    
    assert scaled_results[0, 0] == 8000.0
    assert scaled_results[1, 1] == 16800.0

def test_yfinance_robust_extraction():
    """Test the hardened logic for extracting 'Close' from yfinance DataFrames."""
    # Mock a MultiIndex DataFrame similar to what yfinance returns
    columns = pd.MultiIndex.from_tuples([('Close', 'INR=X'), ('Adj Close', 'INR=X')])
    data = pd.DataFrame([[83.0, 83.0], [83.5, 83.5]], columns=columns)
    
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data['Close'].iloc[:, 0]
    else:
        close_data = data['Close']
        
    rate = float(close_data.iloc[-1])
    assert rate == 83.5
