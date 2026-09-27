"""
Simulation Services Layer
Handles high-level orchestration, caching, and multi-scenario execution.
"""
import streamlit as st
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List

from src.data_loader import fetch_multi_asset_data, compute_statistics
from src.gbm import simulate_portfolio_paths
from src.portfolio import compute_portfolio_value
from src.risk_metrics import compute_var, compute_cvar, compute_drawdowns, compute_sharpe_ratio
from services.currency import get_usd_inr_rate

# --- Internal Institutional Defaults (Previously in config.py) ---
class SimulationConfig:
    def __init__(self, enable_black_swan: bool, enable_regime: bool):
        self.ENABLE_BLACK_SWAN = enable_black_swan
        self.ENABLE_REGIME_SWITCHING = enable_regime
        
        # Black Swan Parameters
        self.BLACK_SWAN_LAMBDA = 1.0
        self.MAX_CRASHES_PER_SIM = 3
        self.CRISIS_PROB_DAILY = 0.005
        self.VOLATILITY_SPIKE_MULTIPLIER = 2.0
        self.BLACK_SWAN_MAGNITUDE_RANGE = (-0.35, -0.15)
        self.CRISIS_DURATION_RANGE = (10, 30)
        
        # Regime Switching Parameters
        self.REGIME_TRANSITION_MATRIX = [[0.99, 0.01], [0.03, 0.97]]
        self.BULL_MU_MULTIPLIER = 1.2
        self.BEAR_MU_MULTIPLIER = 0.8
        self.BULL_SIGMA_MULTIPLIER = 0.9
        self.BEAR_SIGMA_MULTIPLIER = 1.4

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def get_cached_market_data(tickers: List[str], years: int):
    """Fetch and compute statistics with Streamlit caching and USD normalization."""
    prices_df = fetch_multi_asset_data(tickers, years)
    inr_rate = get_usd_inr_rate()
    for col in prices_df.columns:
        if col.endswith(".NS") or col.endswith(".BO"):
            prices_df[col] = prices_df[col] / inr_rate
    S0_vector, daily_mu, daily_cov, corr_df = compute_statistics(prices_df)
    return S0_vector, daily_mu, daily_cov, corr_df, inr_rate

def run_production_simulation(
    initial_investment: float,
    time_horizon: int,
    monthly_withdrawal: float,
    num_simulations: int,
    tickers: List[str],
    weights: List[float],
    market_focus: str = "US",
    enable_black_swan: bool = True,
    enable_regime: bool = True
) -> Dict[str, Any]:
    """
    Executes the full simulation pipeline. 
    Optimization: Returns both Normal and Scenario-specific results for comparison.
    """
    # Map back to internal variable names
    years = time_horizon
    n_sims = num_simulations
    
    # 1. Fetch & Stat Calculation (Cached)
    S0_vector, daily_mu, daily_cov, corr_df, inr_rate = get_cached_market_data(tickers, years)
    
    annualized_mu = daily_mu * 252
    annualized_cov = daily_cov * 252
    sim_days = years * 252
    
    # 2. Config Initialization
    current_config = SimulationConfig(enable_black_swan, enable_regime)
    
    # 3. Execution: Scenario (Black Swan/Regime potentially ON)
    price_paths, regimes, crashes = simulate_portfolio_paths(
        S0_vector=S0_vector,
        mu_vector=annualized_mu,
        cov_matrix=annualized_cov,
        days=sim_days,
        n_simulations=n_sims,
        dt=1/252,
        config=current_config
    )
    
    portfolio_paths = compute_portfolio_value(
        asset_paths=price_paths,
        weights=weights,
        initial_investment=initial_investment,
        rebalance_freq_days=252 
    )
    
    # Apply withdrawals
    if monthly_withdrawal > 0:
        for t in range(1, sim_days):
            if t % 21 == 0:
                portfolio_paths[t, :] -= monthly_withdrawal
                portfolio_paths[t, :] = np.maximum(portfolio_paths[t, :], 0)
    
    # 4. Extract Metrics
    final_values = portfolio_paths[-1, :]
    median_final = np.median(final_values)
    var_95 = compute_var(final_values, initial_investment, 0.95)
    cvar_95 = compute_cvar(final_values, initial_investment, 0.95)
    _, worst_drawdown = compute_drawdowns(portfolio_paths)
    sharpe_ratio = compute_sharpe_ratio(portfolio_paths)
    success_rate = (final_values > 0).sum() / n_sims * 100
    
    return {
        "portfolio_paths": portfolio_paths,
        "final_values": final_values,
        "regimes": regimes,
        "crashes": crashes,
        "metrics": {
            "median_final": median_final,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": abs(worst_drawdown),
            "success_rate": success_rate,
        },
        "statistics": {
            "corr_df": corr_df,
            "S0": S0_vector,
            "mu": annualized_mu,
            "cov": annualized_cov,
            "inr_rate": inr_rate
        },
        "params": {
            "tickers": tickers,
            "weights": weights,
            "days": sim_days,
            "initial_investment": initial_investment
        }
    }
