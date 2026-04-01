"""
Sensitivity Analysis Engine
Dynamically executes an N x M multidimensional execution matrix comparing survival probabilities 
against fluctuating variables (E.g Withdrawals vs Allocations).
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List

from src.gbm import simulate_portfolio_paths
from src.portfolio import compute_portfolio_value

logger = logging.getLogger(__name__)

def run_withdrawal_allocation_sensitivity(
    S0_vector: np.ndarray,
    annualized_mu: np.ndarray,
    annualized_cov: np.ndarray,
    days: int,
    n_sims: int,
    initial_investment: float,
    stock_idx: int,
    bond_idx: int,
    stock_weights: np.ndarray,
    withdrawals: np.ndarray,
    base_weights: List[float],
    config=None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Executes parallel Monte Carlo universes evaluating 2D topological matrix grids.
    
    Returns:
        np.ndarray: Matrix of shape (len(stock_weights), len(withdrawals)) containing survival percentages.
        pd.DataFrame: Flat dataframe for CSV tracking.
    """
    grid_rows = len(stock_weights)
    grid_cols = len(withdrawals)
    
    success_matrix = np.zeros((grid_rows, grid_cols))
    csv_data = []
    
    logger.info(f"Firing up {grid_rows}x{grid_cols} N-Dimensional Sensitivity Scanner...")
    
    for i, w_stock in enumerate(stock_weights):
        # Dynamically inject the new test weight allocation natively
        test_weights = list(base_weights)
        test_weights[stock_idx] = w_stock
        
        # Calculate theoretical remaining balance allowed
        remaining_bound = 1.0 - w_stock
        
        # All non-stock base allocations total
        other_base_sum = sum(base_weights) - base_weights[stock_idx]
        
        # Scale remaining allocations proportionally to fit the remaining dimensional boundary structurally!
        if other_base_sum > 0:
            for idx in range(len(test_weights)):
                if idx != stock_idx:
                    test_weights[idx] = remaining_bound * (base_weights[idx] / other_base_sum)
        else:
            # If everything else was zero, just dump it all into the bond index
            test_weights[bond_idx] = remaining_bound
            
        # Standard mathematical execution identically across the localized allocation
        paths, _, _ = simulate_portfolio_paths(
            S0_vector=S0_vector,
            mu_vector=annualized_mu,
            cov_matrix=annualized_cov,
            days=days,
            n_simulations=n_sims,
            dt=1/252,
            config=config
        )
        
        for j, withdr in enumerate(withdrawals):
            portfolio_vals = compute_portfolio_value(
                asset_paths=paths,
                weights=test_weights,
                initial_investment=initial_investment,
                rebalance_freq_days=21 # Standard monthly mapping
            )
            
            # Replicate standard chronologically drained survival testing
            adjusted_paths = np.copy(portfolio_vals)
            if withdr > 0:
                daily_returns = np.ones_like(portfolio_vals)
                daily_returns[1:] = portfolio_vals[1:] / (portfolio_vals[:-1] + 1e-10)
                for t in range(1, days):
                    adjusted_paths[t] = adjusted_paths[t-1] * daily_returns[t]

                    if t % 21 == 0:
                        adjusted_paths[t] -= withdr
                        adjusted_paths[t] = np.maximum(adjusted_paths[t], 0.0)
                        
            # Evaluate native success rates efficiently
            survival_rate = (adjusted_paths[-1, :] > 0).sum() / n_sims * 100
            success_matrix[i, j] = survival_rate
            
            csv_data.append({
                "Stock_Allocation_Pct": w_stock * 100,
                "Bond_Allocation_Pct": test_weights[bond_idx] * 100,
                "Monthly_Withdrawal": withdr,
                "Survival_Probability": survival_rate
            })
            
    df = pd.DataFrame(csv_data)
    return success_matrix, df
