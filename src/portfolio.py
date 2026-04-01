"""
Portfolio construction and rebalancing logic.
"""
import numpy as np

def compute_portfolio_value(asset_paths: np.ndarray, 
                            weights: list[float], 
                            initial_investment: float = 10000.0, 
                            rebalance_freq_days: int = 0) -> np.ndarray:
    """
    Computes portfolio value across time, given optional time-based rebalancing.

    Args:
        asset_paths (np.ndarray): Shape (days, n_assets, n_simulations).
        weights (list[float]): Target portfolio weights.
        initial_investment (float): Starting portfolio value.
        rebalance_freq_days (int): If > 0, rebalances portfolio periodically to reset weights.

    Returns:
        np.ndarray: Portfolio values of shape (days, n_simulations).
    """
    days, n_assets, n_sims = asset_paths.shape
    weights = np.array(weights)
    
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError(f"Portfolio weights must sum to 1. Got {np.sum(weights)}")
        
    portfolio_values = np.zeros((days, n_sims))
    
    # Capital allocated per asset per simulation natively mapping to (n_assets, n_sims)
    current_capital = initial_investment * weights.reshape(n_assets, 1)
    
    # Unit shares owned per simulation
    # Shares = Capital / Initial Asset Price
    S0_vector = asset_paths[0, :, :] 
    shares = current_capital / S0_vector
    
    for t in range(days):
        current_prices = asset_paths[t, :, :] 
        
        # Portfolio value is total value of shares times the current valuation
        # Summing over assets gives direct aggregate portfolio value over the simulations
        portfolio_values[t, :] = np.sum(shares * current_prices, axis=0)
        
        # Periodic Rebalancing
        if rebalance_freq_days > 0 and t > 0 and t % rebalance_freq_days == 0:
            total_value = portfolio_values[t, :] 
            
            # Rebalance generic capital to exactly align with mathematical target weights
            current_capital = total_value.reshape(1, n_sims) * weights.reshape(n_assets, 1)
            
            # Re-buy shares at current prices, enforcing rebalancing trades
            shares = current_capital / current_prices
            
    return portfolio_values
