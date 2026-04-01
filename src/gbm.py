"""
Multi-Asset Correlated Geometric Brownian Motion (GBM) Engine.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

from src.black_swan import generate_crash_events
from src.regime import generate_regime_sequence

def generate_correlated_randoms(cov_matrix: np.ndarray, days: int, n_simulations: int) -> np.ndarray:
    """
    Applies Cholesky decomposition to translate independent normal variables
    into correlated shocks.
    """
    n_assets = cov_matrix.shape[0]
    
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix not positive definite. Injecting epsilon diagonal stabilizer.")
        epsilon = 1e-8
        cov_matrix_stable = cov_matrix + epsilon * np.eye(n_assets)
        L = np.linalg.cholesky(cov_matrix_stable)
        
    Z = np.random.standard_normal((days, n_assets, n_simulations))
    correlated_Z = np.einsum('ij, djs -> dis', L, Z)
    return correlated_Z

def simulate_portfolio_paths(S0_vector: np.ndarray, 
                             mu_vector: np.ndarray, 
                             cov_matrix: np.ndarray, 
                             days: int, 
                             n_simulations: int,
                             dt: float = 1/252,
                             config=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates multidimensional stock paths via vectorized GBM with isolated
    extreme statistical fat-tail crashes and dynamic market regimes simultaneously natively.
    """
    n_assets = len(S0_vector)
    variances = np.diag(cov_matrix)
    
    correlated_Z = generate_correlated_randoms(cov_matrix, days - 1, n_simulations)
    
    # Phase 4 Extreme Tail Risk Evaluation (Black Swans)
    if config and getattr(config, 'ENABLE_BLACK_SWAN', False):
        crash_muls, bs_vol_muls = generate_crash_events(days - 1, n_simulations, config)
    else:
        crash_muls = np.ones((days - 1, n_simulations))
        bs_vol_muls = np.ones((days - 1, n_simulations))
        
    # Phase 6 Active Market Regimes (Bull/Bear)
    if config and getattr(config, 'ENABLE_REGIME_SWITCHING', False):
        regime_states = generate_regime_sequence(days - 1, n_simulations, getattr(config, 'REGIME_TRANSITION_MATRIX', [[1,0],[0,1]]))
        regime_mu_muls = np.where(regime_states == 0, config.BULL_MU_MULTIPLIER, config.BEAR_MU_MULTIPLIER)
        regime_sig_muls = np.where(regime_states == 0, config.BULL_SIGMA_MULTIPLIER, config.BEAR_SIGMA_MULTIPLIER)
    else:
        regime_states = np.zeros((days - 1, n_simulations), dtype=int)
        regime_mu_muls = np.ones((days - 1, n_simulations))
        regime_sig_muls = np.ones((days - 1, n_simulations))
        
    # Broadcast mathematical variables dynamically onto a multi-asset matrix structure
    # Arrays shape changes: (assets) -> (1, assets, 1) mapping across (days-1, 1, n_sims)
    daily_mu = mu_vector.reshape(1, n_assets, 1) * np.expand_dims(regime_mu_muls, axis=1)
    daily_var = variances.reshape(1, n_assets, 1) * np.expand_dims(regime_sig_muls**2, axis=1)
    
    drift = (daily_mu - 0.5 * daily_var) * dt
    
    # Scale compound dynamic volatility clusters over the stochastic diffusions (Bull/Bear * Crisis)
    total_vol_muls = regime_sig_muls * bs_vol_muls
    vol_muls_expanded = np.expand_dims(total_vol_muls, axis=1) 
    diffusion = correlated_Z * np.sqrt(dt) * vol_muls_expanded
    
    # Compute base fractional log returns scaling exponentially
    daily_returns = np.exp(drift + diffusion)
    
    # Apply instantaneous sudden systemic Black Swan drops onto paths dynamically
    crash_muls_expanded = np.expand_dims(crash_muls, axis=1)
    adjusted_returns = daily_returns * crash_muls_expanded
    
    price_paths = np.zeros((days, n_assets, n_simulations))
    price_paths[0, :, :] = S0_vector.reshape(n_assets, 1)
    
    # Re-normalize continuous geometric transformations recursively across arrays flawlessly
    price_paths[1:, :, :] = S0_vector.reshape(1, n_assets, 1) * np.cumprod(adjusted_returns, axis=0)
    
    # Expose arrays dynamically so utilities can visualize underlying regime contexts later
    return price_paths, regime_states, crash_muls
