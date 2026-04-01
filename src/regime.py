"""
Regime Switching Engine
Simulates Bull (0) and Bear (1) market cycles using Markov Chain probabilities.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_regime_sequence(days: int, n_simulations: int, trans_matrix: list, initial_state: int = 0) -> np.ndarray:
    """
    Generates a Markov chain representing chronological market regimes.
    0 = Bull Market (Growth)
    1 = Bear Market (Contraction)

    Args:
        days: Number of trading days to simulate.
        n_simulations: Number of parallel simulation paths.
        trans_matrix: 2x2 transition probability matrix logic.
                      [[P(0->0), P(0->1)],
                       [P(1->0), P(1->1)]]
        initial_state: 0 or 1.

    Returns:
        np.ndarray: Matrix of shape (days, n_simulations) containing 0 or 1.
    """
    logger.info("Initializing Chronological Markov Regime parameters (Bull vs Bear states)...")
    regimes = np.zeros((days, n_simulations), dtype=int)
    regimes[0, :] = initial_state
    
    trans_arr = np.array(trans_matrix)
    
    for t in range(1, days):
        current_states = regimes[t-1, :]
        probs_to_bear = trans_arr[current_states, 1]
        
        rand_rolls = np.random.random(n_simulations)
        # Process logic transition into Bear (1) if roll falls securely beneath threshold
        new_states = (rand_rolls < probs_to_bear).astype(int)
        regimes[t, :] = new_states
        
    bull_pct = np.sum(regimes == 0) / (days * n_simulations) * 100
    bear_pct = np.sum(regimes == 1) / (days * n_simulations) * 100
    logger.info(f"Regime Topography Output: {bull_pct:.1f}% Bull vs {bear_pct:.1f}% Bear across all chronological projections.")
    
    return regimes
