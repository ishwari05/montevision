"""
Risk analytics module for generating statistical insights.
"""
import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def compute_percentiles(portfolio_paths: np.ndarray, percentiles: list = [5, 50, 95]) -> dict:
    """Computes specific percentiles for the final portfolio values."""
    final_values = portfolio_paths[-1, :]
    return {p: np.percentile(final_values, p) for p in percentiles}

def compute_var(final_values: np.ndarray, initial_investment: float, confidence_level: float = 0.95) -> float:
    """
    Computes Value at Risk (VaR).
    'With 95% confidence, the maximum expected loss is X'.
    """
    alpha = (1.0 - confidence_level) * 100
    percentile_val = np.percentile(final_values, alpha)
    var = initial_investment - percentile_val
    return max(0.0, var)

def compute_cvar(final_values: np.ndarray, initial_investment: float, confidence_level: float = 0.95) -> float:
    """
    Computes Conditional Value at Risk (CVaR).
    The average loss beyond the VaR threshold.
    """
    alpha = (1.0 - confidence_level) * 100
    percentile_val = np.percentile(final_values, alpha)
    
    worst_cases = final_values[final_values <= percentile_val]
    if len(worst_cases) == 0:
        return 0.0
        
    cvar = initial_investment - np.mean(worst_cases)
    return max(0.0, cvar)

def compute_sharpe_ratio(portfolio_paths: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Computes the annualized Sharpe ratio from all simulated portfolio returns."""
    previous_values = portfolio_paths[:-1, :]
    current_values = portfolio_paths[1:, :]
    period_returns = np.divide(
        current_values,
        previous_values,
        out=np.ones_like(current_values, dtype=float),
        where=previous_values != 0,
    ) - 1.0
    excess_returns = period_returns - risk_free_rate / periods_per_year
    return_std = np.std(excess_returns)

    if return_std == 0.0:
        return 0.0

    return float(np.mean(excess_returns) / return_std * np.sqrt(periods_per_year))

def compute_drawdowns(portfolio_paths: np.ndarray) -> tuple[float, float]:
    """
    Computes aggregate drawdown metrics.
    Returns: (average_max_drawdown, worst_drawdown)
    """
    # rolling structural max natively across vertical chronological axis (0)
    rolling_max = np.maximum.accumulate(portfolio_paths, axis=0)
    
    # Drawdowns as negative percentages with stability guard
    drawdowns = (portfolio_paths - rolling_max) / (rolling_max + 1e-10)

    
    # Extrapolate worst max downswings exclusively 
    max_drawdowns_per_sim = np.min(drawdowns, axis=0) 
    
    avg_max_drawdown = np.mean(max_drawdowns_per_sim)
    worst_drawdown = np.min(max_drawdowns_per_sim)
    
    return avg_max_drawdown, worst_drawdown

def simulate_withdrawals(portfolio_paths: np.ndarray, monthly_withdrawal: float) -> float:
    """
    Simulates monthly structural withdrawals against compounding curves.
    Evaluates success rates where capital does not zero out.
    """
    days, n_sims = portfolio_paths.shape
    
    # Determine geometric daily compounding rate matrix (with epsilon guard)
    daily_returns = np.ones_like(portfolio_paths)
    daily_returns[1:] = portfolio_paths[1:] / (portfolio_paths[:-1] + 1e-10)

    
    # Establish dynamic adjusted simulation memory
    adjusted_paths = np.zeros_like(portfolio_paths)
    adjusted_paths[0] = portfolio_paths[0]
    
    for t in range(1, days):
        adjusted_paths[t] = adjusted_paths[t-1] * daily_returns[t]
        
        # Monthly abstraction ~ 21 trading days
        if t % 21 == 0:
            adjusted_paths[t] -= monthly_withdrawal
            # Protect capital bounds artificially above zero physically
            adjusted_paths[t] = np.maximum(adjusted_paths[t], 0.0)
            
    # Success evaluation: Any simulation traversing without snapping capital to 0
    successes = np.all(adjusted_paths > 0, axis=0)
    success_rate = (np.sum(successes) / n_sims) * 100
    return success_rate

def generate_summary_table(portfolio_paths: np.ndarray, initial_investment: float, monthly_withdrawal: float, confidence_level: float = 0.95):
    """Generates the integrated master statistical matrix."""
    final_values = portfolio_paths[-1, :]
    
    percentiles = compute_percentiles(portfolio_paths)
    var = compute_var(final_values, initial_investment, confidence_level)
    cvar = compute_cvar(final_values, initial_investment, confidence_level)
    sharpe_ratio = compute_sharpe_ratio(portfolio_paths)
    avg_dd, worst_dd = compute_drawdowns(portfolio_paths)
    
    if monthly_withdrawal > 0:
        success_rate = simulate_withdrawals(portfolio_paths, monthly_withdrawal)
    else:
        success_rate = 100.0
        
    summary = {
        "Metric": [
            "Initial Investment",
            "Median Final Value",
            "Worst Case (5%)",
            "Best Case (95%)",
            "Sharpe Ratio",
            f"VaR ({int(confidence_level*100)}%)",
            f"CVaR ({int(confidence_level*100)}%)",
            "Success Rate",
            "Avg Max Drawdown",
            "Worst Drawdown"
        ],
        "Value": [
            f"${initial_investment:,.2f}",
            f"${percentiles[50]:,.2f}",
            f"${percentiles[5]:,.2f}",
            f"${percentiles[95]:,.2f}",
            f"{sharpe_ratio:.2f}",
            f"${var:,.2f}",
            f"${cvar:,.2f}",
            f"{success_rate:.1f}%",
            f"{avg_dd*100:.2f}%",
            f"{worst_dd*100:.2f}%"
        ]
    }
    
    df = pd.DataFrame(summary)
    
    # Console Interface Standard Output
    print("\n" + "=" * 45)
    print(" PORTFOLIO RISK & PERFORMANCE SUMMARY")
    print("=" * 45)
    for index, row in df.iterrows():
        print(f"{row['Metric']:<25} {row['Value']:>19}")
    print("=" * 45 + "\n")
    logger.info(f"Interpretation: With {int(confidence_level*100)}% confidence, the maximum expected loss is ${var:,.2f}.")
    
    os.makedirs('data', exist_ok=True)
    df.to_csv("data/risk_summary.csv", index=False)
    
    return df
