"""
Data loader module for multi-asset historical data and statistics.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def fetch_multi_asset_data(tickers: list[str], years: int = 5) -> pd.DataFrame:
    """
    Fetches historical data for multiple correlated assets.

    Args:
        tickers (list[str]): List of stock/bond/commodity tickers.
        years (int): Number of years of historical data to fetch.

    Returns:
        pd.DataFrame: DataFrame containing adjusted close prices for all tickers.
    """
    logger.info(f"Fetching {years} years of historical data for {tickers}...")
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception as e:
        logger.error(f"Failed to fetch data for {tickers}: {e}")
        raise

    if data.empty:
        raise ValueError(f"No data found for tickers {tickers}.")

    # Modern yfinance returns MultiIndex DataFrames
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close']
        else:
            prices = data['Close']
    else:
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']
            
    # If a single ticker was passed by mistake, ensure it's a 2D Frame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        prices.columns = tickers

    # Ensure all columns exist and sequence matches
    prices = prices[tickers]
    
    # Handle missing values: Forward fill first to carry over previous closes (holidays),
    # then drop any remaining NaNs (usually where different assets have different start dates).
    prices = prices.ffill().dropna()
    
    if prices.empty:
        raise ValueError(f"No overlapping data found for tickers {tickers}. Check if one asset has very short history.")
    
    # Cache data to CSV
    import os
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', "multi_asset_historical.csv")
    prices.to_csv(csv_path)
    logger.info(f"Fetched {len(prices)} trading days for {len(tickers)} assets.")
    
    return prices

def compute_statistics(prices_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Calculates statistical parameters required for multi-asset GBM.

    Args:
        prices_df (pd.DataFrame): DataFrame of correlated asset prices.

    Returns:
        tuple: (S0_vector, mu_vector, cov_matrix, corr_df)
    """
    logger.info("Computing returns, covariance, and correlation matrices...")
    
    returns = prices_df.pct_change().dropna()
    
    S0_vector = prices_df.iloc[-1].values
    mu_vector = returns.mean().values
    
    # Daily covariance backing the Cholesky decompositon
    cov_matrix = returns.cov().values
    
    # Correlation DataFrame for clean printing/review
    corr_df = returns.corr()
    
    return S0_vector, mu_vector, cov_matrix, corr_df
