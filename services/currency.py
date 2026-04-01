"""
Currency Conversion Services
Handles real-time exchange rates and localized currency formatting.
"""
import streamlit as st
import yfinance as yf
import logging
from typing import Optional

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)  # Cache rate for 1 hour
def get_usd_inr_rate() -> float:
    """Fetch live USD to INR exchange rate."""
    try:
        # yfinance ticker for USDINR
        data = yf.download("INR=X", period="1d", progress=False)
        if not data.empty:
            # Handle MultiIndex if present
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data['Close'].iloc[:, 0]
            else:
                close_data = data['Close']
            
            rate = float(close_data.iloc[-1])
            return rate

    except Exception as e:
        logger.warning(f"Could not fetch USD/INR rate: {e}. Using fallback 83.0")
    
    return 83.0  # Common fallback rate

def format_currency(value: float, currency: str = "USD") -> str:
    """Formats numeric values with localized symbols and grouping."""
    if currency == "INR":
        # Indian Numbering System (simplified for display)
        return f"₹{value:,.0f}"
    return f"${value:,.0f}"

def format_ticker(ticker: str, market: str) -> str:
    """Appends .NS for Indian tickers unless already present."""
    ticker = ticker.strip().upper()
    if market == "India":
        if not (ticker.endswith(".NS") or ticker.endswith(".BO")):
            ticker += ".NS"
    return ticker
