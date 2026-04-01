# Portfolio Oracle: Advanced Stochastic Risk Engine
### Institutional-grade portfolio simulation and tail-risk analytics

---

## Overview
**Portfolio Oracle** is a sophisticated quantitative finance dashboard designed to model complex market dynamics and portfolio trajectories using stochastic processes. Unlike traditional linear projections, this engine accounts for non-normal distributions, market regime shifts, and "Black Swan" events to provide a realistic assessment of portfolio survival and risk exposure.

By utilizing multi-dimensional Geometric Brownian Motion (GBM) coupled with Markov-chain regime switching, Portfolio Oracle allows investors and analysts to stress-test asset allocations against historical correlations and synthetic volatility shocks.

---

## Features

### Simulation Engine
*   **Correlated Asset Modeling**: Implements Cholesky decomposition to maintain mathematically accurate cross-asset relationships based on historical covariance matrices.
*   **Stochastic Trajectories**: Generates thousands of independent paths for $N$ assets simultaneously using vectorized NumPy operations.
*   **Automated Rebalancing**: Supports periodic capital redistribution to strictly enforce strategic asset allocation targets throughout the simulation horizon.

### Risk & Tail-Event Modeling
*   **Regime Switching**: Employs a Markov process to simulate transitions between distinct market regimes (e.g., Bull and Bear markets) with unique drift and volatility parameters.
*   **Black Swan Simulation**: Models rare, catastrophic market events using Poisson-clustered volatility spikes and significant price gaps.
*   **Advanced Metrics**: Real-time calculation of Value at Risk (VaR), Conditional Value at Risk (CVaR/Expected Shortfall), and Maximum Drawdown across all simulated paths.

### Interactive Visualization
*   **Confidence Channels**: Multi-percentile "Fan Charts" visualizing the probability distribution of wealth over time.
*   **3D Topology**: High-resolution surface plots depicting the evolution of portfolio variance across the simulation set.
*   **Sensitivity Analysis**: Dynamic heatmaps identifying the intersection of withdrawal rates and equity exposure on portfolio survival probability.

### UI/UX
*   **Premium Interface**: A clean, minimalist dashboard built with Streamlit, featuring high-contrast typography and a professional financial aesthetic.
*   **Educational Landing Page**: Integrated onboarding mode providing sample visualizations and conceptual explanations of stochastic modeling for non-technical stakeholders.

---

## How It Works

1.  **Data Acquisition**: Retrieves historical adjusted closing prices for user-defined tickers via the Yahoo Finance API.
2.  **Parameter Extraction**: Calculates annualized returns (drift), volatility, and the correlation matrix.
3.  **Monte Carlo Execution**: 
    *   Generates correlated random shocks.
    *   Applies regime-specific adjustments and Poisson-distributed jump diffusion.
    *   Iteratively updates asset prices based on the GBM SDE: $dS_t = \mu S_t dt + \sigma S_t dW_t$.
4.  **Portfolio Aggregation**: Combines individual asset paths into a unified portfolio value, incorporating cash flows and rebalancing.
5.  **Risk Analysis**: Post-processes the simulation data to extract statistical risk bounds and survival probabilities.

---

## Tech Stack
*   **Language**: Python 3.10+
*   **Quantitative Logic**: NumPy, SciPy, Pandas
*   **Financial Data**: yfinance
*   **Frontend Framework**: Streamlit
*   **Visuals**: Plotly (Interactive Charts)

---

## Project Structure
```text
portfolio-simulator/
├── data/                 # Local cache for processed market data
├── src/                  # Specialized analytics (Sensitivity functions)
├── services/             # Core logic (Simulation Engine, Data Loaders)
├── components/           # UI modules (Landing Page, Metric Cards, Charts)
├── app.py                # Main application entry and UI orchestration
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```

---

## Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/username/portfolio-oracle.git
    cd portfolio-oracle
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

---

## Key Highlights

### Realism Through Non-Linear Modeling
Most simulators assume constant volatility and normal distributions. Portfolio Oracle acknowledges that markets are "fat-tailed" and prone to sudden regime shifts. By integrating jump-diffusion and switching models, the engine provides a more rigorous "worst-case" analysis than standard tools.

### Tail-Risk Decision Support
The inclusion of CVaR and survival heatmaps shifts the focus from "average returns" to "tail-risk mitigation." This supports better decision-making for long-term horizons where avoiding ruin is more critical than maximizing immediate gains.

---

## Future Improvements
*   **Asset Class Expansion**: Direct integration for fixed-income instruments and derivative pricing.
*   **Enhanced Optimization**: Implementation of Black-Litterman and Mean-Variance Efficient Frontier optimization to suggest ideal weights.
*   **Factor Modeling**: Decomposing returns into Fama-French factors for deeper attribution analysis.

---

## Disclaimer
*This software is for educational and research purposes only. It does not constitute financial advice. Quantitative models are based on historical data and theoretical assumptions; future market performance may differ significantly from simulated results.*
