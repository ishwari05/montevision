import streamlit as st
import numpy as np
import pandas as pd

from services.simulation_wrapper import run_production_simulation
from components.metrics import display_summary_metrics
from components.charts import (
    plot_growth_bands, plot_3d_topology, 
    plot_sensitivity_heatmap, render_drawdown_analysis, 
    plot_final_distribution
)
from components.landing import render_landing_page
from src.sensitivity import run_withdrawal_allocation_sensitivity

# --- Page Configuration ---
st.set_page_config(
    page_title="Portfolio Risk Intelligence Oracle",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Light Theme ---
st.markdown("""
<style>
    /* White Background */
    .stApp {
        background-color: #ffffff !important;
        color: #1f2328 !important;
    }

    /* Reduce Top Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }



    /* Light Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f6f8fa !important;
        border-right: 1px solid #d0d7de !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }


    /* Professional Metrics & Panels */
    div[data-testid="stMetricValue"] {
        color: #0969da !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stExpander, .stTabs {
        background-color: #ffffff !important;
        border: 1px solid #d0d7de !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }

    /* Custom Class for Light Cards */
    .glass-card {
        background-color: #ffffff;
        border: 1px solid #d0d7de;
        box-shadow: 0 1px 3px rgba(31,35,40,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Global Typography (Dark Text) */
    h1, h2, h3, h4, h5, h6, p, label {
        font-family: 'Inter', sans-serif !important;
        color: #1f2328 !important;
    }
    
    /* Sidebar Text Fix */
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #1f2328 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "has_run" not in st.session_state:
    st.session_state["has_run"] = False

# --- SIDEBAR: Global Inputs ---
with st.sidebar:
    st.title("Strategies")
    
    # Presets
    with st.expander("Strategy Presets", expanded=False):
        col1, col2 = st.columns(2)
        if col1.button("US Institutional", use_container_width=True):
            st.session_state['tickers_input'] = "SPY, AGG, GLD"
            st.session_state['weights_input'] = "0.6, 0.3, 0.1"
        if col2.button("India Flex", use_container_width=True):
            st.session_state['tickers_input'] = "^NSEI, GOLD, BANKBEES.NS"
            st.session_state['weights_input'] = "0.5, 0.2, 0.3"

    # Core Parameters
    with st.expander("Settings & Simulation Rigor", expanded=True):
        market_focus = st.selectbox("Market Focus", ["US", "India", "Mixed"], help="Target region for data and currency.")
        currency_mode = st.selectbox("Display Currency", ["USD", "INR"], help="Base currency for visualization.")
        initial_investment = st.number_input(f"Capital Initial ({'₹' if currency_mode == 'INR' else '$'})", min_value=1000, value=100000, step=5000)
        time_horizon = st.slider("Horizon (Years)", 1, 30, 10)
        monthly_withdrawal = st.number_input(f"Monthly Outflow ({'₹' if currency_mode == 'INR' else '$'})", value=500, step=100)
        num_simulations = st.select_slider("Simulation Density", options=[500, 1000, 2500, 5000], value=1000)

    # Assets
    st.subheader("Asset Allocation")
    ticker_input = st.text_input("Tickers (CSV)", value=st.session_state.get('tickers_input', "SPY, AGG, GLD"))
    weight_input = st.text_input("Relative Weights", value=st.session_state.get('weights_input', "0.6, 0.3, 0.1"))
    
    # Risk Engines
    st.subheader("Risk Parameters")
    enable_black_swan = st.toggle("Black Swan Modeling", value=True)
    regime_switching = st.toggle("Regime Switching", value=True)

    # Parse inputs
    tickers = [t.strip() for t in ticker_input.split(',')]
    try:
        weights = [float(w.strip()) for w in weight_input.split(',')]
    except ValueError:
        weights = []

    # Validation Layer
    is_valid = True
    if len(tickers) != len(weights):
        st.error("Mismatch: Asset count must match weight count.")
        is_valid = False
    elif abs(sum(weights) - 1.0) > 0.01:
        st.warning(f"Rebalancing required (Sum: {sum(weights):.2f}). Adjusting to 100% internally.")
        weights = [w/sum(weights) for w in weights] if sum(weights) > 0 else []

    st.divider()
    if st.button("RUN MONTE CARLO", type="primary", use_container_width=True, disabled=not is_valid):
        st.session_state["has_run"] = True
        st.rerun()

# --- MAIN PANEL ---
if not st.session_state["has_run"]:
    render_landing_page()
else:
    # Dashboard Header
    st.markdown("# Portfolio Risk Simulator")
    st.markdown("##### Professional stochastic engine for cross-market stress testing and tail-risk analysis.")
    st.divider()

    with st.spinner("Executing stochastic engine..."):
        res = run_production_simulation(
            initial_investment, time_horizon, monthly_withdrawal, num_simulations,
            tickers, weights, market_focus, enable_black_swan, regime_switching
        )
        
        if "error" in res:
            st.error(f"Engine Failure: {res['error']}")
            st.stop()

        # Currency Scaling Logic
        # The simulation engine normalizes assets to USD for statistical consistency.
        # We scale back to INR only if the user explicitly requests INR display.
        multiplier = 1.0
        inr_rate = res['statistics'].get('inr_rate', 83.0)
        
        if currency_mode == "INR":
            multiplier = inr_rate
        
        scaled_results = res['portfolio_paths'] * multiplier

        
        # --- A. Top KPI Grid ---
        display_summary_metrics(res, currency=currency_mode, multiplier=multiplier)
        
        # --- B. Primary Visualization ---
        st.markdown("### Portfolio Growth Confidence Channels")
        fig_fan = plot_growth_bands(scaled_results, currency=currency_mode)
        st.plotly_chart(fig_fan, use_container_width=True)

        # --- C. Secondary Visualizations ---
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_dd = render_drawdown_analysis(scaled_results)
            st.plotly_chart(fig_dd, use_container_width=True)
        with col_c2:
            fig_dist = plot_final_distribution(scaled_results, initial_investment * multiplier, currency_mode)
            st.plotly_chart(fig_dist, use_container_width=True)

        # --- D. Advanced Analytics Expandable ---
        with st.expander("Advanced Risk Analytics & Stress Testing", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Sensitivity Matrix", "3D Evolution", "Correlation"])
            
            with tab1:
                # Setup sensitivity parameters
                stk_idx = 0
                bnd_idx = 1 if len(tickers) > 1 else 0
                stock_test = np.linspace(0.2, 1.0, 5)
                withdraw_test = np.linspace(0, initial_investment * 0.1, 5)
                
                with st.spinner("Calculating sensitivity surface..."):
                    h_matrix, h_df = run_withdrawal_allocation_sensitivity(
                        S0_vector=res['statistics']['S0'],
                        annualized_mu=res['statistics']['mu'],
                        annualized_cov=res['statistics']['cov'],
                        days=res['params']['days'],
                        n_sims=max(100, num_simulations // 10),
                        initial_investment=initial_investment,
                        stock_idx=stk_idx,
                        bond_idx=bnd_idx,
                        stock_weights=stock_test,
                        withdrawals=withdraw_test,
                        base_weights=weights
                    )
                
                fig_h = plot_sensitivity_heatmap(h_matrix, [f"${int(w):,}" for w in withdraw_test], [f"{int(w*100)}%" for w in stock_test])
                st.plotly_chart(fig_h, use_container_width=True)
            
            with tab2:
                fig_3d = plot_3d_topology(scaled_results, currency=currency_mode)
                st.plotly_chart(fig_3d, use_container_width=True)
                
            with tab3:
                st.markdown("#### Realized Asset Correlations")
                st.dataframe(res['statistics']['corr_df'].style.background_gradient(cmap="Blues"), use_container_width=True)

    if st.button("RESET SIMULATION", use_container_width=True):
        st.session_state["has_run"] = False
        st.rerun()
