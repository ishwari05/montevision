"""
Landing Page Component
Educational and high-impact entry point for the Portfolio Risk Simulator.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def render_demo_charts():
    """Generates static/demo data for landing page education."""
    # 1. Fan Chart Demo
    t = np.linspace(0, 10, 100)
    median = 100 * np.exp(0.07 * t)
    upper = median * 1.5
    lower = median * 0.5
    
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=t, y=upper, mode='lines', line=dict(width=0), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=t, y=lower, fill='tonexty', fillcolor='rgba(88, 166, 255, 0.1)', line=dict(width=0), name='Uncertainty Range'))
    fig_fan.add_trace(go.Scatter(x=t, y=median, mode='lines', line=dict(color='#58a6ff', width=3), name='Most Likely Path'))
    
    fig_fan.update_layout(
        title="Portfolio Growth Expectations", showlegend=False, 
        template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0), height=250
    )

    # 2. Drawdown Demo
    dd = -np.abs(np.random.normal(0, 5, 100).cumsum())
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=t, y=dd, fill='tozeroy', fillcolor='rgba(215, 58, 73, 0.2)', line=dict(color='#d73a49')))
    fig_dd.update_layout(
        title="Historical Stress Testing (Drawdowns)", showlegend=False,
        template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0), height=250
    )

    # 3. Distribution Demo
    dist = np.random.lognormal(0, 0.5, 1000)
    fig_dist = px.histogram(dist, nbins=50, color_discrete_sequence=['#58a6ff'])
    fig_dist.update_layout(
        title="Outcome Probability Distribution", showlegend=False,
        template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0), height=250
    )

    return fig_fan, fig_dd, fig_dist

def render_landing_page():
    """Main rendering entry for Landing Mode."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;'>Portfolio Risk Intelligence Engine</h1>
        <p style='font-size: 1.2rem; opacity: 0.8;'>Experience the power of institutional-grade stochastic modeling with regime switching and tail-risk analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = render_demo_charts()

    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(f1, use_container_width=True)
        st.markdown("**1. Growth Projections**  \nGrowth isn't linear. This chart shows the range of possible outcomes based on historical volatility and market drift.")
        
        st.plotly_chart(f3, use_container_width=True)
        st.markdown("**3. Probability of Success**  \nSee the exact frequency of outcomes. We model thousands of 'alternate realities' to find your statistical edge.")

    with col2:
        st.plotly_chart(f2, use_container_width=True)
        st.markdown("**2. Drawdowns & Stress**  \nWhat happens during a market crash? We simulate Black Swan events to identify your portfolio's breaking point.")
        
        st.info("Ready to run your own simulation? \nAdjust your parameters in the sidebar to get started with real-time market data.")
        
        if st.button("Run Your Simulation", use_container_width=True, type="primary"):
            st.session_state["has_run"] = True
            st.rerun()

    st.divider()
    
    st.markdown("### Why Stochastic Modeling?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Adaptive Regimes", "Bull/Bear")
    c1.caption("We model market shifting between high-growth and high-volatility regimes.")
    
    c2.metric("Tail Risk Modeling", "Black Swans")
    c2.caption("Poisson clustered events simulate rare but catastrophic market shocks.")
    
    c3.metric("Correlation Engine", "Cholesky")
    c3.caption("Dynamic cross-asset relationships are preserved across all market simulations.")
