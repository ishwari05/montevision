"""
Clean UI Components: Metrics Scorecards
Implements highly-polished, CSS-styled metric cards for financial dashboards.
"""
import streamlit as st

def render_metric_card(label, value, help_text=None):
    """Renders a metric in a glassmorphism card."""
    st.markdown(f"""
    <div class="glass-card">
        <div style="color: #57606a; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem; margin-bottom: 0.5rem;">{label}</div>
        <div style="color: #0969da; font-size: 2rem; font-weight: 700;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def display_summary_metrics(results: dict, currency: str = "USD", multiplier: float = 1.0):
    """Organizes metrics into a responsive 4-column grid using glass cards."""
    from services.currency import format_currency
    col1, col2, col3, col4 = st.columns(4)
    
    # Extract metrics sub-dict for convenience
    m = results['metrics']
    
    with col1:
        render_metric_card(
            "Median Final Wealth", 
            format_currency(m['median_final'] * multiplier, currency)
        )
    with col2:
        render_metric_card(
            "Success Probability", 
            f"{m['success_rate']:.1f}%"
        )
    with col3:
        render_metric_card(
            "Value at Risk (95%)", 
            format_currency(m['var_95'] * multiplier, currency)
        )
    with col4:
        render_metric_card(
            "Tail Risk (CVaR)", 
            format_currency(m['cvar_95'] * multiplier, currency)
        )
