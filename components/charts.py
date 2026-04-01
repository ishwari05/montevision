"""
Interactive Visualization Components
Production-grade Plotly charts for financial analysis.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_growth_bands(portfolio_paths: np.ndarray, title: str = "Portfolio Growth Confidence Channels", currency: str = "USD"):
    """Institutional Percentile Fan Chart."""
    days = portfolio_paths.shape[0]
    time_axis = np.arange(days)
    
    # Pre-calculate percentiles for performance
    p5 = np.percentile(portfolio_paths, 5, axis=1)
    p25 = np.percentile(portfolio_paths, 25, axis=1)
    p50 = np.median(portfolio_paths, 1)
    p75 = np.percentile(portfolio_paths, 75, axis=1)
    p95 = np.percentile(portfolio_paths, 95, axis=1)
    
    fig = go.Figure()
    
    # Outer Band (5-95)
    fig.add_trace(go.Scatter(x=time_axis, y=p95, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=time_axis, y=p5, fill='tonexty', fillcolor='rgba(88, 166, 255, 0.1)',
        line=dict(width=0), name='95% Confidence Band'
    ))
    
    # Inner Band (25-75)
    fig.add_trace(go.Scatter(x=time_axis, y=p75, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=time_axis, y=p25, fill='tonexty', fillcolor='rgba(88, 166, 255, 0.2)',
        line=dict(width=0), name='50% Confidence Band'
    ))
    
    # Median
    fig.add_trace(go.Scatter(
        x=time_axis, y=p50, mode='lines', line=dict(color='#58a6ff', width=3),
        name='Median Projection'
    ))
    
    symbol = "₹" if currency == "INR" else "$"
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Trading Days",
        yaxis_title=f"Value ({symbol})",
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def render_drawdown_analysis(portfolio_paths: np.ndarray):
    """Visualizes portfolio drawdowns over time."""
    peak = np.maximum.accumulate(portfolio_paths, axis=0)
    drawdowns = (portfolio_paths - peak) / peak
    
    # Calculate median and worst drawdown
    median_dd = np.median(drawdowns, axis=1)
    worst_dd = np.min(drawdowns, axis=1)
    time_axis = np.arange(len(median_dd))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis, y=median_dd * 100, fill='tozeroy', 
        name="Median Drawdown", line=dict(color='#d73a49', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=time_axis, y=worst_dd * 100, name="Worst Case", 
        line=dict(color='rgba(215, 58, 73, 0.3)', dash='dot')
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown Analysis (%)",
        xaxis_title="Trading Days",
        yaxis_title="Drawdown %",
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_final_distribution(portfolio_paths: np.ndarray, initial_investment: float, currency: str = "USD"):
    """Histogram of final wealth outcomes."""
    final_wealth = portfolio_paths[-1, :]
    
    fig = px.histogram(
        final_wealth, nbins=50,
        title="Final Wealth Distribution",
        labels={'value': f'Final Wealth ({currency})', 'count': 'Frequency'},
        color_discrete_sequence=['#58a6ff']
    )
    
    fig.add_vline(x=initial_investment, line_dash="dash", line_color="#8b949e", annotation_text="Initial")
    fig.add_vline(x=np.median(final_wealth), line_color="#58a6ff", annotation_text="Median")
    
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_3d_topology(portfolio_paths: np.ndarray, max_sims: int = 100, currency: str = "USD"):
    """3D Surface plot optimizing for performance and clarity."""
    days, total_sims = portfolio_paths.shape
    n_display = min(max_sims, total_sims)
    
    day_step = max(1, days // 60)
    sim_data = portfolio_paths[::day_step, :n_display].T
    
    fig = go.Figure(data=[go.Surface(
        z=sim_data, 
        colorscale='Viridis',
        lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.9, specular=0.6, fresnel=0.2)
    )])
    
    symbol = "₹" if currency == "INR" else "$"
    fig.update_layout(
        title="3D Evolution Topology",
        scene=dict(
            xaxis_title="Timeline",
            yaxis_title="Path ID",
            zaxis_title=f"Wealth ({symbol})",
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d'),
            zaxis=dict(gridcolor='#30363d')
        ),
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def plot_sensitivity_heatmap(h_matrix: np.ndarray, x_labels: list, y_labels: list):
    """Institutional Heatmap for Sensitivity Analysis."""
    fig = px.imshow(
        h_matrix,
        labels=dict(x="Withdrawal Amount", y="Stock Exposure", color="Survival %"),
        x=x_labels, y=y_labels,
        color_continuous_scale='RdYlGn',
        text_auto=".0f"
    )
    
    fig.update_layout(
        title="Survival Matrix (Withdrawal vs allocation)",
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_nticks=len(x_labels)
    )
    return fig
