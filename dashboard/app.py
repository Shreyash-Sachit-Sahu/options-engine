"""
Options Pricing Engine — Streamlit Dashboard

Premium dark-themed dashboard with 6 panels:
1. Volatility Surface (3D Plotly)
2. Live Greeks Calculator
3. Agent vs Baseline PnL
4. Sharpe Comparison
5. Drawdown Analysis
6. PnL Distribution

Designed for interview demos and real-time monitoring.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import bs_call, bs_put, greeks_call, greeks_put, mc_price
from backtester.vol_surface import build_synthetic_surface, build_vol_surface

# Try importing C++ pricer
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import pricer as cpp_pricer
    BACKEND = "C++"
except ImportError:
    from src.pricer import pricer_py as cpp_pricer
    BACKEND = "Python"


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & THEME
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Options Pricing Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS: Premium Dark Theme ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a1f2e;
    --bg-card-hover: #1f2937;
    --border-subtle: rgba(99, 102, 241, 0.15);
    --border-glow: rgba(99, 102, 241, 0.4);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-cyan: #22d3ee;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --gradient-primary: linear-gradient(135deg, #6366f1, #8b5cf6);
    --gradient-success: linear-gradient(135deg, #10b981, #22d3ee);
    --gradient-danger: linear-gradient(135deg, #f43f5e, #f59e0b);
}

.stApp {
    background: var(--bg-primary);
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding: 1rem 2rem;
    max-width: 1400px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%);
    border-right: 1px solid var(--border-subtle);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary);
}

/* Card styling */
.metric-card {
    background: linear-gradient(135deg, rgba(26, 31, 46, 0.9), rgba(31, 41, 55, 0.7));
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
}

.metric-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}

.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-top: 4px;
}

.metric-delta-positive {
    color: var(--accent-emerald);
    font-size: 0.9rem;
    font-weight: 600;
}

.metric-delta-negative {
    color: var(--accent-rose);
    font-size: 0.9rem;
    font-weight: 600;
}

/* Header */
.header-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border-subtle);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Greek display */
.greek-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 16px 0;
}

.greek-item {
    background: linear-gradient(135deg, rgba(26, 31, 46, 0.95), rgba(31, 41, 55, 0.8));
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s ease;
}

.greek-item:hover {
    border-color: var(--accent-indigo);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.1);
}

.greek-symbol {
    font-size: 1.4rem;
    font-weight: 300;
    color: var(--accent-cyan);
    font-family: 'Times New Roman', serif;
    font-style: italic;
}

.greek-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 4px;
}

.greek-name {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Backend badge */
.backend-badge {
    display: inline-block;
    background: var(--gradient-success);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 600;
    padding: 8px 20px;
}

.stTabs [aria-selected="true"] {
    background: var(--gradient-primary);
    color: white;
    border-color: var(--accent-indigo);
}

/* Slider styling */
.stSlider label {
    color: var(--text-secondary) !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,23,0.8)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(99,102,241,0.1)",
        zerolinecolor="rgba(99,102,241,0.2)",
    ),
    yaxis=dict(
        gridcolor="rgba(99,102,241,0.1)",
        zerolinecolor="rgba(99,102,241,0.2)",
    ),
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)",
        bordercolor="rgba(99,102,241,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
)

COLORS = {
    "SAC": "#6366f1",
    "DeltaHedger": "#22d3ee",
    "StaticHedger": "#f59e0b",
    "RandomAgent": "#f43f5e",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
    <div>
        <div class="header-title">⚡ Options Pricing Engine</div>
        <div class="header-subtitle">Multi-Agent Hedging · C++ Core · Reinforcement Learning</div>
    </div>
    <div class="backend-badge">{BACKEND} Backend</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Interactive Pricing Calculator
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎮 Pricing Calculator")
    st.markdown("Adjust parameters to compute option prices & Greeks in real-time.")

    st.markdown("---")

    S = st.slider("**Spot Price (S)**", 50.0, 500.0, 100.0, 0.5, key="spot_price")
    K = st.slider("**Strike Price (K)**", 50.0, 500.0, 100.0, 0.5, key="strike_price")
    T = st.slider("**Time to Maturity (years)**", 0.01, 2.0, 1.0, 0.01, key="maturity")
    r = st.slider("**Risk-Free Rate**", 0.0, 0.15, 0.05, 0.005, key="risk_free")
    sigma = st.slider("**Volatility (σ)**", 0.05, 1.0, 0.20, 0.01, key="volatility")
    option_type = st.selectbox("**Option Type**", ["Call", "Put"], key="option_type")

    st.markdown("---")

    # Compute prices
    flag = "call" if option_type == "Call" else "put"
    try:
        if flag == "call":
            bs_price = cpp_pricer.bs_call(S, K, T, r, sigma)
            g = cpp_pricer.greeks_call(S, K, T, r, sigma) if hasattr(cpp_pricer, 'greeks_call') else cpp_pricer.greeks(S, K, T, r, sigma, "call")
        else:
            bs_price = cpp_pricer.bs_put(S, K, T, r, sigma)
            g = cpp_pricer.greeks_put(S, K, T, r, sigma) if hasattr(cpp_pricer, 'greeks_put') else cpp_pricer.greeks(S, K, T, r, sigma, "put")

        mc_result = cpp_pricer.mc_price(S, K, T, r, sigma, n_paths=50000, flag=flag)
    except Exception as e:
        if flag == "call":
            bs_price = bs_call(S, K, T, r, sigma)
            g = greeks_call(S, K, T, r, sigma)
        else:
            bs_price = bs_put(S, K, T, r, sigma)
            g = greeks_put(S, K, T, r, sigma)
        mc_result = mc_price(S, K, T, r, sigma, flag=flag)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">BS {option_type} Price</div>
        <div class="metric-value">${bs_price:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MC Price (50k paths)</div>
        <div class="metric-value">${mc_result.price:.4f}</div>
        <div class="metric-delta-{'positive' if abs(mc_result.price - bs_price) < 0.1 else 'negative'}">
            SE: ±{mc_result.std_error:.4f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Moneyness indicator
    moneyness = S / K
    if moneyness > 1.02:
        m_text, m_color = "ITM", "#10b981"
    elif moneyness < 0.98:
        m_text, m_color = "OTM", "#f43f5e"
    else:
        m_text, m_color = "ATM", "#f59e0b"

    st.markdown(f"""
    <div style="text-align: center; margin: 12px 0;">
        <span style="background: {m_color}22; color: {m_color}; padding: 4px 16px;
                     border-radius: 20px; font-weight: 700; font-size: 0.85rem;
                     border: 1px solid {m_color}44;">
            {m_text} · Moneyness: {moneyness:.3f}
        </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT — Dashboard Panels
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Row 1: Greeks Display ────────────────────────────────────────────────────

st.markdown('<div class="section-header">📐 Option Greeks</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="greek-grid">
    <div class="greek-item">
        <div class="greek-symbol">Δ</div>
        <div class="greek-value">{g.delta:+.4f}</div>
        <div class="greek-name">Delta</div>
    </div>
    <div class="greek-item">
        <div class="greek-symbol">Γ</div>
        <div class="greek-value">{g.gamma:.6f}</div>
        <div class="greek-name">Gamma</div>
    </div>
    <div class="greek-item">
        <div class="greek-symbol">ν</div>
        <div class="greek-value">{g.vega:.4f}</div>
        <div class="greek-name">Vega</div>
    </div>
    <div class="greek-item">
        <div class="greek-symbol">Θ</div>
        <div class="greek-value">{g.theta:+.4f}</div>
        <div class="greek-name">Theta</div>
    </div>
    <div class="greek-item">
        <div class="greek-symbol">ρ</div>
        <div class="greek-value">{g.rho:+.4f}</div>
        <div class="greek-name">Rho</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs for dashboard panels ───────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🌊 Vol Surface",
    "📈 Agent Performance",
    "📊 Sharpe & Drawdown",
    "🔔 PnL Distribution"
])


# ─── Tab 1: Volatility Surface ───────────────────────────────────────────────

with tab1:
    st.markdown('<div class="section-header">🌊 Implied Volatility Surface</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col2:
        use_live = st.checkbox("Use live SPY data", value=False, key="live_data")
        surface_spot = st.number_input("Spot", value=float(S), key="surface_spot")

    # Build surface
    if use_live:
        with st.spinner("Fetching live options data..."):
            try:
                surface_df = build_vol_surface(surface_spot)
            except Exception as e:
                st.warning(f"Live data unavailable: {e}. Using synthetic surface.")
                surface_df = build_synthetic_surface(surface_spot)
    else:
        surface_df = build_synthetic_surface(surface_spot)

    with col1:
        # 3D Surface plot
        pivot = surface_df.pivot_table(
            values="iv", index="strike", columns="T", aggfunc="mean"
        )
        pivot = pivot.interpolate(method="linear", axis=0).interpolate(
            method="linear", axis=1
        )
        pivot = pivot.ffill().bfill()

        fig_surface = go.Figure(data=[
            go.Surface(
                z=pivot.values * 100,  # Convert to percentage
                x=pivot.columns.values * 365,  # Days to expiry
                y=pivot.index.values,
                colorscale=[
                    [0.0, "#0c1445"], [0.15, "#1a237e"], [0.3, "#283593"],
                    [0.45, "#3949ab"], [0.55, "#5c6bc0"], [0.65, "#7986cb"],
                    [0.75, "#9fa8da"], [0.85, "#c5cae9"], [0.95, "#e8eaf6"],
                    [1.0, "#f5f5f5"]
                ],
                opacity=0.92,
                contours=dict(
                    z=dict(show=True, color="rgba(99,102,241,0.3)", width=1),
                ),
                hovertemplate=(
                    "<b>Strike</b>: $%{y:.0f}<br>"
                    "<b>DTE</b>: %{x:.0f} days<br>"
                    "<b>IV</b>: %{z:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        ])

        fig_surface.update_layout(
            **PLOTLY_LAYOUT,
            height=550,
            title=dict(text="Implied Volatility Surface", font=dict(size=16)),
            scene=dict(
                xaxis=dict(title="Days to Expiry", backgroundcolor="rgba(10,14,23,0.8)",
                          gridcolor="rgba(99,102,241,0.15)"),
                yaxis=dict(title="Strike ($)", backgroundcolor="rgba(10,14,23,0.8)",
                          gridcolor="rgba(99,102,241,0.15)"),
                zaxis=dict(title="IV (%)", backgroundcolor="rgba(10,14,23,0.8)",
                          gridcolor="rgba(99,102,241,0.15)"),
                bgcolor="rgba(10,14,23,0.9)",
                camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
            ),
        )

        st.plotly_chart(fig_surface, use_container_width=True)

    # Surface statistics
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.metric("Min IV", f"{surface_df['iv'].min()*100:.1f}%")
    with scol2:
        st.metric("Max IV", f"{surface_df['iv'].max()*100:.1f}%")
    with scol3:
        st.metric("ATM IV", f"{surface_df.loc[surface_df['moneyness'].sub(1).abs().idxmin(), 'iv']*100:.1f}%")
    with scol4:
        st.metric("Points", f"{len(surface_df)}")


# ─── Load Evaluation Data (shared across Tabs 2-4) ───────────────────────────
# All chart data is loaded from real evaluation runs — zero hardcoded values.
# If no evaluation has been run yet, we run live simulations on the spot.

RESULTS_PATH = Path(__file__).parent.parent / "agent" / "evaluation_results.json"

@st.cache_data(ttl=300)  # Cache for 5 minutes, then refresh
def load_evaluation_data():
    """
    Load agent evaluation results from disk.
    If no evaluation_results.json exists, run live baseline simulations.
    Returns dict of {agent_name: {metrics + episode_pnls}}.
    """
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        return data, "file"

    # No saved results — run live simulations using actual environment
    from environment.options_env import OptionsHedgingEnv
    from environment.baselines import (
        DeltaHedger, StaticHedger, RandomAgent, evaluate_agent
    )

    env_config = {
        "simulator_type": "gbm", "S0": 100.0, "K": 100.0,
        "T": 30 / 252, "r": 0.05, "sigma": 0.2, "mu": 0.05,
        "n_steps": 30, "transaction_cost": 0.001,
    }
    env = OptionsHedgingEnv(**env_config)
    n_episodes = 200  # fewer for dashboard speed

    agents = {
        "DeltaHedger": DeltaHedger(K=100.0, r=0.05, sigma=0.2, T=30/252),
        "StaticHedger": StaticHedger(),
        "RandomAgent": RandomAgent(seed=42),
    }

    results = {}
    for name, agent in agents.items():
        result = evaluate_agent(env, agent, n_episodes=n_episodes)
        result["agent"] = name
        results[name] = result

    return results, "live_simulation"


eval_data, data_source = load_evaluation_data()
data_source_label = "Evaluation Results" if data_source == "file" else "Live Simulation"


# ─── Tab 2: Agent Performance ────────────────────────────────────────────────

with tab2:
    st.markdown(f'<div class="section-header">📈 Agent vs Baseline — Episode PnL Distribution ({data_source_label})</div>',
                unsafe_allow_html=True)

    # Build PnL comparison from real evaluation data
    fig_pnl = go.Figure()

    agent_order = ["SAC", "DeltaHedger", "StaticHedger", "RandomAgent"]
    available_agents = [a for a in agent_order if a in eval_data]

    for agent_name in available_agents:
        agent_data = eval_data[agent_name]
        episode_pnls = agent_data.get("episode_pnls", [])

        if not episode_pnls:
            continue

        ep_pnls = np.array(episode_pnls)
        n_eps = len(ep_pnls)

        # Sorted cumulative PnL (simulates episodes ordered by performance)
        sorted_pnls = np.sort(ep_pnls)
        cumulative_pnl = np.cumsum(sorted_pnls)
        episodes = list(range(1, n_eps + 1))

        # Confidence band using rolling statistics
        window = max(n_eps // 20, 5)
        rolling_mean = pd.Series(ep_pnls).rolling(window, min_periods=1).mean().values
        rolling_std = pd.Series(ep_pnls).rolling(window, min_periods=1).std().fillna(0).values

        cum_rolling = np.cumsum(rolling_mean)
        cum_upper = np.cumsum(rolling_mean + rolling_std)
        cum_lower = np.cumsum(rolling_mean - rolling_std)

        color = COLORS.get(agent_name, "#6366f1")
        r_hex, g_hex, b_hex = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        # Confidence band
        fig_pnl.add_trace(go.Scatter(
            x=episodes + episodes[::-1],
            y=list(cum_upper) + list(cum_lower[::-1]),
            fill="toself",
            fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.1)",
            line=dict(width=0),
            showlegend=False,
            name=agent_name,
            hoverinfo="skip",
        ))

        # Mean line
        fig_pnl.add_trace(go.Scatter(
            x=episodes,
            y=cum_rolling,
            mode="lines",
            name=agent_name,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{agent_name}</b><br>Episode %{{x}}<br>Cum PnL: %{{y:.4f}}<extra></extra>",
        ))

    n_total = eval_data.get("SAC", eval_data.get("DeltaHedger", {})).get("n_episodes", "N/A")
    fig_pnl.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        title=dict(text=f"Cumulative PnL ({n_total} episodes, {data_source_label})", font=dict(size=14)),
        xaxis_title="Episode",
        yaxis_title="Cumulative PnL ($)",
        hovermode="x unified",
    )

    st.plotly_chart(fig_pnl, use_container_width=True)


# ─── Tab 3: Sharpe & Drawdown ────────────────────────────────────────────────

with tab3:
    col_sharpe, col_dd = st.columns(2)

    with col_sharpe:
        st.markdown('<div class="section-header">📊 Sharpe Ratio Comparison</div>',
                    unsafe_allow_html=True)

        # Load Sharpe data from real evaluation results
        sharpe_data = {k: v.get("sharpe_ratio", 0) for k, v in eval_data.items()}

        agents = list(sharpe_data.keys())
        sharpes = list(sharpe_data.values())

        fig_sharpe = go.Figure(data=[
            go.Bar(
                x=agents,
                y=sharpes,
                marker=dict(
                    color=[COLORS.get(a, "#6366f1") for a in agents],
                    line=dict(width=0),
                    opacity=0.9,
                ),
                text=[f"{s:.2f}" for s in sharpes],
                textposition="outside",
                textfont=dict(size=14, color="#e2e8f0", family="Inter"),
                hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.4f}<extra></extra>",
            )
        ])

        # Target line
        fig_sharpe.add_hline(
            y=1.4, line=dict(color="#10b981", width=2, dash="dash"),
            annotation_text="Target: 1.4",
            annotation_position="top right",
            annotation_font=dict(color="#10b981", size=12),
        )

        fig_sharpe.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            yaxis_title="Sharpe Ratio",
            showlegend=False,
        )

        st.plotly_chart(fig_sharpe, use_container_width=True)

    with col_dd:
        st.markdown('<div class="section-header">📉 Drawdown Analysis</div>',
                    unsafe_allow_html=True)

        # Plot drawdown from real episode PnL data
        fig_dd = go.Figure()

        for agent_name in available_agents:
            ep_pnls = eval_data[agent_name].get("episode_pnls", [])
            if not ep_pnls:
                continue

            pnls_arr = np.array(ep_pnls)
            # Compute drawdown from cumulative PnL across episodes
            cum_pnl = np.cumsum(pnls_arr)
            running_max = np.maximum.accumulate(cum_pnl)
            drawdown = running_max - cum_pnl

            episodes = list(range(1, len(drawdown) + 1))
            color = COLORS.get(agent_name, "#6366f1")
            r_hex, g_hex, b_hex = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

            fig_dd.add_trace(go.Scatter(
                x=episodes,
                y=-drawdown,
                mode="lines",
                name=agent_name,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.15)",
                hovertemplate=f"<b>{agent_name}</b><br>Episode %{{x}}<br>DD: %{{y:.4f}}<extra></extra>",
            ))

        fig_dd.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            xaxis_title="Episode",
            yaxis_title="Drawdown ($)",
            hovermode="x unified",
        )

        st.plotly_chart(fig_dd, use_container_width=True)

    # Metrics row — all computed from real evaluation data
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    sac_sharpe = sharpe_data.get("SAC", 0)
    delta_sharpe = sharpe_data.get("DeltaHedger", 0)
    sac_dd = eval_data.get("SAC", {}).get("mean_max_drawdown", 0)
    delta_dd = eval_data.get("DeltaHedger", {}).get("mean_max_drawdown", 0)
    outperformance = sac_sharpe / max(abs(delta_sharpe), 0.01)

    metrics = [
        ("SAC Sharpe", f"{sac_sharpe:.2f}",
         f"+{(sac_sharpe/max(abs(delta_sharpe), 0.01)-1)*100:.0f}% vs Delta" if delta_sharpe != 0 else "N/A",
         sac_sharpe > delta_sharpe),
        ("Delta Sharpe", f"{delta_sharpe:.2f}", "Baseline", None),
        ("SAC Mean DD", f"${sac_dd:.2f}",
         f"vs Delta ${delta_dd:.2f}",
         sac_dd < delta_dd),
        ("Outperformance", f"{outperformance:.2f}x", "Target: 1.55x",
         outperformance > 1.55),
    ]

    for col, (label, value, delta, is_pos) in zip([mcol1, mcol2, mcol3, mcol4], metrics):
        delta_class = "positive" if is_pos else ("negative" if is_pos is False else "positive")
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta-{delta_class}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Tab 4: PnL Distribution ─────────────────────────────────────────────────

with tab4:
    st.markdown(f'<div class="section-header">🔔 PnL Distribution — All Agents ({data_source_label})</div>',
                unsafe_allow_html=True)

    fig_dist = go.Figure()

    for agent_name in available_agents:
        ep_pnls = eval_data[agent_name].get("episode_pnls", [])
        if not ep_pnls:
            continue

        fig_dist.add_trace(go.Histogram(
            x=ep_pnls,
            name=agent_name,
            nbinsx=50,
            opacity=0.7,
            marker_color=COLORS.get(agent_name, "#6366f1"),
            hovertemplate=f"<b>{agent_name}</b><br>PnL: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
        ))

    n_total = eval_data.get("SAC", eval_data.get("DeltaHedger", {})).get("n_episodes", "N/A")
    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        barmode="overlay",
        xaxis_title="Episode PnL ($)",
        yaxis_title="Frequency",
        title=dict(
            text=f"PnL Distribution ({n_total} episodes, {data_source_label})",
            font=dict(size=14),
        ),
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # Distribution statistics table — computed from real episode data
    dist_stats = []
    for agent_name in available_agents:
        ep_pnls = np.array(eval_data[agent_name].get("episode_pnls", [0]))
        dist_stats.append({
            "Agent": agent_name,
            "Mean PnL": f"${np.mean(ep_pnls):.4f}",
            "Std PnL": f"${np.std(ep_pnls):.4f}",
            "Sharpe": f"{eval_data[agent_name].get('sharpe_ratio', 0):.2f}",
            "Skewness": f"{float(pd.Series(ep_pnls).skew()):.2f}" if len(ep_pnls) > 2 else "N/A",
            "Kurtosis": f"{float(pd.Series(ep_pnls).kurtosis()):.2f}" if len(ep_pnls) > 3 else "N/A",
            "5th %ile": f"${np.percentile(ep_pnls, 5):.4f}",
            "95th %ile": f"${np.percentile(ep_pnls, 95):.4f}",
        })

    st.dataframe(
        pd.DataFrame(dist_stats),
        use_container_width=True,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 12px 0;">
    Options Pricing Engine v1.0 · {BACKEND} Backend ·
    Pricing Core: Black-Scholes + Monte Carlo (50k paths) ·
    RL Agent: SAC (stable-baselines3) ·
    Built with ❤️ using pybind11 + FastAPI + Streamlit
</div>
""", unsafe_allow_html=True)
