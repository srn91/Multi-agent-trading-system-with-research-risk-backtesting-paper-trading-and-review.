"""
Hedge Fund OS — Interactive Dashboard.

Run: streamlit run dashboard.py

Features:
- Live backtest results with interactive charts
- Agent decision explorer (see why every trade was made)
- Paper trading simulator
- Strategy parameter tuning
- Trade journal with filtering
"""
import sys, os, json, time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Hedge Fund OS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🏦 Hedge Fund OS")
st.sidebar.markdown("**Multi-Agent Trading System**")

page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🤖 Agent Decisions",
    "📈 Trade Journal",
    "⚙️ Strategy Lab",
    "📝 Paper Trading",
])

# ============================================================
# DATA LOADING
# ============================================================
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "backtests")

def load_data():
    """Load all backtest results."""
    data = {}

    # Try V6 first, then V5
    for prefix in ["v6", "v5", "v4"]:
        eq_path = os.path.join(REPORTS, f"{prefix}_equity_curve.csv")
        tl_path = os.path.join(REPORTS, f"{prefix}_trade_log.csv")
        ad_path = os.path.join(REPORTS, f"{prefix}_agent_decisions.json")
        mt_path = os.path.join(REPORTS, f"{prefix}_metrics.txt")

        if os.path.exists(eq_path):
            data["version"] = prefix.upper()
            data["equity"] = pd.read_csv(eq_path)
            data["equity"]["date"] = pd.to_datetime(data["equity"]["date"])
            break

    if "equity" not in data:
        # Fallback to generic files
        for name in ["equity_curve.csv", "equity_curve_v2.csv"]:
            p = os.path.join(REPORTS, name)
            if os.path.exists(p):
                data["version"] = "V1"
                data["equity"] = pd.read_csv(p)
                data["equity"]["date"] = pd.to_datetime(data["equity"]["date"])
                break

    # Trade log
    for prefix in ["v6", "v5", "v4", ""]:
        p = os.path.join(REPORTS, f"{prefix}_trade_log.csv" if prefix else "trade_log.csv")
        if os.path.exists(p):
            data["trades"] = pd.read_csv(p)
            break

    # Agent decisions
    for prefix in ["v6", "v5", "v4"]:
        p = os.path.join(REPORTS, f"{prefix}_agent_decisions.json")
        if os.path.exists(p):
            with open(p) as f:
                data["agents"] = json.load(f)
            break

    # Metrics
    for prefix in ["v6", "v5", "v4", ""]:
        p = os.path.join(REPORTS, f"{prefix}_metrics.txt" if prefix else "metrics.txt")
        if os.path.exists(p):
            metrics = {}
            with open(p) as f:
                for line in f:
                    if ":" in line:
                        k, v = line.strip().split(":", 1)
                        metrics[k.strip()] = v.strip()
            data["metrics"] = metrics
            break

    return data


def run_fresh_backtest(top_n, rebal_freq, momentum_lookback, trailing_stop, hard_stop, max_positions):
    """Run a fresh backtest with custom parameters."""
    from src.data.ingest import download_ohlcv, get_spy, DEFAULT_UNIVERSE
    from src.data.features_v4 import compute_v4_features, compute_v4_spy
    from src.backtest.engine_v6 import BacktestEngineV6, V6Config

    tickers = DEFAULT_UNIVERSE[:80]
    spy_raw = get_spy(start="2012-01-01")
    stock_raw = {}
    for t in tickers:
        try:
            df = download_ohlcv(t, start="2012-01-01")
            if not df.empty and len(df) > 252:
                stock_raw[t] = df
        except:
            pass

    spy = compute_v4_spy(spy_raw)
    stock_data = {}
    for t, df in stock_raw.items():
        try:
            stock_data[t] = compute_v4_features(df, spy)
        except:
            pass

    config = V6Config(
        max_positions=max_positions,
        top_n=top_n,
        buy_threshold=top_n,
        sell_threshold=top_n * 2,
        check_frequency=rebal_freq,
        momentum_lookback=momentum_lookback,
        trailing_stop_pct=trailing_stop / 100,
        hard_stop_pct=hard_stop / 100,
    )

    engine = BacktestEngineV6(config)
    metrics = engine.run(stock_data, spy)
    eq = engine.get_equity_curve()
    trades = engine.get_trade_log()

    return metrics, eq, trades, engine.agents.decisions


# ============================================================
# PAGE: DASHBOARD
# ============================================================
def page_dashboard():
    data = load_data()

    st.title(f"📊 Hedge Fund OS — {data.get('version', 'N/A')} Dashboard")

    if "metrics" not in data:
        st.warning("No backtest data found. Run `python3 run_v6_visual.py` first to generate results.")
        return

    m = data["metrics"]

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return", m.get("total_return_pct", "?") + "%")
    col2.metric("CAGR", m.get("cagr_pct", "?") + "%")
    col3.metric("Max Drawdown", m.get("max_drawdown_pct", "?") + "%")
    col4.metric("Sharpe Ratio", m.get("sharpe_ratio", "?"))
    col5.metric("Total Trades", m.get("total_trades", "?"))

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Rate", m.get("win_rate", "?") + "%")
    col2.metric("Profit Factor", m.get("profit_factor", "?"))
    col3.metric("Expectancy", m.get("expectancy_per_trade", "?") + "%")
    col4.metric("Avg Hold", m.get("avg_hold_days", "?") + " days")

    st.divider()

    # Equity curve
    if "equity" in data:
        eq = data["equity"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq["date"], y=eq["equity"],
            mode="lines", name="V6 Strategy",
            line=dict(color="#3B82F6", width=2),
        ))

        # Add starting capital line
        initial = float(m.get("initial_capital", 100000))
        fig.add_hline(y=initial, line_dash="dash", line_color="gray",
                       annotation_text=f"Starting ${initial:,.0f}")

        fig.update_layout(
            title="Equity Curve",
            yaxis_title="Equity ($)",
            xaxis_title="Date",
            template="plotly_dark",
            height=500,
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

        # Drawdown chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=eq["date"], y=eq["drawdown"] * 100,
            fill="tozeroy", name="Drawdown",
            line=dict(color="#EF4444", width=1),
            fillcolor="rgba(239, 68, 68, 0.3)",
        ))
        fig2.update_layout(
            title="Drawdown",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=250,
        )
        st.plotly_chart(fig2, width='stretch')


# ============================================================
# PAGE: AGENT DECISIONS
# ============================================================
def page_agents():
    data = load_data()
    st.title("🤖 Agent Decision Explorer")

    if "agents" not in data:
        st.warning("No agent decisions found. Run `python3 run_v6_visual.py` first.")
        return

    decisions = data["agents"]
    st.info(f"**{len(decisions)} agent decisions logged**")

    # Filter controls
    col1, col2, col3 = st.columns(3)
    agents_list = list(set(d.get("agent", "?") for d in decisions))
    actions_list = list(set(d.get("action", "?") for d in decisions))

    selected_agent = col1.selectbox("Filter by Agent", ["All"] + sorted(agents_list))
    selected_action = col2.selectbox("Filter by Action", ["All"] + sorted(actions_list))
    ticker_filter = col3.text_input("Filter by Ticker", "")

    filtered = decisions
    if selected_agent != "All":
        filtered = [d for d in filtered if d.get("agent") == selected_agent]
    if selected_action != "All":
        filtered = [d for d in filtered if d.get("action") == selected_action]
    if ticker_filter:
        filtered = [d for d in filtered if ticker_filter.upper() in d.get("ticker", "").upper()]

    st.subheader(f"Showing {len(filtered)} decisions")

    # Action distribution
    action_counts = {}
    for d in decisions:
        a = d.get("action", "unknown")
        action_counts[a] = action_counts.get(a, 0) + 1

    if action_counts:
        fig = px.pie(
            names=list(action_counts.keys()),
            values=list(action_counts.values()),
            title="Decision Type Distribution",
        )
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, width='stretch')

    # Decision table
    if filtered:
        display_data = []
        for d in filtered[-100:]:  # show last 100
            row = {
                "Date": d.get("date", ""),
                "Agent": d.get("agent", ""),
                "Ticker": d.get("ticker", ""),
                "Action": d.get("action", ""),
            }
            # Add conviction if present
            if "conviction" in d:
                row["Conviction"] = f"{d['conviction']:.3f}"
            if "sell_pressure" in d:
                row["Sell Pressure"] = f"{d['sell_pressure']:.3f}"
            if "reasoning" in d:
                row["Reasoning"] = str(d["reasoning"])[:80]
            if "scores" in d:
                row["Scores"] = str({k: f"{v:.2f}" for k, v in d["scores"].items()})
            display_data.append(row)

        df = pd.DataFrame(display_data)
        st.dataframe(df, width='stretch', height=500)

    # Detailed view
    st.subheader("Detailed Decision View")
    if filtered:
        idx = st.slider("Decision #", 0, len(filtered) - 1, len(filtered) - 1)
        st.json(filtered[idx])


# ============================================================
# PAGE: TRADE JOURNAL
# ============================================================
def page_trades():
    data = load_data()
    st.title("📈 Trade Journal")

    if "trades" not in data:
        st.warning("No trade data found.")
        return

    trades = data["trades"]
    st.info(f"**{len(trades)} completed trades**")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    winners = trades[trades["pnl_pct"] > 0]
    losers = trades[trades["pnl_pct"] <= 0]
    col1.metric("Winners", len(winners))
    col2.metric("Losers", len(losers))
    col3.metric("Avg Winner", f"{winners['pnl_pct'].mean():+.1f}%" if len(winners) > 0 else "N/A")
    col4.metric("Avg Loser", f"{losers['pnl_pct'].mean():.1f}%" if len(losers) > 0 else "N/A")

    # PnL chart
    fig = go.Figure()
    colors = ["#22C55E" if x > 0 else "#EF4444" for x in trades["pnl_pct"]]
    fig.add_trace(go.Bar(
        x=list(range(len(trades))),
        y=trades["pnl_pct"],
        marker_color=colors,
        name="PnL %",
    ))
    fig.update_layout(
        title="PnL per Trade (%)",
        template="plotly_dark",
        height=350,
    )
    st.plotly_chart(fig, width='stretch')

    # Filters
    col1, col2 = st.columns(2)
    ticker_filter = col1.text_input("Filter by Ticker", "", key="trade_ticker")
    exit_filter = col2.selectbox("Filter by Exit Reason",
                                  ["All"] + list(trades["exit_reason"].unique()))

    filtered = trades
    if ticker_filter:
        filtered = filtered[filtered["ticker"].str.contains(ticker_filter.upper())]
    if exit_filter != "All":
        filtered = filtered[filtered["exit_reason"] == exit_filter]

    # Per-ticker stats
    st.subheader("Per-Ticker Performance")
    ticker_stats = trades.groupby("ticker").agg(
        trades=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        total_pnl=("pnl_dollars", "sum"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
        avg_hold=("hold_days", "mean"),
    ).round(2).sort_values("total_pnl", ascending=False)
    st.dataframe(ticker_stats.head(20), width='stretch')

    # Full trade log
    st.subheader(f"Trade Log ({len(filtered)} trades)")
    st.dataframe(filtered, width='stretch', height=400)


# ============================================================
# PAGE: STRATEGY LAB
# ============================================================
def page_strategy_lab():
    st.title("⚙️ Strategy Lab — Parameter Tuning")
    st.markdown("Change parameters and run a fresh backtest to see how they affect results.")

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Top N stocks to hold", 3, 10, 5)
        rebal_freq = st.slider("Rebalance frequency (days)", 3, 21, 5)
        momentum_lookback = st.slider("Momentum lookback (days)", 21, 252, 63)
    with col2:
        trailing_stop = st.slider("Trailing stop (%)", 10, 35, 22)
        hard_stop = st.slider("Hard stop (%)", 15, 40, 28)
        max_positions = st.slider("Max positions", 3, 10, 5)

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest... (this takes 30-60 seconds)"):
            metrics, eq, trades, agent_log = run_fresh_backtest(
                top_n, rebal_freq, momentum_lookback, trailing_stop, hard_stop, max_positions
            )

        # Results
        st.success("Backtest complete!")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR", f"{metrics.cagr_pct:+.1f}%")
        col2.metric("Sharpe", f"{metrics.sharpe_ratio:.2f}")
        col3.metric("Max DD", f"{metrics.max_drawdown_pct:.1f}%")
        col4.metric("Trades", metrics.total_trades)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Return", f"{metrics.total_return_pct:+.0f}%")
        col2.metric("Win Rate", f"{metrics.win_rate:.0f}%")
        col3.metric("PF", f"{metrics.profit_factor:.2f}")
        col4.metric("Exp/Trade", f"{metrics.expectancy_per_trade:+.2f}%")

        if not eq.empty:
            eq["date"] = pd.to_datetime(eq["date"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq["date"], y=eq["equity"],
                mode="lines", name="Strategy",
                line=dict(color="#3B82F6", width=2),
            ))
            fig.update_layout(
                title=f"Equity Curve — CAGR: {metrics.cagr_pct:+.1f}%, Sharpe: {metrics.sharpe_ratio:.2f}",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Adjust parameters above and click **Run Backtest** to test a new configuration.")


# ============================================================
# PAGE: PAPER TRADING
# ============================================================
def page_paper_trading():
    st.title("📝 Paper Trading Monitor")

    st.markdown("""
    **Paper trading status**: Not yet live.

    To start paper trading:
    1. Run the backtest to confirm strategy parameters
    2. The system will generate daily signals based on current market data
    3. Track simulated trades without risking real capital
    4. After 30-60 paper trades, evaluate performance vs backtest

    **Coming in Phase 2:**
    - Daily signal generation from live market data
    - Automatic paper trade execution
    - Real-time P&L tracking
    - Performance comparison vs backtest expectations
    - Broker API integration for live trading
    """)

    st.divider()

    # Paper trading setup
    st.subheader("Setup Paper Trading")
    col1, col2 = st.columns(2)
    paper_capital = col1.number_input("Paper Trading Capital ($)", value=100000, step=10000)
    paper_risk = col2.slider("Risk per Trade (%)", 1.0, 5.0, 2.0, 0.5)

    st.subheader("Current Market Signal")
    st.info("Run `python3 generate_signals.py` to see today's momentum rankings and trade recommendations.")

    # Show what a signal would look like
    st.subheader("Example Signal Output")
    example = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "action": "BUY",
        "ticker": "NVDA",
        "conviction": 0.82,
        "agents": {
            "momentum": {"score": 0.95, "evidence": "rank 1/99, 63d return +42%"},
            "trend": {"score": 0.88, "evidence": "above all MAs, RS=87"},
            "risk": {"score": 0.65, "evidence": "vol=32%, acceptable"},
            "regime": {"score": 0.72, "evidence": "market score 72/100, mild bull"},
        },
        "sizing": {
            "conviction_multiplier": 1.3,
            "position_value": "$32,500",
            "shares": 250,
            "stop_loss": "$113.75 (-22%)",
        },
    }
    st.json(example)

    st.divider()
    st.subheader("Roadmap to Live Trading")
    st.markdown("""
    | Phase | Status | Description |
    |-------|--------|-------------|
    | ✅ Phase 1 | **Complete** | Backtest engine with agent decisions |
    | ✅ Phase 2 | **Complete** | Strategy optimization (V1→V6) |
    | 🔄 Phase 3 | **Current** | Paper trading simulation |
    | ⬜ Phase 4 | Planned | Live signal generation |
    | ⬜ Phase 5 | Planned | Broker API integration (Alpaca/IBKR) |
    | ⬜ Phase 6 | Planned | Live monitoring dashboard |
    """)


# ============================================================
# ROUTER
# ============================================================
if page == "📊 Dashboard":
    page_dashboard()
elif page == "🤖 Agent Decisions":
    page_agents()
elif page == "📈 Trade Journal":
    page_trades()
elif page == "⚙️ Strategy Lab":
    page_strategy_lab()
elif page == "📝 Paper Trading":
    page_paper_trading()
