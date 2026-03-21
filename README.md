# Hedge Fund OS — Multi-Agent Trading System

> **4 AI Agents | 100 Stocks | 8 Versions | Automated Paper Trading**
>
> A multi-agent trading research system where specialized AI agents collaborate through a controlled decision flow to generate, validate, and execute trading ideas — mimicking a real hedge fund team.

![Summary](reports/charts/00_summary_card.png)

---

## Performance (2012–2026, net of transaction costs)

| Metric | V6 Strategy | SPY Buy & Hold | Delta |
|---|---|---|---|
| **CAGR** | **+21.1%** | +14.0% | +7.1% |
| **Total Return** | **+1,426%** | +453% | +973% |
| **Sharpe Ratio** | **0.84** | ~0.65 | +0.19 |
| **Profit Factor** | **1.34** | — | — |
| **Total Trades** | **494** | 0 | — |
| **Avg Hold** | **33 days** | ∞ | — |
| **Transaction Costs** | **$79,652** | $0 | Realistic |

> $100,000 → $1,525,000 after slippage (0.05%/side) and commissions ($0.005/share). Every trade is scored by 4 independent agents before execution.

![Equity Curve](reports/charts/01_equity_curve.png)

![Drawdown](reports/charts/02_drawdown.png)

---

## Architecture — 4 Autonomous AI Agents

The system operates as a hedge fund team. Each agent has a defined role. The PM Agent only executes when there is sufficient consensus across the team. Every decision produces a typed Signal object with reasoning and confidence score → full audit trail.

![Architecture](reports/charts/09_architecture.png)

| Agent | Role | Key Innovation |
|---|---|---|
| **Momentum Agent** | Multi-timeframe momentum scanner | 3m/6m/12m composite ranking across 100 stocks |
| **Trend/Quality Agent** | Trend health and quality validation | MA structure checks, relative strength scoring |
| **Risk Agent** | Position limits and exposure control | Vol-based sizing, sector caps (max 2), correlation penalty |
| **Regime Agent** | Market regime classification | Bull/Bear/Choppy detection, exposure adjustment |
| **PM Agent** | Final decision maker | Consensus gating, conviction scoring, threshold rebalancing |

### Decision Flow (Every Rebalance)

1. **Momentum Agent** ranks all 100 stocks by 3m/6m/12m composite momentum
2. **Trend/Quality Agent** validates trend health: MA structure, relative strength, trend rising
3. **Risk Agent** enforces position limits, sector caps, and correlation checks
4. **Regime Agent** assesses market environment and adjusts exposure
5. **PM Agent** checks conviction threshold (>0.4) → execute or skip
6. Every step logged with reasoning → **890 agent decisions per backtest**

![Agent Decisions](reports/charts/07_agent_decisions.png)

---

## Version Evolution — From -0.2% to +21.2% CAGR

The system was iteratively improved across 8 versions. Each version fixed a specific structural problem identified through diagnosis.

| Version | Strategy | CAGR | Max DD | Sharpe | Key Fix |
|---|---|---|---|---|---|
| V1 | Breakout only | -0.2% | -21% | -0.01 | Baseline — too many false breakouts |
| V3 | +Pullback, RS filter | +7.6% | -19% | 0.68 | Added pullback entries + relative strength |
| V4 | +Vol targeting, agents | +9.6% | -22% | 0.89 | First agent integration, vol-based sizing |
| V5 | Momentum rotation | +20.4% | -42% | 0.84 | Switched to weekly rotation — first to beat SPY |
| **V6** | **+Agent team, threshold rebal** | **+21.2%** | **-48%** | **0.84** | **55% fewer trades, agents decide every entry/exit** |
| V7 | +Drawdown control, 3-factor | +15.3% | -26% | 0.90 | 46% DD reduction, multi-factor ranking |
| V8 | Regime-adaptive rotation | +15.2% | -34% | 0.83 | Offense/defense mode switching |
| **QMB** | **Quality Momentum Breakout** | **+19.8%** | **-28%** | **0.95** | **Inverse-vol weighting, best risk-adjusted** |

![Version Evolution](reports/charts/06_version_evolution.png)

---

## Production Strategy: Quality Momentum Breakout (QMB)

**Core logic:**
1. Rank 100 US stocks weekly by composite score: momentum (50%) + quality (25%) + inverse-volatility (25%)
2. 4 agents independently evaluate every candidate — conviction gate at 0.4
3. Hold top 7 stocks, sector-diversified (max 2 per sector)
4. Inverse-vol position weighting — stable stocks get bigger allocations
5. 15% trailing stop from peak, 20% hard stop from entry
6. Threshold rebalancing — only trade when rankings change meaningfully

**Why this works:**
- Multi-timeframe momentum (3m/6m/12m composite) captures persistent trends
- Quality filter removes fragile momentum names before they crash
- Low-vol tilt mechanically reduces portfolio volatility → higher Sharpe
- Sector limits prevent concentration blowups (2018, 2022 lessons)
- Threshold rebalancing reduces unnecessary turnover → $53K less in transaction costs vs V5

---

## Trade Analysis

![Trade Analysis](reports/charts/04_trade_analysis.png)

**Key observations:**
- **50% win rate** with asymmetric payoff — winners average +15%, losers average -9%
- **73% of exits via trailing stop** — system lets winners run
- **33-day average hold** — monthly rotation, not day trading
- **Fat right tail** — a few big winners (100%+) drive total returns

---

## Monthly Returns

![Monthly Heatmap](reports/charts/03_monthly_heatmap.png)

---

## Rolling Risk Metrics

![Rolling Sharpe](reports/charts/05_rolling_sharpe.png)

---

## Per-Ticker Performance

![Ticker Performance](reports/charts/08_ticker_performance.png)

---

## Automated Paper Trading

The system runs fully automated via **GitHub Actions + Alpaca API**:

- **Every weekday at 10:00 AM ET**, GitHub Actions triggers `paper_trade.py`
- Downloads fresh market data for 100 stocks
- 4 agents score and rank the universe
- Places buy/sell orders on Alpaca paper trading account
- Logs all decisions as downloadable artifacts
- **Zero manual intervention** — works when laptop is off

### Setup

1. Sign up free at [alpaca.markets](https://app.alpaca.markets/signup)
2. Generate Paper Trading API keys
3. Add keys to GitHub: **Settings → Secrets → Actions**
   - `ALPACA_API_KEY` → your paper API key
   - `ALPACA_SECRET_KEY` → your paper secret key
4. Workflow runs automatically, or trigger manually from **Actions** tab

---

## Quick Start

```bash
# Install
pip3 install -r requirements.txt

# Run backtest with charts
python3 run_v6_visual.py

# Generate today's signals
python3 generate_signals.py

# Launch interactive dashboard
python3 -m streamlit run dashboard.py

# Paper trading (requires Alpaca — free)
cp .env.example .env  # add your API keys
python3 paper_trade.py
```

---

## Project Structure

```
hedge-fund-os/
├── .github/workflows/         # GitHub Actions (automated daily trading)
│   └── paper_trade.yml        # Runs every weekday at 10:00 AM ET
├── dashboard.py               # Streamlit dashboard (5 pages)
├── paper_trade.py             # Alpaca paper trading
├── generate_signals.py        # Daily signal generator
├── run_v6_visual.py           # V6 backtest + charts
├── run_v7_visual.py           # V7 backtest + charts
├── src/
│   ├── agents/                # 6 agent implementations
│   ├── backtest/              # V1-V8 engines (complete evolution)
│   ├── data/                  # Ingestion + feature engineering (V1-V4)
│   ├── decision/              # Orchestrator
│   ├── memory/                # Scoped memory managers
│   └── schemas/               # Structured data models (Pydantic-style)
├── configs/                   # YAML configuration
├── docs/                      # Architecture documentation
├── reports/                   # Backtests + paper trading logs + charts
├── .env.example               # Alpaca API key template
└── tests/                     # Unit tests
```

---

## Known Limitations (honest)

- **Survivorship bias**: Universe uses today's large-cap names — not point-in-time
- **2012-2026 was exceptional**: Dominated by tech/growth momentum bull market
- **Max drawdown**: V6 hits -48%, QMB hits -28% — painful but survivable
- **No shorting**: Long-only limits Sharpe to ~1.0 without leverage
- **Free data only**: yfinance has limitations vs institutional feeds
- **No intraday monitoring**: System runs once daily, does not track positions intraday

---

## Roadmap

- [x] V1-V8 strategy evolution (8 versions)
- [x] Multi-agent decision system (4 agents, 890 decisions per backtest)
- [x] Interactive Streamlit dashboard (5 pages)
- [x] Alpaca paper trading integration
- [x] Automated daily trading via GitHub Actions
- [ ] Options overlay for income + reduced drawdown
- [ ] Live trading with real capital
- [ ] Alternative data integration (NLP, earnings sentiment)
- [ ] Intraday monitoring and real-time stop execution

---

## License

```
Apache License Version 2.0, January 2004
Copyright 2026 Sathwik Rao Nadipelli
```

**Additional Notice:** This repository contains a research and infrastructure framework only. The authors make no claims regarding financial performance, and any example results shown are for research and educational purposes only. This software is not intended to be used as financial advice or as a production trading system without independent validation.
