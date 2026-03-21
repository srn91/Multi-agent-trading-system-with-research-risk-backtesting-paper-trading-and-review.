# Hedge Fund OS

A role-specialized multi-agent trading research system designed to simulate a hedge fund workflow through research, risk control, backtesting, paper trading, and post-trade review.

## What This Project Is

Most trading repos stop at a single strategy or a single backtest. This project takes a different approach: it models trading as a structured decision process where specialized agents handle momentum, trend/quality, risk, market regime, portfolio decisions, and trade review.

The goal is not to build a toy signal bot. The goal is to build a modular research system with:
- clear decision roles
- auditable agent outputs
- backtesting and paper-trading workflows
- post-trade review and continuous refinement

---

## Current Status

| Area | Status |
|---|---|
| Multi-agent scoring pipeline | Implemented |
| Historical backtesting (8 versions) | Implemented |
| Portfolio construction with transaction costs | Implemented |
| Decision artifacts and audit trail | Implemented |
| Interactive Streamlit dashboard | Implemented |
| Paper trading via Alpaca API | Implemented |
| Automated daily execution (GitHub Actions) | Implemented |
| Walk-forward validation | In progress (V7 partial) |
| Live deployment | Not started |

---

## Strategy Evolution

This repo includes multiple research versions developed over time. Earlier versions (V1–V8) document the evolution of the system. The current production candidate is **Quality Momentum Breakout (QMB)**, selected for its risk-adjusted profile over the raw-CAGR leaders.

Historical versions are retained as research milestones, not as the canonical production path.

| Version | Strategy | CAGR | Max DD | Sharpe | Key Fix |
|---|---|---|---|---|---|
| V1 | Breakout only | -0.2% | -21% | -0.01 | Baseline — too many false breakouts |
| V3 | +Pullback, RS filter | +7.6% | -19% | 0.68 | Added pullback entries + relative strength |
| V4 | +Vol targeting, agents | +9.6% | -22% | 0.89 | First agent integration, vol-based sizing |
| V5 | Momentum rotation | +20.4% | -42% | 0.84 | Weekly rotation — first to beat SPY |
| V6 | +Agent team, threshold rebal | +21.2% | -48% | 0.84 | Highest CAGR but unacceptable drawdown |
| V7 | +Drawdown control, 3-factor | +15.3% | -26% | 0.90 | 46% DD reduction via vol targeting |
| V8 | Regime-adaptive rotation | +15.2% | -34% | 0.83 | Offense/defense switching (rejected) |
| **QMB** | **Quality Momentum Breakout** | **+19.8%** | **-28%** | **0.95** | **Production candidate: best risk-adjusted** |

> Results are from historical simulation under stated assumptions (2012–2026, 100 US large-cap stocks, weekly rebalance, 0.05% slippage + $0.005/share commission). These are research results, not guaranteed future performance.

![Equity Curve](reports/charts/01_equity_curve.png)

![Drawdown](reports/charts/02_drawdown.png)

---

## QMB Production Candidate

| Metric | QMB | SPY Buy & Hold |
|---|---|---|
| CAGR | +19.8% | +14.0% |
| Sharpe | 0.95 | ~0.65 |
| Max Drawdown | -28% | ~-33% |
| Profit Factor | 1.40 | — |
| Trades | 722 | 0 |

**Core logic:**
1. Rank 100 US stocks weekly by composite score: momentum (50%) + quality (25%) + inverse-volatility (25%)
2. 4 agents independently evaluate every candidate — conviction gate at 0.4
3. Hold top 7 stocks, sector-diversified (max 2 per sector)
4. Inverse-vol position weighting — stable stocks get bigger allocations
5. 15% trailing stop from peak, 20% hard stop from entry
6. Threshold rebalancing — only trade when rankings change meaningfully

---

## Architecture — 4 Specialized Decision Agents

Each trade is evaluated by 4 role-specialized agents. The PM Agent only executes when there is sufficient consensus. Every step produces a typed Signal object with reasoning and confidence score.

![Architecture](reports/charts/09_architecture.png)

| Agent | Role | What It Checks |
|---|---|---|
| **Momentum Agent** | Multi-timeframe momentum scanner | 3m/6m/12m composite ranking across 100 stocks |
| **Trend/Quality Agent** | Trend health validation | MA structure, relative strength, trend direction |
| **Risk Agent** | Position limits and exposure control | Vol-based sizing, sector caps (max 2), correlation penalty |
| **Regime Agent** | Market regime classification | Bull/Bear/Choppy detection, exposure adjustment |
| **PM Agent** | Final decision maker | Conviction gating (>0.4 to trade), consensus check |

### Decision Flow

1. Momentum Agent ranks all qualifying stocks by composite momentum
2. Trend/Quality Agent validates trend health for each candidate
3. Risk Agent enforces position limits, sector caps, and correlation checks
4. Regime Agent assesses market environment and adjusts exposure
5. PM Agent checks conviction threshold → execute or skip

---

## Example Decision Artifact

Below is a real decision from the 2019-01-24 rebalance, not a mockup:

```json
{
  "action": "BUY",
  "ticker": "ZS",
  "momentum_agent": {
    "score": 1.00,
    "evidence": "rank 1/43, 3m momentum +62%",
    "confidence": 1.00
  },
  "trend_quality_agent": {
    "score": 0.73,
    "evidence": "above 150MA, 150>200MA, RS rank 67/100",
    "confidence": 0.73
  },
  "risk_agent": {
    "score": 0.10,
    "evidence": "vol 46.6% — high but momentum compensates",
    "approved": true,
    "warnings": ["elevated volatility"]
  },
  "regime_agent": {
    "score": 0.78,
    "evidence": "market score 78/100, mild bull regime",
    "confidence": 0.78
  },
  "pm_agent": {
    "conviction": 0.743,
    "threshold": 0.400,
    "decision": "APPROVED",
    "position_size_multiplier": 1.3,
    "reason": "high conviction triggers 30% larger position"
  }
}
```

Full rebalance artifacts are stored in [`reports/decisions/`](reports/decisions/).

![Agent Decisions](reports/charts/07_agent_decisions.png)

---

## Trade Analysis

![Trade Analysis](reports/charts/04_trade_analysis.png)

- ~50% win rate with asymmetric payoff — winners average +15%, losers average -9%
- 73% of exits via trailing stop — system lets winners run
- ~30-day average hold — monthly rotation, not day trading

---

## Monthly Returns

![Monthly Heatmap](reports/charts/03_monthly_heatmap.png)

---

## Rolling Risk Metrics

![Rolling Sharpe](reports/charts/05_rolling_sharpe.png)

---

## Version Evolution

![Version Evolution](reports/charts/06_version_evolution.png)

---

## Per-Ticker Performance

![Ticker Performance](reports/charts/08_ticker_performance.png)

---

## Automated Paper Trading

The system runs via **GitHub Actions + Alpaca API**:

- Every weekday at 10:00 AM ET, GitHub Actions triggers the signal pipeline
- Downloads fresh market data, agents score and rank the universe
- Places buy/sell orders on Alpaca paper trading account
- Logs all decisions as downloadable artifacts

### Setup

1. Sign up free at [alpaca.markets](https://app.alpaca.markets/signup)
2. Generate Paper Trading API keys
3. Add keys to GitHub: **Settings → Secrets → Actions** (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`)
4. Workflow runs on schedule or trigger manually from **Actions** tab

---

## Reproducibility

Run the canonical backtest with:

```bash
python3 run_backtest.py
```

Reference configuration:
- **Strategy**: QMB (Quality Momentum Breakout)
- **Universe**: 100 liquid US large-cap equities
- **Benchmark**: SPY
- **Period**: 2012-01-01 to 2026-03-19
- **Rebalance**: weekly (every 5 trading days)
- **Costs**: 0.05% slippage per side + $0.005/share commission
- **Stops**: 15% trailing from peak, 20% hard from entry

---

## Quick Start

```bash
pip3 install -r requirements.txt
python3 run_backtest.py              # canonical backtest
python3 generate_signals.py          # today's signals
python3 -m streamlit run dashboard.py  # interactive dashboard
python3 paper_trade.py               # paper trading (requires Alpaca)
```

---

## Project Structure

```
├── run_backtest.py             # Canonical backtest entry point
├── dashboard.py                # Streamlit dashboard
├── paper_trade.py              # Alpaca paper trading
├── generate_signals.py         # Daily signal generator
├── .github/workflows/          # GitHub Actions automation
├── src/
│   ├── agents/                 # 6 agent implementations
│   ├── backtest/               # V1-V8 engines
│   ├── data/                   # Ingestion + feature engineering
│   ├── decision/               # Orchestrator
│   ├── memory/                 # Scoped memory managers
│   └── schemas/                # Structured data models
├── configs/                    # Strategy + risk configuration
├── reports/
│   ├── charts/                 # 10 publication-quality visualizations
│   ├── decisions/              # Agent decision artifacts
│   ├── backtests/              # Backtest output data
│   └── paper_trading/          # Paper trading logs
├── research/archive/           # V3-V7 historical runners (preserved)
├── docs/                       # Architecture documentation
├── tests/                      # Unit tests
├── notebooks/                  # Interactive exploration
└── CHANGELOG.md                # Version history
```

---

## Known Limitations

- Results are based on historical simulation and are not live trading results.
- The current system is a research prototype, not a production trading system.
- Performance is sensitive to universe definition, rebalance rules, and cost assumptions.
- Agent outputs are structured and auditable, but the system is still being refined for walk-forward robustness.
- Paper trading is more realistic than backtesting, but still does not capture full live execution behavior.
- Survivorship bias: universe uses today's large-cap names, not point-in-time constituents.
- 2012–2026 was a historically strong US equity bull market.

---

## Roadmap

- [x] V1-V8 strategy evolution
- [x] Role-specialized agent system
- [x] Interactive Streamlit dashboard
- [x] Alpaca paper trading
- [x] GitHub Actions automation
- [x] Decision artifacts and audit trail
- [ ] 90-day paper trading validation
- [ ] Walk-forward out-of-sample testing
- [ ] Agent-level performance attribution
- [ ] Live trading with small capital

---

## License

```
Apache License Version 2.0, January 2004
Copyright 2026 Sathwik Rao Nadipelli
```

This repository contains a research and infrastructure framework only. The authors make no claims regarding financial performance. Example results shown are for research and educational purposes only. This software is not intended to be used as financial advice or as a production trading system without independent validation.
