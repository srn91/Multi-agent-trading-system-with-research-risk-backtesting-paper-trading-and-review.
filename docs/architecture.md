# Hedge Fund OS — Architecture Blueprint

## 1. Project Vision

A modular multi-agent trading research system that simulates a hedge fund team.
Each agent owns a narrow function, operates on scoped data, writes to scoped memory,
and produces structured outputs. Decisions are auditable. Performance is attributed
per agent, per regime, per strategy.

**This is not a chatbot that talks about stocks.**
**This is a decision system that generates, validates, and executes trading ideas.**

---

## 2. V1 Scope — What We Build First

### In Scope
- US equities, daily bars, long-only
- One core strategy: breakout trend-following
- 6 agents: Data Ops, Technical, Regime, Risk, PM, Review
- Structured memory: ticker, regime, portfolio, review
- Backtest engine with realistic assumptions
- Paper trading simulation
- Performance metrics and attribution
- Persistent project memory (primer, history, lessons)

### Out of Scope (v1)
- Live broker integration
- Options / futures / crypto
- Fundamental / news agents
- Multi-strategy portfolio
- Real-time streaming data
- Machine learning models
- Vector DB / embeddings
- External orchestration frameworks

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                         │
│  prices · volume · fundamentals · macro · metadata   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 FEATURE LAYER                        │
│  returns · MAs · ATR · momentum · regime indicators  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  AGENT LAYER                         │
│  Technical · Regime · Risk · PM · Review · Data Ops  │
│  Each agent: scoped inputs → structured output       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               DECISION LAYER                         │
│  Orchestrator aggregates → PM decides → Risk vetoes  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│             EXECUTION LAYER                          │
│  Backtest engine · Paper trading simulator            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              MEMORY LAYER                            │
│  Ticker · Regime · Portfolio · Review · Agent-scoped  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│             REPORTING LAYER                          │
│  Metrics · Attribution · Trade journal · Dashboards   │
└─────────────────────────────────────────────────────┘
```

---

## 4. Agent Roster — V1

### 4.1 Data Ops Agent

| Field | Value |
|-------|-------|
| **Mission** | Collect, clean, validate, and transform market data |
| **Allowed Inputs** | Raw OHLCV, fundamentals, macro indicators, metadata |
| **Forbidden Inputs** | Agent opinions, trade decisions, portfolio state |
| **Memory Scope** | Data quality logs, missing-data flags |
| **Output** | Clean datasets, feature-ready tables, quality report |
| **Success Metric** | Data completeness, freshness, zero downstream errors |
| **Failure Metric** | Missing fields not flagged, stale data passed through |

### 4.2 Technical Agent

| Field | Value |
|-------|-------|
| **Mission** | Identify technically favorable setups (breakouts, trend, compression) |
| **Allowed Inputs** | OHLCV, technical indicators, volatility features |
| **Forbidden Inputs** | Fundamentals, news, sentiment, portfolio PnL |
| **Memory Scope** | Past technical setup outcomes by ticker and regime |
| **Output** | Signal type, entry zone, stop zone, confidence, invalidation triggers |
| **Success Metric** | Precision of approved setups, expectancy by regime |
| **Failure Metric** | False breakouts, late entries, regime mismatch |

### 4.3 Regime Agent

| Field | Value |
|-------|-------|
| **Mission** | Classify current market regime |
| **Allowed Inputs** | Broad market indices, volatility measures, breadth, rates |
| **Forbidden Inputs** | Individual stock signals, portfolio positions |
| **Memory Scope** | Regime history, transition patterns |
| **Output** | Regime label, confidence, supporting indicators |
| **Success Metric** | Regime accuracy vs realized market behavior |
| **Failure Metric** | Late regime calls, false transitions |

Regime labels (v1):
- `trending_up` — broad market rising, low vol
- `trending_down` — broad market falling
- `choppy` — sideways, high whipsaw risk
- `high_vol_crisis` — panic, correlation spike
- `low_vol_grind` — calm, range-bound

### 4.4 Risk Agent

| Field | Value |
|-------|-------|
| **Mission** | Compute allowed risk, enforce position limits, veto dangerous trades |
| **Allowed Inputs** | Portfolio state, volatility, correlation, drawdown, exposure |
| **Forbidden Inputs** | Individual stock fundamentals, news sentiment |
| **Memory Scope** | Drawdown history, exposure history, past veto outcomes |
| **Output** | Allowed risk budget, max position size, concentration warnings, veto signal |
| **Success Metric** | Drawdown control, survival through hostile regimes |
| **Failure Metric** | Excessive drawdown, missed veto on correlated positions |
| **Special Authority** | **Veto power. No debate.** |

### 4.5 PM (Portfolio Manager) Agent

| Field | Value |
|-------|-------|
| **Mission** | Aggregate all agent outputs and make final trade decision |
| **Allowed Inputs** | Summaries from all agents (not raw data) |
| **Forbidden Inputs** | Raw price data, raw features |
| **Memory Scope** | Past decisions and outcomes, decision rationale log |
| **Output** | Action (long/flat), size, stop, thesis, invalidation |
| **Success Metric** | Portfolio-level expectancy, risk-adjusted return |
| **Failure Metric** | Overtrading, ignoring risk veto, thesis drift |

### 4.6 Review Agent

| Field | Value |
|-------|-------|
| **Mission** | Post-trade analysis — what worked, what failed, why |
| **Allowed Inputs** | Trade logs, agent output logs, outcome data, regime context |
| **Forbidden Inputs** | Live market data, current signals |
| **Memory Scope** | Per-agent accuracy, per-regime performance, failure patterns |
| **Output** | Attribution report, agent usefulness scores, refinement suggestions |
| **Success Metric** | Identifies real failure patterns, improves system over time |
| **Failure Metric** | Generic reviews, no actionable corrections |

---

## 5. Memory Schema

### 5.1 Memory Layers

| Layer | Scope | Lifespan | Format |
|-------|-------|----------|--------|
| **Ticker Memory** | Per asset | Persistent | JSON |
| **Regime Memory** | Market-wide | Rolling window | JSON |
| **Portfolio Memory** | Portfolio-wide | Persistent | JSON |
| **Review Memory** | System-wide | Persistent | JSON |
| **Agent Memory** | Per agent | Persistent | JSON |

### 5.2 Ticker Memory Schema

```json
{
  "ticker": "AAPL",
  "last_updated": "2026-03-17",
  "past_setups": [
    {
      "date": "2026-02-10",
      "setup_type": "breakout",
      "entry": 185.20,
      "stop": 178.50,
      "outcome": "winner",
      "pnl_pct": 4.2,
      "regime": "trending_up",
      "hold_days": 12,
      "exit_reason": "trailing_stop"
    }
  ],
  "support_levels": [175.0, 180.0],
  "resistance_levels": [195.0],
  "notes": []
}
```

### 5.3 Portfolio Memory Schema

```json
{
  "last_updated": "2026-03-17",
  "current_positions": [],
  "realized_pnl": [],
  "max_drawdown": -0.035,
  "current_drawdown": -0.01,
  "total_trades": 47,
  "win_rate": 0.42,
  "avg_winner": 0.038,
  "avg_loser": -0.012,
  "exposure_history": []
}
```

---

## 6. Agent Output Schema (Universal)

Every agent returns this structure:

```json
{
  "agent": "technical_agent",
  "ticker": "AAPL",
  "timestamp": "2026-03-17T14:30:00Z",
  "thesis": "20-day breakout from volatility compression",
  "direction": "long",
  "confidence": 0.72,
  "evidence": [
    "close above 20-day high",
    "volume 1.7x 20-day average",
    "ATR compression over 15 days"
  ],
  "risks": [
    "broad market extended",
    "earnings in 5 days"
  ],
  "invalidation": [
    "close back below breakout level within 5 days",
    "regime flips to choppy"
  ],
  "recommendation": "long",
  "memory_used": ["past breakout on 2025-11-03 succeeded in similar regime"],
  "metadata": {}
}
```

---

## 7. Decision Flow

```
For each ticker in universe:

1. Data Ops Agent → clean data + features
2. Regime Agent → classify market state
3. IF regime is hostile → skip (no trade)
4. Technical Agent → score setup
5. IF no valid setup → skip
6. Risk Agent → compute allowed size, check exposure
7. IF Risk vetoes → skip
8. PM Agent → final decision (long / flat / size / stop / thesis)
9. Log everything
10. Execute in backtest or paper trade
11. After outcome → Review Agent audits
12. Write results to memory
```

---

## 8. Core Strategy — V1 Breakout System

### Universe
- US equities
- Close > $10
- Avg daily dollar volume > $5M

### Market Regime Filter
- SPY close > 200-day MA

### Stock Trend Filter
- Close > 150-day MA
- 150-day MA > 200-day MA
- 200-day MA rising over last 20 days

### Base Detection
- 15–40 day consolidation
- Base depth ≤ 12%
- ATR compression present (20-day ATR/Close declining)

### Breakout Trigger
- Close > prior 20-day high
- Volume ≥ 1.5x 20-day average
- Close in top 25% of daily range

### Entry
- Next day open after valid breakout close

### Stop Loss
- Lower of: (base low − 0.5 × ATR) or (entry × 0.92)

### Position Sizing
- Risk 0.5%–1.0% of account per trade
- Shares = floor(account_risk / per_share_risk)

### Exit Rules
- Failed breakout: if close falls below breakout level within 5 days → exit
- Trailing stop: 10-day low or 20-day EMA, whichever is tighter
- Hard max hold: 60 trading days

---

## 9. Backtest Requirements

### Realism Rules
- Entry at next-day open (not breakout close)
- Include 0.05% slippage per side
- Include $0.005/share commission estimate
- No lookahead bias
- Survivorship-bias-free universe (if available)
- Max 10 concurrent positions

### Test Periods
- In-sample: 2012–2020
- Out-of-sample: 2021–2025
- Walk-forward: rolling 2-year train / 6-month test

### Required Metrics
- Total return, CAGR
- Max drawdown, avg drawdown
- Sharpe ratio, Sortino ratio
- Win rate, avg winner, avg loser
- Profit factor, expectancy per trade
- Exposure %, number of trades
- Average hold time
- Longest losing streak
- Performance by regime

---

## 10. Evaluation Plan

### Per-Strategy
- Does it beat buy-and-hold on risk-adjusted basis?
- Does it survive bear markets?
- Is drawdown tolerable?
- Is it robust across parameter variations?

### Per-Agent
- Agreement rate with profitable trades
- Disagreement rate with losing trades
- False positive / false negative rates
- Usefulness by regime

### System-Level
- Backtest vs paper trade discrepancy
- Decision audit trail completeness
- Memory quality over time

---

## 11. Roadmap

### Phase 1 — Core Engine (Weeks 1–3)
- Data ingestion (yfinance)
- Feature engineering
- Breakout strategy rules
- Backtest engine with metrics
- Trade logging

### Phase 2 — Agent Layer (Weeks 4–6)
- Agent base class + output schema
- Technical, Regime, Risk, PM, Review agents
- Orchestrator / decision flow
- Agent-scoped memory

### Phase 3 — Memory + Continuity (Weeks 7–8)
- Ticker, regime, portfolio memory
- primer.md / project_memory.md / lessons.md
- Session state management
- Git hooks for auto-logging

### Phase 4 — Paper Trading (Weeks 9–10)
- Simulated live execution
- Daily signal generation
- Trade journal
- Performance review loop

### Phase 5 — Expansion (Future)
- Event/news agent
- Fundamental agent
- Multi-strategy portfolio
- Vol targeting
- Factor attribution
- Dashboard UI
- Broker integration

---

## 12. Non-Negotiable Rules

1. No agent approves its own work.
2. Risk Agent has veto power. Always.
3. Every trade has a thesis and invalidation condition before entry.
4. Backtest before paper trade. Paper trade before real money.
5. Memory is scoped — no universal blob.
6. Every decision is logged with agent, evidence, and outcome.
7. Parameters are not optimized on out-of-sample data.
8. The system must handle "do nothing" as a valid output.
9. Build in phases. Ship working pieces. Don't architect forever.
10. If the backtest can't survive 2022, the strategy doesn't deserve capital.
