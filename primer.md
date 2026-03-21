# Project Primer

## Project
Hedge Fund OS — A modular multi-agent trading research system that simulates a hedge fund team with role-specialized agents, scoped memory, backtesting, paper trading, and performance attribution.

## Current Phase
Phase 1 — Core backtest engine and v1 breakout strategy

## Completed
- Project architecture blueprint (docs/architecture.md)
- Repository structure created
- Agent base class and v1 agent stubs (Technical, Regime, Risk, PM, Review)
- Data ingestion module (yfinance)
- Feature engineering pipeline (MAs, ATR, volume, breakout, base, trend)
- Backtest engine with realistic execution assumptions
- Strategy configuration (YAML)
- Agent configuration (YAML)
- Risk configuration (YAML)
- Structured schemas (signals, trades, portfolio state, metrics)
- README with project overview

## In Progress
- Running first backtest on 50-stock universe
- Validating feature engineering correctness
- Verifying signal generation logic

## Blocked
- Nothing currently blocked

## Latest Decisions
- Start with US equities, long-only, daily bars
- Use 6 agents for v1: Data Ops, Technical, Regime, Risk, PM, Review
- Use yfinance for prototype data (replace with paid source later)
- Backtest entry at next-day open with 0.05% slippage
- Risk per trade: 1%, max 10 positions
- Memory scoped per agent, per ticker, per regime, per portfolio

## Next Immediate Tasks
1. Run first backtest and validate results
2. Review exit reason distribution
3. Check if signal generation produces reasonable trade count
4. Add basic tests for feature computation
5. Begin wiring agents into orchestrated decision flow

## Open Questions
- Should regime agent be purely rule-based or include a simple model in v2?
- What is the right failed-breakout window (3 vs 5 vs 7 days)?
- Should we add sector exposure tracking in v1 or defer to v2?

## Known Constraints
- US equities only (v1)
- Daily bars only (v1)
- No live trading until paper-trading validation passes
- No ML models in v1 — rules-based only
- Max 10 concurrent positions
