# Project Memory

Append-only log of project milestones, decisions, and changes.
Updated automatically via git hooks and manually at session end.

---

## 2026-03-17 — Project Initialization

### Summary
Created the full Hedge Fund OS repository from scratch:
- Architecture blueprint defining v1 scope, agents, memory, decision flow
- Complete repo scaffold with all directories and configs
- Working backtest engine with breakout trend-following strategy
- 6 agent stubs: Technical, Regime, Risk, PM, Review, Data Ops
- Feature engineering pipeline: MAs, ATR, volume, breakout, base detection, trend filter
- Structured Pydantic-style schemas for all data types
- YAML configs for strategy, agents, and risk
- Session continuity system: primer.md, project_memory.md, lessons.md
- Git hook for automatic commit logging

### Key Decisions
- V1 scope locked: US equities, long-only, daily bars, 6 agents
- Breakout strategy parameters: 150/200 MA trend, 20-day breakout, 1.5x volume
- Backtest realism: next-day open entry, 0.05% slippage, commissions
- Memory architecture: scoped per agent, per ticker, per regime, per portfolio
- Build order: backtest first → agents → memory → paper trading

### Architecture
- src/data/ — ingestion and features
- src/agents/ — agent implementations
- src/backtest/ — backtest engine
- src/schemas/ — structured data models
- configs/ — YAML configuration
- memory/ — persistent agent and trade memory
- docs/ — architecture and design documentation
