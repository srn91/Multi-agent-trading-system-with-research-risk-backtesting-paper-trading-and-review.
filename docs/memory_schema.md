# Memory Schema

## Design Principles

1. **Scoped** — Each memory type lives in its own directory and file
2. **Bounded** — Max entries per memory store prevents unbounded growth
3. **Timestamped** — Every entry has a timestamp for chronological ordering
4. **Append-only** — New entries are appended; old entries age out naturally
5. **JSON-based** — Human-readable, easy to inspect and debug

## Memory Types

### Ticker Memory (`memory/ticker_memory/{TICKER}.json`)
Per-asset history of setups, trades, and outcomes.

### Regime Memory (`memory/regime_memory/regime_history.json`)
Rolling log of market regime classifications and transitions.

### Portfolio Memory (`memory/trade_journal/`)
Portfolio-level trade log and daily snapshots.

### Agent Memory (`memory/agent_memories/{agent_name}_memory.json`)
Per-agent log of decisions, predictions, and outcomes.

## Session Memory (Human-Readable)

### `primer.md`
Where we left off. Rewritten at session end.

### `project_memory.md`
Append-only project timeline. Auto-updated by git hook.

### `tasks/lessons.md`
Operational rules and corrections. Read at session start.

### `state/session_state.json`
Machine-readable current state for scripts and automation.
