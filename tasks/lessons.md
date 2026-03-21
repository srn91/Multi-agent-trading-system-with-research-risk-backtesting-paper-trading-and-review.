# Tasks and Lessons

Rules learned from mistakes, corrections, and hard-won insights.
Read at session start. Applied before touching any code.

---

## Coding Rules
- Never let an agent approve its own trade.
- Every agent must return structured AgentOutput — no free-form dicts.
- Risk Agent has veto authority. PM cannot override.
- Backtest Agent cannot optimize parameters — it only tests what it receives.
- All file paths use pathlib, not string concatenation.
- Every function that touches data must handle NaN gracefully.

## Architecture Rules
- Keep workflow agents separate from decision agents.
- Scope memory by ticker, regime, portfolio, and review — never a universal blob.
- Agents get only the data they need. Forbidden inputs are enforced.
- Decision flow is sequential: Data → Regime → Technical → Risk → PM.
- Review Agent runs post-trade, not in the live decision loop.

## Data Rules
- Validate data freshness before feature computation.
- Log missing fields explicitly — silent failures are bugs.
- No signal generation on incomplete inputs.
- Always check for NaN after rolling window calculations.
- yfinance multi-level columns must be flattened before use.

## Backtest Rules
- Entry at next-day open, not breakout close (avoids lookahead).
- Include slippage (0.05% per side) and commissions ($0.005/share).
- Never test on out-of-sample data during development.
- Track exit reasons — they reveal strategy weaknesses faster than returns.
- A strategy with great return and terrible drawdown is still unusable.

## Evaluation Rules
- Track performance by agent and by regime.
- Compare backtest vs paper trade discrepancies.
- Do not judge strategy by total return alone — use Sharpe, drawdown, expectancy.
- Win rate alone is meaningless without avg winner/loser ratio.

## Project Rules
- Build portfolio-grade code, not course-grade demos.
- Prioritize clean architecture over flashy extras.
- Ship in phases, not as one giant system.
- Update primer.md at end of every working session.
- Never add more than 2 new agents per phase.

## Mistakes to Avoid
- Do not build 18 agents before proving one signal works.
- Do not add vector DB, RL, or ML before the rules-based system is validated.
- Do not optimize parameters on out-of-sample data.
- Do not trust a backtest that only works in bull markets.
- Do not skip paper trading — backtests lie, timestamps don't.
