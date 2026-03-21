# Evaluation Plan

## Strategy Evaluation

### Required Metrics
- Total return, CAGR
- Max drawdown, average drawdown
- Sharpe ratio, Sortino ratio
- Win rate, avg winner %, avg loser %
- Profit factor, expectancy per trade
- Exposure %, number of trades
- Average hold time, longest losing streak
- Performance by regime

### Passing Criteria (v1)
- CAGR > 8% (must beat risk-free)
- Max drawdown < -25%
- Sharpe > 0.5
- Profit factor > 1.3
- Win rate > 30% (with avg winner > 2x avg loser)
- Survives 2022 bear market without catastrophic drawdown

## Agent Evaluation

For each agent, track:
- Agreement rate with profitable trades
- Disagreement rate with losing trades
- False positive rate (recommended trade that lost)
- False negative rate (missed a winning setup)
- Usefulness by regime (trending vs choppy vs crisis)

## System-Level Evaluation

- Backtest vs paper trade discrepancy < 20%
- Decision audit trail is 100% complete
- Memory quality improves over time (fewer repeated mistakes)
- Exit reason distribution is healthy (not dominated by stop losses)

## Test Periods

| Period | Purpose | Dates |
|--------|---------|-------|
| In-sample | Build and light tuning | 2012–2020 |
| Out-of-sample | Honest validation | 2021–2025 |
| Walk-forward | Rolling robustness | 2-year train / 6-month test |
| Paper trading | Live simulation | 30–60 trades minimum |
