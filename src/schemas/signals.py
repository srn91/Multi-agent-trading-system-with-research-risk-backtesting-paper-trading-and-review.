"""
Structured schemas for the entire system.
Every agent input/output, every trade, every decision uses these.
No free-form dicts. No mystery keys.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CHOPPY = "choppy"
    HIGH_VOL_CRISIS = "high_vol_crisis"
    LOW_VOL_GRIND = "low_vol_grind"
    UNKNOWN = "unknown"


class ExitReason(Enum):
    TRAILING_STOP = "trailing_stop"
    FAILED_BREAKOUT = "failed_breakout"
    STOP_LOSS = "stop_loss"
    MAX_HOLD = "max_hold"
    REGIME_CHANGE = "regime_change"
    MANUAL = "manual"


@dataclass
class AgentOutput:
    """Universal structured output from any agent."""
    agent: str
    ticker: str
    timestamp: str
    thesis: str
    direction: Direction
    confidence: float  # 0.0 to 1.0
    evidence: list[str]
    risks: list[str]
    invalidation: list[str]
    recommendation: Direction
    memory_used: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class BreakoutSignal:
    """Output from the breakout scanner."""
    ticker: str
    signal_date: date
    breakout_close: float
    breakout_high: float
    base_low: float
    base_high: float
    base_length: int
    base_depth_pct: float
    volume_ratio: float  # today vol / 20d avg vol
    close_range_pct: float  # where close sits in day's range (0-1)
    atr: float
    ma_150: float
    ma_200: float
    spy_above_200: bool
    trend_valid: bool
    base_valid: bool
    breakout_valid: bool
    is_valid: bool  # all conditions met


@dataclass
class TradeEntry:
    """A trade that has been entered."""
    ticker: str
    entry_date: date
    entry_price: float
    stop_price: float
    shares: int
    risk_dollars: float
    risk_pct: float
    thesis: str
    signal: BreakoutSignal
    regime: Regime


@dataclass
class TradeExit:
    """A completed trade."""
    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    stop_price: float
    shares: int
    pnl_dollars: float
    pnl_pct: float
    hold_days: int
    exit_reason: ExitReason
    regime_at_entry: Regime
    regime_at_exit: Regime
    max_favorable: float  # best unrealized gain
    max_adverse: float  # worst unrealized loss


@dataclass
class PortfolioState:
    """Current portfolio snapshot."""
    date: date
    equity: float
    cash: float
    open_positions: list[TradeEntry]
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    wins: int
    losses: int
    current_drawdown: float
    max_drawdown: float
    peak_equity: float


@dataclass
class BacktestMetrics:
    """Final backtest results."""
    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    avg_winner_pct: float
    avg_loser_pct: float
    profit_factor: float
    expectancy_per_trade: float
    total_trades: int
    avg_hold_days: float
    max_consecutive_losses: int
    exposure_pct: float
    trades_per_year: float
