"""
Backtest Engine — The Truth Machine.

This is where strategy ideas die or survive.
No opinions. No hope. Just math.

Rules:
- Entry at next-day open (not breakout close)
- Slippage: 0.05% per side
- Commission: $0.005/share
- No lookahead bias
- Max concurrent positions enforced
- Position sizing by fixed fractional risk
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.schemas.signals import (
    BacktestMetrics,
    BreakoutSignal,
    ExitReason,
    Regime,
    TradeEntry,
    TradeExit,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """All strategy and execution parameters in one place."""

    # Capital
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01  # 1% risk per trade

    # Execution realism
    slippage_pct: float = 0.0005  # 0.05% per side
    commission_per_share: float = 0.005

    # Position limits
    max_positions: int = 10
    max_position_pct: float = 0.20  # max 20% of equity in one name

    # Strategy: trend filter
    ma_fast: int = 150
    ma_slow: int = 200
    ma_slope_lookback: int = 20

    # Strategy: base detection
    base_min_len: int = 15
    base_max_len: int = 40
    base_max_depth: float = 0.12  # 12%

    # Strategy: breakout
    breakout_lookback: int = 20
    min_volume_ratio: float = 1.5
    min_close_range_pct: float = 0.25  # close in top 25% of day's range

    # Strategy: stop loss
    stop_atr_mult: float = 0.5
    max_stop_pct: float = 0.08  # 8% max stop distance

    # Strategy: exits
    failed_breakout_days: int = 5
    trailing_stop_lookback: int = 10  # 10-day low
    max_hold_days: int = 60

    # Market regime
    spy_ma: int = 200


class BacktestEngine:
    """
    Event-driven backtest engine for the breakout strategy.

    Walk through each day. Scan for signals. Manage positions. Track everything.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.peak_equity = self.config.initial_capital

        self.open_positions: list[dict] = []
        self.closed_trades: list[TradeExit] = []
        self.equity_curve: list[dict] = []
        self.signals_log: list[dict] = []

    def run(
        self,
        stock_data: dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
    ) -> BacktestMetrics:
        """
        Run the full backtest.

        Args:
            stock_data: dict of {ticker: DataFrame} with features already computed
            spy_data: SPY DataFrame with features already computed
        """
        # Get all trading dates from SPY
        dates = spy_data.index.tolist()
        logger.info(f"Running backtest: {dates[0].date()} to {dates[-1].date()}, {len(stock_data)} tickers")

        for i, current_date in enumerate(dates):
            if i < 252:  # Need at least 1 year of data for indicators
                continue

            # --- Update portfolio mark-to-market ---
            self._update_portfolio(current_date, stock_data)

            # --- Check exits on open positions ---
            self._check_exits(current_date, stock_data, spy_data)

            # --- Check for new entries ---
            if len(self.open_positions) < self.config.max_positions:
                spy_above_ma = self._spy_regime_ok(current_date, spy_data)
                if spy_above_ma:
                    self._scan_for_entries(current_date, dates, i, stock_data, spy_data)

            # --- Record equity curve ---
            self._record_equity(current_date, stock_data)

        # --- Compute final metrics ---
        return self._compute_metrics(dates[0].date(), dates[-1].date())

    def _spy_regime_ok(self, dt, spy_data: pd.DataFrame) -> bool:
        """Check if SPY is above its 200-day MA."""
        if dt not in spy_data.index:
            return False
        row = spy_data.loc[dt]
        if pd.isna(row.get("ma_200", np.nan)):
            return False
        return row["close"] > row["ma_200"]

    def _scan_for_entries(self, current_date, dates, idx, stock_data, spy_data):
        """Scan all stocks for valid breakout signals on current_date."""
        cfg = self.config

        # Get tickers we're not already holding
        held_tickers = {p["ticker"] for p in self.open_positions}

        for ticker, df in stock_data.items():
            if ticker in held_tickers:
                continue
            if len(self.open_positions) >= cfg.max_positions:
                break
            if current_date not in df.index:
                continue

            row = df.loc[current_date]

            # --- Validate all conditions ---
            # Trend filter
            if not row.get("trend_valid", False):
                continue

            # Base detection
            if not row.get("in_base", False):
                continue
            if pd.isna(row.get("base_depth_pct", np.nan)):
                continue
            if row["base_depth_pct"] > cfg.base_max_depth:
                continue

            # Breakout
            if not row.get("breakout", False):
                continue

            # Volume confirmation
            vol_ratio = row.get("vol_ratio", 0)
            if pd.isna(vol_ratio) or vol_ratio < cfg.min_volume_ratio:
                continue

            # Close in top portion of range
            close_range = row.get("close_range_pct", 0)
            if pd.isna(close_range) or close_range < cfg.min_close_range_pct:
                continue

            # ATR compression
            if not row.get("atr_compression", False):
                continue

            # --- Signal is valid ---
            signal = BreakoutSignal(
                ticker=ticker,
                signal_date=current_date.date() if hasattr(current_date, 'date') else current_date,
                breakout_close=row["close"],
                breakout_high=row["high"],
                base_low=row.get("base_low", row["low"]),
                base_high=row.get("base_high", row["high"]),
                base_length=int(row.get("base_length", 20)),
                base_depth_pct=row["base_depth_pct"],
                volume_ratio=vol_ratio,
                close_range_pct=close_range,
                atr=row.get("atr", 0),
                ma_150=row.get("ma_150", 0),
                ma_200=row.get("ma_200", 0),
                spy_above_200=True,
                trend_valid=True,
                base_valid=True,
                breakout_valid=True,
                is_valid=True,
            )

            # --- Compute entry for NEXT DAY ---
            if idx + 1 >= len(dates):
                continue
            next_date = dates[idx + 1]
            if next_date not in df.index:
                continue

            next_row = df.loc[next_date]
            entry_price = next_row["open"] * (1 + cfg.slippage_pct)  # slippage on entry

            # --- Compute stop ---
            atr_val = row.get("atr", 0)
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            base_low = row.get("base_low", row["low"])
            if pd.isna(base_low):
                base_low = row["low"]

            stop_from_base = base_low - cfg.stop_atr_mult * atr_val
            stop_from_pct = entry_price * (1 - cfg.max_stop_pct)
            stop_price = max(stop_from_base, stop_from_pct)

            if stop_price >= entry_price:
                continue  # Invalid: stop above entry

            # --- Position sizing ---
            per_share_risk = entry_price - stop_price
            if per_share_risk <= 0:
                continue

            account_risk = self.equity * cfg.risk_per_trade
            shares = int(account_risk / per_share_risk)

            if shares <= 0:
                continue

            # Cap position size
            position_value = shares * entry_price
            max_position = self.equity * cfg.max_position_pct
            if position_value > max_position:
                shares = int(max_position / entry_price)
            if shares <= 0:
                continue

            # Check we have enough cash
            cost = shares * entry_price + shares * cfg.commission_per_share
            if cost > self.cash:
                shares = int((self.cash - 100) / (entry_price + cfg.commission_per_share))
            if shares <= 0:
                continue

            # --- Execute entry ---
            actual_cost = shares * entry_price + shares * cfg.commission_per_share
            self.cash -= actual_cost

            position = {
                "ticker": ticker,
                "entry_date": next_date,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "shares": shares,
                "risk_dollars": shares * per_share_risk,
                "risk_pct": per_share_risk / entry_price,
                "breakout_level": row["close"],
                "signal": signal,
                "max_price": entry_price,
                "min_price": entry_price,
                "days_held": 0,
            }
            self.open_positions.append(position)

            self.signals_log.append({
                "date": current_date,
                "ticker": ticker,
                "action": "entry",
                "price": entry_price,
                "stop": stop_price,
                "shares": shares,
            })

            logger.debug(f"ENTRY: {ticker} @ {entry_price:.2f}, stop {stop_price:.2f}, {shares} shares")

    def _check_exits(self, current_date, stock_data, spy_data):
        """Check all open positions for exit conditions."""
        cfg = self.config
        positions_to_close = []

        for pos in self.open_positions:
            ticker = pos["ticker"]
            df = stock_data.get(ticker)
            if df is None or current_date not in df.index:
                continue

            row = df.loc[current_date]
            current_price = row["close"]
            pos["days_held"] += 1

            # Track max/min for favorable/adverse excursion
            pos["max_price"] = max(pos["max_price"], row["high"])
            pos["min_price"] = min(pos["min_price"], row["low"])

            exit_reason = None

            # 1. Stop loss hit (check against low)
            if row["low"] <= pos["stop_price"]:
                exit_reason = ExitReason.STOP_LOSS
                current_price = pos["stop_price"] * (1 - cfg.slippage_pct)

            # 2. Failed breakout: close below breakout level within N days
            elif (
                pos["days_held"] <= cfg.failed_breakout_days
                and row["close"] < pos["breakout_level"]
            ):
                exit_reason = ExitReason.FAILED_BREAKOUT

            # 3. Trailing stop: close below 10-day low
            elif pos["days_held"] > cfg.failed_breakout_days:
                loc = df.index.get_loc(current_date)
                if loc >= cfg.trailing_stop_lookback:
                    trail_low = df["low"].iloc[
                        loc - cfg.trailing_stop_lookback : loc
                    ].min()
                    if row["close"] < trail_low:
                        exit_reason = ExitReason.TRAILING_STOP

            # 4. Max hold period
            if pos["days_held"] >= cfg.max_hold_days:
                exit_reason = ExitReason.MAX_HOLD

            # 5. Regime change
            if not self._spy_regime_ok(current_date, spy_data):
                exit_reason = ExitReason.REGIME_CHANGE

            if exit_reason:
                positions_to_close.append((pos, current_price, exit_reason, current_date))

        # Execute exits
        for pos, exit_price, reason, dt in positions_to_close:
            self._close_position(pos, exit_price, reason, dt)

    def _close_position(self, pos, exit_price, reason, exit_date):
        """Close a position and record the trade."""
        cfg = self.config
        exit_price_after_slip = exit_price * (1 - cfg.slippage_pct)
        proceeds = pos["shares"] * exit_price_after_slip - pos["shares"] * cfg.commission_per_share
        self.cash += proceeds

        pnl_dollars = (exit_price_after_slip - pos["entry_price"]) * pos["shares"]
        pnl_pct = (exit_price_after_slip - pos["entry_price"]) / pos["entry_price"]

        trade = TradeExit(
            ticker=pos["ticker"],
            entry_date=pos["entry_date"].date() if hasattr(pos["entry_date"], 'date') else pos["entry_date"],
            exit_date=exit_date.date() if hasattr(exit_date, 'date') else exit_date,
            entry_price=pos["entry_price"],
            exit_price=exit_price_after_slip,
            stop_price=pos["stop_price"],
            shares=pos["shares"],
            pnl_dollars=pnl_dollars,
            pnl_pct=pnl_pct,
            hold_days=pos["days_held"],
            exit_reason=reason,
            regime_at_entry=Regime.UNKNOWN,
            regime_at_exit=Regime.UNKNOWN,
            max_favorable=(pos["max_price"] - pos["entry_price"]) / pos["entry_price"],
            max_adverse=(pos["min_price"] - pos["entry_price"]) / pos["entry_price"],
        )
        self.closed_trades.append(trade)
        self.open_positions.remove(pos)

        logger.debug(
            f"EXIT: {pos['ticker']} @ {exit_price_after_slip:.2f}, "
            f"PnL: {pnl_pct:+.2%}, reason: {reason.value}"
        )

    def _update_portfolio(self, current_date, stock_data):
        """Update portfolio equity based on current prices."""
        unrealized = 0.0
        for pos in self.open_positions:
            df = stock_data.get(pos["ticker"])
            if df is not None and current_date in df.index:
                current_price = df.loc[current_date, "close"]
                unrealized += (current_price - pos["entry_price"]) * pos["shares"]

        self.equity = self.cash + sum(
            pos["shares"] * pos["entry_price"] for pos in self.open_positions
        ) + unrealized

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def _record_equity(self, current_date, stock_data):
        """Record daily equity for the equity curve."""
        dd = (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.equity_curve.append({
            "date": current_date,
            "equity": self.equity,
            "cash": self.cash,
            "positions": len(self.open_positions),
            "drawdown": dd,
        })

    def _compute_metrics(self, start_date, end_date) -> BacktestMetrics:
        """Compute all performance metrics from closed trades and equity curve."""
        trades = self.closed_trades
        eq = pd.DataFrame(self.equity_curve)

        if len(trades) == 0:
            logger.warning("No trades executed.")
            return BacktestMetrics(
                start_date=start_date, end_date=end_date,
                initial_capital=self.config.initial_capital,
                final_equity=self.equity,
                total_return_pct=0, cagr_pct=0, max_drawdown_pct=0,
                avg_drawdown_pct=0, sharpe_ratio=0, sortino_ratio=0,
                win_rate=0, avg_winner_pct=0, avg_loser_pct=0,
                profit_factor=0, expectancy_per_trade=0, total_trades=0,
                avg_hold_days=0, max_consecutive_losses=0, exposure_pct=0,
                trades_per_year=0,
            )

        # Basic returns
        total_return = (self.equity - self.config.initial_capital) / self.config.initial_capital
        years = max((end_date - start_date).days / 365.25, 0.1)
        cagr = (1 + total_return) ** (1 / years) - 1

        # Drawdown
        if not eq.empty:
            max_dd = eq["drawdown"].min()
            avg_dd = eq.loc[eq["drawdown"] < 0, "drawdown"].mean() if (eq["drawdown"] < 0).any() else 0
        else:
            max_dd = 0
            avg_dd = 0

        # Win/loss stats
        pnls = [t.pnl_pct for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        win_rate = len(winners) / len(trades) if trades else 0
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = np.mean(losers) if losers else 0

        # Profit factor
        gross_profit = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
        gross_loss = abs(sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        expectancy = np.mean(pnls) if pnls else 0

        # Sharpe / Sortino from equity curve
        if not eq.empty and len(eq) > 1:
            daily_returns = eq["equity"].pct_change().dropna()
            sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
            downside = daily_returns[daily_returns < 0]
            sortino = (
                daily_returns.mean() / downside.std() * np.sqrt(252)
                if len(downside) > 0 and downside.std() > 0
                else 0
            )
        else:
            sharpe = 0
            sortino = 0

        # Hold time
        hold_days = [t.hold_days for t in trades]
        avg_hold = np.mean(hold_days) if hold_days else 0

        # Max consecutive losses
        max_consec_losses = 0
        current_streak = 0
        for p in pnls:
            if p <= 0:
                current_streak += 1
                max_consec_losses = max(max_consec_losses, current_streak)
            else:
                current_streak = 0

        # Exposure
        if not eq.empty:
            exposure = (eq["positions"] > 0).mean()
        else:
            exposure = 0

        trades_per_year = len(trades) / years if years > 0 else 0

        metrics = BacktestMetrics(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital,
            final_equity=round(self.equity, 2),
            total_return_pct=round(total_return * 100, 2),
            cagr_pct=round(cagr * 100, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            avg_drawdown_pct=round(avg_dd * 100, 2) if avg_dd else 0,
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            win_rate=round(win_rate * 100, 2),
            avg_winner_pct=round(avg_winner * 100, 2),
            avg_loser_pct=round(avg_loser * 100, 2),
            profit_factor=round(profit_factor, 2),
            expectancy_per_trade=round(expectancy * 100, 2),
            total_trades=len(trades),
            avg_hold_days=round(avg_hold, 1),
            max_consecutive_losses=max_consec_losses,
            exposure_pct=round(exposure * 100, 2),
            trades_per_year=round(trades_per_year, 1),
        )

        return metrics

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve)

    def get_trade_log(self) -> pd.DataFrame:
        """Return all closed trades as DataFrame."""
        if not self.closed_trades:
            return pd.DataFrame()
        records = []
        for t in self.closed_trades:
            records.append({
                "ticker": t.ticker,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "shares": t.shares,
                "pnl_dollars": round(t.pnl_dollars, 2),
                "pnl_pct": round(t.pnl_pct * 100, 2),
                "hold_days": t.hold_days,
                "exit_reason": t.exit_reason.value,
                "max_favorable_pct": round(t.max_favorable * 100, 2),
                "max_adverse_pct": round(t.max_adverse * 100, 2),
            })
        return pd.DataFrame(records)

    def print_summary(self, metrics: BacktestMetrics):
        """Print a clean summary of backtest results."""
        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Period:          {metrics.start_date} → {metrics.end_date}")
        print(f"  Initial Capital: ${metrics.initial_capital:,.0f}")
        print(f"  Final Equity:    ${metrics.final_equity:,.0f}")
        print(f"  Total Return:    {metrics.total_return_pct:+.2f}%")
        print(f"  CAGR:            {metrics.cagr_pct:+.2f}%")
        print(f"  Max Drawdown:    {metrics.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:   {metrics.sortino_ratio:.2f}")
        print("-" * 60)
        print(f"  Total Trades:    {metrics.total_trades}")
        print(f"  Win Rate:        {metrics.win_rate:.1f}%")
        print(f"  Avg Winner:      {metrics.avg_winner_pct:+.2f}%")
        print(f"  Avg Loser:       {metrics.avg_loser_pct:.2f}%")
        print(f"  Profit Factor:   {metrics.profit_factor:.2f}")
        print(f"  Expectancy:      {metrics.expectancy_per_trade:+.2f}% per trade")
        print("-" * 60)
        print(f"  Avg Hold:        {metrics.avg_hold_days:.1f} days")
        print(f"  Max Consec Loss: {metrics.max_consecutive_losses}")
        print(f"  Exposure:        {metrics.exposure_pct:.1f}%")
        print(f"  Trades/Year:     {metrics.trades_per_year:.1f}")
        print("=" * 60 + "\n")
