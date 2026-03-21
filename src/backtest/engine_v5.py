"""
Backtest Engine V5 — Beat The Market.

Philosophy change from V1-V4:
- FEWER trades, BIGGER winners, LONGER holds
- Monthly momentum rotation as the CORE strategy (not a side dish)
- Hold winners for months, not days
- Wide trailing stops (15-20% from peak, not 10-day low)
- Go to CASH in bear markets (don't fight the trend)
- Concentrated portfolio (8-10 names max)
- Transaction costs: 0.05% slippage + $0.005/share commission (unchanged)

What killed V1-V4:
- 125 trades/year × 0.1% round-trip cost = 12.5% annual drag
- 35% win rate with 10-day holds = never catching the big move
- Trailing stops too tight = selling winners early

V5 targets:
- 20-30 trades/year
- Hold 30-90 days average
- 45%+ win rate
- Capture 80%+ of major moves
"""
import logging
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from src.schemas.signals import BacktestMetrics, ExitReason, Regime, TradeExit

logger = logging.getLogger(__name__)


@dataclass
class V5Config:
    initial_capital: float = 100_000.0

    # Execution costs (REALISTIC — same as V1-V4)
    slippage_pct: float = 0.0005       # 0.05% per side
    commission_per_share: float = 0.005 # $0.005/share

    # Portfolio
    max_positions: int = 10
    max_position_pct: float = 0.15     # 15% max per name
    risk_per_trade: float = 0.02       # 2% risk (wider stops = need more room)

    # MOMENTUM ROTATION (core strategy)
    rebal_frequency: int = 21          # monthly rebalance
    momentum_lookback: int = 252       # 12-month momentum
    momentum_skip: int = 21            # skip most recent month
    top_n: int = 10                    # hold top 10
    min_price: float = 10.0
    require_uptrend: bool = True       # must be above 200MA

    # STOPS (wider than V1-V4)
    trailing_stop_pct: float = 0.15    # 15% trailing from peak
    hard_stop_pct: float = 0.20        # 20% max loss per position

    # REGIME (go to cash in bear markets)
    regime_ma: int = 200
    regime_ma_short: int = 50
    bear_market_cash: bool = True      # 100% cash when SPY < 200MA

    # VOL TARGETING
    vol_target: float = 0.12           # 12% annualized
    vol_targeting: bool = True
    vol_lookback: int = 20


class BacktestEngineV5:
    def __init__(self, config=None):
        self.config = config or V5Config()
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.peak_equity = self.config.initial_capital

        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.signals_log = []
        self.agent_decisions = []
        self.rebal_count = 0
        self.days_since_rebal = 999

    def run(self, stock_data, spy_data):
        dates = spy_data.index.tolist()
        logger.info(f"V5: {dates[0].date()} to {dates[-1].date()}, {len(stock_data)} tickers")

        for i, dt in enumerate(dates):
            if i < 260:  # need 252 + buffer for momentum calc
                continue

            # Mark to market
            self._update_portfolio(dt, stock_data)

            # Check regime
            in_bear = self._is_bear_market(dt, spy_data)

            # Bear market: close everything and wait
            if in_bear and self.config.bear_market_cash:
                if self.open_positions:
                    self._close_all(dt, stock_data, "bear_market")
                    self.agent_decisions.append({
                        "date": str(dt.date()), "agent": "regime",
                        "action": "close_all_bear_market",
                        "reasoning": "SPY below 200MA — go to cash",
                    })
                self._record_equity(dt, stock_data)
                continue

            # Check trailing stops on existing positions
            self._check_stops(dt, stock_data)

            # Monthly rebalance
            self.days_since_rebal += 1
            if self.days_since_rebal >= self.config.rebal_frequency:
                self._rebalance(dt, dates, i, stock_data, spy_data)
                self.days_since_rebal = 0
                self.rebal_count += 1

            self._record_equity(dt, stock_data)

        return self._compute_metrics(dates[0].date(), dates[-1].date())

    def _is_bear_market(self, dt, spy):
        """Simple but effective: SPY below 200MA = bear market."""
        if dt not in spy.index: return False
        row = spy.loc[dt]
        ma200 = row.get("ma_200", np.nan)
        if pd.isna(ma200): return False
        close = row["close"]

        # Primary: below 200MA
        below_200 = close < ma200

        # Secondary: 50MA below 200MA (death cross)
        ma50 = row.get("ma_50", np.nan)
        death_cross = False
        if not pd.isna(ma50):
            death_cross = ma50 < ma200

        # Be aggressive about going to cash: either signal triggers it
        return below_200

    def _rebalance(self, dt, dates, idx, stock_data, spy_data):
        """
        Monthly momentum rotation.
        Rank all stocks by 12-month return (skip last month).
        Buy top N. Sell anything not in top N.
        """
        cfg = self.config
        lookback = cfg.momentum_lookback
        skip = cfg.momentum_skip

        # Rank stocks by momentum
        rankings = []
        for ticker, df in stock_data.items():
            if dt not in df.index: continue
            loc = df.index.get_loc(dt)
            if loc < lookback + skip: continue

            # 12-month return, skipping most recent month
            price_now = df.iloc[loc - skip]["close"]
            price_past = df.iloc[loc - lookback]["close"]
            if price_past <= 0: continue
            momentum = (price_now / price_past) - 1

            # Current price check
            current_price = df.iloc[loc]["close"]
            if current_price < cfg.min_price: continue

            # Must be in uptrend (above 200MA)
            ma200 = df.iloc[loc].get("ma_200", 0)
            if pd.isna(ma200): ma200 = 0
            if cfg.require_uptrend:
                if ma200 <= 0 or current_price < ma200: continue

            # Relative strength
            rs = df.iloc[loc].get("rs_rank_score", 50)
            if pd.isna(rs): rs = 50

            # Volatility for sizing
            vol = df.iloc[loc].get("realized_vol_20", 0.20)
            if pd.isna(vol) or vol <= 0: vol = 0.20

            rankings.append({
                "ticker": ticker,
                "momentum": momentum,
                "rs": rs,
                "price": current_price,
                "vol": vol,
                "ma200": ma200 if not pd.isna(ma200) else 0,
            })

        if not rankings: return

        # Sort by momentum (strongest first)
        rankings.sort(key=lambda x: x["momentum"], reverse=True)
        target_tickers = [r["ticker"] for r in rankings[:cfg.top_n]]
        target_set = set(target_tickers)

        # Log agent decision
        self.agent_decisions.append({
            "date": str(dt.date()), "agent": "pm",
            "action": "rebalance",
            "reasoning": f"top {cfg.top_n}: {', '.join(target_tickers[:5])}...",
            "rankings": [{"ticker": r["ticker"], "mom": f"{r['momentum']*100:+.1f}%"}
                          for r in rankings[:cfg.top_n]],
        })

        # SELL: positions not in target list
        for pos in list(self.open_positions):
            if pos["ticker"] not in target_set:
                if dt in stock_data[pos["ticker"]].index:
                    price = stock_data[pos["ticker"]].loc[dt, "close"]
                    self._close(pos, price, ExitReason.TRAILING_STOP, dt, "rotation_out")

        # BUY: target tickers not already held
        held_tickers = {p["ticker"] for p in self.open_positions}
        if idx + 1 >= len(dates): return
        next_date = dates[idx + 1]

        # Vol targeting: scale total portfolio risk
        spy_vol = 0.15
        if dt in spy_data.index:
            sv = spy_data.loc[dt].get("realized_vol_20", 0.15)
            if not pd.isna(sv) and sv > 0: spy_vol = sv

        vol_scalar = 1.0
        if cfg.vol_targeting:
            vol_scalar = min(cfg.vol_target / spy_vol, 1.5)
            vol_scalar = max(vol_scalar, 0.3)

        # Equal weight among target positions, adjusted by vol
        target_allocation = (1.0 / cfg.top_n) * vol_scalar
        target_allocation = min(target_allocation, cfg.max_position_pct)

        for ticker in target_tickers:
            if ticker in held_tickers: continue
            if len(self.open_positions) >= cfg.max_positions: break

            df = stock_data.get(ticker)
            if df is None or next_date not in df.index: continue

            entry_price = df.loc[next_date, "open"] * (1 + cfg.slippage_pct)
            if entry_price <= 0: continue

            # Position size by allocation target
            target_value = self.equity * target_allocation
            shares = int(target_value / entry_price)
            if shares <= 0: continue

            # Cost check
            cost = shares * entry_price + shares * cfg.commission_per_share
            if cost > self.cash * 0.95:
                shares = int((self.cash * 0.9) / (entry_price + cfg.commission_per_share))
            if shares <= 0: continue

            # Stop price: 15-20% below entry
            stop_price = entry_price * (1 - cfg.hard_stop_pct)

            self.cash -= shares * entry_price + shares * cfg.commission_per_share
            self.open_positions.append({
                "ticker": ticker,
                "entry_date": next_date,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "shares": shares,
                "max_price": entry_price,
                "min_price": entry_price,
                "days_held": 0,
                "signal_type": "momentum",
            })
            self.signals_log.append({
                "date": dt, "ticker": ticker, "type": "momentum",
                "quality": 70,
            })

    def _check_stops(self, dt, stock_data):
        """Wide trailing stop: 15% from peak. No daily micro-stops."""
        cfg = self.config
        to_close = []

        for pos in self.open_positions:
            df = stock_data.get(pos["ticker"])
            if df is None or dt not in df.index: continue

            row = df.loc[dt]
            pos["days_held"] += 1
            pos["max_price"] = max(pos["max_price"], row["high"])
            pos["min_price"] = min(pos["min_price"], row["low"])

            # Trailing stop from peak
            trail_stop = pos["max_price"] * (1 - cfg.trailing_stop_pct)
            effective_stop = max(trail_stop, pos["stop_price"])

            if row["low"] <= effective_stop:
                exit_price = effective_stop * (1 - cfg.slippage_pct)
                to_close.append((pos, exit_price, ExitReason.STOP_LOSS, dt))

            # Also check if stock drops below 200MA (trend break)
            ma200 = row.get("ma_200", 0)
            if not pd.isna(ma200) and ma200 > 0 and row["close"] < ma200:
                # Only exit if we're also losing money
                pnl = (row["close"] - pos["entry_price"]) / pos["entry_price"]
                if pnl < -0.05:  # down more than 5% AND below 200MA
                    to_close.append((pos, row["close"], ExitReason.REGIME_CHANGE, dt))

        closed_tickers = set()
        for pos, price, reason, d in to_close:
            if pos["ticker"] not in closed_tickers and pos in self.open_positions:
                self._close(pos, price, reason, d)
                closed_tickers.add(pos["ticker"])

    def _close(self, pos, exit_price, reason, exit_date, note=""):
        cfg = self.config
        epa = exit_price * (1 - cfg.slippage_pct)
        self.cash += pos["shares"] * epa - pos["shares"] * cfg.commission_per_share
        pnl_d = (epa - pos["entry_price"]) * pos["shares"]
        pnl_p = (epa - pos["entry_price"]) / pos["entry_price"]

        self.closed_trades.append(TradeExit(
            ticker=pos["ticker"],
            entry_date=pos["entry_date"].date() if hasattr(pos["entry_date"], "date") else pos["entry_date"],
            exit_date=exit_date.date() if hasattr(exit_date, "date") else exit_date,
            entry_price=pos["entry_price"], exit_price=epa, stop_price=pos["stop_price"],
            shares=pos["shares"], pnl_dollars=pnl_d, pnl_pct=pnl_p,
            hold_days=pos["days_held"], exit_reason=reason,
            regime_at_entry=Regime.UNKNOWN, regime_at_exit=Regime.UNKNOWN,
            max_favorable=(pos["max_price"] - pos["entry_price"]) / pos["entry_price"],
            max_adverse=(pos["min_price"] - pos["entry_price"]) / pos["entry_price"],
        ))

        self.agent_decisions.append({
            "date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
            "agent": "pm", "ticker": pos["ticker"],
            "action": f"sell_{reason.value}",
            "reasoning": f"pnl={pnl_p*100:+.1f}%, held {pos['days_held']}d, {note}",
        })

        self.open_positions.remove(pos)

    def _close_all(self, dt, stock_data, note=""):
        for pos in list(self.open_positions):
            df = stock_data.get(pos["ticker"])
            if df is not None and dt in df.index:
                self._close(pos, df.loc[dt, "close"], ExitReason.REGIME_CHANGE, dt, note)

    def _update_portfolio(self, dt, stock_data):
        ur = 0.0
        for p in self.open_positions:
            df = stock_data.get(p["ticker"])
            if df is not None and dt in df.index:
                ur += (df.loc[dt, "close"] - p["entry_price"]) * p["shares"]
        self.equity = self.cash + sum(p["shares"] * p["entry_price"] for p in self.open_positions) + ur
        if self.equity > self.peak_equity: self.peak_equity = self.equity

    def _record_equity(self, dt, stock_data):
        dd = (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.equity_curve.append({"date": dt, "equity": self.equity, "cash": self.cash,
                                   "positions": len(self.open_positions), "drawdown": dd})

    def _compute_metrics(self, sd, ed):
        trades = self.closed_trades
        eq = pd.DataFrame(self.equity_curve)
        if not trades:
            return BacktestMetrics(start_date=sd, end_date=ed, initial_capital=self.config.initial_capital,
                final_equity=self.equity, total_return_pct=0, cagr_pct=0, max_drawdown_pct=0,
                avg_drawdown_pct=0, sharpe_ratio=0, sortino_ratio=0, win_rate=0, avg_winner_pct=0,
                avg_loser_pct=0, profit_factor=0, expectancy_per_trade=0, total_trades=0,
                avg_hold_days=0, max_consecutive_losses=0, exposure_pct=0, trades_per_year=0)

        tr = (self.equity - self.config.initial_capital) / self.config.initial_capital
        yrs = max((ed - sd).days / 365.25, 0.1)
        cagr = (1 + tr) ** (1 / yrs) - 1
        mdd = eq["drawdown"].min() if not eq.empty else 0
        add = eq.loc[eq["drawdown"] < 0, "drawdown"].mean() if not eq.empty and (eq["drawdown"] < 0).any() else 0
        pnls = [t.pnl_pct for t in trades]
        w = [p for p in pnls if p > 0]; l = [p for p in pnls if p <= 0]
        wr = len(w) / len(trades)
        aw = np.mean(w) if w else 0; al = np.mean(l) if l else 0
        gp = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
        gl = abs(sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0))
        pf = gp / gl if gl > 0 else float("inf")
        exp = np.mean(pnls)
        if not eq.empty and len(eq) > 1:
            dr = eq["equity"].pct_change().dropna()
            sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
            ds = dr[dr < 0]
            so = dr.mean() / ds.std() * np.sqrt(252) if len(ds) > 0 and ds.std() > 0 else 0
        else: sh = so = 0
        hd = [t.hold_days for t in trades]
        mc = cs = 0
        for p in pnls:
            if p <= 0: cs += 1; mc = max(mc, cs)
            else: cs = 0
        ex = (eq["positions"] > 0).mean() if not eq.empty else 0

        # Transaction cost accounting
        total_commission = sum(t.shares * self.config.commission_per_share * 2 for t in trades)
        total_slippage = sum(abs(t.entry_price * t.shares * self.config.slippage_pct * 2) for t in trades)

        return BacktestMetrics(start_date=sd, end_date=ed, initial_capital=self.config.initial_capital,
            final_equity=round(self.equity, 2), total_return_pct=round(tr*100, 2), cagr_pct=round(cagr*100, 2),
            max_drawdown_pct=round(mdd*100, 2), avg_drawdown_pct=round(add*100, 2) if add else 0,
            sharpe_ratio=round(sh, 2), sortino_ratio=round(so, 2), win_rate=round(wr*100, 2),
            avg_winner_pct=round(aw*100, 2), avg_loser_pct=round(al*100, 2), profit_factor=round(pf, 2),
            expectancy_per_trade=round(exp*100, 2), total_trades=len(trades),
            avg_hold_days=round(np.mean(hd), 1) if hd else 0, max_consecutive_losses=mc,
            exposure_pct=round(ex*100, 2), trades_per_year=round(len(trades)/yrs, 1))

    def get_equity_curve(self): return pd.DataFrame(self.equity_curve)
    def get_trade_log(self):
        if not self.closed_trades: return pd.DataFrame()
        return pd.DataFrame([{
            "ticker": t.ticker, "entry_date": t.entry_date, "exit_date": t.exit_date,
            "entry_price": round(t.entry_price, 2), "exit_price": round(t.exit_price, 2),
            "shares": t.shares, "pnl_dollars": round(t.pnl_dollars, 2),
            "pnl_pct": round(t.pnl_pct * 100, 2), "hold_days": t.hold_days,
            "exit_reason": t.exit_reason.value,
            "max_favorable_pct": round(t.max_favorable * 100, 2),
            "max_adverse_pct": round(t.max_adverse * 100, 2),
        } for t in self.closed_trades])

    def print_summary(self, m):
        total_comm = sum(t.shares * self.config.commission_per_share * 2 for t in self.closed_trades)
        total_slip = sum(abs(t.entry_price * t.shares * self.config.slippage_pct * 2) for t in self.closed_trades)
        print("\n" + "=" * 65)
        print("  BACKTEST V5 — BEAT THE MARKET")
        print("  Monthly Momentum Rotation | Wide Stops | Go-to-Cash Regime")
        print("=" * 65)
        print(f"  Period:          {m.start_date} → {m.end_date}")
        print(f"  Initial Capital: ${m.initial_capital:,.0f}")
        print(f"  Final Equity:    ${m.final_equity:,.0f}")
        print(f"  Total Return:    {m.total_return_pct:+.2f}%")
        print(f"  CAGR:            {m.cagr_pct:+.2f}%")
        print(f"  Max Drawdown:    {m.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio:    {m.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:   {m.sortino_ratio:.2f}")
        print("-" * 65)
        print(f"  Total Trades:    {m.total_trades}")
        print(f"  Win Rate:        {m.win_rate:.1f}%")
        print(f"  Avg Winner:      {m.avg_winner_pct:+.2f}%")
        print(f"  Avg Loser:       {m.avg_loser_pct:.2f}%")
        print(f"  Profit Factor:   {m.profit_factor:.2f}")
        print(f"  Expectancy:      {m.expectancy_per_trade:+.2f}% per trade")
        print("-" * 65)
        print(f"  Avg Hold:        {m.avg_hold_days:.1f} days")
        print(f"  Max Consec Loss: {m.max_consecutive_losses}")
        print(f"  Exposure:        {m.exposure_pct:.1f}%")
        print(f"  Trades/Year:     {m.trades_per_year:.1f}")
        print("-" * 65)
        print(f"  TRANSACTION COSTS (realistic):")
        print(f"    Commissions:   ${total_comm:,.2f}")
        print(f"    Slippage:      ${total_slip:,.2f}")
        print(f"    Total Costs:   ${total_comm + total_slip:,.2f}")
        print(f"    Cost/Trade:    ${(total_comm + total_slip) / max(m.total_trades, 1):,.2f}")
        print(f"  Rebalances:      {self.rebal_count}")
        print(f"  Agent Decisions:  {len(self.agent_decisions)}")
        print("=" * 65)

        if self.closed_trades:
            print(f"\n  Exit Reasons:")
            reasons = defaultdict(int)
            for t in self.closed_trades: reasons[t.exit_reason.value] += 1
            for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"    {r:25s} {c:4d} ({c/len(self.closed_trades)*100:.1f}%)")
        print()
