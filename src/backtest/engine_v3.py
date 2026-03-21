"""
Backtest Engine V3 — Dual Strategy with Quality Ranking.

What changed vs v1/v2:
1. TWO strategies: breakout + mean-reversion pullback
2. Relative strength filter (stock must outperform SPY)
3. Breakout quality scoring (only take top-scored setups)
4. Multi-day breakout confirmation (2 consecutive closes)
5. Pullback-to-MA strategy for uptrending stocks
6. Momentum ranking to prioritize best ideas
7. Adaptive position sizing based on signal quality
"""
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.schemas.signals import BacktestMetrics, ExitReason, Regime, TradeExit

logger = logging.getLogger(__name__)

@dataclass
class V3Config:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    slippage_pct: float = 0.0005
    commission_per_share: float = 0.005
    max_positions: int = 10
    max_position_pct: float = 0.20
    ma_fast: int = 150
    ma_slow: int = 200
    breakout_lookback: int = 20
    min_volume_ratio: float = 1.5
    min_close_range_pct: float = 0.25
    base_max_depth: float = 0.10
    min_breakout_quality: float = 40.0
    require_2day_confirm: bool = True
    min_rs_score: float = 50.0
    min_momentum_score: float = 40.0
    pullback_enabled: bool = True
    stop_atr_mult: float = 0.5
    max_stop_pct: float = 0.08
    failed_breakout_days: int = 4
    trailing_stop_lookback: int = 10
    max_hold_days: int = 60
    scale_risk_by_quality: bool = True

class BacktestEngineV3:
    def __init__(self, config=None):
        self.config = config or V3Config()
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.signals_log = []

    def run(self, stock_data, spy_data):
        dates = spy_data.index.tolist()
        logger.info(f"V3 backtest: {dates[0].date()} to {dates[-1].date()}, {len(stock_data)} tickers")
        for i, dt in enumerate(dates):
            if i < 252:
                continue
            self._update_portfolio(dt, stock_data)
            self._check_exits(dt, stock_data, spy_data)
            if len(self.open_positions) < self.config.max_positions:
                if self._spy_ok(dt, spy_data):
                    self._scan_and_rank(dt, dates, i, stock_data, spy_data)
            self._record_equity(dt, stock_data)
        return self._compute_metrics(dates[0].date(), dates[-1].date())

    def _spy_ok(self, dt, spy):
        if dt not in spy.index: return False
        r = spy.loc[dt]
        return not pd.isna(r.get("ma_200", np.nan)) and r["close"] > r["ma_200"]

    def _scan_and_rank(self, dt, dates, idx, stock_data, spy_data):
        cfg = self.config
        held = {p["ticker"] for p in self.open_positions}
        slots = cfg.max_positions - len(self.open_positions)
        if slots <= 0: return
        candidates = []
        for ticker, df in stock_data.items():
            if ticker in held or dt not in df.index: continue
            if idx + 1 >= len(dates): continue
            nd = dates[idx + 1]
            if nd not in df.index: continue
            row = df.loc[dt]
            nr = df.loc[nd]
            c = self._check_breakout(row, ticker, cfg)
            if c:
                c["next_open"] = nr["open"]; c["next_date"] = nd; c["row"] = row
                candidates.append(c)
            if cfg.pullback_enabled:
                c2 = self._check_pullback(row, ticker, cfg)
                if c2:
                    c2["next_open"] = nr["open"]; c2["next_date"] = nd; c2["row"] = row
                    candidates.append(c2)
        if not candidates: return
        for c in candidates:
            q = c.get("quality", 50); m = c.get("momentum", 50); rs = c.get("rs_score", 50)
            bonus = 10 if c["signal_type"] == "pullback" else 0
            c["rank_score"] = q * 0.4 + m * 0.3 + rs * 0.3 + bonus
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)
        for c in candidates[:slots]:
            self._execute_entry(c, dt)

    def _check_breakout(self, row, ticker, cfg):
        if not row.get("trend_valid", False): return None
        if not row.get("in_base", False): return None
        bd = row.get("base_depth_pct", np.nan)
        if pd.isna(bd) or bd > cfg.base_max_depth: return None
        if cfg.require_2day_confirm:
            if not row.get("breakout_confirmed", False): return None
        else:
            if not row.get("breakout", False): return None
        vr = row.get("vol_ratio", 0)
        if pd.isna(vr) or vr < cfg.min_volume_ratio: return None
        crp = row.get("close_range_pct", 0)
        if pd.isna(crp) or crp < cfg.min_close_range_pct: return None
        if not row.get("atr_compression", False): return None
        rs = row.get("rs_rank_score", 0)
        if pd.isna(rs) or rs < cfg.min_rs_score: return None
        mom = row.get("momentum_score", 0)
        if pd.isna(mom) or mom < cfg.min_momentum_score: return None
        qual = row.get("breakout_quality", 0)
        if pd.isna(qual) or qual < cfg.min_breakout_quality: return None
        atr = row.get("atr", 0)
        if pd.isna(atr) or atr <= 0: return None
        bl = row.get("base_low", row["low"])
        if pd.isna(bl): bl = row["low"]
        return {"ticker": ticker, "signal_type": "breakout", "quality": qual,
                "momentum": mom, "rs_score": rs, "breakout_close": row["close"],
                "base_low": bl, "atr": atr}

    def _check_pullback(self, row, ticker, cfg):
        if not row.get("pullback_signal", False): return None
        if not row.get("trend_close_above_150", False): return None
        if not (row.get("ma_150", 0) > row.get("ma_200", 0)): return None
        rs = row.get("rs_rank_score", 0)
        if pd.isna(rs) or rs < 30: return None
        atr = row.get("atr", 0)
        if pd.isna(atr) or atr <= 0: return None
        mom = row.get("momentum_score", 50)
        return {"ticker": ticker, "signal_type": "pullback", "quality": 60,
                "momentum": mom if not pd.isna(mom) else 50, "rs_score": rs,
                "breakout_close": row["close"],
                "base_low": row.get("ma_50", row["close"] * 0.95), "atr": atr}

    def _execute_entry(self, c, signal_date):
        cfg = self.config
        ep = c["next_open"] * (1 + cfg.slippage_pct)
        atr = c["atr"]
        if c["signal_type"] == "breakout":
            sp = max(c["base_low"] - cfg.stop_atr_mult * atr, ep * (1 - cfg.max_stop_pct))
        else:
            sp = max(c["base_low"] - 0.3 * atr, ep * (1 - 0.05))
        if sp >= ep: return
        psr = ep - sp
        if psr <= 0: return
        rp = cfg.risk_per_trade
        if cfg.scale_risk_by_quality:
            q = c.get("quality", 50)
            if q >= 70: rp *= 1.25
            elif q < 40: rp *= 0.75
        ar = self.equity * rp
        shares = int(ar / psr)
        if shares <= 0: return
        pv = shares * ep
        mx = self.equity * cfg.max_position_pct
        if pv > mx: shares = int(mx / ep)
        if shares <= 0: return
        cost = shares * ep + shares * cfg.commission_per_share
        if cost > self.cash:
            shares = int((self.cash - 100) / (ep + cfg.commission_per_share))
        if shares <= 0: return
        self.cash -= shares * ep + shares * cfg.commission_per_share
        self.open_positions.append({
            "ticker": c["ticker"], "entry_date": c["next_date"], "entry_price": ep,
            "stop_price": sp, "shares": shares, "risk_dollars": shares * psr,
            "breakout_level": c["breakout_close"], "signal_type": c["signal_type"],
            "quality": c.get("quality", 50), "max_price": ep, "min_price": ep, "days_held": 0,
        })
        self.signals_log.append({"date": signal_date, "ticker": c["ticker"],
                                  "type": c["signal_type"], "quality": c.get("quality", 0)})

    def _check_exits(self, dt, stock_data, spy_data):
        cfg = self.config
        to_close = []
        for pos in self.open_positions:
            df = stock_data.get(pos["ticker"])
            if df is None or dt not in df.index: continue
            row = df.loc[dt]
            cp = row["close"]
            pos["days_held"] += 1
            pos["max_price"] = max(pos["max_price"], row["high"])
            pos["min_price"] = min(pos["min_price"], row["low"])
            reason = None
            if row["low"] <= pos["stop_price"]:
                reason = ExitReason.STOP_LOSS
                cp = pos["stop_price"] * (1 - cfg.slippage_pct)
            elif pos["signal_type"] == "breakout" and pos["days_held"] <= cfg.failed_breakout_days and row["close"] < pos["breakout_level"]:
                reason = ExitReason.FAILED_BREAKOUT
            elif pos["days_held"] > cfg.failed_breakout_days:
                loc = df.index.get_loc(dt)
                if loc >= cfg.trailing_stop_lookback:
                    tl = df["low"].iloc[loc - cfg.trailing_stop_lookback:loc].min()
                    if pos["signal_type"] == "pullback":
                        ema = row.get("ema_20", 0)
                        if not pd.isna(ema) and ema > 0: tl = max(tl, ema * 0.98)
                    if row["close"] < tl: reason = ExitReason.TRAILING_STOP
            if pos["signal_type"] == "pullback" and reason is None:
                rps = pos["entry_price"] - pos["stop_price"]
                tgt = pos["entry_price"] + 2.5 * rps
                if row["high"] >= tgt:
                    cp = tgt; reason = ExitReason.TRAILING_STOP
            if pos["days_held"] >= cfg.max_hold_days: reason = ExitReason.MAX_HOLD
            if not self._spy_ok(dt, spy_data):
                ur = (cp - pos["entry_price"]) / pos["entry_price"]
                if ur > 0 or pos["days_held"] > 10: reason = ExitReason.REGIME_CHANGE
            if reason: to_close.append((pos, cp, reason, dt))
        for pos, ep, r, d in to_close:
            self._close(pos, ep, r, d)

    def _close(self, pos, exit_price, reason, exit_date):
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
            shares=pos["shares"], pnl_dollars=pnl_d, pnl_pct=pnl_p, hold_days=pos["days_held"],
            exit_reason=reason, regime_at_entry=Regime.UNKNOWN, regime_at_exit=Regime.UNKNOWN,
            max_favorable=(pos["max_price"] - pos["entry_price"]) / pos["entry_price"],
            max_adverse=(pos["min_price"] - pos["entry_price"]) / pos["entry_price"],
        ))
        self.open_positions.remove(pos)

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
        print("\n" + "=" * 60)
        print("  BACKTEST V3 — DUAL STRATEGY + QUALITY RANKING")
        print("=" * 60)
        print(f"  Period:          {m.start_date} → {m.end_date}")
        print(f"  Initial Capital: ${m.initial_capital:,.0f}")
        print(f"  Final Equity:    ${m.final_equity:,.0f}")
        print(f"  Total Return:    {m.total_return_pct:+.2f}%")
        print(f"  CAGR:            {m.cagr_pct:+.2f}%")
        print(f"  Max Drawdown:    {m.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio:    {m.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:   {m.sortino_ratio:.2f}")
        print("-" * 60)
        print(f"  Total Trades:    {m.total_trades}")
        print(f"  Win Rate:        {m.win_rate:.1f}%")
        print(f"  Avg Winner:      {m.avg_winner_pct:+.2f}%")
        print(f"  Avg Loser:       {m.avg_loser_pct:.2f}%")
        print(f"  Profit Factor:   {m.profit_factor:.2f}")
        print(f"  Expectancy:      {m.expectancy_per_trade:+.2f}% per trade")
        print("-" * 60)
        print(f"  Avg Hold:        {m.avg_hold_days:.1f} days")
        print(f"  Max Consec Loss: {m.max_consecutive_losses}")
        print(f"  Exposure:        {m.exposure_pct:.1f}%")
        print(f"  Trades/Year:     {m.trades_per_year:.1f}")
        print("=" * 60)
        if self.signals_log:
            bo = sum(1 for s in self.signals_log if s["type"] == "breakout")
            pb = sum(1 for s in self.signals_log if s["type"] == "pullback")
            print(f"\n  Signal Breakdown: {bo} breakouts, {pb} pullbacks")
        if self.closed_trades:
            print(f"  Exit Reasons:")
            reasons = {}
            for t in self.closed_trades:
                r = t.exit_reason.value; reasons[r] = reasons.get(r, 0) + 1
            for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"    {r:25s} {c:4d} ({c/len(self.closed_trades)*100:.1f}%)")
        print()
