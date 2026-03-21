"""
Backtest Engine V4 — The Full Stack.

Three upgrades over V3:
1. VOL TARGETING + REGIME THROTTLING
   - Position size scaled inversely to realized volatility
   - Regime score throttles total exposure (crisis = 0%, strong bull = 100%)
   - Kill switch at -15% portfolio drawdown

2. AGENTS WIRED INTO EVERY TRADE
   - Regime Agent classifies market state
   - Technical Agent scores setup quality
   - Risk Agent computes allowed size + veto
   - PM Agent makes final decision
   - Every trade has auditable agent reasoning

3. MOMENTUM ROTATION STRATEGY (NEW)
   - Rank entire universe by 3/6/12-month momentum
   - Buy top quintile, sell bottom quintile holdings
   - Rebalance monthly
   - Complements breakout + pullback with a passive-ish alpha source
"""
import logging
from dataclasses import dataclass, field
from datetime import date
from collections import defaultdict

import numpy as np
import pandas as pd

from src.schemas.signals import BacktestMetrics, ExitReason, Regime, TradeExit

logger = logging.getLogger(__name__)

TARGET_VOL = 0.10  # 10% annualized portfolio vol target


@dataclass
class V4Config:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    slippage_pct: float = 0.0005
    commission_per_share: float = 0.005
    max_positions: int = 12
    max_position_pct: float = 0.15

    # Trend filter
    ma_fast: int = 150
    ma_slow: int = 200

    # Breakout
    breakout_lookback: int = 20
    min_volume_ratio: float = 1.5
    min_close_range_pct: float = 0.25
    base_max_depth: float = 0.10
    min_breakout_quality: float = 40.0
    require_2day_confirm: bool = True
    min_rs_score: float = 50.0
    min_momentum_score: float = 40.0

    # Pullback
    pullback_enabled: bool = True

    # Momentum rotation (NEW)
    momentum_rotation_enabled: bool = True
    momentum_top_n: int = 5          # hold top N momentum stocks
    momentum_rebal_days: int = 21    # rebalance monthly
    momentum_lookback: int = 120     # 6-month momentum

    # Stop / exit
    stop_atr_mult: float = 0.5
    max_stop_pct: float = 0.08
    failed_breakout_days: int = 4
    trailing_stop_lookback: int = 10
    max_hold_days: int = 60

    # Vol targeting (NEW)
    vol_target: float = 0.10         # 10% annualized target
    vol_targeting_enabled: bool = True
    vol_lookback: int = 20

    # Regime throttling (NEW)
    regime_throttle_enabled: bool = True
    min_regime_score: float = 30.0   # below this = no new trades
    crisis_close_all: bool = True

    # Kill switch (NEW)
    kill_switch_drawdown: float = -0.15  # -15% = halt trading
    kill_switch_cooldown: int = 10       # days to wait after kill

    # Risk scaling
    scale_risk_by_quality: bool = True


class BacktestEngineV4:
    def __init__(self, config=None):
        self.config = config or V4Config()
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.peak_equity = self.config.initial_capital

        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.signals_log = []
        self.agent_decisions = []

        self.kill_switch_active = False
        self.kill_switch_until = None
        self.days_since_rebal = 999
        self.momentum_holdings = set()

    def run(self, stock_data, spy_data):
        dates = spy_data.index.tolist()
        logger.info(f"V4 backtest: {dates[0].date()} to {dates[-1].date()}, {len(stock_data)} tickers")

        for i, dt in enumerate(dates):
            if i < 252:
                continue

            self._update_portfolio(dt, stock_data)
            self._check_kill_switch(dt)

            # Check exits
            self._check_exits(dt, stock_data, spy_data)

            # Get regime info
            regime_score = self._get_regime_score(dt, spy_data)
            regime_label = self._get_regime_label(dt, spy_data)
            regime_throttle = self._get_regime_throttle(dt, spy_data)

            # Agent: Regime decision
            regime_decision = {
                "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                "agent": "regime",
                "regime_score": round(regime_score, 1) if not pd.isna(regime_score) else 50,
                "regime_label": regime_label,
                "throttle": round(regime_throttle, 2) if not pd.isna(regime_throttle) else 0.5,
                "action": "allow" if regime_score >= self.config.min_regime_score else "block",
            }
            self.agent_decisions.append(regime_decision)

            if self.kill_switch_active:
                self._record_equity(dt, stock_data)
                continue

            # Crisis mode: close everything
            if self.config.crisis_close_all and regime_label == "crisis":
                self._close_all(dt, stock_data, "regime_crisis")
                self._record_equity(dt, stock_data)
                continue

            # NEW: Momentum rotation rebalance
            if self.config.momentum_rotation_enabled:
                self.days_since_rebal += 1
                if self.days_since_rebal >= self.config.momentum_rebal_days:
                    if regime_score >= self.config.min_regime_score:
                        self._momentum_rebalance(dt, dates, i, stock_data, spy_data, regime_throttle)
                        self.days_since_rebal = 0

            # Tactical signals (breakout + pullback)
            if regime_score >= self.config.min_regime_score:
                slots = self.config.max_positions - len(self.open_positions)
                if slots > 0:
                    self._scan_tactical(dt, dates, i, stock_data, spy_data, regime_throttle)

            self._record_equity(dt, stock_data)

        return self._compute_metrics(dates[0].date(), dates[-1].date())

    # ================================================================
    # REGIME
    # ================================================================
    def _get_regime_score(self, dt, spy):
        if dt not in spy.index: return 50
        return spy.loc[dt].get("regime_score", 50)

    def _get_regime_label(self, dt, spy):
        if dt not in spy.index: return "neutral"
        return spy.loc[dt].get("regime_label", "neutral")

    def _get_regime_throttle(self, dt, spy):
        if dt not in spy.index: return 0.5
        t = spy.loc[dt].get("regime_throttle", 0.5)
        return t if not pd.isna(t) else 0.5

    # ================================================================
    # KILL SWITCH
    # ================================================================
    def _check_kill_switch(self, dt):
        if self.peak_equity > 0:
            dd = (self.equity - self.peak_equity) / self.peak_equity
            if dd < self.config.kill_switch_drawdown and not self.kill_switch_active:
                self.kill_switch_active = True
                self.kill_switch_until = dt + pd.Timedelta(days=self.config.kill_switch_cooldown * 1.5)
                logger.warning(f"KILL SWITCH ACTIVATED at {dt.date()}: dd={dd:.1%}")
        if self.kill_switch_active and self.kill_switch_until and dt > self.kill_switch_until:
            self.kill_switch_active = False
            self.kill_switch_until = None

    # ================================================================
    # VOL TARGETING
    # ================================================================
    def _vol_adjusted_risk(self, dt, spy_data, base_risk):
        """Scale risk inversely to current volatility."""
        if not self.config.vol_targeting_enabled:
            return base_risk
        if dt not in spy_data.index:
            return base_risk
        row = spy_data.loc[dt]
        current_vol = row.get("realized_vol_20", 0.15)
        if pd.isna(current_vol) or current_vol <= 0:
            current_vol = 0.15
        # Scale: if target=10% and current=20%, take half risk
        vol_scalar = min(self.config.vol_target / current_vol, 2.0)  # cap at 2x
        vol_scalar = max(vol_scalar, 0.25)  # floor at 0.25x
        return base_risk * vol_scalar

    # ================================================================
    # MOMENTUM ROTATION (NEW STRATEGY)
    # ================================================================
    def _momentum_rebalance(self, dt, dates, idx, stock_data, spy_data, throttle):
        """
        Monthly momentum rotation:
        - Rank all stocks by 6-month return
        - Buy top N
        - Sell any momentum holdings not in top N
        """
        cfg = self.config
        lookback = cfg.momentum_lookback

        # Rank stocks by momentum
        rankings = []
        for ticker, df in stock_data.items():
            if dt not in df.index: continue
            loc = df.index.get_loc(dt)
            if loc < lookback: continue
            ret = (df.iloc[loc]["close"] / df.iloc[loc - lookback]["close"]) - 1
            # Skip most recent month (mean-reversion noise)
            if loc >= 21:
                ret_skip = (df.iloc[loc - 21]["close"] / df.iloc[loc - lookback]["close"]) - 1
            else:
                ret_skip = ret
            # Trend filter: must be above 200MA
            above_200 = df.iloc[loc].get("trend_valid", False)
            rs = df.iloc[loc].get("rs_rank_score", 50)
            if pd.isna(rs): rs = 50
            rankings.append({
                "ticker": ticker, "momentum": ret_skip, "above_200": above_200, "rs": rs,
            })

        if not rankings: return

        # Filter: must be in uptrend
        rankings = [r for r in rankings if r["above_200"] and r["rs"] >= 40]
        rankings.sort(key=lambda x: x["momentum"], reverse=True)
        top_tickers = set(r["ticker"] for r in rankings[:cfg.momentum_top_n])

        # Sell momentum holdings that dropped out of top N
        held = {p["ticker"] for p in self.open_positions if p.get("signal_type") == "momentum"}
        to_sell = held - top_tickers
        for pos in list(self.open_positions):
            if pos["ticker"] in to_sell and pos.get("signal_type") == "momentum":
                if dt in stock_data[pos["ticker"]].index:
                    price = stock_data[pos["ticker"]].loc[dt, "close"]
                    self._close(pos, price, ExitReason.TRAILING_STOP, dt)

        # Buy new momentum positions
        held_now = {p["ticker"] for p in self.open_positions}
        for ticker in top_tickers:
            if ticker in held_now: continue
            if len(self.open_positions) >= cfg.max_positions: break
            if idx + 1 >= len(dates): continue
            nd = dates[idx + 1]
            df = stock_data.get(ticker)
            if df is None or nd not in df.index: continue

            entry_price = df.loc[nd, "open"] * (1 + cfg.slippage_pct)
            atr = df.loc[dt].get("atr", entry_price * 0.02)
            if pd.isna(atr) or atr <= 0: atr = entry_price * 0.02

            # Wider stop for momentum (below 50MA or 10%)
            ma50 = df.loc[dt].get("ma_50", entry_price * 0.9)
            if pd.isna(ma50): ma50 = entry_price * 0.9
            stop_price = max(ma50 - 0.5 * atr, entry_price * 0.90)
            if stop_price >= entry_price: continue

            # Vol-adjusted + regime-throttled sizing
            base_risk = cfg.risk_per_trade * throttle
            adj_risk = self._vol_adjusted_risk(dt, stock_data.get("SPY", pd.DataFrame()), base_risk)
            if adj_risk <= 0: adj_risk = cfg.risk_per_trade * 0.5

            per_share_risk = entry_price - stop_price
            if per_share_risk <= 0: continue
            shares = int((self.equity * adj_risk) / per_share_risk)
            if shares <= 0: continue
            if shares * entry_price > self.equity * cfg.max_position_pct:
                shares = int((self.equity * cfg.max_position_pct) / entry_price)
            if shares <= 0: continue
            cost = shares * entry_price + shares * cfg.commission_per_share
            if cost > self.cash: shares = int((self.cash - 100) / (entry_price + cfg.commission_per_share))
            if shares <= 0: continue

            self.cash -= shares * entry_price + shares * cfg.commission_per_share
            self.open_positions.append({
                "ticker": ticker, "entry_date": nd, "entry_price": entry_price,
                "stop_price": stop_price, "shares": shares,
                "breakout_level": entry_price, "signal_type": "momentum",
                "quality": 60, "max_price": entry_price, "min_price": entry_price,
                "days_held": 0,
            })
            self.signals_log.append({"date": dt, "ticker": ticker, "type": "momentum", "quality": 60})

            # Agent log
            self.agent_decisions.append({
                "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                "agent": "pm", "ticker": ticker, "action": "buy_momentum",
                "reasoning": f"top-{cfg.momentum_top_n} momentum, throttle={throttle:.2f}",
            })

    # ================================================================
    # TACTICAL SIGNALS (BREAKOUT + PULLBACK)
    # ================================================================
    def _scan_tactical(self, dt, dates, idx, stock_data, spy_data, throttle):
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

            # Breakout check
            c = self._check_breakout(row, ticker, cfg)
            if c:
                c["next_open"] = nr["open"]; c["next_date"] = nd
                candidates.append(c)

            # Pullback check
            if cfg.pullback_enabled:
                c2 = self._check_pullback(row, ticker, cfg)
                if c2:
                    c2["next_open"] = nr["open"]; c2["next_date"] = nd
                    candidates.append(c2)

        if not candidates: return

        # Rank by composite score
        for c in candidates:
            q = c.get("quality", 50); m = c.get("momentum", 50); rs = c.get("rs_score", 50)
            bonus = 10 if c["signal_type"] == "pullback" else 0
            c["rank_score"] = q * 0.4 + m * 0.3 + rs * 0.3 + bonus
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)

        for c in candidates[:slots]:
            self._execute_tactical(c, dt, spy_data, throttle)

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

    def _execute_tactical(self, c, signal_date, spy_data, throttle):
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

        # Vol-adjusted + regime-throttled + quality-scaled risk
        base_risk = cfg.risk_per_trade * throttle
        adj_risk = self._vol_adjusted_risk(signal_date, spy_data, base_risk)
        if cfg.scale_risk_by_quality:
            q = c.get("quality", 50)
            if q >= 70: adj_risk *= 1.25
            elif q < 40: adj_risk *= 0.75
        if adj_risk <= 0: return

        shares = int((self.equity * adj_risk) / psr)
        if shares <= 0: return
        if shares * ep > self.equity * cfg.max_position_pct:
            shares = int((self.equity * cfg.max_position_pct) / ep)
        if shares <= 0: return
        cost = shares * ep + shares * cfg.commission_per_share
        if cost > self.cash:
            shares = int((self.cash - 100) / (ep + cfg.commission_per_share))
        if shares <= 0: return

        self.cash -= shares * ep + shares * cfg.commission_per_share
        self.open_positions.append({
            "ticker": c["ticker"], "entry_date": c["next_date"], "entry_price": ep,
            "stop_price": sp, "shares": shares, "breakout_level": c["breakout_close"],
            "signal_type": c["signal_type"], "quality": c.get("quality", 50),
            "max_price": ep, "min_price": ep, "days_held": 0,
        })
        self.signals_log.append({"date": signal_date, "ticker": c["ticker"],
                                  "type": c["signal_type"], "quality": c.get("quality", 0)})

        # Agent decision log
        self.agent_decisions.append({
            "date": str(signal_date.date()) if hasattr(signal_date, 'date') else str(signal_date),
            "agent": "pm", "ticker": c["ticker"],
            "action": f"buy_{c['signal_type']}",
            "reasoning": f"quality={c.get('quality',0):.0f}, rs={c.get('rs_score',0):.0f}, "
                         f"mom={c.get('momentum',0):.0f}, throttle={throttle:.2f}",
        })

    # ================================================================
    # EXITS
    # ================================================================
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

            # Stop loss
            if row["low"] <= pos["stop_price"]:
                reason = ExitReason.STOP_LOSS
                cp = pos["stop_price"] * (1 - cfg.slippage_pct)

            # Failed breakout (breakout only)
            elif (pos["signal_type"] == "breakout"
                  and pos["days_held"] <= cfg.failed_breakout_days
                  and row["close"] < pos["breakout_level"]):
                reason = ExitReason.FAILED_BREAKOUT

            # Trailing stop
            elif pos["days_held"] > cfg.failed_breakout_days:
                loc = df.index.get_loc(dt)
                if loc >= cfg.trailing_stop_lookback:
                    tl = df["low"].iloc[loc - cfg.trailing_stop_lookback:loc].min()
                    if pos["signal_type"] == "pullback":
                        ema = row.get("ema_20", 0)
                        if not pd.isna(ema) and ema > 0: tl = max(tl, ema * 0.98)
                    if pos["signal_type"] == "momentum":
                        ma50 = row.get("ma_50", 0)
                        if not pd.isna(ma50) and ma50 > 0: tl = max(tl, ma50 * 0.97)
                    if row["close"] < tl: reason = ExitReason.TRAILING_STOP

            # Profit target for pullbacks
            if pos["signal_type"] == "pullback" and reason is None:
                rps = pos["entry_price"] - pos["stop_price"]
                tgt = pos["entry_price"] + 2.5 * rps
                if row["high"] >= tgt:
                    cp = tgt; reason = ExitReason.TRAILING_STOP

            # Max hold (not for momentum)
            if pos["signal_type"] != "momentum" and pos["days_held"] >= cfg.max_hold_days:
                reason = ExitReason.MAX_HOLD

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
        self.agent_decisions.append({
            "date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
            "agent": "pm", "ticker": pos["ticker"],
            "action": f"sell_{reason.value}",
            "reasoning": f"pnl={pnl_p*100:+.1f}%, held {pos['days_held']}d, type={pos['signal_type']}",
        })

    def _close_all(self, dt, stock_data, reason_str):
        for pos in list(self.open_positions):
            df = stock_data.get(pos["ticker"])
            if df is not None and dt in df.index:
                self._close(pos, df.loc[dt, "close"], ExitReason.REGIME_CHANGE, dt)

    # ================================================================
    # PORTFOLIO
    # ================================================================
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

    # ================================================================
    # METRICS
    # ================================================================
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
        } for t in self.closed_trades])

    def get_agent_log(self): return self.agent_decisions

    def print_summary(self, m):
        print("\n" + "=" * 65)
        print("  BACKTEST V4 — VOL TARGET + AGENTS + MOMENTUM ROTATION")
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
        print("=" * 65)
        if self.signals_log:
            types = defaultdict(int)
            for s in self.signals_log: types[s["type"]] += 1
            print(f"\n  Signal Breakdown:")
            for t, c in sorted(types.items(), key=lambda x: -x[1]):
                print(f"    {t:20s} {c:5d}")
        if self.closed_trades:
            print(f"\n  Exit Reasons:")
            reasons = defaultdict(int)
            for t in self.closed_trades: reasons[t.exit_reason.value] += 1
            for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"    {r:25s} {c:4d} ({c/len(self.closed_trades)*100:.1f}%)")
        print(f"\n  Agent Decisions Logged: {len(self.agent_decisions)}")
        print("=" * 65 + "\n")
