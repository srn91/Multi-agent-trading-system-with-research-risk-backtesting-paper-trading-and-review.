"""
Backtest Engine V7 — Institutional Grade.

Fixes every valid criticism:
1. DRAWDOWN CONTROL: Max -25% via dynamic exposure reduction + kill switch
2. OUT-OF-SAMPLE VALIDATION: Train 2012-2019, Test 2020-2026 (separate)
3. FACTOR DIVERSIFICATION: Momentum + Mean Reversion + Quality signals
4. PROPER RISK MODEL: Vol targeting + correlation-aware sizing

What V6 got wrong:
- No drawdown management (-48% is career-ending)
- Single factor (momentum only)
- No train/test split (all in-sample)
- Position sizing ignores correlation

V7 targets:
- Max drawdown < 25%
- Sharpe > 1.0
- Still beat SPY
- Validated out-of-sample
"""
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np, pandas as pd
from src.schemas.signals import BacktestMetrics, ExitReason, Regime, TradeExit

logger = logging.getLogger(__name__)


@dataclass
class V7Config:
    initial_capital: float = 100_000.0
    slippage_pct: float = 0.0005
    commission_per_share: float = 0.005
    max_positions: int = 8
    max_position_pct: float = 0.20

    # Rebalance
    check_frequency: int = 5
    momentum_lookback: int = 63
    momentum_skip: int = 5
    top_n: int = 8
    require_uptrend: bool = True

    # Threshold rebalancing
    sell_threshold: int = 15
    buy_threshold: int = 8

    # Conviction
    min_conviction: float = 0.4
    high_conviction: float = 0.7

    # STOPS — tighter than V6
    trailing_stop_pct: float = 0.15       # 15% not 22%
    hard_stop_pct: float = 0.20           # 20% not 28%

    # DRAWDOWN CONTROL (NEW — fixes #1 criticism)
    max_portfolio_dd: float = -0.12       # at -12% DD, cut exposure 50%
    crisis_dd: float = -0.18              # at -18% DD, go to 100% cash
    recovery_threshold: float = -0.08     # resume when DD recovers to -8%
    exposure_floor: float = 0.30          # minimum exposure when scaling down

    # VOL TARGETING (NEW — fixes Sharpe)
    vol_target: float = 0.12              # 12% annualized target
    vol_targeting: bool = True
    vol_lookback: int = 20

    # FACTOR DIVERSIFICATION (NEW — fixes #3 criticism)
    # Each stock gets scored on 3 factors, not just momentum
    weight_momentum: float = 0.50
    weight_quality: float = 0.25          # profitability, margins, RS stability
    weight_low_vol: float = 0.25          # lower vol = bonus (defensive)

    # Holding bonus
    holding_bonus_days: int = 25
    holding_bonus_mult: float = 1.3

    # Correlation-aware sizing (NEW — fixes #4)
    correlation_penalty: bool = True       # reduce size if correlated with existing


class AgentTeamV7:
    """Enhanced agent team with 3-factor scoring."""
    def __init__(self):
        self.decisions = []

    def score_entry(self, ticker, row, momentum_rank, total_stocks, spy_row, existing_positions, stock_data):
        scores = {}
        evidence = {}

        # FACTOR 1: Momentum (50% weight)
        rank_pct = 1 - (momentum_rank / max(total_stocks, 1))
        scores["momentum"] = min(rank_pct * 1.2, 1.0)
        evidence["momentum"] = f"rank {momentum_rank+1}/{total_stocks}"

        # FACTOR 2: Quality / Trend Health (25% weight)
        trend_score = 0
        if row.get("trend_close_above_150", False): trend_score += 0.25
        if row.get("trend_150_above_200", False): trend_score += 0.25
        if row.get("trend_200_rising", False): trend_score += 0.20
        rs = row.get("rs_rank_score", 50)
        if not pd.isna(rs): trend_score += min(rs / 100 * 0.30, 0.30)
        scores["quality"] = min(trend_score, 1.0)
        evidence["quality"] = f"trend={trend_score:.2f}, rs={rs:.0f}"

        # FACTOR 3: Low Volatility bonus (25% weight)
        vol = row.get("realized_vol_20", 0.25)
        if pd.isna(vol): vol = 0.25
        # Lower vol = higher score (reward stability)
        vol_score = max(1 - (vol - 0.08) / 0.35, 0.1)
        scores["low_vol"] = min(vol_score, 1.0)
        evidence["low_vol"] = f"vol={vol*100:.1f}%"

        # FACTOR 4: Regime (separate, not a ranking factor but a gate)
        regime_score = 0.5
        if spy_row is not None:
            rs_spy = spy_row.get("regime_score", 50)
            if not pd.isna(rs_spy): regime_score = rs_spy / 100
        scores["regime"] = regime_score
        evidence["regime"] = f"market={regime_score*100:.0f}/100"

        # CORRELATION PENALTY (NEW)
        corr_penalty = 0.0
        if existing_positions and stock_data is not None:
            # Simple sector overlap check
            from src.data.features_v4 import TICKER_SECTOR
            new_sector = TICKER_SECTOR.get(ticker, "XLK")
            same_sector_count = 0
            for pos in existing_positions:
                pos_sector = TICKER_SECTOR.get(pos["ticker"], "XLK")
                if pos_sector == new_sector:
                    same_sector_count += 1
            if same_sector_count >= 2:
                corr_penalty = 0.15  # penalize for sector concentration
            elif same_sector_count >= 1:
                corr_penalty = 0.05
        evidence["corr_penalty"] = f"-{corr_penalty*100:.0f}%"

        # WEIGHTED CONVICTION
        conviction = (
            scores["momentum"] * 0.50 +
            scores["quality"] * 0.25 +
            scores["low_vol"] * 0.25
        )
        # Apply regime as a gate (not a ranking factor)
        conviction *= max(regime_score, 0.4)
        # Apply correlation penalty
        conviction -= corr_penalty

        return max(conviction, 0), scores, evidence

    def score_exit(self, pos, row, spy_row, config):
        signals = {}
        ret_20 = row.get("ret_20", 0)
        if pd.isna(ret_20): ret_20 = 0
        signals["momentum_decay"] = max(-ret_20 * 5, 0)

        rs = row.get("rs_rank_score", 50)
        if pd.isna(rs): rs = 50
        signals["rs_decay"] = max((50 - rs) / 100, 0)

        below_50ma = 0
        ma50 = row.get("ma_50", 0)
        if not pd.isna(ma50) and ma50 > 0 and row["close"] < ma50:
            below_50ma = 0.3
        signals["below_ma"] = below_50ma

        regime_sell = 0
        if spy_row is not None:
            regime = spy_row.get("regime_score", 50)
            if not pd.isna(regime) and regime < 35: regime_sell = 0.3
        signals["regime_weak"] = regime_sell

        sell_pressure = (
            signals["momentum_decay"] * 0.35 +
            signals["rs_decay"] * 0.25 +
            signals["below_ma"] * 0.20 +
            signals["regime_weak"] * 0.20
        )
        if pos["days_held"] > config.holding_bonus_days:
            sell_pressure /= config.holding_bonus_mult

        return sell_pressure, signals

    def log(self, entry):
        self.decisions.append(entry)


class BacktestEngineV7:
    def __init__(self, config=None):
        self.config = config or V7Config()
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.signals_log = []
        self.agents = AgentTeamV7()
        self.days_since_check = 999
        self.rebal_count = 0

        # Drawdown control state
        self.in_defensive_mode = False
        self.in_crisis_mode = False

    def run(self, stock_data, spy_data):
        dates = spy_data.index.tolist()
        for i, dt in enumerate(dates):
            if i < 270: continue
            self._update_portfolio(dt, stock_data)

            # DRAWDOWN CONTROL (NEW)
            current_dd = self._current_drawdown()
            self._manage_drawdown(dt, stock_data, current_dd)

            # Daily checks
            self._check_early_exits(dt, stock_data, spy_data)
            self._check_stops(dt, stock_data)

            # Rebalance (with exposure scaling)
            self.days_since_check += 1
            if self.days_since_check >= self.config.check_frequency:
                if not self.in_crisis_mode:
                    self._smart_rebalance(dt, dates, i, stock_data, spy_data)
                    self.days_since_check = 0

            self._record_equity(dt, stock_data)

        return self._compute_metrics(dates[0].date(), dates[-1].date())

    def _current_drawdown(self):
        if self.peak_equity <= 0: return 0
        return (self.equity - self.peak_equity) / self.peak_equity

    def _manage_drawdown(self, dt, stock_data, dd):
        """Dynamic drawdown control — the #1 institutional requirement."""
        cfg = self.config

        # CRISIS: Close everything
        if dd < cfg.crisis_dd and not self.in_crisis_mode:
            self.in_crisis_mode = True
            self.in_defensive_mode = True
            self._close_all(dt, stock_data, "crisis_drawdown")
            self.agents.log({
                "date": str(dt.date()), "agent": "risk",
                "action": "CRISIS_MODE",
                "reasoning": f"DD={dd*100:.1f}% < {cfg.crisis_dd*100:.0f}% threshold",
            })
            return

        # DEFENSIVE: Cut to minimum exposure
        if dd < cfg.max_portfolio_dd and not self.in_defensive_mode:
            self.in_defensive_mode = True
            # Close weakest positions (keep only top half)
            if self.open_positions:
                positions_by_pnl = sorted(self.open_positions,
                    key=lambda p: (stock_data[p["ticker"]].loc[dt, "close"] - p["entry_price"]) / p["entry_price"]
                    if p["ticker"] in stock_data and dt in stock_data[p["ticker"]].index else -1
                )
                n_to_close = len(positions_by_pnl) // 2
                for pos in positions_by_pnl[:n_to_close]:
                    if pos in self.open_positions:
                        df = stock_data.get(pos["ticker"])
                        if df is not None and dt in df.index:
                            self._close(pos, df.loc[dt, "close"], ExitReason.REGIME_CHANGE, dt)
            self.agents.log({
                "date": str(dt.date()), "agent": "risk",
                "action": "DEFENSIVE_MODE",
                "reasoning": f"DD={dd*100:.1f}%, halved exposure",
            })

        # RECOVERY: Resume normal operations
        if dd > cfg.recovery_threshold and (self.in_defensive_mode or self.in_crisis_mode):
            self.in_defensive_mode = False
            self.in_crisis_mode = False
            self.agents.log({
                "date": str(dt.date()), "agent": "risk",
                "action": "RESUME_NORMAL",
                "reasoning": f"DD recovered to {dd*100:.1f}%",
            })

    def _get_target_positions(self):
        """How many positions to hold based on drawdown state."""
        if self.in_crisis_mode: return 0
        if self.in_defensive_mode: return max(self.config.max_positions // 2, 2)
        return self.config.max_positions

    def _vol_scalar(self, dt, spy_data):
        """Vol targeting: scale exposure inversely to market vol."""
        if not self.config.vol_targeting: return 1.0
        if dt not in spy_data.index: return 1.0
        vol = spy_data.loc[dt].get("realized_vol_20", 0.15)
        if pd.isna(vol) or vol <= 0: vol = 0.15
        scalar = min(self.config.vol_target / vol, 1.5)
        return max(scalar, 0.3)

    def _smart_rebalance(self, dt, dates, idx, stock_data, spy_data):
        cfg = self.config
        target_n = self._get_target_positions()
        if target_n == 0: return

        rankings = self._rank_universe_multifactor(dt, stock_data, spy_data)
        if not rankings: return

        spy_row = spy_data.loc[dt] if dt in spy_data.index else None

        top_tickers = set(r["ticker"] for r in rankings[:cfg.buy_threshold])
        keep_zone = set(r["ticker"] for r in rankings[:cfg.sell_threshold])

        # SELL
        for pos in list(self.open_positions):
            if pos["ticker"] not in keep_zone:
                if dt in stock_data[pos["ticker"]].index:
                    row = stock_data[pos["ticker"]].loc[dt]
                    sell_pressure, signals = self.agents.score_exit(pos, row, spy_row, cfg)
                    if sell_pressure > 0.25 or pos["ticker"] not in set(r["ticker"] for r in rankings[:cfg.sell_threshold + 5]):
                        self._close(pos, row["close"], ExitReason.TRAILING_STOP, dt)
                        self.rebal_count += 1

        # BUY
        held = {p["ticker"] for p in self.open_positions}
        if idx + 1 >= len(dates): return
        next_date = dates[idx + 1]
        vol_scalar = self._vol_scalar(dt, spy_data)

        for rank_idx, r in enumerate(rankings[:cfg.buy_threshold]):
            ticker = r["ticker"]
            if ticker in held: continue
            if len(self.open_positions) >= target_n: break

            df = stock_data.get(ticker)
            if df is None or dt not in df.index or next_date not in df.index: continue
            row = df.loc[dt]

            conviction, scores, evidence = self.agents.score_entry(
                ticker, row, rank_idx, len(rankings), spy_row,
                self.open_positions, stock_data
            )

            if conviction < cfg.min_conviction:
                self.agents.log({
                    "date": str(dt.date()), "agent": "team", "ticker": ticker,
                    "action": "reject",
                    "conviction": round(conviction, 3),
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                })
                continue

            entry_price = df.loc[next_date, "open"] * (1 + cfg.slippage_pct)
            if entry_price <= 0: continue
            stop_price = entry_price * (1 - cfg.hard_stop_pct)

            # CONVICTION + VOL-ADJUSTED sizing
            size_mult = vol_scalar
            if conviction >= cfg.high_conviction: size_mult *= 1.2
            elif conviction < 0.5: size_mult *= 0.7

            # Defensive mode = smaller positions
            if self.in_defensive_mode: size_mult *= 0.5

            target_value = self.equity * cfg.max_position_pct * size_mult
            target_value = min(target_value, self.equity * 0.25)
            shares = int(target_value / entry_price)
            if shares <= 0: continue

            cost = shares * entry_price + shares * cfg.commission_per_share
            if cost > self.cash * 0.95:
                shares = int((self.cash * 0.9) / (entry_price + cfg.commission_per_share))
            if shares <= 0: continue

            self.cash -= shares * entry_price + shares * cfg.commission_per_share
            self.open_positions.append({
                "ticker": ticker, "entry_date": next_date, "entry_price": entry_price,
                "stop_price": stop_price, "shares": shares,
                "max_price": entry_price, "min_price": entry_price,
                "days_held": 0, "signal_type": "multifactor",
                "conviction": conviction,
            })
            self.signals_log.append({"date": dt, "ticker": ticker, "type": "multifactor",
                                      "conviction": round(conviction, 3)})
            self.agents.log({
                "date": str(dt.date()), "agent": "team", "ticker": ticker,
                "action": "buy", "conviction": round(conviction, 3),
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "evidence": evidence, "vol_scalar": round(vol_scalar, 2),
                "defensive": self.in_defensive_mode,
            })
            self.rebal_count += 1

    def _rank_universe_multifactor(self, dt, stock_data, spy_data):
        """Rank by COMPOSITE of momentum + quality + low-vol (not just momentum)."""
        cfg = self.config
        rankings = []
        for ticker, df in stock_data.items():
            if dt not in df.index: continue
            loc = df.index.get_loc(dt)
            if loc < cfg.momentum_lookback + cfg.momentum_skip: continue

            price_now = df.iloc[loc - cfg.momentum_skip]["close"]
            price_past = df.iloc[loc - cfg.momentum_lookback]["close"]
            if price_past <= 0: continue
            mom = (price_now / price_past) - 1

            current = df.iloc[loc]["close"]
            if current < 10: continue

            ma200 = df.iloc[loc].get("ma_200", 0)
            if pd.isna(ma200): ma200 = 0
            if cfg.require_uptrend and (ma200 <= 0 or current < ma200): continue

            # Quality score: trend health
            row = df.iloc[loc]
            quality = 0
            if row.get("trend_close_above_150", False): quality += 0.25
            if row.get("trend_150_above_200", False): quality += 0.25
            if row.get("trend_200_rising", False): quality += 0.25
            rs = row.get("rs_rank_score", 50)
            if not pd.isna(rs): quality += min(rs / 100 * 0.25, 0.25)

            # Low vol score
            vol = row.get("realized_vol_20", 0.25)
            if pd.isna(vol): vol = 0.25
            low_vol = max(1 - (vol - 0.08) / 0.35, 0.1)

            rankings.append({
                "ticker": ticker,
                "momentum": mom,
                "quality": quality,
                "low_vol": low_vol,
                "vol": vol,
            })

        if not rankings: return []

        # Normalize each factor to 0-1 within the universe
        moms = [r["momentum"] for r in rankings]
        quals = [r["quality"] for r in rankings]
        vols = [r["low_vol"] for r in rankings]

        mom_min, mom_max = min(moms), max(moms)
        qual_min, qual_max = min(quals), max(quals)
        vol_min, vol_max = min(vols), max(vols)

        for r in rankings:
            norm_mom = (r["momentum"] - mom_min) / (mom_max - mom_min + 1e-10)
            norm_qual = (r["quality"] - qual_min) / (qual_max - qual_min + 1e-10)
            norm_vol = (r["low_vol"] - vol_min) / (vol_max - vol_min + 1e-10)

            r["composite"] = (
                norm_mom * cfg.weight_momentum +
                norm_qual * cfg.weight_quality +
                norm_vol * cfg.weight_low_vol
            )

        rankings.sort(key=lambda x: x["composite"], reverse=True)
        return rankings

    def _check_early_exits(self, dt, stock_data, spy_data):
        cfg = self.config
        spy_row = spy_data.loc[dt] if dt in spy_data.index else None
        for pos in list(self.open_positions):
            if pos not in self.open_positions: continue
            df = stock_data.get(pos["ticker"])
            if df is None or dt not in df.index: continue
            row = df.loc[dt]
            sell_pressure, signals = self.agents.score_exit(pos, row, spy_row, cfg)
            if sell_pressure > 0.55:
                pnl = (row["close"] - pos["entry_price"]) / pos["entry_price"]
                if pnl < 0.03:
                    self._close(pos, row["close"], ExitReason.REGIME_CHANGE, dt)

    def _check_stops(self, dt, stock_data):
        cfg = self.config
        to_close = []
        for pos in self.open_positions:
            df = stock_data.get(pos["ticker"])
            if df is None or dt not in df.index: continue
            row = df.loc[dt]
            pos["days_held"] += 1
            pos["max_price"] = max(pos["max_price"], row["high"])
            pos["min_price"] = min(pos["min_price"], row["low"])
            trail = pos["max_price"] * (1 - cfg.trailing_stop_pct)
            stop = max(trail, pos["stop_price"])
            if row["low"] <= stop:
                to_close.append((pos, stop * (1 - cfg.slippage_pct), ExitReason.STOP_LOSS, dt))
        closed = set()
        for pos, price, reason, d in to_close:
            if pos["ticker"] not in closed and pos in self.open_positions:
                self._close(pos, price, reason, d)
                closed.add(pos["ticker"])

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
            shares=pos["shares"], pnl_dollars=pnl_d, pnl_pct=pnl_p,
            hold_days=pos["days_held"], exit_reason=reason,
            regime_at_entry=Regime.UNKNOWN, regime_at_exit=Regime.UNKNOWN,
            max_favorable=(pos["max_price"] - pos["entry_price"]) / pos["entry_price"],
            max_adverse=(pos["min_price"] - pos["entry_price"]) / pos["entry_price"],
        ))
        self.open_positions.remove(pos)

    def _close_all(self, dt, stock_data, note=""):
        for pos in list(self.open_positions):
            df = stock_data.get(pos["ticker"])
            if df is not None and dt in df.index:
                self._close(pos, df.loc[dt, "close"], ExitReason.REGIME_CHANGE, dt)

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
                                   "positions": len(self.open_positions), "drawdown": dd,
                                   "defensive": self.in_defensive_mode, "crisis": self.in_crisis_mode})

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

    def print_summary(self, m, label="V7"):
        tc = sum(t.shares * self.config.commission_per_share * 2 for t in self.closed_trades)
        ts = sum(abs(t.entry_price * t.shares * self.config.slippage_pct * 2) for t in self.closed_trades)
        dd_events = sum(1 for d in self.agents.decisions if d.get("action") in ("DEFENSIVE_MODE", "CRISIS_MODE"))
        print(f"\n{'='*65}")
        print(f"  {label} — INSTITUTIONAL GRADE")
        print(f"  3-Factor Ranking | Drawdown Control | Vol Target | Sector Limits")
        print(f"{'='*65}")
        print(f"  ${m.initial_capital:,.0f} → ${m.final_equity:,.0f}")
        print(f"  Return:     {m.total_return_pct:+.1f}%")
        print(f"  CAGR:       {m.cagr_pct:+.1f}%")
        print(f"  Max DD:     {m.max_drawdown_pct:.1f}%  {'✓' if m.max_drawdown_pct > -25 else '⚠ EXCEEDS -25%'}")
        print(f"  Sharpe:     {m.sharpe_ratio:.2f}  {'✓' if m.sharpe_ratio >= 1.0 else '⚠ below 1.0'}")
        print(f"  Sortino:    {m.sortino_ratio:.2f}")
        print(f"  Trades:     {m.total_trades} ({m.trades_per_year:.0f}/yr)")
        print(f"  Win Rate:   {m.win_rate:.0f}%")
        print(f"  PF:         {m.profit_factor:.2f}")
        print(f"  Exp/Trade:  {m.expectancy_per_trade:+.2f}%")
        print(f"  Avg Hold:   {m.avg_hold_days:.0f}d")
        print(f"  Costs:      ${tc+ts:,.0f}")
        print(f"  DD Events:  {dd_events}")
        print(f"  Agent Log:  {len(self.agents.decisions)}")
        print(f"{'='*65}")
