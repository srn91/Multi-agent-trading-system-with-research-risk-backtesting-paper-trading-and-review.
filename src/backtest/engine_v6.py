"""
Backtest Engine V6 — Disciplined Alpha.

Philosophy: Trade less. Hold longer. Exit smarter. Agents decide.

V5 problem: 1,091 trades, $132K costs, weekly churn.
V6 solution:
  1. THRESHOLD REBALANCING — only trade when rankings change meaningfully
  2. AGENT-DRIVEN DECISIONS — 4 agents vote on every entry and exit
  3. EARLY WARNING EXITS — detect momentum decay BEFORE price crashes
  4. CONVICTION SIZING — higher confidence = bigger position
  5. HOLDING BONUS — stocks held >30 days need stronger reason to sell

Transaction costs: 0.05% slippage + $0.005/share (unchanged, realistic)
"""
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np, pandas as pd
from src.schemas.signals import BacktestMetrics, ExitReason, Regime, TradeExit

logger = logging.getLogger(__name__)


@dataclass
class V6Config:
    initial_capital: float = 100_000.0
    slippage_pct: float = 0.0005
    commission_per_share: float = 0.005
    max_positions: int = 5
    max_position_pct: float = 0.25
    base_risk: float = 0.04

    # Rebalance: check weekly but only trade if rankings shift
    check_frequency: int = 5
    momentum_lookback: int = 63
    momentum_skip: int = 5
    top_n: int = 5
    require_uptrend: bool = True

    # THRESHOLD: stock must drop out of top N+buffer to get sold
    # Prevents selling #5 when it becomes #6 for one week
    sell_threshold: int = 10       # only sell if dropped below top 10
    buy_threshold: int = 5         # only buy if in top 5

    # Conviction scoring thresholds
    min_conviction: float = 0.4    # 0-1 score, below this = no trade
    high_conviction: float = 0.7   # above this = 1.5x position size

    # Stops (wide)
    trailing_stop_pct: float = 0.22
    hard_stop_pct: float = 0.28

    # Early warning: sell before stop if momentum decays
    momentum_decay_exit: bool = True
    decay_lookback: int = 10       # check 10-day momentum
    decay_threshold: float = -0.08 # if 10-day return < -8% AND RS dropping

    # Holding bonus: harder to sell after 30 days
    holding_bonus_days: int = 30
    holding_bonus_multiplier: float = 1.3  # need 30% stronger sell signal


class AgentTeam:
    """
    4 agents work as a team on every decision.
    Each agent scores 0-1. PM aggregates and decides.
    """
    def __init__(self):
        self.decisions = []

    def score_entry(self, ticker, row, momentum_rank, total_stocks, spy_row):
        """4 agents independently score a potential entry."""
        scores = {}
        evidence = {}

        # AGENT 1: Momentum Agent — is this a top mover?
        rank_pct = 1 - (momentum_rank / max(total_stocks, 1))
        scores["momentum"] = min(rank_pct * 1.2, 1.0)
        evidence["momentum"] = f"rank {momentum_rank+1}/{total_stocks} ({rank_pct*100:.0f}th pctl)"

        # AGENT 2: Trend Agent — is the trend structure healthy?
        trend_score = 0
        if row.get("trend_close_above_150", False): trend_score += 0.3
        if row.get("trend_150_above_200", False): trend_score += 0.3
        if row.get("trend_200_rising", False): trend_score += 0.2
        rs = row.get("rs_rank_score", 50)
        if not pd.isna(rs): trend_score += min(rs / 100 * 0.2, 0.2)
        scores["trend"] = min(trend_score, 1.0)
        evidence["trend"] = f"trend={trend_score:.2f}, rs={rs:.0f}"

        # AGENT 3: Risk Agent — is volatility acceptable?
        vol = row.get("realized_vol_20", 0.20)
        if pd.isna(vol): vol = 0.20
        # Lower vol = higher score (less risky)
        vol_score = max(1 - (vol - 0.10) / 0.40, 0.1)
        scores["risk"] = min(vol_score, 1.0)
        evidence["risk"] = f"vol={vol*100:.1f}%, score={vol_score:.2f}"

        # AGENT 4: Regime Agent — is the market supportive?
        regime_score = 0.5
        if spy_row is not None:
            rs_spy = spy_row.get("regime_score", 50)
            if not pd.isna(rs_spy):
                regime_score = rs_spy / 100
        scores["regime"] = regime_score
        evidence["regime"] = f"market={regime_score*100:.0f}/100"

        # PM AGGREGATION: weighted average
        weights = {"momentum": 0.35, "trend": 0.25, "risk": 0.15, "regime": 0.25}
        conviction = sum(scores[k] * weights[k] for k in scores)

        return conviction, scores, evidence

    def score_exit(self, pos, row, spy_row, config):
        """Score whether to exit a position. Higher = stronger sell signal."""
        signals = {}

        # Signal 1: Momentum decay
        ret_10 = row.get("ret_20", 0)  # use 20-day return as proxy
        if pd.isna(ret_10): ret_10 = 0
        signals["momentum_decay"] = max(-ret_10 * 5, 0)  # scale: -10% = 0.5 sell signal

        # Signal 2: RS deterioration
        rs = row.get("rs_rank_score", 50)
        if pd.isna(rs): rs = 50
        signals["rs_decay"] = max((50 - rs) / 100, 0)  # below 50 RS = sell pressure

        # Signal 3: Below key MA
        below_50ma = 0
        ma50 = row.get("ma_50", 0)
        if not pd.isna(ma50) and ma50 > 0 and row["close"] < ma50:
            below_50ma = 0.3
        signals["below_ma"] = below_50ma

        # Signal 4: Regime deterioration
        regime_sell = 0
        if spy_row is not None:
            regime = spy_row.get("regime_score", 50)
            if not pd.isna(regime) and regime < 35:
                regime_sell = 0.3
        signals["regime_weak"] = regime_sell

        # Aggregate sell pressure
        sell_pressure = (
            signals["momentum_decay"] * 0.35 +
            signals["rs_decay"] * 0.25 +
            signals["below_ma"] * 0.20 +
            signals["regime_weak"] * 0.20
        )

        # Holding bonus: harder to sell stocks held > N days
        if pos["days_held"] > config.holding_bonus_days:
            sell_pressure /= config.holding_bonus_multiplier

        return sell_pressure, signals

    def log(self, entry):
        self.decisions.append(entry)


class BacktestEngineV6:
    def __init__(self, config=None):
        self.config = config or V6Config()
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.signals_log = []
        self.agents = AgentTeam()
        self.days_since_check = 999
        self.rebal_count = 0

    def run(self, stock_data, spy_data):
        dates = spy_data.index.tolist()
        for i, dt in enumerate(dates):
            if i < 270: continue
            self._update_portfolio(dt, stock_data)

            # Daily: check early warning exits
            self._check_early_exits(dt, stock_data, spy_data)
            # Daily: check hard stops
            self._check_stops(dt, stock_data)

            # Periodic: check if rebalance needed
            self.days_since_check += 1
            if self.days_since_check >= self.config.check_frequency:
                self._smart_rebalance(dt, dates, i, stock_data, spy_data)
                self.days_since_check = 0

            self._record_equity(dt, stock_data)

        return self._compute_metrics(dates[0].date(), dates[-1].date())

    def _smart_rebalance(self, dt, dates, idx, stock_data, spy_data):
        """Only trade when rankings change meaningfully."""
        cfg = self.config
        rankings = self._rank_universe(dt, stock_data)
        if not rankings: return

        spy_row = spy_data.loc[dt] if dt in spy_data.index else None

        top_tickers = set(r["ticker"] for r in rankings[:cfg.buy_threshold])
        keep_zone = set(r["ticker"] for r in rankings[:cfg.sell_threshold])

        # SELL: only if dropped below threshold AND agent agrees
        for pos in list(self.open_positions):
            if pos["ticker"] not in keep_zone:
                # Agent team scores the exit
                if dt in stock_data[pos["ticker"]].index:
                    row = stock_data[pos["ticker"]].loc[dt]
                    sell_pressure, signals = self.agents.score_exit(pos, row, spy_row, cfg)

                    # Require meaningful sell pressure (not just ranking noise)
                    if sell_pressure > 0.25 or pos["ticker"] not in set(r["ticker"] for r in rankings[:cfg.sell_threshold + 5]):
                        price = row["close"]
                        self._close(pos, price, ExitReason.TRAILING_STOP, dt)
                        self.agents.log({
                            "date": str(dt.date()), "agent": "team", "ticker": pos["ticker"],
                            "action": "sell_rotation", "sell_pressure": round(sell_pressure, 3),
                            "signals": {k: round(v, 3) for k, v in signals.items()},
                        })
                        self.rebal_count += 1

        # BUY: only if in top N AND high conviction AND slot available
        held = {p["ticker"] for p in self.open_positions}
        if idx + 1 >= len(dates): return
        next_date = dates[idx + 1]

        for rank_idx, r in enumerate(rankings[:cfg.buy_threshold]):
            ticker = r["ticker"]
            if ticker in held: continue
            if len(self.open_positions) >= cfg.max_positions: break

            df = stock_data.get(ticker)
            if df is None or dt not in df.index or next_date not in df.index: continue

            row = df.loc[dt]

            # Agent team scores the entry
            conviction, scores, evidence = self.agents.score_entry(
                ticker, row, rank_idx, len(rankings), spy_row
            )

            if conviction < cfg.min_conviction:
                self.agents.log({
                    "date": str(dt.date()), "agent": "team", "ticker": ticker,
                    "action": "reject_low_conviction",
                    "conviction": round(conviction, 3), "scores": {k: round(v, 3) for k, v in scores.items()},
                })
                continue

            # Execute entry
            entry_price = df.loc[next_date, "open"] * (1 + cfg.slippage_pct)
            if entry_price <= 0: continue

            atr = row.get("atr", entry_price * 0.02)
            if pd.isna(atr) or atr <= 0: atr = entry_price * 0.02
            stop_price = entry_price * (1 - cfg.hard_stop_pct)

            # Conviction-based sizing
            size_mult = 1.0
            if conviction >= cfg.high_conviction:
                size_mult = 1.3
            elif conviction < 0.5:
                size_mult = 0.7

            target_value = self.equity * cfg.max_position_pct * size_mult
            target_value = min(target_value, self.equity * 0.30)  # cap at 30%
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
                "days_held": 0, "signal_type": "momentum",
                "conviction": conviction,
            })
            self.signals_log.append({"date": dt, "ticker": ticker, "type": "momentum",
                                      "conviction": round(conviction, 3)})
            self.agents.log({
                "date": str(dt.date()), "agent": "team", "ticker": ticker,
                "action": "buy",
                "conviction": round(conviction, 3),
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "evidence": evidence,
                "size_mult": round(size_mult, 2),
            })
            self.rebal_count += 1

    def _check_early_exits(self, dt, stock_data, spy_data):
        """Agent-driven early warning: exit before stop hits."""
        if not self.config.momentum_decay_exit: return
        cfg = self.config
        spy_row = spy_data.loc[dt] if dt in spy_data.index else None

        for pos in list(self.open_positions):
            if pos not in self.open_positions: continue
            df = stock_data.get(pos["ticker"])
            if df is None or dt not in df.index: continue
            row = df.loc[dt]

            sell_pressure, signals = self.agents.score_exit(pos, row, spy_row, cfg)

            # Strong sell signal = exit early (before trailing stop)
            if sell_pressure > 0.55:
                pnl = (row["close"] - pos["entry_price"]) / pos["entry_price"]
                # Only early-exit if we're losing money or barely positive
                if pnl < 0.03:
                    self._close(pos, row["close"], ExitReason.REGIME_CHANGE, dt)
                    self.agents.log({
                        "date": str(dt.date()), "agent": "risk",
                        "ticker": pos["ticker"], "action": "early_warning_exit",
                        "sell_pressure": round(sell_pressure, 3),
                        "pnl": round(pnl * 100, 2),
                        "signals": {k: round(v, 3) for k, v in signals.items()},
                    })

    def _check_stops(self, dt, stock_data):
        """Hard trailing stop from peak."""
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
                exit_price = stop * (1 - cfg.slippage_pct)
                to_close.append((pos, exit_price, ExitReason.STOP_LOSS, dt))

        closed = set()
        for pos, price, reason, d in to_close:
            if pos["ticker"] not in closed and pos in self.open_positions:
                self._close(pos, price, reason, d)
                closed.add(pos["ticker"])

    def _rank_universe(self, dt, stock_data):
        """Rank all stocks by momentum."""
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

            rankings.append({"ticker": ticker, "momentum": mom})

        rankings.sort(key=lambda x: x["momentum"], reverse=True)
        return rankings

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
        } for t in self.closed_trades])

    def print_summary(self, m):
        tc = sum(t.shares * self.config.commission_per_share * 2 for t in self.closed_trades)
        ts = sum(abs(t.entry_price * t.shares * self.config.slippage_pct * 2) for t in self.closed_trades)
        print("\n" + "=" * 65)
        print("  V6 — DISCIPLINED ALPHA")
        print("  Threshold Rebal | Agent Team | Early Warning | Conviction Size")
        print("=" * 65)
        print(f"  Period:          {m.start_date} → {m.end_date}")
        print(f"  ${m.initial_capital:,.0f} → ${m.final_equity:,.0f}")
        print(f"  Total Return:    {m.total_return_pct:+.2f}%")
        print(f"  CAGR:            {m.cagr_pct:+.2f}%")
        print(f"  Max Drawdown:    {m.max_drawdown_pct:.2f}%")
        print(f"  Sharpe:          {m.sharpe_ratio:.2f}")
        print(f"  Sortino:         {m.sortino_ratio:.2f}")
        print("-" * 65)
        print(f"  Trades:          {m.total_trades}  ({m.trades_per_year:.0f}/year)")
        print(f"  Win Rate:        {m.win_rate:.1f}%")
        print(f"  Avg Winner:      {m.avg_winner_pct:+.2f}%")
        print(f"  Avg Loser:       {m.avg_loser_pct:.2f}%")
        print(f"  Profit Factor:   {m.profit_factor:.2f}")
        print(f"  Expectancy:      {m.expectancy_per_trade:+.2f}%/trade")
        print(f"  Avg Hold:        {m.avg_hold_days:.0f} days")
        print("-" * 65)
        print(f"  COSTS: ${tc+ts:,.0f} total (${(tc+ts)/max(m.total_trades,1):,.0f}/trade)")
        print(f"  Rebalance actions: {self.rebal_count}")
        print(f"  Agent decisions:   {len(self.agents.decisions)}")
        if self.closed_trades:
            reasons = defaultdict(int)
            for t in self.closed_trades: reasons[t.exit_reason.value] += 1
            print(f"  Exits: ", end="")
            print(" | ".join(f"{r}:{c}" for r, c in sorted(reasons.items(), key=lambda x:-x[1])))
        print("=" * 65 + "\n")
