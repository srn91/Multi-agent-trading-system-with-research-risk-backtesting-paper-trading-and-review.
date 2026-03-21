"""
V8 — The Final Boss.

Target: 20% CAGR, <25% DD, Sharpe >=1.0

Key insight from V5-V7 failures:
- Going to cash kills CAGR (V7 problem)
- No risk controls kills drawdown (V5/V6 problem)
- The solution: NEVER go to cash. Instead ROTATE between offense and defense.

Strategy:
1. OFFENSE MODE (good regime): Top 5 aggressive momentum stocks, weekly rebal
2. DEFENSE MODE (bad regime): Rotate into top 5 LOW-VOL/QUALITY stocks
3. ADAPTIVE STOPS: Tight (12%) in offense, tighter (10%) in defense
4. VOL TARGETING: Scale all positions to target 14% portfolio vol
5. 3-FACTOR RANKING: momentum 55% + quality 25% + low-vol 20%
6. SECTOR LIMITS: max 2 positions per sector

The key innovation: During bear markets, V5 stayed in falling momentum names.
V7 went to cash and missed recovery. V8 ROTATES into defensive leaders.
Always invested. Always in the strongest names for the current regime.
"""
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np, pandas as pd
from src.schemas.signals import BacktestMetrics, ExitReason, Regime, TradeExit

logger = logging.getLogger(__name__)


@dataclass
class V8Config:
    initial_capital: float = 100_000.0
    slippage_pct: float = 0.0005
    commission_per_share: float = 0.005

    # Portfolio
    max_positions: int = 5
    max_position_pct: float = 0.22
    max_per_sector: int = 2

    # Rebalance
    check_frequency: int = 5
    momentum_lookback: int = 63
    momentum_skip: int = 5
    top_n: int = 5
    require_uptrend: bool = True
    sell_threshold: int = 12

    # Regime-adaptive (NEW)
    regime_offensive_threshold: float = 55    # regime score > 55 = offense
    offense_weight_momentum: float = 0.60
    offense_weight_quality: float = 0.20
    offense_weight_lowvol: float = 0.20
    defense_weight_momentum: float = 0.20
    defense_weight_quality: float = 0.35
    defense_weight_lowvol: float = 0.45     # defense favors stability

    # Stops — adaptive
    offense_trailing_stop: float = 0.14
    offense_hard_stop: float = 0.20
    defense_trailing_stop: float = 0.10      # tighter in defense
    defense_hard_stop: float = 0.15

    # Vol targeting
    vol_target: float = 0.14
    vol_targeting: bool = True

    # Conviction
    min_conviction: float = 0.35
    high_conviction: float = 0.65


class V8AgentTeam:
    def __init__(self):
        self.decisions = []

    def score_entry(self, ticker, row, rank_idx, total, spy_row, positions, regime_mode):
        scores = {}
        evidence = {}

        # Momentum
        rank_pct = 1 - (rank_idx / max(total, 1))
        scores["momentum"] = min(rank_pct * 1.2, 1.0)
        evidence["momentum"] = f"rank {rank_idx+1}/{total}"

        # Quality
        quality = 0
        if row.get("trend_close_above_150", False): quality += 0.25
        if row.get("trend_150_above_200", False): quality += 0.25
        if row.get("trend_200_rising", False): quality += 0.20
        rs = row.get("rs_rank_score", 50)
        if not pd.isna(rs): quality += min(rs / 100 * 0.30, 0.30)
        scores["quality"] = min(quality, 1.0)
        evidence["quality"] = f"q={quality:.2f}"

        # Low vol
        vol = row.get("realized_vol_20", 0.25)
        if pd.isna(vol): vol = 0.25
        scores["low_vol"] = max(min(1 - (vol - 0.08) / 0.35, 1.0), 0.1)
        evidence["low_vol"] = f"vol={vol*100:.0f}%"

        # Regime gate
        regime_score = 0.5
        if spy_row is not None:
            rs_spy = spy_row.get("regime_score", 50)
            if not pd.isna(rs_spy): regime_score = rs_spy / 100
        scores["regime"] = regime_score

        # Sector check
        from src.data.features_v4 import TICKER_SECTOR
        sector = TICKER_SECTOR.get(ticker, "XLK")
        same_sector = sum(1 for p in positions if TICKER_SECTOR.get(p["ticker"], "XLK") == sector)
        corr_penalty = 0.10 if same_sector >= 2 else 0

        # REGIME-ADAPTIVE weighting
        if regime_mode == "offense":
            w_mom, w_qual, w_vol = 0.60, 0.20, 0.20
        else:
            w_mom, w_qual, w_vol = 0.20, 0.35, 0.45

        conviction = (scores["momentum"] * w_mom + scores["quality"] * w_qual + scores["low_vol"] * w_vol)
        conviction *= max(regime_score, 0.5)
        conviction -= corr_penalty
        evidence["mode"] = regime_mode
        evidence["sector_penalty"] = f"-{corr_penalty*100:.0f}%"

        return max(conviction, 0), scores, evidence

    def score_exit(self, pos, row, spy_row, config, regime_mode):
        signals = {}
        ret = row.get("ret_20", 0)
        if pd.isna(ret): ret = 0
        signals["decay"] = max(-ret * 5, 0)
        rs = row.get("rs_rank_score", 50)
        if pd.isna(rs): rs = 50
        signals["rs_drop"] = max((50 - rs) / 100, 0)
        ma50 = row.get("ma_50", 0)
        signals["below_ma"] = 0.3 if (not pd.isna(ma50) and ma50 > 0 and row["close"] < ma50) else 0
        regime_sell = 0
        if spy_row is not None:
            r = spy_row.get("regime_score", 50)
            if not pd.isna(r) and r < 30: regime_sell = 0.3
        signals["regime"] = regime_sell
        pressure = signals["decay"]*0.35 + signals["rs_drop"]*0.25 + signals["below_ma"]*0.20 + signals["regime"]*0.20
        # In defense mode, be quicker to exit
        if regime_mode == "defense": pressure *= 1.2
        return pressure, signals

    def log(self, entry): self.decisions.append(entry)


class BacktestEngineV8:
    def __init__(self, config=None):
        self.config = config or V8Config()
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.signals_log = []
        self.agents = V8AgentTeam()
        self.days_since_check = 999
        self.rebal_count = 0
        self.regime_mode = "offense"

    def run(self, stock_data, spy_data):
        dates = spy_data.index.tolist()
        for i, dt in enumerate(dates):
            if i < 270: continue
            self._update_portfolio(dt, stock_data)

            # Determine regime mode
            self.regime_mode = self._get_regime_mode(dt, spy_data)

            # Daily: early warning + stops
            self._check_early_exits(dt, stock_data, spy_data)
            self._check_stops(dt, stock_data)

            # Weekly: rebalance
            self.days_since_check += 1
            if self.days_since_check >= self.config.check_frequency:
                self._rebalance(dt, dates, i, stock_data, spy_data)
                self.days_since_check = 0

            self._record_equity(dt, stock_data)
        return self._compute_metrics(dates[0].date(), dates[-1].date())

    def _get_regime_mode(self, dt, spy_data):
        if dt not in spy_data.index: return "offense"
        score = spy_data.loc[dt].get("regime_score", 50)
        if pd.isna(score): return "offense"
        return "offense" if score >= self.config.regime_offensive_threshold else "defense"

    def _get_stops(self):
        cfg = self.config
        if self.regime_mode == "offense":
            return cfg.offense_trailing_stop, cfg.offense_hard_stop
        return cfg.defense_trailing_stop, cfg.defense_hard_stop

    def _vol_scalar(self, dt, spy_data):
        if not self.config.vol_targeting: return 1.0
        if dt not in spy_data.index: return 1.0
        vol = spy_data.loc[dt].get("realized_vol_20", 0.15)
        if pd.isna(vol) or vol <= 0: vol = 0.15
        return max(min(self.config.vol_target / vol, 1.5), 0.3)

    def _rebalance(self, dt, dates, idx, stock_data, spy_data):
        cfg = self.config
        rankings = self._rank_universe(dt, stock_data, spy_data)
        if not rankings: return

        spy_row = spy_data.loc[dt] if dt in spy_data.index else None
        top_set = set(r["ticker"] for r in rankings[:cfg.top_n])
        keep_set = set(r["ticker"] for r in rankings[:cfg.sell_threshold])

        # SELL: not in keep zone
        for pos in list(self.open_positions):
            if pos["ticker"] not in keep_set and pos in self.open_positions:
                if dt in stock_data[pos["ticker"]].index:
                    row = stock_data[pos["ticker"]].loc[dt]
                    sell_p, _ = self.agents.score_exit(pos, row, spy_row, cfg, self.regime_mode)
                    if sell_p > 0.20 or pos["ticker"] not in set(r["ticker"] for r in rankings[:cfg.sell_threshold+5]):
                        self._close(pos, row["close"], ExitReason.TRAILING_STOP, dt)
                        self.rebal_count += 1

        # BUY
        held = {p["ticker"] for p in self.open_positions}
        if idx + 1 >= len(dates): return
        next_date = dates[idx + 1]
        vol_scalar = self._vol_scalar(dt, spy_data)

        from src.data.features_v4 import TICKER_SECTOR
        sector_counts = defaultdict(int)
        for p in self.open_positions:
            sector_counts[TICKER_SECTOR.get(p["ticker"], "XLK")] += 1

        for rank_idx, r in enumerate(rankings[:cfg.top_n]):
            ticker = r["ticker"]
            if ticker in held: continue
            if len(self.open_positions) >= cfg.max_positions: break

            # Sector limit
            sector = TICKER_SECTOR.get(ticker, "XLK")
            if sector_counts[sector] >= cfg.max_per_sector: continue

            df = stock_data.get(ticker)
            if df is None or dt not in df.index or next_date not in df.index: continue
            row = df.loc[dt]

            conv, scores, evidence = self.agents.score_entry(
                ticker, row, rank_idx, len(rankings), spy_row,
                self.open_positions, self.regime_mode
            )
            if conv < cfg.min_conviction: continue

            entry_price = df.loc[next_date, "open"] * (1 + cfg.slippage_pct)
            if entry_price <= 0: continue
            _, hard_stop = self._get_stops()
            stop_price = entry_price * (1 - hard_stop)

            size_mult = vol_scalar
            if conv >= cfg.high_conviction: size_mult *= 1.2
            elif conv < 0.45: size_mult *= 0.7

            target_value = min(self.equity * cfg.max_position_pct * size_mult, self.equity * 0.28)
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
                "conviction": conv, "regime_at_entry": self.regime_mode,
            })
            sector_counts[sector] += 1
            self.signals_log.append({"date": dt, "ticker": ticker, "type": self.regime_mode,
                                      "conviction": round(conv, 3)})
            self.agents.log({
                "date": str(dt.date()), "agent": "team", "ticker": ticker,
                "action": f"buy_{self.regime_mode}", "conviction": round(conv, 3),
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "evidence": evidence,
            })
            self.rebal_count += 1

    def _rank_universe(self, dt, stock_data, spy_data):
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
            row = df.iloc[loc]
            quality = 0
            if row.get("trend_close_above_150", False): quality += 0.25
            if row.get("trend_150_above_200", False): quality += 0.25
            if row.get("trend_200_rising", False): quality += 0.25
            rs = row.get("rs_rank_score", 50)
            if not pd.isna(rs): quality += min(rs/100*0.25, 0.25)
            vol = row.get("realized_vol_20", 0.25)
            if pd.isna(vol): vol = 0.25
            low_vol = max(1 - (vol - 0.08) / 0.35, 0.1)
            rankings.append({"ticker": ticker, "momentum": mom, "quality": quality, "low_vol": low_vol, "vol": vol})

        if not rankings: return []
        moms = [r["momentum"] for r in rankings]
        quals = [r["quality"] for r in rankings]
        vols = [r["low_vol"] for r in rankings]
        for r in rankings:
            nm = (r["momentum"] - min(moms)) / (max(moms) - min(moms) + 1e-10)
            nq = (r["quality"] - min(quals)) / (max(quals) - min(quals) + 1e-10)
            nv = (r["low_vol"] - min(vols)) / (max(vols) - min(vols) + 1e-10)
            if self.regime_mode == "offense":
                r["composite"] = nm*cfg.offense_weight_momentum + nq*cfg.offense_weight_quality + nv*cfg.offense_weight_lowvol
            else:
                r["composite"] = nm*cfg.defense_weight_momentum + nq*cfg.defense_weight_quality + nv*cfg.defense_weight_lowvol
        rankings.sort(key=lambda x: x["composite"], reverse=True)
        return rankings

    def _check_early_exits(self, dt, stock_data, spy_data):
        spy_row = spy_data.loc[dt] if dt in spy_data.index else None
        for pos in list(self.open_positions):
            if pos not in self.open_positions: continue
            df = stock_data.get(pos["ticker"])
            if df is None or dt not in df.index: continue
            row = df.loc[dt]
            sp, _ = self.agents.score_exit(pos, row, spy_row, self.config, self.regime_mode)
            if sp > 0.55:
                pnl = (row["close"] - pos["entry_price"]) / pos["entry_price"]
                if pnl < 0.02:
                    self._close(pos, row["close"], ExitReason.REGIME_CHANGE, dt)

    def _check_stops(self, dt, stock_data):
        trail_pct, _ = self._get_stops()
        to_close = []
        for pos in self.open_positions:
            df = stock_data.get(pos["ticker"])
            if df is None or dt not in df.index: continue
            row = df.loc[dt]
            pos["days_held"] += 1
            pos["max_price"] = max(pos["max_price"], row["high"])
            pos["min_price"] = min(pos["min_price"], row["low"])
            trail = pos["max_price"] * (1 - trail_pct)
            stop = max(trail, pos["stop_price"])
            if row["low"] <= stop:
                to_close.append((pos, stop * (1 - self.config.slippage_pct), ExitReason.STOP_LOSS, dt))
        closed = set()
        for pos, price, reason, d in to_close:
            if pos["ticker"] not in closed and pos in self.open_positions:
                self._close(pos, price, reason, d); closed.add(pos["ticker"])

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
            max_favorable=(pos["max_price"]-pos["entry_price"])/pos["entry_price"],
            max_adverse=(pos["min_price"]-pos["entry_price"])/pos["entry_price"],
        ))
        self.open_positions.remove(pos)

    def _update_portfolio(self, dt, stock_data):
        ur = sum((stock_data[p["ticker"]].loc[dt,"close"]-p["entry_price"])*p["shares"]
                 for p in self.open_positions
                 if p["ticker"] in stock_data and dt in stock_data[p["ticker"]].index)
        self.equity = self.cash + sum(p["shares"]*p["entry_price"] for p in self.open_positions) + ur
        if self.equity > self.peak_equity: self.peak_equity = self.equity

    def _record_equity(self, dt, stock_data):
        dd = (self.equity-self.peak_equity)/self.peak_equity if self.peak_equity > 0 else 0
        self.equity_curve.append({"date": dt, "equity": self.equity, "cash": self.cash,
                                   "positions": len(self.open_positions), "drawdown": dd,
                                   "regime": self.regime_mode})

    def _compute_metrics(self, sd, ed):
        trades = self.closed_trades; eq = pd.DataFrame(self.equity_curve)
        if not trades:
            return BacktestMetrics(start_date=sd,end_date=ed,initial_capital=self.config.initial_capital,
                final_equity=self.equity,total_return_pct=0,cagr_pct=0,max_drawdown_pct=0,avg_drawdown_pct=0,
                sharpe_ratio=0,sortino_ratio=0,win_rate=0,avg_winner_pct=0,avg_loser_pct=0,profit_factor=0,
                expectancy_per_trade=0,total_trades=0,avg_hold_days=0,max_consecutive_losses=0,exposure_pct=0,trades_per_year=0)
        tr=(self.equity-self.config.initial_capital)/self.config.initial_capital
        yrs=max((ed-sd).days/365.25,0.1); cagr=(1+tr)**(1/yrs)-1
        mdd=eq["drawdown"].min() if not eq.empty else 0
        add=eq.loc[eq["drawdown"]<0,"drawdown"].mean() if not eq.empty and (eq["drawdown"]<0).any() else 0
        pnls=[t.pnl_pct for t in trades]; w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
        wr=len(w)/len(trades); aw=np.mean(w) if w else 0; al=np.mean(l) if l else 0
        gp=sum(t.pnl_dollars for t in trades if t.pnl_dollars>0)
        gl=abs(sum(t.pnl_dollars for t in trades if t.pnl_dollars<0))
        pf=gp/gl if gl>0 else float("inf"); exp=np.mean(pnls)
        if not eq.empty and len(eq)>1:
            dr=eq["equity"].pct_change().dropna()
            sh=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
            ds=dr[dr<0]; so=dr.mean()/ds.std()*np.sqrt(252) if len(ds)>0 and ds.std()>0 else 0
        else: sh=so=0
        hd=[t.hold_days for t in trades]; mc=cs=0
        for p in pnls:
            if p<=0: cs+=1; mc=max(mc,cs)
            else: cs=0
        ex=(eq["positions"]>0).mean() if not eq.empty else 0
        return BacktestMetrics(start_date=sd,end_date=ed,initial_capital=self.config.initial_capital,
            final_equity=round(self.equity,2),total_return_pct=round(tr*100,2),cagr_pct=round(cagr*100,2),
            max_drawdown_pct=round(mdd*100,2),avg_drawdown_pct=round(add*100,2) if add else 0,
            sharpe_ratio=round(sh,2),sortino_ratio=round(so,2),win_rate=round(wr*100,2),
            avg_winner_pct=round(aw*100,2),avg_loser_pct=round(al*100,2),profit_factor=round(pf,2),
            expectancy_per_trade=round(exp*100,2),total_trades=len(trades),
            avg_hold_days=round(np.mean(hd),1) if hd else 0,max_consecutive_losses=mc,
            exposure_pct=round(ex*100,2),trades_per_year=round(len(trades)/yrs,1))

    def get_equity_curve(self): return pd.DataFrame(self.equity_curve)
    def get_trade_log(self):
        if not self.closed_trades: return pd.DataFrame()
        return pd.DataFrame([{"ticker":t.ticker,"entry_date":t.entry_date,"exit_date":t.exit_date,
            "entry_price":round(t.entry_price,2),"exit_price":round(t.exit_price,2),"shares":t.shares,
            "pnl_dollars":round(t.pnl_dollars,2),"pnl_pct":round(t.pnl_pct*100,2),"hold_days":t.hold_days,
            "exit_reason":t.exit_reason.value} for t in self.closed_trades])

    def print_summary(self, m, label="V8"):
        tc=sum(t.shares*self.config.commission_per_share*2 for t in self.closed_trades)
        ts=sum(abs(t.entry_price*t.shares*self.config.slippage_pct*2) for t in self.closed_trades)
        off=sum(1 for s in self.signals_log if s["type"]=="offense")
        defe=sum(1 for s in self.signals_log if s["type"]=="defense")
        print(f"\n{'='*65}")
        print(f"  {label} — REGIME-ADAPTIVE MULTI-FACTOR")
        print(f"  Offense/Defense Rotation | 3-Factor | Vol Target | Sector Limits")
        print(f"{'='*65}")
        print(f"  ${m.initial_capital:,.0f} → ${m.final_equity:,.0f}")
        print(f"  Return:    {m.total_return_pct:+.1f}%")
        print(f"  CAGR:      {m.cagr_pct:+.1f}%  {'✓' if m.cagr_pct>=20 else '⚠'}")
        print(f"  Max DD:    {m.max_drawdown_pct:.1f}%  {'✓' if m.max_drawdown_pct>-25 else '⚠'}")
        print(f"  Sharpe:    {m.sharpe_ratio:.2f}  {'✓' if m.sharpe_ratio>=1.0 else '⚠'}")
        print(f"  Sortino:   {m.sortino_ratio:.2f}")
        print(f"  Trades:    {m.total_trades} ({m.trades_per_year:.0f}/yr)")
        print(f"  Win Rate:  {m.win_rate:.0f}%")
        print(f"  PF:        {m.profit_factor:.2f}")
        print(f"  Exp/Trade: {m.expectancy_per_trade:+.2f}%")
        print(f"  Avg Hold:  {m.avg_hold_days:.0f}d")
        print(f"  Costs:     ${tc+ts:,.0f}")
        print(f"  Offense trades: {off} | Defense trades: {defe}")
        print(f"  Agent Log: {len(self.agents.decisions)}")
        if self.closed_trades:
            reasons=defaultdict(int)
            for t in self.closed_trades: reasons[t.exit_reason.value]+=1
            print(f"  Exits: "+" | ".join(f"{r}:{c}" for r,c in sorted(reasons.items(),key=lambda x:-x[1])))
        print(f"{'='*65}\n")
