"""
Hedge Fund OS — V7 Institutional Grade (FIXED).

Run: python3 run_v7_visual.py

FIXES:
- Forces fresh data download (deletes stale cache)
- Saves ALL required files for dashboard (metrics, equity, trades, agent log)
- Shows debug info if something goes wrong
"""
import sys, os, time, logging, platform, json, shutil
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.ingest import download_ohlcv, get_spy, DEFAULT_UNIVERSE
from src.data.features_v4 import compute_v4_features, compute_v4_spy
from src.backtest.engine_v7 import BacktestEngineV7, V7Config

TICKERS = DEFAULT_UNIVERSE[:100]
START = '2012-01-01'
CAPITAL = 100_000
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'backtests')
DATA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
os.makedirs(REPORTS, exist_ok=True)

plt.style.use('dark_background')
plt.rcParams.update({'figure.figsize': (16, 8), 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.2})


def main():
    print("\n" + "=" * 65)
    print("  HEDGE FUND OS — V7: DISCIPLINED ALPHA")
    print("  3-Factor | Vol Target | Sector Limits | OOS Validated")
    print("=" * 65)

    # STEP 0: Clear stale cache to force fresh downloads
    print("\n[0/5] Clearing stale data cache...")
    if os.path.exists(DATA_CACHE):
        shutil.rmtree(DATA_CACHE)
        print("  Cache cleared. Fresh data will be downloaded.")
    os.makedirs(DATA_CACHE, exist_ok=True)

    # STEP 1: Download fresh data
    print("\n[1/5] Downloading 100 stocks (fresh)...")
    t0 = time.time()
    spy_raw = get_spy(start=START, force=True)
    if spy_raw.empty:
        print("ERROR: Cannot download SPY. Check internet.")
        return

    stock_raw = {}
    for i, t in enumerate(TICKERS):
        try:
            df = download_ohlcv(t, start=START, force=True)
            if not df.empty and len(df) > 252:
                stock_raw[t] = df
        except:
            pass
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(TICKERS)}...")
    print(f"  Got {len(stock_raw)} tickers in {time.time()-t0:.0f}s")

    # STEP 2: Compute features
    print("\n[2/5] Computing V4 features...")
    t0 = time.time()
    spy = compute_v4_spy(spy_raw)
    stock_data = {}
    for t, df in stock_raw.items():
        try:
            stock_data[t] = compute_v4_features(df, spy)
        except Exception as e:
            print(f"  WARNING: {t} feature error: {e}")
    print(f"  {len(stock_data)} tickers in {time.time()-t0:.1f}s")

    # DEBUG: Verify ranking works
    test_dt = spy.index[1000]
    from src.backtest.engine_v7 import BacktestEngineV7 as _E
    _test = _E(V7Config())
    _rankings = _test._rank_universe(test_dt, stock_data)
    print(f"  DEBUG: {len(_rankings)} stocks qualify at {test_dt.date()}")
    if len(_rankings) < 10:
        print("  WARNING: Very few stocks qualifying. Check data.")

    # STEP 3: Run V7
    print("\n[3/5] Running V7 backtest...")
    t0 = time.time()
    config = V7Config(
        max_portfolio_dd=-0.50, crisis_dd=-0.60, recovery_threshold=-0.40,
        trailing_stop_pct=0.12, hard_stop_pct=0.18,
        max_positions=8, top_n=8, buy_threshold=8, sell_threshold=15,
        vol_targeting=True, vol_target=0.14,
        weight_momentum=0.55, weight_quality=0.25, weight_low_vol=0.20,
    )
    engine = BacktestEngineV7(config)
    metrics = engine.run(stock_data, spy)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Print summary
    tc = sum(t.shares * config.commission_per_share * 2 for t in engine.closed_trades)
    ts = sum(abs(t.entry_price * t.shares * config.slippage_pct * 2) for t in engine.closed_trades)
    print("\n" + "=" * 65)
    print("  V7 RESULTS")
    print("=" * 65)
    print(f"  ${CAPITAL:,.0f} → ${metrics.final_equity:,.0f}")
    print(f"  Total Return:    {metrics.total_return_pct:+.2f}%")
    print(f"  CAGR:            {metrics.cagr_pct:+.2f}%")
    print(f"  Max Drawdown:    {metrics.max_drawdown_pct:.2f}%")
    print(f"  Sharpe:          {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino:         {metrics.sortino_ratio:.2f}")
    print(f"  Trades:          {metrics.total_trades} ({metrics.trades_per_year:.0f}/yr)")
    print(f"  Win Rate:        {metrics.win_rate:.1f}%")
    print(f"  Avg Winner:      {metrics.avg_winner_pct:+.2f}%")
    print(f"  Avg Loser:       {metrics.avg_loser_pct:.2f}%")
    print(f"  Profit Factor:   {metrics.profit_factor:.2f}")
    print(f"  Expectancy:      {metrics.expectancy_per_trade:+.2f}%/trade")
    print(f"  Avg Hold:        {metrics.avg_hold_days:.0f} days")
    print(f"  Costs:           ${tc+ts:,.0f}")
    print(f"  Agent Decisions: {len(engine.agents.decisions)}")
    if engine.closed_trades:
        reasons = defaultdict(int)
        for t in engine.closed_trades:
            reasons[t.exit_reason.value] += 1
        print(f"  Exits: " + " | ".join(f"{r}:{c}" for r, c in sorted(reasons.items(), key=lambda x: -x[1])))
    print("=" * 65)

    # SANITY CHECK
    if metrics.total_trades < 50:
        print(f"\n  ⚠️  WARNING: Only {metrics.total_trades} trades. Expected 400+.")
        print(f"  This may indicate a data issue. Try deleting data/raw/ and re-running.")

    eq = engine.get_equity_curve()
    trades = engine.get_trade_log()

    # SPY comparison
    spy_i = 270
    spy_sp = spy.iloc[spy_i]['close']
    spy_ep = spy.iloc[-1]['close']
    spy_yrs = (spy.index[-1] - spy.index[spy_i]).days / 365.25
    spy_cagr = ((spy_ep / spy_sp) ** (1 / spy_yrs) - 1) * 100

    # ============================================================
    # SAVE ALL FILES THE DASHBOARD NEEDS
    # ============================================================
    print("\n  Saving all dashboard files...")

    # 1. Equity curve
    if not eq.empty:
        eq.to_csv(os.path.join(REPORTS, 'v7_equity_curve.csv'), index=False)
        print(f"  ✓ v7_equity_curve.csv")

    # 2. Trade log
    if not trades.empty:
        trades.to_csv(os.path.join(REPORTS, 'v7_trade_log.csv'), index=False)
        print(f"  ✓ v7_trade_log.csv")

    # 3. METRICS FILE (this was missing before!)
    metrics_dict = metrics.__dict__.copy()
    # Convert date objects to strings
    for k, v in metrics_dict.items():
        if hasattr(v, 'isoformat'):
            metrics_dict[k] = str(v)
    with open(os.path.join(REPORTS, 'v7_metrics.txt'), 'w') as f:
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"  ✓ v7_metrics.txt")

    # 4. Agent decisions
    with open(os.path.join(REPORTS, 'v7_agent_decisions.json'), 'w') as f:
        json.dump(engine.agents.decisions[-500:], f, indent=2, default=str)
    print(f"  ✓ v7_agent_decisions.json ({len(engine.agents.decisions)} decisions)")

    # ============================================================
    # CHARTS
    # ============================================================
    print("\n[4/5] Generating charts...")
    charts = []

    # Chart 1: Equity vs SPY
    if not eq.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle('HEDGE FUND OS V7 — Institutional Grade', fontsize=18, fontweight='bold', y=0.98)
        dates = pd.to_datetime(eq['date'])
        ax1.plot(dates, eq['equity'], color='#3B82F6', linewidth=2, label='V7 Strategy')
        spy_eq = [CAPITAL * (spy.iloc[j]['close'] / spy_sp) for j in range(spy_i, len(spy))]
        spy_dates = spy.index[spy_i:len(spy)]
        ax1.plot(spy_dates, spy_eq, color='#F59E0B', linewidth=1.5, alpha=0.7, label='SPY Buy & Hold')
        ax1.axhline(y=CAPITAL, color='gray', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Equity ($)')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.set_title(
            f"V7: {metrics.total_return_pct:+.0f}% ({metrics.cagr_pct:+.1f}% CAGR, Sharpe {metrics.sharpe_ratio:.2f}) | "
            f"SPY: +{(spy_ep/spy_sp-1)*100:.0f}% ({spy_cagr:+.1f}% CAGR) | "
            f"Only {metrics.total_trades} trades, {len(engine.agents.decisions)} agent decisions",
            fontsize=12)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.fill_between(dates, eq['drawdown'] * 100, 0, color='#EF4444', alpha=0.6)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        p = os.path.join(REPORTS, 'v7_equity_vs_spy.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        plt.close()
        charts.append(p)
        print("  ✓ v7_equity_vs_spy.png")

    # Chart 2: Trade analysis
    if not trades.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'V7 Trade Analysis — {metrics.total_trades} trades, {metrics.win_rate:.0f}% win rate, {metrics.avg_hold_days:.0f}d avg hold',
                     fontsize=14, fontweight='bold')
        ax = axes[0, 0]
        c = ['#22C55E' if x > 0 else '#EF4444' for x in trades['pnl_pct']]
        ax.bar(range(len(trades)), trades['pnl_pct'], color=c, alpha=0.7, width=1.0)
        ax.set_title('PnL per Trade (%)')
        ax.axhline(y=0, color='white', linewidth=0.5)
        ax = axes[0, 1]
        ec = trades['exit_reason'].value_counts()
        ax.pie(ec.values, labels=ec.index, autopct='%1.0f%%',
               colors=['#3B82F6', '#EF4444', '#F59E0B', '#22C55E', '#8B5CF6'][:len(ec)],
               textprops={'fontsize': 9})
        ax.set_title('Exit Reasons')
        ax = axes[1, 0]
        ax.hist(trades['hold_days'], bins=30, color='#3B82F6', alpha=0.7, edgecolor='white')
        ax.axvline(x=trades['hold_days'].mean(), color='#F59E0B', linestyle='--',
                   label=f"Avg: {trades['hold_days'].mean():.0f}d")
        ax.legend()
        ax.set_title('Hold Time Distribution')
        ax.set_xlabel('Days')
        ax = axes[1, 1]
        w = trades[trades['pnl_pct'] > 0]['pnl_pct']
        l = trades[trades['pnl_pct'] <= 0]['pnl_pct']
        ax.hist(l, bins=25, color='#EF4444', alpha=0.6, label=f'Losers ({len(l)})')
        ax.hist(w, bins=25, color='#22C55E', alpha=0.6, label=f'Winners ({len(w)})')
        ax.set_title('PnL Distribution')
        ax.legend()
        ax.axvline(x=0, color='white', linewidth=0.5)
        plt.tight_layout()
        p = os.path.join(REPORTS, 'v7_trade_analysis.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        plt.close()
        charts.append(p)
        print("  ✓ v7_trade_analysis.png")

    # Open charts
    print("\n[5/5] Opening charts...")
    for p in charts:
        try:
            if platform.system() == 'Darwin':
                os.system(f'open "{p}"')
        except:
            pass

    beat = "YES ✓" if metrics.cagr_pct > spy_cagr else "NO"
    print("\n" + "=" * 65)
    print("  V7 COMPLETE")
    print("=" * 65)
    print(f"  V7:  ${CAPITAL:,.0f} → ${metrics.final_equity:,.0f} ({metrics.cagr_pct:+.1f}% CAGR)")
    print(f"  SPY: ${CAPITAL:,.0f} → ${CAPITAL*(spy_ep/spy_sp):,.0f} ({spy_cagr:+.1f}% CAGR)")
    print(f"  BEATS SPY? {beat}")
    print(f"  Costs: ${tc+ts:,.0f} | Agents: {len(engine.agents.decisions)} decisions")
    print(f"\n  Dashboard ready: streamlit run dashboard.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
