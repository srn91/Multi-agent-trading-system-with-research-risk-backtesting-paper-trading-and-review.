"""
Hedge Fund OS — V4 Full Stack Backtest with Visuals.

Run: python3 run_v4_visual.py

V4 = V3 + Vol Targeting + Regime Throttling + Agent Decisions + Momentum Rotation
"""
import sys, os, time, logging, platform, json
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.ingest import download_ohlcv, get_spy, DEFAULT_UNIVERSE
from src.data.features_v4 import compute_v4_features, compute_v4_spy
from src.backtest.engine_v4 import BacktestEngineV4, V4Config

TICKERS = DEFAULT_UNIVERSE[:60]
START, END = '2012-01-01', None
CAPITAL = 100_000
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'backtests')
os.makedirs(REPORTS, exist_ok=True)

plt.style.use('dark_background')
plt.rcParams.update({'figure.figsize': (16, 8), 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.2})

def main():
    print("\n" + "=" * 65)
    print("  HEDGE FUND OS — V4 FULL STACK BACKTEST")
    print("  Vol Targeting | Regime Throttle | Agents | Momentum Rotation")
    print("=" * 65)
    print(f"  Universe: {len(TICKERS)} stocks  |  Capital: ${CAPITAL:,.0f}")
    print("=" * 65)

    # Download
    print("\n[1/6] Downloading data...")
    t0 = time.time()
    spy_raw = get_spy(start=START, end=END)
    stock_raw = {}
    for i, t in enumerate(TICKERS):
        try:
            df = download_ohlcv(t, start=START, end=END)
            if not df.empty and len(df) > 252: stock_raw[t] = df
        except: pass
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(TICKERS)}...")
    print(f"  Got {len(stock_raw)} tickers in {time.time()-t0:.0f}s")

    # V4 Features
    print("\n[2/6] Computing V4 features...")
    t0 = time.time()
    spy = compute_v4_spy(spy_raw)
    stock_data = {}
    for t, df in stock_raw.items():
        try: stock_data[t] = compute_v4_features(df, spy)
        except: pass
    print(f"  {len(stock_data)} tickers in {time.time()-t0:.1f}s")

    # Run V4
    print("\n[3/6] Running V4 backtest...")
    t0 = time.time()
    config = V4Config(
        initial_capital=CAPITAL, risk_per_trade=0.01, max_positions=12,
        max_position_pct=0.15, min_volume_ratio=1.5, base_max_depth=0.10,
        min_breakout_quality=40, require_2day_confirm=True,
        min_rs_score=50, min_momentum_score=40,
        pullback_enabled=True,
        momentum_rotation_enabled=True, momentum_top_n=5, momentum_rebal_days=21,
        vol_targeting_enabled=True, vol_target=0.10,
        regime_throttle_enabled=True, min_regime_score=30,
        kill_switch_drawdown=-0.15,
        scale_risk_by_quality=True,
    )
    engine = BacktestEngineV4(config)
    metrics = engine.run(stock_data, spy)
    print(f"  Done in {time.time()-t0:.1f}s")
    engine.print_summary(metrics)

    eq = engine.get_equity_curve()
    trades = engine.get_trade_log()
    agent_log = engine.get_agent_log()

    # Save data
    if not eq.empty: eq.to_csv(os.path.join(REPORTS, 'v4_equity_curve.csv'), index=False)
    if not trades.empty: trades.to_csv(os.path.join(REPORTS, 'v4_trade_log.csv'), index=False)
    with open(os.path.join(REPORTS, 'v4_metrics.txt'), 'w') as f:
        for k, v in metrics.__dict__.items(): f.write(f"{k}: {v}\n")
    with open(os.path.join(REPORTS, 'v4_agent_decisions.json'), 'w') as f:
        json.dump(agent_log[-200:], f, indent=2, default=str)

    # Charts
    print("\n[4/6] Generating charts...")
    charts = []

    # Chart 1: Equity + Drawdown + Regime
    if not eq.empty:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle('HEDGE FUND OS — V4 Results', fontsize=18, fontweight='bold', y=0.98)
        dates = pd.to_datetime(eq['date'])
        ax1.plot(dates, eq['equity'], color='#3B82F6', linewidth=1.5, label='V4 Strategy')
        ax1.axhline(y=CAPITAL, color='gray', linestyle='--', alpha=0.5, label=f'Starting ${CAPITAL:,.0f}')
        ax1.fill_between(dates, CAPITAL, eq['equity'], where=eq['equity']>=CAPITAL, alpha=0.1, color='#22C55E')
        ax1.fill_between(dates, CAPITAL, eq['equity'], where=eq['equity']<CAPITAL, alpha=0.1, color='#EF4444')
        ax1.set_ylabel('Equity ($)')
        ax1.legend(loc='upper left')
        ax1.set_title(f"Return: {metrics.total_return_pct:+.1f}% | CAGR: {metrics.cagr_pct:+.1f}% | "
                       f"Sharpe: {metrics.sharpe_ratio:.2f} | Max DD: {metrics.max_drawdown_pct:.1f}%", fontsize=13)
        ax1.xaxis.set_major_locator(mdates.YearLocator()); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.fill_between(dates, eq['drawdown']*100, 0, color='#EF4444', alpha=0.6)
        ax2.set_ylabel('Drawdown (%)')
        ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.fill_between(dates, eq['positions'], 0, color='#8B5CF6', alpha=0.4)
        ax3.set_ylabel('Positions'); ax3.set_xlabel('Date')
        ax3.xaxis.set_major_locator(mdates.YearLocator()); ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        p = os.path.join(REPORTS, 'v4_equity_curve.png'); plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); charts.append(p)
        print("  Saved v4_equity_curve.png")

    # Chart 2: Trade Analysis
    if not trades.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('V4 Trade Analysis', fontsize=16, fontweight='bold')
        ax = axes[0,0]
        colors = ['#22C55E' if x > 0 else '#EF4444' for x in trades['pnl_pct']]
        ax.bar(range(len(trades)), trades['pnl_pct'], color=colors, alpha=0.7, width=1.0)
        ax.set_title(f'PnL per Trade — {len(trades)} total'); ax.axhline(y=0, color='white', linewidth=0.5)
        ax = axes[0,1]
        ec = trades['exit_reason'].value_counts()
        ax.pie(ec.values, labels=ec.index, autopct='%1.0f%%',
               colors=['#3B82F6','#EF4444','#F59E0B','#22C55E','#8B5CF6'][:len(ec)], textprops={'fontsize':9})
        ax.set_title('Exit Reasons')
        ax = axes[1,0]
        ax.hist(trades['hold_days'], bins=25, color='#3B82F6', alpha=0.7, edgecolor='white')
        ax.set_title('Hold Time'); ax.set_xlabel('Days')
        ax = axes[1,1]
        w = trades[trades['pnl_pct']>0]['pnl_pct']; l = trades[trades['pnl_pct']<=0]['pnl_pct']
        ax.hist(l, bins=30, color='#EF4444', alpha=0.6, label=f'Losers ({len(l)})')
        ax.hist(w, bins=30, color='#22C55E', alpha=0.6, label=f'Winners ({len(w)})')
        ax.set_title('PnL Distribution'); ax.legend()
        plt.tight_layout()
        p = os.path.join(REPORTS, 'v4_trade_analysis.png'); plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); charts.append(p)
        print("  Saved v4_trade_analysis.png")

    # Chart 3: Strategy breakdown by signal type
    if not trades.empty and engine.signals_log:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('V4 Strategy Breakdown by Signal Type', fontsize=16, fontweight='bold')
        types = defaultdict(int)
        for s in engine.signals_log: types[s["type"]] += 1
        ax1.bar(types.keys(), types.values(), color=['#3B82F6', '#22C55E', '#F59E0B'][:len(types)])
        ax1.set_title('Signals by Type'); ax1.set_ylabel('Count')
        # PnL by signal type (approximate from trade log matching)
        type_pnl = defaultdict(list)
        for s in engine.signals_log:
            for _, tr in trades.iterrows():
                if tr['ticker'] == s['ticker']:
                    type_pnl[s['type']].append(tr['pnl_pct'])
                    break
        if type_pnl:
            labels = list(type_pnl.keys())
            avgs = [np.mean(v) if v else 0 for v in type_pnl.values()]
            colors = ['#22C55E' if a > 0 else '#EF4444' for a in avgs]
            ax2.bar(labels, avgs, color=colors)
            ax2.set_title('Avg PnL % by Signal Type'); ax2.axhline(y=0, color='white', linewidth=0.5)
        plt.tight_layout()
        p = os.path.join(REPORTS, 'v4_strategy_breakdown.png'); plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); charts.append(p)
        print("  Saved v4_strategy_breakdown.png")

    # Chart 4: Monthly heatmap
    if not eq.empty:
        eqc = eq.copy(); eqc['date'] = pd.to_datetime(eqc['date']); eqc = eqc.set_index('date')
        monthly = eqc['equity'].resample('ME').last().pct_change() * 100
        mdf = pd.DataFrame({'year': monthly.index.year, 'month': monthly.index.month, 'ret': monthly.values}).dropna()
        if len(mdf) > 0:
            pivot = mdf.pivot_table(index='year', columns='month', values='ret')
            pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            fig, ax = plt.subplots(figsize=(16, 8))
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
            ax.set_xticks(range(12)); ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
            for i in range(len(pivot.index)):
                for j in range(12):
                    v = pivot.values[i,j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9,
                                color='black' if abs(v)<3 else 'white', fontweight='bold')
            plt.colorbar(im, label='Monthly Return (%)', shrink=0.8)
            ax.set_title('V4 Monthly Returns (%)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            p = os.path.join(REPORTS, 'v4_monthly_heatmap.png'); plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); charts.append(p)
            print("  Saved v4_monthly_heatmap.png")

    # Summary
    print(f"\n[5/6] Agent decisions logged: {len(agent_log)}")
    print(f"  Sample decisions (last 5):")
    for d in agent_log[-5:]:
        print(f"    {d.get('date','')} | {d.get('agent','')} | {d.get('ticker','')} | {d.get('action','')} | {d.get('reasoning','')[:60]}")

    print(f"\n[6/6] Opening charts...")
    for p in charts:
        try:
            if platform.system() == 'Darwin': os.system(f'open "{p}"')
            elif platform.system() == 'Windows': os.startfile(p)
        except: pass

    print("\n" + "=" * 65)
    print("  V4 BACKTEST COMPLETE")
    print("=" * 65)
    print(f"  ${CAPITAL:,.0f} → ${metrics.final_equity:,.0f} ({metrics.total_return_pct:+.1f}%)")
    print(f"  CAGR: {metrics.cagr_pct:+.1f}% | Sharpe: {metrics.sharpe_ratio:.2f} | Max DD: {metrics.max_drawdown_pct:.1f}%")
    print(f"  Agents: {len(agent_log)} decisions logged | Strategies: 3 (breakout + pullback + momentum)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
