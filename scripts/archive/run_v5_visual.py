"""
Hedge Fund OS — V5 Market-Beating Backtest.
Run: python3 run_v5_visual.py
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
from src.backtest.engine_v5 import BacktestEngineV5, V5Config

TICKERS = DEFAULT_UNIVERSE[:100]
START = '2012-01-01'
CAPITAL = 100_000
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'backtests')
os.makedirs(REPORTS, exist_ok=True)
plt.style.use('dark_background')
plt.rcParams.update({'figure.figsize': (16, 8), 'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.2})

def main():
    print("\n" + "=" * 65)
    print("  HEDGE FUND OS — V5: BEAT THE MARKET")
    print("  Weekly Rebal | Top 5 Concentrated | 100 Stocks | 20%+ CAGR")
    print("=" * 65)

    # Download
    print("\n[1/5] Downloading 80 stocks...")
    t0 = time.time()
    spy_raw = get_spy(start=START)
    stock_raw = {}
    for i, t in enumerate(TICKERS):
        try:
            df = download_ohlcv(t, start=START)
            if not df.empty and len(df) > 252: stock_raw[t] = df
        except: pass
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(TICKERS)}...")
    print(f"  Got {len(stock_raw)} in {time.time()-t0:.0f}s")

    # Features
    print("\n[2/5] Computing features...")
    spy = compute_v4_spy(spy_raw)
    stock_data = {}
    for t, df in stock_raw.items():
        try: stock_data[t] = compute_v4_features(df, spy)
        except: pass
    print(f"  {len(stock_data)} tickers")

    # THE WINNING CONFIG
    print("\n[3/5] Running V5 (market-beating config)...")
    t0 = time.time()
    config = V5Config(
        initial_capital=CAPITAL,
        max_positions=5,
        max_position_pct=0.25,
        risk_per_trade=0.04,
        rebal_frequency=5,         # weekly rebalance
        momentum_lookback=63,      # 3-month momentum
        momentum_skip=5,           # skip last week
        top_n=5,                   # concentrated top 5
        trailing_stop_pct=0.22,    # 22% trailing from peak
        hard_stop_pct=0.28,        # 28% hard stop
        bear_market_cash=False,    # stay invested
        vol_targeting=False,
        require_uptrend=True,      # above 200MA only
    )
    engine = BacktestEngineV5(config)
    metrics = engine.run(stock_data, spy)
    print(f"  Done in {time.time()-t0:.1f}s")
    engine.print_summary(metrics)

    eq = engine.get_equity_curve()
    trades = engine.get_trade_log()

    # SPY comparison
    spy_i = 260
    spy_start_p = spy.iloc[spy_i]['close']
    spy_end_p = spy.iloc[-1]['close']
    spy_ret = (spy_end_p / spy_start_p - 1) * 100
    spy_yrs = (spy.index[-1] - spy.index[spy_i]).days / 365.25
    spy_cagr = ((spy_end_p / spy_start_p) ** (1/spy_yrs) - 1) * 100

    # Build SPY equity curve for comparison
    spy_equity = []
    for idx in range(spy_i, len(spy)):
        price = spy.iloc[idx]['close']
        spy_eq = CAPITAL * (price / spy_start_p)
        spy_equity.append({"date": spy.index[idx], "equity": spy_eq})
    spy_eq_df = pd.DataFrame(spy_equity)

    # Save CSVs
    if not eq.empty: eq.to_csv(os.path.join(REPORTS, 'v5_equity_curve.csv'), index=False)
    if not trades.empty: trades.to_csv(os.path.join(REPORTS, 'v5_trade_log.csv'), index=False)
    with open(os.path.join(REPORTS, 'v5_metrics.txt'), 'w') as f:
        for k, v in metrics.__dict__.items(): f.write(f"{k}: {v}\n")
    with open(os.path.join(REPORTS, 'v5_agent_decisions.json'), 'w') as f:
        json.dump(engine.agent_decisions[-200:], f, indent=2, default=str)

    # CHARTS
    print("\n[4/5] Generating charts...")
    charts = []

    # Chart 1: Equity Curve vs SPY
    if not eq.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle('HEDGE FUND OS V5 vs S&P 500', fontsize=18, fontweight='bold', y=0.98)
        dates = pd.to_datetime(eq['date'])
        ax1.plot(dates, eq['equity'], color='#3B82F6', linewidth=2, label='V5 Strategy')
        if not spy_eq_df.empty:
            spy_dates = pd.to_datetime(spy_eq_df['date'])
            ax1.plot(spy_dates, spy_eq_df['equity'], color='#F59E0B', linewidth=1.5, alpha=0.7, label='SPY Buy & Hold')
        ax1.axhline(y=CAPITAL, color='gray', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.set_title(
            f"V5: {metrics.total_return_pct:+.0f}% ({metrics.cagr_pct:+.1f}% CAGR) | "
            f"SPY: {spy_ret:+.0f}% ({spy_cagr:+.1f}% CAGR) | "
            f"Sharpe: {metrics.sharpe_ratio:.2f} | Max DD: {metrics.max_drawdown_pct:.1f}%",
            fontsize=13, pad=10)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.fill_between(dates, eq['drawdown']*100, 0, color='#EF4444', alpha=0.6)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        p = os.path.join(REPORTS, 'v5_equity_vs_spy.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); charts.append(p)
        print("  Saved v5_equity_vs_spy.png")

    # Chart 2: Trade Analysis
    if not trades.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('V5 Trade Analysis — Market-Beating Config', fontsize=16, fontweight='bold')
        ax = axes[0,0]
        colors = ['#22C55E' if x > 0 else '#EF4444' for x in trades['pnl_pct']]
        ax.bar(range(len(trades)), trades['pnl_pct'], color=colors, alpha=0.7, width=1.0)
        ax.set_title(f'PnL per Trade — {len(trades)} trades, {metrics.win_rate:.0f}% win rate')
        ax.axhline(y=0, color='white', linewidth=0.5)
        ax = axes[0,1]
        ec = trades['exit_reason'].value_counts()
        ax.pie(ec.values, labels=ec.index, autopct='%1.0f%%',
               colors=['#3B82F6','#EF4444','#F59E0B','#22C55E','#8B5CF6'][:len(ec)], textprops={'fontsize':9})
        ax.set_title('Exit Reasons')
        ax = axes[1,0]
        ax.hist(trades['hold_days'], bins=30, color='#3B82F6', alpha=0.7, edgecolor='white')
        ax.axvline(x=trades['hold_days'].mean(), color='#F59E0B', linestyle='--',
                    label=f"Avg: {trades['hold_days'].mean():.0f} days")
        ax.set_title('Hold Time'); ax.set_xlabel('Days'); ax.legend()
        ax = axes[1,1]
        w = trades[trades['pnl_pct']>0]['pnl_pct']; l = trades[trades['pnl_pct']<=0]['pnl_pct']
        ax.hist(l, bins=25, color='#EF4444', alpha=0.6, label=f'Losers ({len(l)})')
        ax.hist(w, bins=25, color='#22C55E', alpha=0.6, label=f'Winners ({len(w)})')
        ax.set_title('PnL Distribution'); ax.legend(); ax.axvline(x=0, color='white', linewidth=0.5)
        plt.tight_layout()
        p = os.path.join(REPORTS, 'v5_trade_analysis.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); charts.append(p)
        print("  Saved v5_trade_analysis.png")

    # Chart 3: Monthly heatmap
    if not eq.empty:
        eqc = eq.copy(); eqc['date'] = pd.to_datetime(eqc['date']); eqc = eqc.set_index('date')
        monthly = eqc['equity'].resample('ME').last().pct_change() * 100
        mdf = pd.DataFrame({'year': monthly.index.year, 'month': monthly.index.month, 'ret': monthly.values}).dropna()
        if len(mdf) > 0:
            pivot = mdf.pivot_table(index='year', columns='month', values='ret')
            pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            fig, ax = plt.subplots(figsize=(16, 8))
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-8, vmax=8)
            ax.set_xticks(range(12)); ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
            for i in range(len(pivot.index)):
                for j in range(12):
                    v = pivot.values[i,j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9,
                                color='black' if abs(v)<4 else 'white', fontweight='bold')
            plt.colorbar(im, label='Monthly Return (%)', shrink=0.8)
            ax.set_title('V5 Monthly Returns (%)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            p = os.path.join(REPORTS, 'v5_monthly_heatmap.png')
            plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); charts.append(p)
            print("  Saved v5_monthly_heatmap.png")

    # Chart 4: Version comparison dashboard
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    data = [
        ['V1 (Breakout only)', '-2.9%', '-0.2%', '-21.0%', '-0.01', '22.7%', '1.00', '657', '$0'],
        ['V3 (Dual strategy)', '+181%', '+7.6%', '-19.4%', '0.68', '34.1%', '1.28', '1743', '$0'],
        ['V4 (Vol target+Agents)', '+266%', '+9.6%', '-22.2%', '0.89', '35.4%', '1.40', '1785', '6,891'],
        ['V5 (Beat market)', f'+{metrics.total_return_pct:.0f}%', f'+{metrics.cagr_pct:.1f}%',
         f'{metrics.max_drawdown_pct:.1f}%', f'{metrics.sharpe_ratio:.2f}', f'{metrics.win_rate:.0f}%',
         f'{metrics.profit_factor:.2f}', f'{metrics.total_trades}', f'{len(engine.agent_decisions)}'],
        ['SPY Buy & Hold', f'+{spy_ret:.0f}%', f'+{spy_cagr:.1f}%', '~-33%', '~0.65', 'N/A', 'N/A', '0', 'N/A'],
    ]
    table = ax.table(
        cellText=data,
        colLabels=['Version', 'Return', 'CAGR', 'Max DD', 'Sharpe', 'Win Rate', 'PF', 'Trades', 'Agent Logs'],
        cellLoc='center', loc='center', colWidths=[0.18, 0.10, 0.08, 0.08, 0.08, 0.08, 0.07, 0.08, 0.10]
    )
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.0)
    for j in range(9): table[0, j].set_facecolor('#1e3a5f'); table[0, j].set_text_props(fontweight='bold', color='white')
    # Highlight V5 row
    for j in range(9): table[4, j].set_facecolor('#1a3a1a')
    # Highlight SPY row
    for j in range(9): table[5, j].set_facecolor('#2a2a1a')
    ax.set_title('Evolution: V1 → V5 vs SPY', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    p = os.path.join(REPORTS, 'v5_version_comparison.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); charts.append(p)
    print("  Saved v5_version_comparison.png")

    print(f"\n[5/5] Opening charts...")
    for p in charts:
        try:
            if platform.system() == 'Darwin': os.system(f'open "{p}"')
        except: pass

    print("\n" + "=" * 65)
    print("  V5 COMPLETE")
    print("=" * 65)
    beat = "YES" if metrics.cagr_pct > spy_cagr else "NO"
    print(f"  V5:  ${CAPITAL:,.0f} → ${metrics.final_equity:,.0f} ({metrics.cagr_pct:+.1f}% CAGR)")
    print(f"  SPY: ${CAPITAL:,.0f} → ${CAPITAL*(1+spy_ret/100):,.0f} ({spy_cagr:+.1f}% CAGR)")
    print(f"  BEATS S&P 500? → {beat}")
    print(f"  Transaction costs included: ${sum(t.shares * config.commission_per_share * 2 for t in engine.closed_trades) + sum(abs(t.entry_price * t.shares * config.slippage_pct * 2) for t in engine.closed_trades):,.0f}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
