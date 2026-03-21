"""
Hedge Fund OS — V3 Backtest with Full Visual Report.

Run this:
    python3 run_v3_visual.py

It will:
1. Download data for 60 stocks
2. Compute V3 features (relative strength, momentum, pullback, quality scoring)
3. Run the V3 dual-strategy backtest
4. Generate charts (equity curve, drawdown, trade analysis, monthly heatmap)
5. Save everything to reports/
6. Open the charts automatically
"""

import sys
import os
import time
import logging
import platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend first
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.WARNING)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.ingest import download_ohlcv, get_spy, DEFAULT_UNIVERSE
from src.data.features import compute_spy_features
from src.data.features_v3 import compute_v3_features
from src.backtest.engine_v3 import BacktestEngineV3, V3Config

# ============================================================
# CONFIGURATION — CHANGE THESE TO EXPERIMENT
# ============================================================
TICKERS = DEFAULT_UNIVERSE[:60]
START_DATE = '2012-01-01'
END_DATE = None  # None = latest available
INITIAL_CAPITAL = 100_000

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 'backtests')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Style
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2


def main():
    print("\n" + "=" * 65)
    print("  HEDGE FUND OS — V3 VISUAL BACKTEST")
    print("  Dual Strategy: Breakout + Pullback | Quality Ranking")
    print("=" * 65)
    print(f"  Universe:    {len(TICKERS)} stocks")
    print(f"  Period:      {START_DATE} → {'latest' if END_DATE is None else END_DATE}")
    print(f"  Capital:     ${INITIAL_CAPITAL:,.0f}")
    print("=" * 65)

    # ---- Step 1: Download Data ----
    print("\n[1/5] Downloading market data...")
    t0 = time.time()

    spy_raw = get_spy(start=START_DATE, end=END_DATE)
    if spy_raw.empty:
        print("ERROR: Could not download SPY. Check your internet connection.")
        return

    stock_raw = {}
    for i, ticker in enumerate(TICKERS):
        try:
            df = download_ohlcv(ticker, start=START_DATE, end=END_DATE)
            if not df.empty and len(df) > 252:
                stock_raw[ticker] = df
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  Downloaded {i + 1}/{len(TICKERS)}...")

    print(f"  Got {len(stock_raw)}/{len(TICKERS)} tickers in {time.time() - t0:.0f}s")

    # ---- Step 2: Compute V3 Features ----
    print("\n[2/5] Computing V3 features (relative strength, momentum, pullback, quality)...")
    t0 = time.time()

    spy = compute_spy_features(spy_raw)
    stock_data = {}
    for ticker, df in stock_raw.items():
        try:
            stock_data[ticker] = compute_v3_features(df, spy)
        except Exception:
            pass

    print(f"  V3 features for {len(stock_data)} tickers in {time.time() - t0:.1f}s")

    # ---- Step 3: Run V3 Backtest ----
    print("\n[3/5] Running V3 backtest...")
    t0 = time.time()

    config = V3Config(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=0.01,
        max_positions=10,
        min_volume_ratio=1.5,
        base_max_depth=0.10,
        min_breakout_quality=40,
        require_2day_confirm=True,
        min_rs_score=50,
        min_momentum_score=40,
        pullback_enabled=True,
        failed_breakout_days=4,
        trailing_stop_lookback=10,
        scale_risk_by_quality=True,
    )

    engine = BacktestEngineV3(config)
    metrics = engine.run(stock_data, spy)

    print(f"  Completed in {time.time() - t0:.1f}s")

    # Print summary
    engine.print_summary(metrics)

    # Get results
    equity_curve = engine.get_equity_curve()
    trade_log = engine.get_trade_log()

    # Save CSVs
    if not equity_curve.empty:
        equity_curve.to_csv(os.path.join(REPORTS_DIR, 'v3_equity_curve.csv'), index=False)
    if not trade_log.empty:
        trade_log.to_csv(os.path.join(REPORTS_DIR, 'v3_trade_log.csv'), index=False)
    with open(os.path.join(REPORTS_DIR, 'v3_metrics.txt'), 'w') as f:
        for k, v in metrics.__dict__.items():
            f.write(f"{k}: {v}\n")

    # ---- Step 4: Generate Charts ----
    print("\n[4/5] Generating charts...")

    charts_generated = []

    # ---- CHART 1: Equity Curve + Drawdown ----
    if not equity_curve.empty:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                              gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle('HEDGE FUND OS — V3 Backtest Results', fontsize=18, fontweight='bold', y=0.98)

        dates = pd.to_datetime(equity_curve['date'])

        # Equity curve
        ax1.plot(dates, equity_curve['equity'], color='#3B82F6', linewidth=1.5, label='V3 Strategy')
        ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
        ax1.fill_between(dates, INITIAL_CAPITAL, equity_curve['equity'],
                         where=equity_curve['equity'] >= INITIAL_CAPITAL,
                         alpha=0.1, color='#22C55E')
        ax1.fill_between(dates, INITIAL_CAPITAL, equity_curve['equity'],
                         where=equity_curve['equity'] < INITIAL_CAPITAL,
                         alpha=0.1, color='#EF4444')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.set_title(
            f'Total Return: {metrics.total_return_pct:+.1f}%  |  '
            f'CAGR: {metrics.cagr_pct:+.1f}%  |  '
            f'Sharpe: {metrics.sharpe_ratio:.2f}  |  '
            f'Max DD: {metrics.max_drawdown_pct:.1f}%',
            fontsize=13, pad=10
        )
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Drawdown
        ax2.fill_between(dates, equity_curve['drawdown'] * 100, 0, color='#EF4444', alpha=0.6)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Positions open
        ax3.fill_between(dates, equity_curve['positions'], 0, color='#8B5CF6', alpha=0.4)
        ax3.set_ylabel('Open Positions', fontsize=11)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, 'v3_equity_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        charts_generated.append(path)
        print(f"  Saved: v3_equity_curve.png")

    # ---- CHART 2: Trade Analysis ----
    if not trade_log.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Trade Analysis — V3', fontsize=16, fontweight='bold')

        # PnL Distribution
        ax = axes[0, 0]
        colors = ['#22C55E' if x > 0 else '#EF4444' for x in trade_log['pnl_pct']]
        ax.bar(range(len(trade_log)), trade_log['pnl_pct'], color=colors, alpha=0.7, width=1.0)
        ax.set_title(f'PnL per Trade (%) — {len(trade_log)} trades', fontsize=12)
        ax.set_xlabel('Trade #')
        ax.set_ylabel('PnL %')
        ax.axhline(y=0, color='white', linewidth=0.5)

        # Exit Reason Breakdown
        ax = axes[0, 1]
        exit_counts = trade_log['exit_reason'].value_counts()
        colors_pie = ['#3B82F6', '#EF4444', '#F59E0B', '#22C55E', '#8B5CF6', '#EC4899']
        wedges, texts, autotexts = ax.pie(
            exit_counts.values, labels=exit_counts.index, autopct='%1.0f%%',
            colors=colors_pie[:len(exit_counts)], textprops={'fontsize': 9}
        )
        ax.set_title('Exit Reasons', fontsize=12)

        # Hold Time Distribution
        ax = axes[1, 0]
        ax.hist(trade_log['hold_days'], bins=25, color='#3B82F6', alpha=0.7, edgecolor='white')
        ax.set_title('Hold Time Distribution', fontsize=12)
        ax.set_xlabel('Days Held')
        ax.set_ylabel('Count')
        ax.axvline(x=trade_log['hold_days'].mean(), color='#F59E0B', linestyle='--',
                    label=f"Avg: {trade_log['hold_days'].mean():.0f} days")
        ax.legend()

        # PnL Histogram
        ax = axes[1, 1]
        winners = trade_log[trade_log['pnl_pct'] > 0]['pnl_pct']
        losers = trade_log[trade_log['pnl_pct'] <= 0]['pnl_pct']
        ax.hist(losers, bins=30, color='#EF4444', alpha=0.6, label=f'Losers ({len(losers)})')
        ax.hist(winners, bins=30, color='#22C55E', alpha=0.6, label=f'Winners ({len(winners)})')
        ax.set_title('PnL Distribution', fontsize=12)
        ax.set_xlabel('PnL %')
        ax.set_ylabel('Count')
        ax.axvline(x=0, color='white', linewidth=0.5)
        ax.legend()

        plt.tight_layout()
        path = os.path.join(REPORTS_DIR, 'v3_trade_analysis.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        charts_generated.append(path)
        print(f"  Saved: v3_trade_analysis.png")

    # ---- CHART 3: Monthly Returns Heatmap ----
    if not equity_curve.empty:
        eq = equity_curve.copy()
        eq['date'] = pd.to_datetime(eq['date'])
        eq = eq.set_index('date')

        monthly = eq['equity'].resample('ME').last().pct_change() * 100
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'ret': monthly.values
        }).dropna()

        if len(monthly_df) > 0:
            pivot = monthly_df.pivot_table(index='year', columns='month', values='ret')
            pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Add annual return column
            annual = eq['equity'].resample('YE').last().pct_change() * 100
            pivot['YEAR'] = [annual.get(pd.Timestamp(f'{y}-12-31'), np.nan) for y in pivot.index]

            fig, ax = plt.subplots(figsize=(16, 8))
            im = ax.imshow(pivot.values[:, :12], cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)

            ax.set_xticks(range(12))
            ax.set_xticklabels(pivot.columns[:12])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)

            for i in range(len(pivot.index)):
                for j in range(12):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        color = 'black' if abs(val) < 3 else 'white'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                fontsize=9, color=color, fontweight='bold')

            plt.colorbar(im, label='Monthly Return (%)', shrink=0.8)
            ax.set_title('Monthly Returns Heatmap (%) — V3 Strategy', fontsize=14, fontweight='bold')

            plt.tight_layout()
            path = os.path.join(REPORTS_DIR, 'v3_monthly_heatmap.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            charts_generated.append(path)
            print(f"  Saved: v3_monthly_heatmap.png")

    # ---- CHART 4: Key Metrics Dashboard ----
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    metrics_data = [
        ['Total Return', f'{metrics.total_return_pct:+.2f}%'],
        ['CAGR', f'{metrics.cagr_pct:+.2f}%'],
        ['Max Drawdown', f'{metrics.max_drawdown_pct:.2f}%'],
        ['Sharpe Ratio', f'{metrics.sharpe_ratio:.2f}'],
        ['Sortino Ratio', f'{metrics.sortino_ratio:.2f}'],
        ['Win Rate', f'{metrics.win_rate:.1f}%'],
        ['Avg Winner', f'{metrics.avg_winner_pct:+.2f}%'],
        ['Avg Loser', f'{metrics.avg_loser_pct:.2f}%'],
        ['Profit Factor', f'{metrics.profit_factor:.2f}'],
        ['Expectancy', f'{metrics.expectancy_per_trade:+.2f}%'],
        ['Total Trades', f'{metrics.total_trades}'],
        ['Trades/Year', f'{metrics.trades_per_year:.1f}'],
        ['Avg Hold', f'{metrics.avg_hold_days:.1f} days'],
        ['Max Consec Loss', f'{metrics.max_consecutive_losses}'],
        ['Exposure', f'{metrics.exposure_pct:.1f}%'],
    ]

    # Signal breakdown
    if engine.signals_log:
        bo = sum(1 for s in engine.signals_log if s["type"] == "breakout")
        pb = sum(1 for s in engine.signals_log if s["type"] == "pullback")
        metrics_data.append(['Breakout Signals', str(bo)])
        metrics_data.append(['Pullback Signals', str(pb)])

    table = ax.table(
        cellText=metrics_data,
        colLabels=['Metric', 'Value'],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Color the cells
    for i in range(len(metrics_data)):
        val_text = metrics_data[i][1]
        if val_text.startswith('+'):
            table[i + 1, 1].set_facecolor('#1a3a1a')
        elif val_text.startswith('-'):
            table[i + 1, 1].set_facecolor('#3a1a1a')

    for j in range(2):
        table[0, j].set_facecolor('#1e3a5f')
        table[0, j].set_text_props(fontweight='bold', color='white')

    ax.set_title('V3 Performance Dashboard', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, 'v3_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts_generated.append(path)
    print(f"  Saved: v3_dashboard.png")

    # ---- Step 5: Summary ----
    print("\n[5/5] Done!")
    print(f"\n  All reports saved to: {REPORTS_DIR}")
    print(f"  Files generated:")
    print(f"    • v3_equity_curve.png")
    print(f"    • v3_trade_analysis.png")
    print(f"    • v3_monthly_heatmap.png")
    print(f"    • v3_dashboard.png")
    print(f"    • v3_equity_curve.csv")
    print(f"    • v3_trade_log.csv")
    print(f"    • v3_metrics.txt")

    # Try to open the charts
    if charts_generated:
        print(f"\n  Opening charts...")
        for path in charts_generated:
            try:
                if platform.system() == 'Darwin':
                    os.system(f'open "{path}"')
                elif platform.system() == 'Windows':
                    os.startfile(path)
                else:
                    os.system(f'xdg-open "{path}"')
            except Exception:
                pass

    print("\n" + "=" * 65)
    print("  V3 BACKTEST COMPLETE")
    print("=" * 65)
    print(f"  ${INITIAL_CAPITAL:,.0f} → ${metrics.final_equity:,.0f} ({metrics.total_return_pct:+.1f}%)")
    print(f"  CAGR: {metrics.cagr_pct:+.1f}%  |  Sharpe: {metrics.sharpe_ratio:.2f}  |  Max DD: {metrics.max_drawdown_pct:.1f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
