"""
Main backtest runner.

Usage:
    python -m src.backtest.run
    python -m src.backtest.run --tickers AAPL,MSFT,NVDA --start 2015-01-01
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.ingest import download_ohlcv, get_spy, DEFAULT_UNIVERSE
from src.data.features import compute_all_features, compute_spy_features
from src.backtest.engine import BacktestEngine, BacktestConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_backtest(
    tickers: list[str] | None = None,
    start: str = "2012-01-01",
    end: str | None = None,
    risk_per_trade: float = 0.01,
    initial_capital: float = 100_000,
    max_positions: int = 10,
    save_results: bool = True,
):
    """Run the full breakout backtest pipeline."""

    if tickers is None:
        tickers = DEFAULT_UNIVERSE[:50]  # Start with top 50 for speed

    print(f"\n{'='*60}")
    print(f"  HEDGE FUND OS — Breakout Backtest")
    print(f"{'='*60}")
    print(f"  Universe:  {len(tickers)} stocks")
    print(f"  Period:    {start} → {end or 'latest'}")
    print(f"  Capital:   ${initial_capital:,.0f}")
    print(f"  Risk/Trade: {risk_per_trade*100:.1f}%")
    print(f"  Max Positions: {max_positions}")
    print(f"{'='*60}\n")

    # --- Step 1: Download data ---
    print("[1/4] Downloading market data...")
    t0 = time.time()

    spy_raw = get_spy(start=start, end=end)
    if spy_raw.empty:
        print("ERROR: Could not download SPY data. Aborting.")
        return None

    stock_data_raw = {}
    for i, ticker in enumerate(tickers):
        try:
            df = download_ohlcv(ticker, start=start, end=end)
            if not df.empty and len(df) > 252:
                stock_data_raw[ticker] = df
        except Exception as e:
            logger.warning(f"Skip {ticker}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  Downloaded {i+1}/{len(tickers)}...")

    print(f"  Got {len(stock_data_raw)}/{len(tickers)} tickers in {time.time()-t0:.1f}s")

    # --- Step 2: Compute features ---
    print("\n[2/4] Computing features...")
    t0 = time.time()

    spy = compute_spy_features(spy_raw)

    stock_data = {}
    for ticker, df in stock_data_raw.items():
        try:
            stock_data[ticker] = compute_all_features(df)
        except Exception as e:
            logger.warning(f"Feature computation failed for {ticker}: {e}")

    print(f"  Features computed for {len(stock_data)} tickers in {time.time()-t0:.1f}s")

    # --- Step 3: Run backtest ---
    print("\n[3/4] Running backtest...")
    t0 = time.time()

    config = BacktestConfig(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        max_positions=max_positions,
    )
    engine = BacktestEngine(config)
    metrics = engine.run(stock_data, spy)

    print(f"  Backtest completed in {time.time()-t0:.1f}s")

    # --- Step 4: Report ---
    print("\n[4/4] Results")
    engine.print_summary(metrics)

    # Save results
    if save_results:
        reports_dir = Path(__file__).resolve().parent.parent.parent / "reports" / "backtests"
        reports_dir.mkdir(parents=True, exist_ok=True)

        eq = engine.get_equity_curve()
        trades = engine.get_trade_log()

        if not eq.empty:
            eq.to_csv(reports_dir / "equity_curve.csv", index=False)
            print(f"  Equity curve saved to reports/backtests/equity_curve.csv")

        if not trades.empty:
            trades.to_csv(reports_dir / "trade_log.csv", index=False)
            print(f"  Trade log saved to reports/backtests/trade_log.csv")

        # Save metrics summary
        with open(reports_dir / "metrics.txt", "w") as f:
            f.write("BACKTEST METRICS\n")
            f.write("=" * 40 + "\n")
            for k, v in metrics.__dict__.items():
                f.write(f"{k}: {v}\n")
        print(f"  Metrics saved to reports/backtests/metrics.txt")

    # Exit reason breakdown
    if engine.closed_trades:
        print("\n  Exit Reason Breakdown:")
        reasons = {}
        for t in engine.closed_trades:
            r = t.exit_reason.value
            reasons[r] = reasons.get(r, 0) + 1
        for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r:25s} {count:4d} ({count/len(engine.closed_trades)*100:.1f}%)")

    return metrics, engine


def main():
    parser = argparse.ArgumentParser(description="Hedge Fund OS — Breakout Backtest")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--start", type=str, default="2012-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--max-positions", type=int, default=10)
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None

    run_backtest(
        tickers=tickers,
        start=args.start,
        end=args.end,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_positions=args.max_positions,
    )


if __name__ == "__main__":
    main()
