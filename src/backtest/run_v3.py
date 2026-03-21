"""
V3 Backtest Runner — Agent-Integrated, Dual Strategy

Usage:
    python -m src.backtest.run_v3
"""

import logging
import time

from src.data.ingest import download_ohlcv, get_spy, DEFAULT_UNIVERSE
from src.data.enhanced_features import (
    compute_enhanced_features,
    add_relative_strength,
)
from src.data.features import compute_spy_features
from src.backtest.engine_v3 import BacktestEngineV3, V3Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_v3_backtest(
    tickers: list[str] | None = None,
    start: str = "2012-01-01",
    end: str | None = None,
):
    if tickers is None:
        tickers = DEFAULT_UNIVERSE[:60]

    print(f"\n{'='*60}")
    print(f"  HEDGE FUND OS — V3 BACKTEST")
    print(f"  Agent-Integrated | Dual Strategy | Relative Strength")
    print(f"{'='*60}")
    print(f"  Universe:  {len(tickers)} stocks")
    print(f"  Period:    {start} → {end or 'latest'}")
    print(f"  Strategies: Breakout + Pullback-to-MA")
    print(f"  Agents:    Regime + Technical + Risk (3 active)")
    print(f"{'='*60}\n")

    # Step 1: Download
    print("[1/5] Downloading market data...")
    t0 = time.time()
    spy_raw = get_spy(start=start, end=end)
    if spy_raw.empty:
        print("ERROR: No SPY data.")
        return

    stock_data_raw = {}
    for i, ticker in enumerate(tickers):
        try:
            df = download_ohlcv(ticker, start=start, end=end)
            if not df.empty and len(df) > 252:
                stock_data_raw[ticker] = df
        except:
            pass
        if (i + 1) % 20 == 0:
            print(f"  Downloaded {i+1}/{len(tickers)}...")
    print(f"  Got {len(stock_data_raw)}/{len(tickers)} tickers in {time.time()-t0:.1f}s")

    # Step 2: Compute enhanced features
    print("\n[2/5] Computing enhanced features...")
    t0 = time.time()
    spy = compute_spy_features(spy_raw)

    stock_data = {}
    for ticker, df in stock_data_raw.items():
        try:
            stock_data[ticker] = compute_enhanced_features(df)
        except Exception as e:
            logger.warning(f"Failed: {ticker}: {e}")
    print(f"  Features for {len(stock_data)} tickers in {time.time()-t0:.1f}s")

    # Step 3: Compute relative strength rankings
    print("\n[3/5] Computing relative strength rankings...")
    t0 = time.time()
    stock_data = add_relative_strength(stock_data, lookback=126)
    print(f"  RS rankings computed in {time.time()-t0:.1f}s")

    # Step 4: Run v3 backtest
    print("\n[4/5] Running V3 backtest with 3 agents...")
    t0 = time.time()

    config = V3Config(
        initial_capital=100_000,
        risk_per_trade=0.01,
        max_positions=10,
        min_rs_rank=0.70,
        base_max_depth=0.10,
        min_volume_ratio=1.5,
        require_confirmed_breakout=True,
        enable_pullback=True,
        use_chandelier_exit=True,
        failed_breakout_days=5,
        use_agents=True,
    )

    engine = BacktestEngineV3(config)
    metrics = engine.run(stock_data, spy)
    print(f"  Completed in {time.time()-t0:.1f}s")

    # Step 5: Report
    print("\n[5/5] Results")
    engine.print_summary(metrics)

    # Exit reason breakdown
    if engine.closed_trades:
        print("  Exit Reason Breakdown:")
        reasons = {}
        for t in engine.closed_trades:
            r = t.exit_reason.value
            reasons[r] = reasons.get(r, 0) + 1
        for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r:25s} {count:4d} ({count/len(engine.closed_trades)*100:.1f}%)")

    # Save
    from pathlib import Path
    reports_dir = Path(__file__).resolve().parent.parent.parent / "reports" / "backtests"
    reports_dir.mkdir(parents=True, exist_ok=True)

    eq = engine.get_equity_curve()
    trades = engine.get_trade_log()
    if not eq.empty:
        eq.to_csv(reports_dir / "equity_curve_v3.csv", index=False)
    if not trades.empty:
        trades.to_csv(reports_dir / "trade_log_v3.csv", index=False)
    with open(reports_dir / "metrics_v3.txt", "w") as f:
        for k, v in metrics.__dict__.items():
            f.write(f"{k}: {v}\n")
    print(f"\n  Results saved to reports/backtests/")

    return metrics, engine


if __name__ == "__main__":
    run_v3_backtest()
