"""
Daily Signal Generator — What to buy/sell today.

Run: python3 generate_signals.py

Generates today's momentum rankings and trade recommendations
using the V6 agent team scoring system.
"""
import sys, os, json, logging
from datetime import datetime
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from src.data.ingest import download_ohlcv, get_spy, DEFAULT_UNIVERSE
from src.data.features_v4 import compute_v4_features, compute_v4_spy
from src.backtest.engine_v6 import AgentTeam

TICKERS = DEFAULT_UNIVERSE[:100]


def main():
    print("\n" + "=" * 65)
    print("  HEDGE FUND OS — DAILY SIGNAL GENERATOR")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # Download latest data
    print("\nDownloading latest data...")
    spy_raw = get_spy(start="2024-01-01")
    spy = compute_v4_spy(spy_raw)

    stock_data = {}
    for t in TICKERS:
        try:
            df = download_ohlcv(t, start="2024-01-01", force=True)
            if not df.empty and len(df) > 60:
                stock_data[t] = compute_v4_features(df, spy)
        except:
            pass

    print(f"Got {len(stock_data)} tickers")

    # Get latest date
    dt = spy.index[-1]
    spy_row = spy.loc[dt]
    print(f"\nLatest data: {dt.date()}")

    # Regime check
    regime_score = spy_row.get("regime_score", 50)
    regime_label = spy_row.get("regime_label", "unknown")
    print(f"Market regime: {regime_label} (score: {regime_score:.0f}/100)")

    # Rank universe by momentum
    rankings = []
    for ticker, df in stock_data.items():
        if dt not in df.index:
            continue
        loc = df.index.get_loc(dt)
        if loc < 68:
            continue

        # 3-month momentum, skip 5 days
        price_now = df.iloc[loc - 5]["close"]
        price_past = df.iloc[loc - 63]["close"]
        if price_past <= 0:
            continue
        momentum = (price_now / price_past) - 1

        current = df.iloc[loc]["close"]
        if current < 10:
            continue

        ma200 = df.iloc[loc].get("ma_200", 0)
        if pd.isna(ma200) or current < ma200:
            continue

        row = df.iloc[loc]
        rankings.append({
            "ticker": ticker,
            "price": round(current, 2),
            "momentum_3m": round(momentum * 100, 1),
            "rs_score": round(row.get("rs_rank_score", 50), 1),
            "vol_20d": round(row.get("realized_vol_20", 0.2) * 100, 1),
            "above_200ma": True,
            "row": row,
        })

    rankings.sort(key=lambda x: x["momentum_3m"], reverse=True)

    # Agent scoring for top 10
    agents = AgentTeam()
    print("\n" + "=" * 65)
    print("  TOP 10 MOMENTUM STOCKS — AGENT SCORES")
    print("=" * 65)

    fmt = "{:<6s} {:>8s} {:>8s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>9s}"
    print(fmt.format("Ticker", "Price", "Mom 3m", "MomA", "TrdA", "RskA", "RegA", "Conv", "Signal"))
    print("-" * 80)

    signals = []
    for i, r in enumerate(rankings[:10]):
        row = r["row"]
        conviction, scores, evidence = agents.score_entry(
            r["ticker"], row, i, len(rankings), spy_row
        )

        signal = "STRONG BUY" if conviction >= 0.7 else "BUY" if conviction >= 0.4 else "WATCH"

        print(fmt.format(
            r["ticker"], f"${r['price']}", f"{r['momentum_3m']:+.1f}%",
            f"{scores['momentum']:.2f}", f"{scores['trend']:.2f}",
            f"{scores['risk']:.2f}", f"{scores['regime']:.2f}",
            f"{conviction:.2f}", signal,
        ))

        signals.append({
            "rank": i + 1,
            "ticker": r["ticker"],
            "price": r["price"],
            "momentum_3m": r["momentum_3m"],
            "conviction": round(conviction, 3),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "signal": signal,
            "evidence": evidence,
        })

    # Full rankings
    print(f"\n  Full momentum rankings: {len(rankings)} qualifying stocks")
    print(f"  Top 5 to HOLD: {', '.join(r['ticker'] for r in rankings[:5])}")

    # Save signals
    output = {
        "date": str(dt.date()),
        "regime": {"score": round(regime_score, 1), "label": regime_label},
        "top_signals": signals,
        "full_ranking_count": len(rankings),
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reports", "backtests", "daily_signals.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Signals saved to: reports/backtests/daily_signals.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
