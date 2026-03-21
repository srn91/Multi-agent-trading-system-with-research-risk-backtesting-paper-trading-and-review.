"""
Alpaca Paper Trading Integration.

Setup:
1. Go to https://app.alpaca.markets/signup
2. Create a FREE account (paper trading is free)
3. Go to API Keys → Generate New Key
4. Copy API Key and Secret Key
5. Create a file called .env in the project root:
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets

Run: python3 paper_trade.py
"""
import sys, os, json, logging, time
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# Try to load Alpaca
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-trade-api not installed. Run: pip3 install alpaca-trade-api")

from src.data.ingest import download_ohlcv, get_spy, DEFAULT_UNIVERSE
from src.data.features_v4 import compute_v4_features, compute_v4_spy, TICKER_SECTOR
from src.backtest.engine_v6 import AgentTeam

# ============================================================
# CONFIGURATION
# ============================================================
UNIVERSE = DEFAULT_UNIVERSE[:100]
TOP_N = 7
MAX_SECTOR = 2
MOMENTUM_LOOKBACK = 63
MOMENTUM_SKIP = 5
VOL_TARGET = 0.15
TRAIL_STOP_PCT = 0.15
HARD_STOP_PCT = 0.20
W_MOM = 0.50
W_QUAL = 0.25
W_VOL = 0.25
MIN_CONVICTION = 0.40
SELL_THRESHOLD = 14

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "paper_trading")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_env():
    """Load API keys from .env file."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


def get_alpaca_client():
    """Connect to Alpaca paper trading API."""
    env = load_env()
    api_key = env.get("ALPACA_API_KEY", os.environ.get("ALPACA_API_KEY", ""))
    secret_key = env.get("ALPACA_SECRET_KEY", os.environ.get("ALPACA_SECRET_KEY", ""))
    base_url = env.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        return None

    try:
        api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        account = api.get_account()
        logger.info(f"Connected to Alpaca: ${float(account.equity):,.2f} equity, "
                     f"${float(account.buying_power):,.2f} buying power")
        return api
    except Exception as e:
        logger.error(f"Alpaca connection failed: {e}")
        return None


def generate_signals():
    """Generate today's trading signals using the QMB strategy."""
    logger.info("Downloading market data...")
    spy_raw = get_spy(start="2024-01-01", force=True)
    spy = compute_v4_spy(spy_raw)

    stock_data = {}
    for t in UNIVERSE:
        try:
            df = download_ohlcv(t, start="2024-01-01", force=True)
            if not df.empty and len(df) > 60:
                stock_data[t] = compute_v4_features(df, spy)
        except:
            pass

    dt = spy.index[-1]
    spy_row = spy.loc[dt]
    logger.info(f"Latest data: {dt.date()}, {len(stock_data)} stocks")

    # Regime check
    regime_score = spy_row.get("regime_score", 50)
    regime_label = spy_row.get("regime_label", "unknown")
    logger.info(f"Regime: {regime_label} (score: {regime_score:.0f}/100)")

    # Rank universe with multi-factor scoring
    rankings = []
    for ticker, df in stock_data.items():
        if dt not in df.index:
            continue
        loc = df.index.get_loc(dt)
        if loc < 260:
            continue

        row = df.iloc[loc]
        current = row["close"]
        if current < 10:
            continue

        ma200 = row.get("ma_200", 0)
        if pd.isna(ma200) or current < ma200:
            continue

        skip = MOMENTUM_SKIP
        m3 = (df.iloc[loc - skip]["close"] / df.iloc[loc - 63]["close"] - 1) if loc >= 63 + skip else 0
        m6 = (df.iloc[loc - skip]["close"] / df.iloc[loc - 126]["close"] - 1) if loc >= 126 + skip else 0
        m12 = (df.iloc[loc - skip]["close"] / df.iloc[loc - 252]["close"] - 1) if loc >= 252 + skip else 0
        mom = m3 * 0.4 + m6 * 0.35 + m12 * 0.25

        quality = 0
        if row.get("trend_close_above_150", False): quality += 0.25
        if row.get("trend_150_above_200", False): quality += 0.25
        if row.get("trend_200_rising", False): quality += 0.25
        rs = row.get("rs_rank_score", 50)
        if not pd.isna(rs): quality += min(rs / 100 * 0.25, 0.25)

        vol = row.get("realized_vol_20", 0.25)
        if pd.isna(vol) or vol <= 0: vol = 0.25

        rankings.append({
            "ticker": ticker, "price": round(current, 2),
            "momentum": mom, "quality": quality, "vol": vol,
        })

    if not rankings:
        logger.warning("No stocks qualify.")
        return [], regime_label, regime_score

    # Normalize and score
    moms = [r["momentum"] for r in rankings]
    quals = [r["quality"] for r in rankings]
    vols = [r["vol"] for r in rankings]

    for r in rankings:
        nm = (r["momentum"] - min(moms)) / (max(moms) - min(moms) + 1e-10)
        nq = (r["quality"] - min(quals)) / (max(quals) - min(quals) + 1e-10)
        nv = 1 - (r["vol"] - min(vols)) / (max(vols) - min(vols) + 1e-10)
        r["score"] = nm * W_MOM + nq * W_QUAL + nv * W_VOL

    rankings.sort(key=lambda x: x["score"], reverse=True)

    # Select top N with sector limits
    selected = []
    sector_count = defaultdict(int)
    for r in rankings:
        if len(selected) >= TOP_N:
            break
        sector = TICKER_SECTOR.get(r["ticker"], "XLK")
        if sector_count[sector] >= MAX_SECTOR:
            continue
        selected.append(r)
        sector_count[sector] += 1

    # Agent scoring
    agents = AgentTeam()
    for i, r in enumerate(selected):
        row = stock_data[r["ticker"]].iloc[-1]
        conv, scores, evidence = agents.score_entry(
            r["ticker"], row, i, len(rankings), spy_row
        )
        r["conviction"] = round(conv, 3)
        r["agent_scores"] = {k: round(v, 3) for k, v in scores.items()}
        r["signal"] = "STRONG BUY" if conv >= 0.7 else "BUY" if conv >= MIN_CONVICTION else "WATCH"

    return selected, regime_label, regime_score


def execute_paper_trades(api, signals, regime_label):
    """Execute paper trades on Alpaca."""
    if api is None:
        logger.warning("No Alpaca connection. Showing signals only.")
        return

    account = api.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)

    # Get current positions
    positions = api.list_positions()
    current_holdings = {p.symbol: {
        "qty": int(p.qty),
        "avg_price": float(p.avg_entry_price),
        "market_value": float(p.market_value),
        "unrealized_pnl": float(p.unrealized_pl),
        "pnl_pct": float(p.unrealized_plpc) * 100,
    } for p in positions}

    logger.info(f"Current positions: {len(current_holdings)}")
    for sym, pos in current_holdings.items():
        logger.info(f"  {sym}: {pos['qty']} shares, PnL: {pos['pnl_pct']:+.1f}%")

    target_tickers = set(s["ticker"] for s in signals if s["signal"] != "WATCH")
    held_tickers = set(current_holdings.keys())

    # SELL: positions not in target list
    for sym in held_tickers:
        if sym not in target_tickers:
            pos = current_holdings[sym]
            logger.info(f"SELL {sym}: {pos['qty']} shares (rotation out, PnL: {pos['pnl_pct']:+.1f}%)")
            try:
                api.submit_order(
                    symbol=sym,
                    qty=pos["qty"],
                    side="sell",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"  ✓ Sell order submitted for {sym}")
            except Exception as e:
                logger.error(f"  ✗ Failed to sell {sym}: {e}")

    # BUY: signals not currently held
    # Inverse-vol weighting
    to_buy = [s for s in signals if s["ticker"] not in held_tickers and s["signal"] != "WATCH"]
    if not to_buy:
        logger.info("No new buys needed.")
        return

    inv_vols = [(s, 1.0 / max(s["vol"], 0.05)) for s in to_buy]
    total_iv = sum(iv for _, iv in inv_vols)

    for s, iv in inv_vols:
        weight = (iv / total_iv) * min(buying_power / equity, 0.90)
        weight = min(weight, 0.18)  # max 18% per position
        target_value = equity * weight
        shares = int(target_value / s["price"])

        if shares <= 0:
            continue

        logger.info(f"BUY {s['ticker']}: {shares} shares @ ~${s['price']:.2f} "
                     f"(conv={s['conviction']:.2f}, signal={s['signal']})")
        try:
            api.submit_order(
                symbol=s["ticker"],
                qty=shares,
                side="buy",
                type="market",
                time_in_force="day",
            )
            logger.info(f"  ✓ Buy order submitted for {s['ticker']}")
        except Exception as e:
            logger.error(f"  ✗ Failed to buy {s['ticker']}: {e}")


def check_stops(api):
    """Check trailing stops on existing positions."""
    if api is None:
        return

    positions = api.list_positions()
    for pos in positions:
        pnl_pct = float(pos.unrealized_plpc)
        # Check if below hard stop
        if pnl_pct < -HARD_STOP_PCT:
            logger.warning(f"STOP LOSS: {pos.symbol} at {pnl_pct*100:+.1f}% — selling")
            try:
                api.submit_order(
                    symbol=pos.symbol,
                    qty=int(pos.qty),
                    side="sell",
                    type="market",
                    time_in_force="day",
                )
            except Exception as e:
                logger.error(f"Failed to stop-loss {pos.symbol}: {e}")


def save_paper_trade_log(signals, regime_label, regime_score, api=None):
    """Save paper trading log."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "regime": {"label": regime_label, "score": round(regime_score, 1)},
        "signals": signals,
    }

    if api:
        try:
            account = api.get_account()
            log_entry["account"] = {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
            }
            positions = api.list_positions()
            log_entry["positions"] = [{
                "symbol": p.symbol,
                "qty": int(p.qty),
                "avg_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "pnl_pct": float(p.unrealized_plpc) * 100,
            } for p in positions]
        except:
            pass

    # Append to daily log
    log_path = os.path.join(REPORTS_DIR, "paper_trade_log.json")
    existing = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                existing = json.load(f)
        except:
            existing = []
    existing.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info(f"Log saved to {log_path}")


def main():
    print("\n" + "=" * 65)
    print("  HEDGE FUND OS — PAPER TRADING")
    print("  QMB 7-Position | Multi-Factor | Agent-Scored")
    print("=" * 65)

    # Connect to Alpaca
    api = None
    if ALPACA_AVAILABLE:
        api = get_alpaca_client()
        if api is None:
            print("\n  ⚠ Alpaca not configured. Running in SIGNAL-ONLY mode.")
            print("  To enable paper trading:")
            print("    1. Sign up at https://app.alpaca.markets/signup (free)")
            print("    2. Generate API keys")
            print("    3. Create .env file with:")
            print("       ALPACA_API_KEY=your_key")
            print("       ALPACA_SECRET_KEY=your_secret")
            print("       ALPACA_BASE_URL=https://paper-api.alpaca.markets")
    else:
        print("\n  ⚠ alpaca-trade-api not installed.")
        print("  Run: pip3 install alpaca-trade-api")
        print("  Running in SIGNAL-ONLY mode.\n")

    # Generate signals
    print("\n[1/3] Generating signals...")
    signals, regime_label, regime_score = generate_signals()

    if not signals:
        print("  No signals generated. Market may be closed or no qualifying stocks.")
        return

    # Display signals
    print(f"\n  Regime: {regime_label} (score: {regime_score:.0f}/100)")
    print(f"\n  {'Rank':<6} {'Ticker':<8} {'Price':>8} {'Mom':>8} {'Score':>7} {'Conv':>7} {'Signal':<12}")
    print("  " + "-" * 65)
    for i, s in enumerate(signals):
        print(f"  {i+1:<6} {s['ticker']:<8} ${s['price']:>7.2f} {s['momentum']*100:>+7.1f}% "
              f"{s['score']:>6.3f} {s['conviction']:>6.3f} {s['signal']}")

    print(f"\n  Top {TOP_N} to hold: {', '.join(s['ticker'] for s in signals[:TOP_N])}")

    # Execute paper trades
    if api:
        print("\n[2/3] Executing paper trades...")
        check_stops(api)
        execute_paper_trades(api, signals, regime_label)
    else:
        print("\n[2/3] Skipping execution (no Alpaca connection)")

    # Save log
    print("\n[3/3] Saving trade log...")
    save_paper_trade_log(signals, regime_label, regime_score, api)

    print("\n" + "=" * 65)
    print("  PAPER TRADING COMPLETE")
    print("=" * 65)
    if api:
        account = api.get_account()
        print(f"  Account equity: ${float(account.equity):,.2f}")
        print(f"  Positions: {len(api.list_positions())}")
    print(f"  Signals: {len(signals)}")
    print(f"  Log: reports/paper_trading/paper_trade_log.json")
    print(f"\n  Run this daily to track paper trading performance.")
    print(f"  Set up a cron job: 0 10 * * 1-5 cd /path/to/hedge-fund-os && python3 paper_trade.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
