"""
V4 Features — Everything V3 had plus:
- Momentum rotation scoring across universe
- Sector ETF momentum for rotation strategy
- Enhanced regime detection (VIX-based, breadth, trend strength)
- Volatility targeting features
"""

import numpy as np
import pandas as pd
from src.data.features import compute_all_features, compute_spy_features
from src.data.features_v3 import (
    add_relative_strength,
    add_momentum_score,
    add_pullback_signal,
    add_breakout_quality_score,
    add_multiday_breakout_confirm,
)

pd.set_option('future.no_silent_downcasting', True)


# === SECTOR ETFs for Momentum Rotation ===
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication",
}

# Map tickers to sectors (approximate)
TICKER_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLC", "AMZN": "XLY", "NVDA": "XLK",
    "META": "XLC", "TSLA": "XLY", "JPM": "XLF", "V": "XLF", "UNH": "XLV",
    "HD": "XLY", "MA": "XLF", "NFLX": "XLC", "COST": "XLP", "ADBE": "XLK",
    "CRM": "XLK", "AMD": "XLK", "AVGO": "XLK", "LLY": "XLV", "ORCL": "XLK",
    "NOW": "XLK", "ISRG": "XLV", "GS": "XLF", "CAT": "XLI", "DE": "XLI",
    "BA": "XLI", "LOW": "XLY", "TGT": "XLY", "NKE": "XLY", "MCD": "XLY",
    "DHR": "XLV", "PG": "XLP", "JNJ": "XLV", "MRK": "XLV", "PEP": "XLP",
    "TMO": "XLV", "ABT": "XLV", "ACN": "XLK", "TXN": "XLK", "QCOM": "XLK",
    "INTC": "XLK", "CSCO": "XLK", "IBM": "XLK", "AMAT": "XLK", "BKNG": "XLY",
    "ADP": "XLK", "MDLZ": "XLP", "GILD": "XLV", "CME": "XLF", "BLK": "XLF",
    "SCHW": "XLF", "MMM": "XLI", "RTX": "XLI", "BRK-B": "XLF", "DIS": "XLC",
    "GE": "XLI", "HON": "XLI", "UPS": "XLI", "FDX": "XLI", "WMT": "XLP",
    "SBUX": "XLY", "YUM": "XLY", "CMG": "XLY", "SYK": "XLV", "ZTS": "XLV",
    "CL": "XLP", "KO": "XLP", "PFE": "XLV", "ABBV": "XLV", "BMY": "XLV",
    "AMGN": "XLV", "REGN": "XLV", "VRTX": "XLV", "MRNA": "XLV",
    "PANW": "XLK", "CRWD": "XLK", "SNOW": "XLK", "DDOG": "XLK", "ZS": "XLK",
    "NET": "XLK", "FTNT": "XLK", "MELI": "XLY", "SQ": "XLF", "PYPL": "XLF",
    "SHOP": "XLK", "ABNB": "XLY", "UBER": "XLY", "DASH": "XLY", "COIN": "XLF",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    "CEG": "XLU", "VST": "XLU", "ENPH": "XLK", "SEDG": "XLK", "FSLR": "XLK",
}


def add_vol_targeting_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features needed for volatility targeting."""
    df = df.copy()
    # Realized vol (annualized)
    df["realized_vol_10"] = df["close"].pct_change().rolling(10).std() * np.sqrt(252)
    df["realized_vol_20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
    df["realized_vol_60"] = df["close"].pct_change().rolling(60).std() * np.sqrt(252)

    # Vol ratio: current vs long-term (>1 = elevated vol)
    df["vol_regime_ratio"] = df["realized_vol_20"] / df["realized_vol_60"]
    df["vol_regime_ratio"] = df["vol_regime_ratio"].fillna(1.0)

    return df


def add_enhanced_regime(spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced regime classification.
    Uses multiple signals, not just SPY > 200MA.
    """
    df = spy_df.copy()

    # Trend layers
    df["spy_above_200"] = df["close"] > df.get("ma_200", df["close"])
    df["spy_above_50"] = df["close"] > df.get("ma_50", df["close"])
    df["spy_above_20"] = df["close"] > df.get("ma_20", df["close"])

    # Momentum
    df["spy_mom_20"] = df["close"].pct_change(20)
    df["spy_mom_60"] = df["close"].pct_change(60)

    # Breadth proxy: % of days up in last 20
    df["spy_up_days_20"] = df["close"].diff().rolling(20).apply(lambda x: (x > 0).sum() / len(x))

    # Realized vol
    df["spy_vol_20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

    # Regime score: 0 (hostile) to 100 (perfect)
    score = pd.Series(50.0, index=df.index)
    score += df["spy_above_200"].astype(float) * 15
    score += df["spy_above_50"].astype(float) * 10
    score += df["spy_above_20"].astype(float) * 5
    score += np.clip(df["spy_mom_20"] * 200, -15, 15)
    score += np.clip(df["spy_mom_60"] * 100, -10, 10)
    score -= np.clip((df["spy_vol_20"] - 0.15) * 100, 0, 20)  # penalize high vol
    score += np.clip((df["spy_up_days_20"] - 0.5) * 30, -10, 10)

    df["regime_score"] = score.clip(0, 100)

    # Regime labels
    df["regime_label"] = "neutral"
    df.loc[df["regime_score"] >= 70, "regime_label"] = "strong_bull"
    df.loc[(df["regime_score"] >= 55) & (df["regime_score"] < 70), "regime_label"] = "mild_bull"
    df.loc[(df["regime_score"] >= 35) & (df["regime_score"] < 55), "regime_label"] = "choppy"
    df.loc[(df["regime_score"] >= 20) & (df["regime_score"] < 35), "regime_label"] = "mild_bear"
    df.loc[df["regime_score"] < 20, "regime_label"] = "crisis"

    # Throttle factor: how much of normal risk to take
    df["regime_throttle"] = np.clip(df["regime_score"] / 70, 0.0, 1.0)

    return df


def add_momentum_rotation_rank(all_stocks: dict, spy_df: pd.DataFrame) -> dict:
    """
    Cross-sectional momentum ranking across the entire universe.
    Rank stocks by composite momentum. Top quintile gets priority.
    """
    # Collect momentum scores for all stocks on common dates
    common_dates = spy_df.index

    for ticker, df in all_stocks.items():
        if "momentum_score" in df.columns:
            # Already ranked within-stock, now we rank across stocks
            pass

    # For each date, rank all stocks by momentum score
    # We do this after feature computation, in the engine
    return all_stocks


def compute_v4_features(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> pd.DataFrame:
    """Full V4 feature pipeline."""
    # V1 base features
    df = compute_all_features(df)

    # V3 features
    if spy_df is not None:
        df = add_relative_strength(df, spy_df)
    else:
        df["rs_rank_score"] = 50.0

    df = add_momentum_score(df)
    df = add_pullback_signal(df)
    df = add_breakout_quality_score(df)
    df = add_multiday_breakout_confirm(df)

    # V4 new features
    df = add_vol_targeting_features(df)

    return df


def compute_v4_spy(spy_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced SPY features for V4."""
    spy = compute_spy_features(spy_raw)
    spy = add_enhanced_regime(spy)
    spy = add_vol_targeting_features(spy)
    return spy
