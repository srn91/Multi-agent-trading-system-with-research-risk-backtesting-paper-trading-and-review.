"""
Enhanced features — v3 additions.
Adds relative strength, momentum ranking, pullback detection,
and chandelier exit levels.
"""

import numpy as np
import pandas as pd


def add_relative_strength(stock_data: dict[str, pd.DataFrame], lookback: int = 126) -> dict[str, pd.DataFrame]:
    """
    Compute relative strength ranking across the universe.
    Ranks each stock by its N-day return vs all other stocks.
    Only stocks in the top percentile get traded.

    Args:
        stock_data: dict of {ticker: DataFrame} with 'close' column
        lookback: days for momentum calculation (126 = ~6 months)

    Returns:
        Same dict with 'rs_rank' column added (0.0 = worst, 1.0 = best)
    """
    # Collect all momentum scores by date
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    for dt in all_dates:
        scores = {}
        for ticker, df in stock_data.items():
            if dt not in df.index:
                continue
            loc = df.index.get_loc(dt)
            if loc < lookback:
                continue
            ret = (df["close"].iloc[loc] / df["close"].iloc[loc - lookback]) - 1
            if not np.isnan(ret):
                scores[ticker] = ret

        if len(scores) < 5:
            continue

        # Rank: 1.0 = best momentum, 0.0 = worst
        sorted_tickers = sorted(scores.keys(), key=lambda t: scores[t])
        n = len(sorted_tickers)
        for rank, ticker in enumerate(sorted_tickers):
            if dt in stock_data[ticker].index:
                stock_data[ticker].loc[dt, "rs_rank"] = rank / (n - 1) if n > 1 else 0.5
                stock_data[ticker].loc[dt, "rs_return"] = scores[ticker]

    # Fill NaN for early dates
    for ticker, df in stock_data.items():
        if "rs_rank" not in df.columns:
            df["rs_rank"] = np.nan
        if "rs_return" not in df.columns:
            df["rs_return"] = np.nan

    return stock_data


def add_chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """
    Chandelier Exit: trailing stop based on highest high minus ATR multiple.
    Much better than a crude N-day low trailing stop.

    Long chandelier = highest_high(N) - mult * ATR(N)
    """
    df = df.copy()
    df["highest_high_22"] = df["high"].rolling(period).max()
    atr = df.get("atr")
    if atr is not None:
        df["chandelier_exit"] = df["highest_high_22"] - mult * atr
    else:
        # Compute ATR inline
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_calc = tr.rolling(period).mean()
        df["chandelier_exit"] = df["highest_high_22"] - mult * atr_calc
    return df


def add_pullback_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect pullback-to-moving-average setups.
    This is the SECOND strategy: mean-reversion within an uptrend.

    A valid pullback:
    - Stock is in uptrend (above 150/200 MA, trend_valid)
    - Price pulls back to touch or approach the 20 or 50-day MA
    - RSI dips below 40 then recovers above 40
    - Volume dries up during pullback (lower than average)
    """
    df = df.copy()

    # Distance from key MAs
    if "ma_20" in df.columns:
        df["dist_from_20ma"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    if "ma_50" in df.columns:
        df["dist_from_50ma"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # RSI recovery: was below 40, now above 40
    df["rsi_was_below_40"] = df["rsi"].rolling(5).min() < 40
    df["rsi_recovering"] = (df["rsi"] > 40) & df["rsi_was_below_40"]

    # Pullback detection: close within 2% of 20-day MA while in uptrend
    df["near_20ma"] = df["dist_from_20ma"].abs() < 0.02 if "dist_from_20ma" in df.columns else False
    df["near_50ma"] = df["dist_from_50ma"].abs() < 0.03 if "dist_from_50ma" in df.columns else False

    # Volume drying up during pullback
    df["low_volume"] = df.get("vol_ratio", 1.0) < 0.8

    # Valid pullback setup
    df["pullback_setup"] = (
        df.get("trend_valid", False)
        & (df["near_20ma"] | df["near_50ma"])
        & df["rsi_recovering"]
    )

    return df


def add_consecutive_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Require 2 consecutive closes above the breakout level.
    Reduces false breakouts significantly.
    """
    df = df.copy()
    if "breakout" in df.columns:
        df["breakout_confirmed"] = df["breakout"] & df["breakout"].shift(1)
    else:
        df["breakout_confirmed"] = False
    return df


def compute_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run enhanced feature pipeline on a single stock."""
    from src.data.features import compute_all_features
    df = compute_all_features(df)
    df = add_chandelier_exit(df)
    df = add_pullback_features(df)
    df = add_consecutive_breakout(df)
    return df
