"""
Feature engineering module.
Computes all technical features needed by the breakout strategy and agents.
Pure pandas — no side effects, no API calls, no opinions.
"""

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add key moving averages."""
    df = df.copy()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_150"] = df["close"].rolling(150).mean()
    df["ma_200"] = df["close"].rolling(200).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df


def add_atr(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Average True Range."""
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(period).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features."""
    df = df.copy()
    df["vol_20_avg"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_20_avg"]
    df["dollar_volume"] = df["close"] * df["volume"]
    df["dollar_vol_20_avg"] = df["dollar_volume"].rolling(20).mean()
    return df


def add_breakout_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Add breakout detection features."""
    df = df.copy()

    # Rolling highs and lows
    df["high_20"] = df["high"].rolling(lookback).max()
    df["low_20"] = df["low"].rolling(lookback).min()

    # Close position within day's range (0 = low, 1 = high)
    day_range = df["high"] - df["low"]
    df["close_range_pct"] = np.where(
        day_range > 0,
        (df["close"] - df["low"]) / day_range,
        0.5,
    )

    # Breakout signal: close above prior 20-day high
    df["prior_high_20"] = df["high"].shift(1).rolling(lookback).max()
    df["breakout"] = df["close"] > df["prior_high_20"]

    return df


def add_base_features(df: pd.DataFrame, min_len: int = 15, max_len: int = 40) -> pd.DataFrame:
    """
    Detect consolidation bases.
    A base is a period where price compresses into a tight range.
    """
    df = df.copy()
    df["base_high"] = np.nan
    df["base_low"] = np.nan
    df["base_length"] = 0
    df["base_depth_pct"] = np.nan
    df["in_base"] = False
    df["atr_compression"] = False

    # ATR compression: current ATR% < ATR% from 20 days ago
    if "atr_pct" in df.columns:
        df["atr_compression"] = df["atr_pct"] < df["atr_pct"].shift(20)

    # Rolling base detection
    for window in [min_len, 20, 25, 30, max_len]:
        if window > len(df):
            continue
        col_h = f"_base_high_{window}"
        col_l = f"_base_low_{window}"
        df[col_h] = df["close"].rolling(window).max()
        df[col_l] = df["close"].rolling(window).min()
        depth = (df[col_h] - df[col_l]) / df[col_h]

        # Mark as valid base if depth <= 12%
        mask = (depth <= 0.12) & (depth > 0)
        # Update base fields where this is the tightest base found
        update_mask = mask & (
            df["base_depth_pct"].isna() | (depth < df["base_depth_pct"])
        )
        df.loc[update_mask, "base_high"] = df.loc[update_mask, col_h]
        df.loc[update_mask, "base_low"] = df.loc[update_mask, col_l]
        df.loc[update_mask, "base_length"] = window
        df.loc[update_mask, "base_depth_pct"] = depth[update_mask]
        df.loc[update_mask, "in_base"] = True

        df.drop(columns=[col_h, col_l], inplace=True)

    return df


def add_trend_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend filter conditions:
    1. Close > 150-day MA
    2. 150-day MA > 200-day MA
    3. 200-day MA rising over last 20 days
    4. Close above 52-week midpoint
    """
    df = df.copy()

    df["ma_200_slope"] = df["ma_200"] - df["ma_200"].shift(20)
    df["week_52_high"] = df["high"].rolling(252).max()
    df["week_52_low"] = df["low"].rolling(252).min()
    df["week_52_mid"] = (df["week_52_high"] + df["week_52_low"]) / 2

    df["trend_close_above_150"] = df["close"] > df["ma_150"]
    df["trend_150_above_200"] = df["ma_150"] > df["ma_200"]
    df["trend_200_rising"] = df["ma_200_slope"] > 0
    df["trend_above_52w_mid"] = df["close"] > df["week_52_mid"]

    df["trend_valid"] = (
        df["trend_close_above_150"]
        & df["trend_150_above_200"]
        & df["trend_200_rising"]
        & df["trend_above_52w_mid"]
    )

    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features used by the regime agent (applied to SPY or market index)."""
    df = df.copy()
    df["spy_above_200"] = df["close"] > df["ma_200"]
    df["spy_above_50"] = df["close"] > df["ma_50"]

    # Volatility regime
    df["realized_vol_20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

    # Breadth proxy (for single-stock this is just trend direction)
    df["momentum_20"] = df["close"].pct_change(20)
    df["momentum_60"] = df["close"].pct_change(60)

    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline on a single stock's OHLCV dataframe."""
    df = add_moving_averages(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_breakout_features(df)
    df = add_base_features(df)
    df = add_trend_filter(df)
    return df


def compute_spy_features(spy_df: pd.DataFrame) -> pd.DataFrame:
    """Compute features specifically for SPY (market regime)."""
    spy = add_moving_averages(spy_df)
    spy = add_atr(spy)
    spy = add_regime_features(spy)
    return spy
