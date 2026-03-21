"""
Enhanced Feature Engineering — V3.

New features that v1/v2 didn't have:
- Relative strength vs SPY
- Momentum ranking score
- Pullback-to-MA detection (mean reversion signal)
- Multi-day breakout confirmation
- Volatility regime classification
- Tighter base quality scoring
"""

import numpy as np
import pandas as pd
from src.data.features import compute_all_features, compute_spy_features


def add_relative_strength(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Relative strength vs SPY.
    Stocks outperforming the market have higher probability of continued outperformance.
    This is the single most important filter missing from v1.
    """
    df = df.copy()

    # Align dates
    common_dates = df.index.intersection(spy_df.index)
    if len(common_dates) < 50:
        df["rs_50"] = np.nan
        df["rs_120"] = np.nan
        df["rs_rank_score"] = 0.0
        return df

    spy_close = spy_df.loc[common_dates, "close"]
    stock_close = df.loc[common_dates, "close"]

    # Relative strength = stock return / SPY return over lookback
    for period, col in [(50, "rs_50"), (120, "rs_120")]:
        stock_ret = stock_close.pct_change(period)
        spy_ret = spy_close.pct_change(period)
        # RS > 1 means outperforming SPY
        rs = (1 + stock_ret) / (1 + spy_ret)
        df.loc[common_dates, col] = rs

    # Composite RS score (0-100 scale)
    rs50 = df["rs_50"].rank(pct=True) * 100
    rs120 = df["rs_120"].rank(pct=True) * 100
    df["rs_rank_score"] = (rs50 * 0.6 + rs120 * 0.4).fillna(0)

    return df


def add_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-timeframe momentum composite.
    Combines short, medium, and long momentum into one score.
    """
    df = df.copy()

    # Returns over multiple lookbacks
    df["ret_20"] = df["close"].pct_change(20)
    df["ret_50"] = df["close"].pct_change(50)
    df["ret_120"] = df["close"].pct_change(120)
    df["ret_250"] = df["close"].pct_change(250)

    # Skip most recent 5 days (mean reversion noise)
    df["ret_20_skip5"] = df["close"].shift(5).pct_change(15)

    # Momentum score: weighted combination
    mom = (
        df["ret_20_skip5"].rank(pct=True) * 0.2
        + df["ret_50"].rank(pct=True) * 0.3
        + df["ret_120"].rank(pct=True) * 0.3
        + df["ret_250"].rank(pct=True) * 0.2
    )
    df["momentum_score"] = (mom * 100).fillna(0)

    return df


def add_pullback_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean reversion pullback signal.
    Detects when a stock in a strong uptrend pulls back to support.
    This is a SECOND strategy type — complements the breakout.

    Logic:
    - Stock is in uptrend (above 150 MA, 150 > 200)
    - Price pulls back to or near the 20-day or 50-day MA
    - RSI drops below 40 (oversold in context of uptrend)
    - Volume dries up during pullback
    - Then a reversal candle appears (close > open, close in upper half)
    """
    df = df.copy()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Distance from key MAs
    df["dist_from_20ma"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["dist_from_50ma"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    # Pullback to MA zone: within 2% of 20-day or 50-day MA
    near_20ma = df["dist_from_20ma"].abs() < 0.02
    near_50ma = df["dist_from_50ma"].abs() < 0.03

    # In uptrend
    uptrend = (
        (df["close"] > df["ma_150"])
        & (df["ma_150"] > df["ma_200"])
    )

    # Oversold in context
    oversold = df["rsi_14"] < 40

    # Volume drying up (below average)
    low_vol = df["vol_ratio"] < 0.8

    # Reversal candle
    reversal = (df["close"] > df["open"]) & (df["close_range_pct"] > 0.6)

    # Pullback signal
    df["pullback_signal"] = (
        uptrend
        & (near_20ma | near_50ma)
        & (oversold | low_vol)
        & reversal
    )

    return df


def add_breakout_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score breakout quality 0-100 instead of binary pass/fail.
    Higher score = higher probability of follow-through.
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    # Volume strength (0-25 points)
    vol = df.get("vol_ratio", pd.Series(0, index=df.index))
    score += np.clip((vol - 1.0) * 25, 0, 25)

    # Close position in range (0-20 points)
    crp = df.get("close_range_pct", pd.Series(0, index=df.index))
    score += np.clip(crp * 20, 0, 20)

    # ATR compression (0-15 points)
    if "atr_compression" in df.columns:
        score += df["atr_compression"].astype(float) * 15

    # Base tightness (0-20 points) — tighter = better
    depth = df.get("base_depth_pct", pd.Series(0.15, index=df.index))
    score += np.clip((0.15 - depth) / 0.15 * 20, 0, 20)

    # Trend strength (0-20 points)
    trend_count = (
        df.get("trend_close_above_150", pd.Series(False, index=df.index)).astype(float)
        + df.get("trend_150_above_200", pd.Series(False, index=df.index)).astype(float)
        + df.get("trend_200_rising", pd.Series(False, index=df.index)).astype(float)
        + df.get("trend_above_52w_mid", pd.Series(False, index=df.index)).astype(float)
    )
    score += trend_count * 5

    df["breakout_quality"] = score.clip(0, 100)

    return df


def add_multiday_breakout_confirm(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Multi-day breakout confirmation.
    Instead of single-day breakout, require price to HOLD above breakout level
    for 2 consecutive closes. Reduces false breakouts significantly.
    """
    df = df.copy()
    prior_high = df["high"].shift(1).rolling(lookback).max()

    # Day 1: close above prior high
    day1_breakout = df["close"] > prior_high
    # Day 2: previous day was also a breakout close
    day2_confirm = day1_breakout & day1_breakout.shift(1).fillna(False)

    df["breakout_confirmed"] = day2_confirm

    return df


def compute_v3_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full v3 feature pipeline for a single stock."""
    # Start with v1 features
    df = compute_all_features(df)

    # Add v3 enhancements
    if spy_df is not None:
        df = add_relative_strength(df, spy_df)
    else:
        df["rs_rank_score"] = 50.0

    df = add_momentum_score(df)
    df = add_pullback_signal(df)
    df = add_breakout_quality_score(df)
    df = add_multiday_breakout_confirm(df)

    return df
