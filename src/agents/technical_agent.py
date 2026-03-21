"""
Technical Agent — Identifies technically favorable setups.

Allowed: OHLCV, technical indicators, volatility features
Forbidden: fundamentals, news, sentiment, portfolio PnL
"""

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.schemas.signals import AgentOutput, Direction


class TechnicalAgent(BaseAgent):
    name = "technical_agent"
    allowed_inputs = ["ohlcv", "indicators", "volatility", "base_features", "breakout_features"]
    forbidden_inputs = ["fundamentals", "news", "sentiment", "portfolio_pnl", "portfolio_state"]

    def analyze(self, data: dict) -> AgentOutput:
        """
        Analyze a single stock for breakout setup quality.

        Expected data keys:
        - ticker: str
        - row: current day's feature row (pd.Series)
        - recent: last N days of data (pd.DataFrame)
        """
        ticker = data["ticker"]
        row = data["row"]
        recent = data.get("recent", pd.DataFrame())

        evidence = []
        risks = []
        invalidation = []
        confidence = 0.0

        # --- Trend assessment ---
        trend_score = 0
        if row.get("trend_close_above_150", False):
            trend_score += 1
            evidence.append("close above 150-day MA")
        if row.get("trend_150_above_200", False):
            trend_score += 1
            evidence.append("150-day MA above 200-day MA")
        if row.get("trend_200_rising", False):
            trend_score += 1
            evidence.append("200-day MA rising")
        if row.get("trend_above_52w_mid", False):
            trend_score += 1

        # --- Base quality ---
        base_score = 0
        if row.get("in_base", False):
            base_score += 1
            evidence.append(f"base depth: {row.get('base_depth_pct', 0)*100:.1f}%")
        if row.get("atr_compression", False):
            base_score += 1
            evidence.append("ATR compression present")

        # --- Breakout quality ---
        breakout_score = 0
        vol_ratio = row.get("vol_ratio", 0)
        if not pd.isna(vol_ratio) and vol_ratio >= 1.5:
            breakout_score += 1
            evidence.append(f"volume {vol_ratio:.1f}x average")
        close_range = row.get("close_range_pct", 0)
        if not pd.isna(close_range) and close_range >= 0.25:
            breakout_score += 1
            evidence.append(f"close in top {(1-close_range)*100:.0f}% of range")

        # --- Risk factors ---
        atr_pct = row.get("atr_pct", 0)
        if not pd.isna(atr_pct) and atr_pct > 0.04:
            risks.append(f"high volatility (ATR {atr_pct*100:.1f}%)")

        # Check if extended from MAs
        if row.get("ma_50", 0) > 0:
            extension = (row["close"] - row["ma_50"]) / row["ma_50"]
            if extension > 0.15:
                risks.append(f"extended {extension*100:.0f}% above 50-day MA")

        # --- Confidence scoring ---
        total_score = trend_score + base_score + breakout_score
        max_score = 8
        confidence = min(total_score / max_score, 1.0)

        # --- Direction ---
        if confidence >= 0.5 and trend_score >= 3 and breakout_score >= 1:
            direction = Direction.LONG
        else:
            direction = Direction.FLAT

        # --- Invalidation ---
        invalidation.append("close back below breakout level within 5 days")
        invalidation.append("200-day MA turns negative")
        if row.get("base_low"):
            invalidation.append(f"price below base low ({row['base_low']:.2f})")

        # --- Check memory for past performance on this ticker ---
        memory_notes = []
        for mem in self.memory[-10:]:
            if mem.get("ticker") == ticker:
                memory_notes.append(
                    f"past {mem.get('setup_type', 'setup')} on {mem.get('date', '?')}: "
                    f"{mem.get('outcome', '?')}"
                )

        return AgentOutput(
            agent=self.name,
            ticker=ticker,
            timestamp=str(row.name) if hasattr(row, "name") else "",
            thesis=f"Breakout from {row.get('base_length', '?')}-day base with trend confirmation",
            direction=direction,
            confidence=round(confidence, 2),
            evidence=evidence,
            risks=risks,
            invalidation=invalidation,
            recommendation=direction,
            memory_used=memory_notes,
        )
