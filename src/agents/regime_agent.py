"""
Regime Agent — Classifies current market state.

Allowed: Broad market indices, volatility, breadth, rates
Forbidden: Individual stock signals, portfolio positions
"""

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.schemas.signals import AgentOutput, Direction, Regime


class RegimeAgent(BaseAgent):
    name = "regime_agent"
    allowed_inputs = ["spy_data", "vix", "breadth", "rates"]
    forbidden_inputs = ["stock_signals", "portfolio_positions", "individual_stock_data"]

    def classify_regime(self, spy_row: pd.Series) -> Regime:
        """Classify market regime from SPY features."""
        above_200 = spy_row.get("spy_above_200", False)
        above_50 = spy_row.get("spy_above_50", False)
        vol = spy_row.get("realized_vol_20", 0)
        mom_20 = spy_row.get("momentum_20", 0)
        mom_60 = spy_row.get("momentum_60", 0)

        if pd.isna(vol):
            vol = 0.15

        # Crisis: high vol + falling
        if vol > 0.30 and mom_20 < -0.05:
            return Regime.HIGH_VOL_CRISIS

        # Trending down
        if not above_200 and mom_60 < -0.05:
            return Regime.TRENDING_DOWN

        # Choppy: below 50 but above 200, or flat momentum
        if above_200 and not above_50 and abs(mom_20) < 0.02:
            return Regime.CHOPPY

        # Low vol grind
        if vol < 0.10 and above_200:
            return Regime.LOW_VOL_GRIND

        # Trending up
        if above_200 and above_50 and mom_20 > 0:
            return Regime.TRENDING_UP

        return Regime.UNKNOWN

    def analyze(self, data: dict) -> AgentOutput:
        spy_row = data["spy_row"]
        regime = self.classify_regime(spy_row)

        evidence = []
        if spy_row.get("spy_above_200", False):
            evidence.append("SPY above 200-day MA")
        if spy_row.get("spy_above_50", False):
            evidence.append("SPY above 50-day MA")

        vol = spy_row.get("realized_vol_20", 0)
        if not pd.isna(vol):
            evidence.append(f"20-day realized vol: {vol*100:.1f}%")

        # Regime determines if trading is allowed
        hostile = regime in (Regime.HIGH_VOL_CRISIS, Regime.TRENDING_DOWN)
        direction = Direction.FLAT if hostile else Direction.LONG

        confidence = 0.8 if regime != Regime.UNKNOWN else 0.3

        return AgentOutput(
            agent=self.name,
            ticker="SPY",
            timestamp=str(spy_row.name) if hasattr(spy_row, "name") else "",
            thesis=f"Market regime: {regime.value}",
            direction=direction,
            confidence=confidence,
            evidence=evidence,
            risks=["regime can shift rapidly on macro shocks"],
            invalidation=["regime classification changes"],
            recommendation=direction,
        )
