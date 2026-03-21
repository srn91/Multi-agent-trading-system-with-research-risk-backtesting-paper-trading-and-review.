"""
Data Ops Agent — Collect, clean, validate, and transform market data.

This agent is plumbing, not prophecy.
It never gives trade opinions.

Allowed: Raw OHLCV, fundamentals, macro, metadata
Forbidden: Agent opinions, trade decisions, portfolio state
"""

import logging

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.schemas.signals import AgentOutput, Direction

logger = logging.getLogger(__name__)


class DataOpsAgent(BaseAgent):
    name = "data_ops_agent"
    allowed_inputs = ["raw_ohlcv", "fundamentals", "macro", "metadata"]
    forbidden_inputs = ["agent_opinions", "trade_decisions", "portfolio_state"]

    def analyze(self, data: dict) -> AgentOutput:
        """
        Validate data quality for a ticker.

        Expected data keys:
        - ticker: str
        - df: pd.DataFrame with OHLCV
        """
        ticker = data.get("ticker", "UNKNOWN")
        df = data.get("df", pd.DataFrame())

        evidence = []
        risks = []

        if df.empty:
            return AgentOutput(
                agent=self.name, ticker=ticker, timestamp="",
                thesis="No data available",
                direction=Direction.FLAT, confidence=0.0,
                evidence=["empty dataframe"], risks=["cannot proceed"],
                invalidation=[], recommendation=Direction.FLAT,
            )

        # Check row count
        evidence.append(f"{len(df)} trading days available")

        # Check for missing values
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            risks.append(f"{total_nulls} missing values across columns")
            for col, count in null_counts.items():
                if count > 0:
                    risks.append(f"  {col}: {count} nulls")

        # Check for zero volume days
        if "volume" in df.columns:
            zero_vol = (df["volume"] == 0).sum()
            if zero_vol > 0:
                risks.append(f"{zero_vol} zero-volume days")

        # Check data freshness
        last_date = df.index[-1]
        evidence.append(f"last data point: {last_date}")

        # Check minimum price
        if "close" in df.columns:
            min_price = df["close"].min()
            if min_price < 10:
                risks.append(f"min price ${min_price:.2f} — below $10 filter")

        # Quality score
        quality = 1.0
        if total_nulls > 0:
            quality -= 0.2
        if len(df) < 252:
            quality -= 0.3
            risks.append("less than 1 year of data")

        direction = Direction.LONG if quality > 0.5 else Direction.FLAT

        return AgentOutput(
            agent=self.name,
            ticker=ticker,
            timestamp="",
            thesis=f"Data quality: {quality:.0%}",
            direction=direction,
            confidence=round(quality, 2),
            evidence=evidence,
            risks=risks,
            invalidation=["data source becomes unavailable"],
            recommendation=direction,
        )
