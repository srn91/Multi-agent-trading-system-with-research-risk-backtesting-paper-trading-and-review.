"""
Tests for feature engineering and backtest engine.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


def make_sample_ohlcv(days: int = 300, base_price: float = 50.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    dates = pd.bdate_range(start="2023-01-01", periods=days)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, days)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        "high": prices * (1 + np.random.uniform(0.001, 0.02, days)),
        "low": prices * (1 - np.random.uniform(0.001, 0.02, days)),
        "close": prices,
        "volume": np.random.randint(500_000, 5_000_000, days).astype(float),
    }, index=dates)

    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


class TestFeatures:
    def test_moving_averages(self):
        from src.data.features import add_moving_averages
        df = make_sample_ohlcv(300)
        result = add_moving_averages(df)
        assert "ma_20" in result.columns
        assert "ma_150" in result.columns
        assert "ma_200" in result.columns
        assert result["ma_20"].iloc[25] > 0  # Should have values after 20 days
        assert pd.isna(result["ma_200"].iloc[100])  # Not enough data yet

    def test_atr(self):
        from src.data.features import add_atr
        df = make_sample_ohlcv(100)
        result = add_atr(df)
        assert "atr" in result.columns
        assert "atr_pct" in result.columns
        assert result["atr"].iloc[-1] > 0

    def test_volume_features(self):
        from src.data.features import add_volume_features
        df = make_sample_ohlcv(100)
        result = add_volume_features(df)
        assert "vol_ratio" in result.columns
        assert "dollar_vol_20_avg" in result.columns

    def test_breakout_features(self):
        from src.data.features import add_breakout_features
        df = make_sample_ohlcv(100)
        result = add_breakout_features(df)
        assert "breakout" in result.columns
        assert "close_range_pct" in result.columns
        assert result["close_range_pct"].between(0, 1).all()

    def test_trend_filter(self):
        from src.data.features import add_moving_averages, add_trend_filter
        df = make_sample_ohlcv(300)
        df = add_moving_averages(df)
        result = add_trend_filter(df)
        assert "trend_valid" in result.columns
        assert result["trend_valid"].dtype == bool

    def test_full_pipeline(self):
        from src.data.features import compute_all_features
        df = make_sample_ohlcv(300)
        result = compute_all_features(df)
        required = ["ma_150", "ma_200", "atr", "vol_ratio", "breakout",
                     "trend_valid", "in_base", "close_range_pct"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"


class TestSchemas:
    def test_breakout_signal_creation(self):
        from src.schemas.signals import BreakoutSignal
        signal = BreakoutSignal(
            ticker="AAPL", signal_date=date(2024, 1, 15),
            breakout_close=185.0, breakout_high=186.0,
            base_low=178.0, base_high=184.0,
            base_length=25, base_depth_pct=0.033,
            volume_ratio=1.8, close_range_pct=0.85,
            atr=2.5, ma_150=175.0, ma_200=170.0,
            spy_above_200=True, trend_valid=True,
            base_valid=True, breakout_valid=True, is_valid=True,
        )
        assert signal.ticker == "AAPL"
        assert signal.is_valid

    def test_backtest_config_defaults(self):
        from src.backtest.engine import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000
        assert cfg.risk_per_trade == 0.01
        assert cfg.max_positions == 10
        assert cfg.slippage_pct == 0.0005


class TestAgents:
    def test_technical_agent_output(self):
        from src.agents.technical_agent import TechnicalAgent
        from src.schemas.signals import Direction
        agent = TechnicalAgent()

        # Create a mock row with all features
        row = pd.Series({
            "close": 100, "high": 101, "low": 99,
            "trend_close_above_150": True,
            "trend_150_above_200": True,
            "trend_200_rising": True,
            "trend_above_52w_mid": True,
            "in_base": True,
            "base_depth_pct": 0.08,
            "atr_compression": True,
            "vol_ratio": 1.8,
            "close_range_pct": 0.9,
            "breakout": True,
            "atr_pct": 0.02,
            "ma_50": 95,
            "base_low": 96,
            "base_length": 25,
        })
        row.name = pd.Timestamp("2024-01-15")

        output = agent.run({"ticker": "TEST", "row": row})
        assert output is not None
        assert output.agent == "technical_agent"
        assert output.ticker == "TEST"
        assert output.direction == Direction.LONG
        assert output.confidence > 0.5
        assert len(output.evidence) > 0

    def test_risk_agent_veto(self):
        from src.agents.risk_agent import RiskAgent
        agent = RiskAgent()
        output = agent.run({
            "ticker": "TEST",
            "current_drawdown": -0.20,
            "open_positions": 10,
            "max_positions": 10,
            "current_exposure_pct": 0.95,
        })
        assert output is not None
        assert output.metadata.get("veto") is True

    def test_risk_agent_pass(self):
        from src.agents.risk_agent import RiskAgent
        agent = RiskAgent()
        output = agent.run({
            "ticker": "TEST",
            "current_drawdown": -0.05,
            "open_positions": 3,
            "max_positions": 10,
            "current_exposure_pct": 0.30,
        })
        assert output is not None
        assert output.metadata.get("veto") is False
