"""
Risk Agent — Enforces position limits, computes risk budget, vetoes trades.

SPECIAL AUTHORITY: Veto power. No debate.

Allowed: Portfolio state, volatility, correlation, drawdown, exposure
Forbidden: Individual stock fundamentals, news sentiment
"""

from src.agents.base_agent import BaseAgent
from src.schemas.signals import AgentOutput, Direction


class RiskAgent(BaseAgent):
    name = "risk_agent"
    allowed_inputs = ["portfolio_state", "volatility", "correlation", "drawdown", "exposure"]
    forbidden_inputs = ["fundamentals", "news", "sentiment"]

    # Hard limits
    MAX_PORTFOLIO_DRAWDOWN = -0.15  # -15% portfolio drawdown → reduce exposure
    MAX_SINGLE_POSITION_PCT = 0.20  # 20% max per position
    MAX_SECTOR_PCT = 0.40  # 40% max per sector
    MAX_CORRELATION_OVERLAP = 0.80

    def analyze(self, data: dict) -> AgentOutput:
        """
        Evaluate whether a proposed trade passes risk checks.

        Expected data keys:
        - ticker: str
        - proposed_risk_pct: float
        - current_drawdown: float
        - open_positions: int
        - max_positions: int
        - portfolio_equity: float
        - current_exposure_pct: float
        """
        ticker = data.get("ticker", "UNKNOWN")
        current_dd = data.get("current_drawdown", 0)
        open_pos = data.get("open_positions", 0)
        max_pos = data.get("max_positions", 10)
        exposure = data.get("current_exposure_pct", 0)

        evidence = []
        risks = []
        veto = False

        # Check drawdown
        if current_dd < self.MAX_PORTFOLIO_DRAWDOWN:
            veto = True
            risks.append(f"portfolio drawdown {current_dd*100:.1f}% exceeds limit")

        # Check position count
        if open_pos >= max_pos:
            veto = True
            risks.append(f"at max positions ({open_pos}/{max_pos})")

        # Check exposure
        if exposure > 0.95:
            veto = True
            risks.append(f"portfolio nearly fully invested ({exposure*100:.0f}%)")

        evidence.append(f"current drawdown: {current_dd*100:.1f}%")
        evidence.append(f"open positions: {open_pos}/{max_pos}")
        evidence.append(f"exposure: {exposure*100:.0f}%")

        direction = Direction.FLAT if veto else Direction.LONG
        confidence = 0.9

        return AgentOutput(
            agent=self.name,
            ticker=ticker,
            timestamp="",
            thesis="VETO — risk limits breached" if veto else "Risk within acceptable bounds",
            direction=direction,
            confidence=confidence,
            evidence=evidence,
            risks=risks,
            invalidation=["drawdown recovers", "positions freed up"],
            recommendation=direction,
            metadata={"veto": veto},
        )
