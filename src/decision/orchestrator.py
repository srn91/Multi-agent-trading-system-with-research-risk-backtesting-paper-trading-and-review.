"""
Orchestrator — Runs agents in sequence and produces trade decisions.

Decision flow:
1. Regime Agent → classify market
2. IF hostile regime → skip all stocks
3. For each candidate stock:
   a. Technical Agent → score setup
   b. Risk Agent → check limits, veto if needed
   c. PM Agent → final decision
4. Log everything
5. Post-trade: Review Agent audits
"""

import logging
from datetime import datetime
from typing import Optional

from src.agents.technical_agent import TechnicalAgent
from src.agents.regime_agent import RegimeAgent
from src.agents.risk_agent import RiskAgent
from src.agents.pm_agent import PMAgent
from src.agents.review_agent import ReviewAgent
from src.schemas.signals import AgentOutput, Direction, Regime

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordinates agent execution in the correct order.
    Enforces decision flow rules.
    """

    def __init__(self):
        self.technical = TechnicalAgent()
        self.regime = RegimeAgent()
        self.risk = RiskAgent()
        self.pm = PMAgent()
        self.review = ReviewAgent()
        self.decision_log: list[dict] = []

    def evaluate_market(self, spy_row) -> AgentOutput:
        """Step 1: Get regime classification."""
        return self.regime.run({"spy_row": spy_row})

    def evaluate_stock(
        self,
        ticker: str,
        stock_row,
        recent_data,
        regime_output: AgentOutput,
        portfolio_state: dict,
    ) -> Optional[AgentOutput]:
        """
        Steps 2-4: Evaluate a single stock through the full agent pipeline.

        Returns PM decision or None if blocked.
        """
        # --- Technical Agent ---
        tech_output = self.technical.run({
            "ticker": ticker,
            "row": stock_row,
            "recent": recent_data,
        })
        if tech_output is None or tech_output.direction != Direction.LONG:
            return None

        # --- Risk Agent ---
        risk_output = self.risk.run({
            "ticker": ticker,
            "current_drawdown": portfolio_state.get("current_drawdown", 0),
            "open_positions": portfolio_state.get("open_positions", 0),
            "max_positions": portfolio_state.get("max_positions", 10),
            "current_exposure_pct": portfolio_state.get("exposure_pct", 0),
        })
        if risk_output is None:
            return None
        if risk_output.metadata.get("veto", False):
            logger.info(f"  {ticker}: VETOED by Risk Agent")
            self._log_decision(ticker, "vetoed", [regime_output, tech_output, risk_output])
            return None

        # --- PM Agent ---
        pm_output = self.pm.run({
            "ticker": ticker,
            "agent_outputs": [regime_output, tech_output, risk_output],
        })

        self._log_decision(
            ticker,
            pm_output.direction.value if pm_output else "error",
            [regime_output, tech_output, risk_output, pm_output] if pm_output else [],
        )

        return pm_output

    def run_review(self, trades: list, period: str) -> AgentOutput:
        """Post-trade review."""
        return self.review.run({
            "trades": trades,
            "period": period,
        })

    def _log_decision(self, ticker: str, decision: str, outputs: list):
        """Log the full decision chain for auditability."""
        self.decision_log.append({
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "decision": decision,
            "agents": [
                {
                    "agent": o.agent,
                    "direction": o.direction.value,
                    "confidence": o.confidence,
                    "thesis": o.thesis,
                }
                for o in outputs if o is not None
            ],
        })

    def get_decision_log(self) -> list[dict]:
        return self.decision_log
