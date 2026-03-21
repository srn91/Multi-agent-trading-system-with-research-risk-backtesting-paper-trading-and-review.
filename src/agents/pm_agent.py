"""
PM (Portfolio Manager) Agent — Makes the final trade decision.

Aggregates all agent outputs. Rules-based, not creative.
Cannot override Risk Agent veto.

Allowed: Summaries from all agents (not raw data)
Forbidden: Raw price data, raw features
"""

from src.agents.base_agent import BaseAgent
from src.schemas.signals import AgentOutput, Direction


class PMAgent(BaseAgent):
    name = "pm_agent"
    allowed_inputs = ["agent_outputs", "portfolio_summary"]
    forbidden_inputs = ["raw_ohlcv", "raw_features", "raw_indicators"]

    # Decision thresholds
    MIN_AGENTS_AGREE = 2  # at least N agents must recommend long
    MIN_CONFIDENCE = 0.5

    def analyze(self, data: dict) -> AgentOutput:
        """
        Aggregate agent outputs and decide.

        Expected data keys:
        - ticker: str
        - agent_outputs: list[AgentOutput]
        """
        ticker = data.get("ticker", "UNKNOWN")
        outputs: list[AgentOutput] = data.get("agent_outputs", [])

        evidence = []
        risks = []
        long_votes = 0
        veto = False
        avg_confidence = 0.0

        for out in outputs:
            evidence.append(f"{out.agent}: {out.direction.value} (conf={out.confidence:.2f})")
            if out.direction == Direction.LONG:
                long_votes += 1
            avg_confidence += out.confidence

            # Risk agent veto is absolute
            if out.agent == "risk_agent" and out.metadata.get("veto", False):
                veto = True
                risks.append("RISK AGENT VETO")

            risks.extend(out.risks)

        avg_confidence = avg_confidence / len(outputs) if outputs else 0

        # Decision logic
        if veto:
            direction = Direction.FLAT
            thesis = "Blocked by Risk Agent veto"
        elif long_votes >= self.MIN_AGENTS_AGREE and avg_confidence >= self.MIN_CONFIDENCE:
            direction = Direction.LONG
            thesis = f"Approved: {long_votes}/{len(outputs)} agents agree, avg confidence {avg_confidence:.2f}"
        else:
            direction = Direction.FLAT
            thesis = f"Rejected: insufficient agreement ({long_votes}/{len(outputs)})"

        return AgentOutput(
            agent=self.name,
            ticker=ticker,
            timestamp="",
            thesis=thesis,
            direction=direction,
            confidence=round(avg_confidence, 2),
            evidence=evidence,
            risks=risks[:5],  # cap risk list
            invalidation=["any agent flips to flat", "risk limits breached"],
            recommendation=direction,
            metadata={"long_votes": long_votes, "veto": veto},
        )
