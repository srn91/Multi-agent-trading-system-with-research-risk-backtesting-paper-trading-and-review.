"""
Review Agent — Post-trade attribution and learning.

Analyzes what worked, what failed, and why.
Writes findings back to agent memories and review memory.

Allowed: Trade logs, agent output logs, outcome data, regime context
Forbidden: Live market data, current signals
"""

import json
from collections import Counter
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.schemas.signals import AgentOutput, Direction, ExitReason, TradeExit


class ReviewAgent(BaseAgent):
    name = "review_agent"
    allowed_inputs = ["trade_log", "agent_log", "outcomes", "regime_context"]
    forbidden_inputs = ["live_data", "current_signals", "current_prices"]

    def analyze(self, data: dict) -> AgentOutput:
        """
        Review a batch of completed trades.

        Expected data keys:
        - trades: list[TradeExit]
        - period: str description of review period
        """
        trades: list[TradeExit] = data.get("trades", [])
        period = data.get("period", "unknown")

        if not trades:
            return AgentOutput(
                agent=self.name, ticker="PORTFOLIO", timestamp="",
                thesis="No trades to review",
                direction=Direction.FLAT, confidence=0.0,
                evidence=[], risks=[], invalidation=[],
                recommendation=Direction.FLAT,
            )

        # --- Compute review stats ---
        winners = [t for t in trades if t.pnl_pct > 0]
        losers = [t for t in trades if t.pnl_pct <= 0]
        win_rate = len(winners) / len(trades)

        exit_reasons = Counter(t.exit_reason.value for t in trades)
        avg_hold_winners = sum(t.hold_days for t in winners) / len(winners) if winners else 0
        avg_hold_losers = sum(t.hold_days for t in losers) / len(losers) if losers else 0

        evidence = [
            f"reviewed {len(trades)} trades ({period})",
            f"win rate: {win_rate*100:.1f}%",
            f"avg winner hold: {avg_hold_winners:.0f} days",
            f"avg loser hold: {avg_hold_losers:.0f} days",
        ]

        for reason, count in exit_reasons.most_common():
            evidence.append(f"exit reason '{reason}': {count} trades")

        # --- Identify failure patterns ---
        risks = []
        if win_rate < 0.35:
            risks.append("win rate dangerously low — check signal quality")
        if exit_reasons.get("failed_breakout", 0) > len(trades) * 0.3:
            risks.append("too many failed breakouts — tighten base criteria")
        if exit_reasons.get("stop_loss", 0) > len(trades) * 0.4:
            risks.append("stops hit frequently — review stop placement")

        # --- Store review in memory ---
        self.add_memory({
            "period": period,
            "total_trades": len(trades),
            "win_rate": round(win_rate, 3),
            "exit_reasons": dict(exit_reasons),
            "risks_identified": risks,
        })

        return AgentOutput(
            agent=self.name,
            ticker="PORTFOLIO",
            timestamp="",
            thesis=f"Review of {len(trades)} trades: {win_rate*100:.1f}% win rate",
            direction=Direction.FLAT,
            confidence=0.8,
            evidence=evidence,
            risks=risks,
            invalidation=[],
            recommendation=Direction.FLAT,
            metadata={"exit_breakdown": dict(exit_reasons)},
        )
