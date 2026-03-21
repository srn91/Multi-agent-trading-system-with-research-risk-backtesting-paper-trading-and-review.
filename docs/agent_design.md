# Agent Design Document

## Design Principle

Each agent is a digital employee with:
- A job description (mission)
- A fixed input set (allowed data)
- A limited memory scope (what it remembers)
- A specific output schema (AgentOutput)
- A decision boundary (what it can and cannot decide)
- A performance metric (how we judge it)

## V1 Agent Summary

| Agent | Mission | Veto Power | Memory Scope |
|-------|---------|------------|--------------|
| Data Ops | Clean and validate data | No | Data quality logs |
| Technical | Identify breakout setups | No | Past setup outcomes |
| Regime | Classify market state | No | Regime transitions |
| Risk | Enforce limits, veto danger | **Yes** | Drawdown, exposure |
| PM | Final trade decision | No (within risk) | Decision outcomes |
| Review | Post-trade attribution | No | Agent accuracy |

## Agent Interaction Rules

1. Agents communicate only through structured AgentOutput objects
2. No agent reads another agent's raw memory
3. PM sees agent summaries, not raw data
4. Risk Agent veto is absolute — PM cannot override
5. Review Agent runs after trades close, not during live decisions

## Adding New Agents (v2+)

To add a new agent:
1. Create `src/agents/new_agent.py` inheriting from `BaseAgent`
2. Define `name`, `allowed_inputs`, `forbidden_inputs`
3. Implement `analyze(data) -> AgentOutput`
4. Add to `configs/agent_config.yaml`
5. Wire into `src/decision/orchestrator.py`
6. Create memory file in `memory/agent_memories/`
7. Add tests in `tests/test_agents.py`
