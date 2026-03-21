"""
Base Agent — Every agent inherits this.
Enforces structured output, scoped inputs, and memory access.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from src.schemas.signals import AgentOutput, Direction

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parent.parent.parent / "memory" / "agent_memories"


class BaseAgent(ABC):
    """
    Base class for all agents.

    Rules:
    - Every agent must declare allowed_inputs and forbidden_inputs
    - Every agent must produce AgentOutput
    - Every agent reads/writes only its own memory file
    - No agent approves its own work
    """

    name: str = "base_agent"
    allowed_inputs: list[str] = []
    forbidden_inputs: list[str] = []

    def __init__(self):
        self.memory: list[dict] = []
        self._load_memory()

    def _memory_path(self) -> Path:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        return MEMORY_DIR / f"{self.name}_memory.json"

    def _load_memory(self):
        """Load agent's scoped memory from disk."""
        path = self._memory_path()
        if path.exists():
            try:
                with open(path) as f:
                    self.memory = json.load(f)
                logger.debug(f"{self.name}: loaded {len(self.memory)} memory entries")
            except Exception as e:
                logger.warning(f"{self.name}: failed to load memory: {e}")
                self.memory = []

    def _save_memory(self):
        """Persist agent's memory to disk."""
        path = self._memory_path()
        with open(path, "w") as f:
            json.dump(self.memory, f, indent=2, default=str)

    def add_memory(self, entry: dict):
        """Add an entry to agent memory. Auto-saves."""
        entry["timestamp"] = datetime.now().isoformat()
        entry["agent"] = self.name
        self.memory.append(entry)
        # Keep memory bounded
        if len(self.memory) > 1000:
            self.memory = self.memory[-500:]
        self._save_memory()

    def validate_inputs(self, data: dict) -> bool:
        """Check that no forbidden inputs are present."""
        for key in self.forbidden_inputs:
            if key in data:
                logger.warning(f"{self.name}: forbidden input '{key}' detected. Rejecting.")
                return False
        return True

    @abstractmethod
    def analyze(self, data: dict) -> AgentOutput:
        """
        Core analysis function. Must be implemented by each agent.

        Args:
            data: dict of allowed inputs for this agent

        Returns:
            AgentOutput with structured thesis, confidence, evidence, etc.
        """
        pass

    def run(self, data: dict) -> AgentOutput | None:
        """
        Run the agent: validate inputs → analyze → return output.
        """
        if not self.validate_inputs(data):
            return None
        try:
            output = self.analyze(data)
            logger.info(f"{self.name}: {output.ticker} → {output.direction.value} (conf={output.confidence:.2f})")
            return output
        except Exception as e:
            logger.error(f"{self.name}: analysis failed: {e}")
            return None
