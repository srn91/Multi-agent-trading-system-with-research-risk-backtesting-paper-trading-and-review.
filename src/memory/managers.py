"""
Memory Managers — Scoped persistent memory for the system.

Each memory type has its own manager, its own storage, and its own rules.
No universal blob. No cross-contamination.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MEMORY_BASE = Path(__file__).resolve().parent.parent.parent / "memory"


class MemoryManager:
    """Base memory manager with JSON persistence."""

    def __init__(self, subdir: str, max_entries: int = 1000):
        self.dir = MEMORY_BASE / subdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

    def _path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.dir / f"{safe_key}.json"

    def load(self, key: str) -> list[dict]:
        path = self._path(key)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load memory {key}: {e}")
        return []

    def save(self, key: str, data: list[dict]):
        if len(data) > self.max_entries:
            data = data[-self.max_entries:]
        path = self._path(key)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def append(self, key: str, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        data = self.load(key)
        data.append(entry)
        self.save(key, data)

    def get_recent(self, key: str, n: int = 10) -> list[dict]:
        data = self.load(key)
        return data[-n:]


class TickerMemory(MemoryManager):
    """Per-ticker memory: past setups, outcomes, support/resistance."""

    def __init__(self):
        super().__init__("ticker_memory", max_entries=500)

    def log_trade(self, ticker: str, trade_data: dict):
        self.append(ticker, {
            "type": "trade",
            **trade_data,
        })

    def log_setup(self, ticker: str, setup_data: dict):
        self.append(ticker, {
            "type": "setup",
            **setup_data,
        })

    def get_past_trades(self, ticker: str, n: int = 20) -> list[dict]:
        data = self.load(ticker)
        return [d for d in data if d.get("type") == "trade"][-n:]


class RegimeMemory(MemoryManager):
    """Market regime history and transition patterns."""

    def __init__(self):
        super().__init__("regime_memory", max_entries=1000)

    def log_regime(self, regime: str, indicators: dict):
        self.append("regime_history", {
            "regime": regime,
            **indicators,
        })

    def get_regime_history(self, n: int = 100) -> list[dict]:
        return self.get_recent("regime_history", n)


class PortfolioMemory(MemoryManager):
    """Portfolio-level memory: PnL, drawdown, exposure history."""

    def __init__(self):
        super().__init__("trade_journal", max_entries=2000)

    def log_trade(self, trade_data: dict):
        self.append("portfolio_trades", trade_data)

    def log_daily_snapshot(self, snapshot: dict):
        self.append("daily_snapshots", snapshot)

    def get_all_trades(self) -> list[dict]:
        return self.load("portfolio_trades")

    def get_recent_snapshots(self, n: int = 30) -> list[dict]:
        return self.get_recent("daily_snapshots", n)


class ReviewMemory(MemoryManager):
    """Review and attribution memory."""

    def __init__(self):
        super().__init__("agent_memories", max_entries=500)

    def log_review(self, review_data: dict):
        self.append("reviews", review_data)

    def get_reviews(self, n: int = 20) -> list[dict]:
        return self.get_recent("reviews", n)
