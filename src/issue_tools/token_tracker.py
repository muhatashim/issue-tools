"""Token usage tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import json
from collections import defaultdict


@dataclass(slots=True)
class TokenUsage:
    """Metadata returned from a model call."""

    model: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: Optional[int] = None

    def combined(self) -> int:
        total = self.total_tokens
        if total is not None:
            return total
        return self.prompt_tokens + self.response_tokens


class TokenTracker:
    """Aggregate token usage and cost estimates."""

    def __init__(self, path: Path, cost_per_1k_tokens: Dict[str, float]):
        self.path = path
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self._lifetime_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
        })
        self._session_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
        })
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return
        with self.path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for model, stats in raw.items():
            bucket = self._lifetime_usage[model]
            bucket.update({
                "prompt_tokens": int(stats.get("prompt_tokens", 0)),
                "response_tokens": int(stats.get("response_tokens", 0)),
                "total_tokens": int(stats.get("total_tokens", 0)),
                "cost": float(stats.get("cost", 0.0)),
            })

    def _record_to_bucket(self, bucket: Dict[str, Dict[str, float]], usage: TokenUsage) -> None:
        stats = bucket[usage.model]
        stats["prompt_tokens"] += usage.prompt_tokens
        stats["response_tokens"] += usage.response_tokens
        stats["total_tokens"] += usage.combined()
        stats["cost"] += self._estimate_cost(usage.model, usage.combined())

    def record(self, usage: TokenUsage) -> None:
        """Record usage from a single call."""

        self._record_to_bucket(self._session_usage, usage)
        self._record_to_bucket(self._lifetime_usage, usage)

    def extend(self, usages: Iterable[TokenUsage]) -> None:
        for usage in usages:
            self.record(usage)

    def _estimate_cost(self, model: str, tokens: int) -> float:
        cost_per = self.cost_per_1k_tokens.get(model)
        if cost_per is None:
            return 0.0
        return (tokens / 1000.0) * cost_per

    def save(self) -> None:
        payload = {
            model: {
                "prompt_tokens": stats["prompt_tokens"],
                "response_tokens": stats["response_tokens"],
                "total_tokens": stats["total_tokens"],
                "cost": stats["cost"],
            }
            for model, stats in self._lifetime_usage.items()
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def session_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            model: dict(stats)
            for model, stats in self._session_usage.items()
        }

    def lifetime_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            model: dict(stats)
            for model, stats in self._lifetime_usage.items()
        }

