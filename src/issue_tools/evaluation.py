"""Evaluation utilities for semantic retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import json

from .search import SearchService


@dataclass(slots=True)
class EvaluationCase:
    query: str
    expected_numbers: List[int]


@dataclass(slots=True)
class EvaluationResult:
    case: EvaluationCase
    matches: List[int]
    precision_at_k: float
    reciprocal_rank: float


@dataclass(slots=True)
class EvaluationSummary:
    results: List[EvaluationResult]
    mean_precision: float
    mean_reciprocal_rank: float


def load_cases(path: Path) -> List[EvaluationCase]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = []
    for item in payload:
        cases.append(
            EvaluationCase(
                query=item["query"],
                expected_numbers=[int(num) for num in item.get("expected_numbers", [])],
            )
        )
    return cases


def evaluate(
    search: SearchService,
    repo: str,
    cases: Iterable[EvaluationCase],
    *,
    top_k: int = 5,
) -> EvaluationSummary:
    results: List[EvaluationResult] = []
    precision_total = 0.0
    reciprocal_total = 0.0

    for case in cases:
        matches = search.search(repo, case.query, limit=top_k)
        numbers = [match.number for match in matches]
        precision = _precision_at_k(numbers, case.expected_numbers, top_k)
        reciprocal = _reciprocal_rank(numbers, case.expected_numbers)
        precision_total += precision
        reciprocal_total += reciprocal
        results.append(
            EvaluationResult(
                case=case,
                matches=numbers,
                precision_at_k=precision,
                reciprocal_rank=reciprocal,
            )
        )

    count = len(results) or 1
    return EvaluationSummary(
        results=results,
        mean_precision=precision_total / count,
        mean_reciprocal_rank=reciprocal_total / count,
    )


def _precision_at_k(results: List[int], expected: List[int], k: int) -> float:
    if not results:
        return 0.0
    top_results = results[:k]
    hits = sum(1 for value in top_results if value in expected)
    denominator = min(k, len(top_results))
    if denominator == 0:
        return 0.0
    return hits / denominator


def _reciprocal_rank(results: List[int], expected: List[int]) -> float:
    for index, value in enumerate(results, start=1):
        if value in expected:
            return 1.0 / index
    return 0.0

