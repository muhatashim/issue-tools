"""Parsing helpers for GitHub-style filters embedded in search queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Set, Tuple, TYPE_CHECKING
import re
import shlex

if TYPE_CHECKING:  # pragma: no cover - import cycle in type checking only
    from .vector_store import StoredDocument


_DATE_PATTERN = re.compile(r"^(?P<op>>=|<=|>|<)?(?P<value>.+)$")


@dataclass(slots=True)
class FilterCriteria:
    """Represents structured filters extracted from a query string."""

    labels: Set[str] = field(default_factory=set)
    states: Set[str] = field(default_factory=set)
    types: Set[str] = field(default_factory=set)
    numbers: Set[int] = field(default_factory=set)
    updated_filter: Optional[Tuple[str, str]] = None
    created_filter: Optional[Tuple[str, str]] = None
    author: Optional[str] = None
    repo: Optional[str] = None

    def with_repo(self, repo: Optional[str]) -> "FilterCriteria":
        clone = FilterCriteria(
            labels=set(self.labels),
            states=set(self.states),
            types=set(self.types),
            numbers=set(self.numbers),
            updated_filter=self.updated_filter,
            created_filter=self.created_filter,
            author=self.author,
            repo=repo or self.repo,
        )
        return clone

    def matches(self, document: "StoredDocument") -> bool:
        if self.repo and document.repo != self.repo:
            return False
        if self.labels and not set(label.lower() for label in document.labels).issuperset(
            {label.lower() for label in self.labels}
        ):
            return False
        if self.states and document.state.lower() not in self.states:
            return False
        if self.types and document.doc_type not in self.types:
            return False
        if self.numbers and document.number not in self.numbers:
            return False
        if self.author and (document.author or "") != self.author:
            return False
        if self.updated_filter and not _compare_iso8601(document.updated_at, self.updated_filter):
            return False
        if self.created_filter and not _compare_iso8601(document.created_at, self.created_filter):
            return False
        return True


def parse_query_with_filters(query: str) -> tuple[str, FilterCriteria]:
    """Split a raw query string into free text and structured filters."""

    tokens = shlex.split(query)
    criteria = FilterCriteria()
    free_text: List[str] = []

    for token in tokens:
        if ":" not in token:
            free_text.append(token)
            continue
        key, raw_value = token.split(":", 1)
        key = key.lower()
        value = raw_value.strip()
        if not value:
            continue
        if key in {"label", "labels", "tag", "tags"}:
            criteria.labels.add(value)
        elif key in {"state"}:
            criteria.states.add(value.lower())
        elif key in {"is", "issue", "type"}:
            handled = False
            normalized = _normalize_type(value)
            if normalized:
                criteria.types.add(normalized)
                handled = True
            if not handled:
                state = _normalize_state(value)
                if state:
                    criteria.states.add(state)
                    handled = True
            if handled:
                continue
        elif key in {"pr", "pull", "pull_request", "pull-request"}:
            criteria.types.add("pull_request")
        elif key in {"updated", "date"}:
            parsed = _parse_date_filter(value)
            if parsed:
                criteria.updated_filter = parsed
        elif key in {"created"}:
            parsed = _parse_date_filter(value)
            if parsed:
                criteria.created_filter = parsed
        elif key in {"author"}:
            criteria.author = value
        elif key in {"number", "issue_number"}:
            try:
                criteria.numbers.add(int(value))
            except ValueError:
                continue
        elif key == "repo":
            criteria.repo = value
        else:
            free_text.append(token)

    return " ".join(free_text).strip(), criteria


def _normalize_type(value: str) -> Optional[str]:
    normalized = value.lower()
    if normalized in {"issue", "issues"}:
        return "issue"
    if normalized in {"pr", "pull", "pull_request", "pull-requests", "pull-request"}:
        return "pull_request"
    return None


def _normalize_state(value: str) -> Optional[str]:
    normalized = value.lower()
    if normalized in {"open", "closed"}:
        return normalized
    return None


def _parse_date_filter(value: str) -> Optional[Tuple[str, str]]:
    match = _DATE_PATTERN.match(value)
    if not match:
        return None
    op = match.group("op") or ">="
    raw_value = match.group("value")
    normalized = _normalize_datetime(raw_value)
    if not normalized:
        return None
    return op, normalized


def _normalize_datetime(value: str) -> Optional[str]:
    value = value.strip()
    if not value:
        return None
    try:
        if value.endswith("Z") or value.endswith("z"):
            dt = datetime.fromisoformat(value.rstrip("Zz") + "+00:00")
        elif "T" in value:
            dt = datetime.fromisoformat(value)
        else:
            dt = datetime.fromisoformat(value + "T00:00:00")
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compare_iso8601(timestamp: str, condition: Tuple[str, str]) -> bool:
    op, value = condition
    left = timestamp or ""
    if not left:
        return False
    if op == ">=":
        return left >= value
    if op == ">":
        return left > value
    if op == "<=":
        return left <= value
    if op == "<":
        return left < value
    return False

