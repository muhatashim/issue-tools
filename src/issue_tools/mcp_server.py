"""FastMCP server exposing issue-tools commands for agents."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, TypeVar

from fastmcp import FastMCP

from .commands import (
    ClusterParams,
    CommandError,
    EvaluateParams,
    FilterParams,
    IndexParams,
    InspectDBParams,
    SearchParams,
    StreamParams,
    TokenUsageSummary,
    cluster_repository,
    evaluate_repository,
    filter_repository,
    index_repository,
    inspect_database,
    search_repository,
    stream_repository,
    summarize_token_usage,
)
from .config import load_config
from .evaluation import EvaluationCase
from .runtime import ConfigurationError, application_services
from .token_tracker import TokenTracker

server = FastMCP("issue-tools")

T = TypeVar("T")


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def _with_services(
    func: Callable[[Any], T], *, require_gemini_api_key: bool = True
) -> T:
    try:
        with application_services(
            console=None, require_gemini_api_key=require_gemini_api_key
        ) as services:
            return func(services)
    except ConfigurationError as exc:
        raise RuntimeError(str(exc)) from exc
    except CommandError as exc:
        raise ValueError(str(exc)) from exc


@server.tool("index")
def index_tool(
    repo: str | None = None,
    force: bool = False,
    limit: int | None = None,
    include_pulls: bool = True,
    filter_query: str | None = None,
) -> Any:
    result = _with_services(
        lambda services: index_repository(
            services,
            IndexParams(
                repo=repo,
                force=force,
                limit=limit,
                include_pulls=include_pulls,
                filter_query=filter_query,
            ),
        )
    )
    return _to_serializable(result)


@server.tool("filter")
def filter_tool(
    query: str, repo: str | None = None, limit: int = 10
) -> Any:
    result = _with_services(
        lambda services: filter_repository(
            services, FilterParams(query=query, repo=repo, limit=limit)
        ),
        require_gemini_api_key=False,
    )
    return _to_serializable(result)


@server.tool("search")
def search_tool(
    query: str, repo: str | None = None, limit: int = 5
) -> Any:
    result = _with_services(
        lambda services: search_repository(
            services, SearchParams(query=query, repo=repo, limit=limit)
        )
    )
    return _to_serializable(result)


@server.tool("cluster")
def cluster_tool(
    repo: str | None = None, query: str | None = None, k: int = 3
) -> Any:
    result = _with_services(
        lambda services: cluster_repository(
            services, ClusterParams(repo=repo, query=query, k=k)
        )
    )
    return _to_serializable(result)


@server.tool("stream")
def stream_tool(repo: str | None = None) -> Any:
    result = _with_services(
        lambda services: stream_repository(services, StreamParams(repo=repo))
    )
    return _to_serializable(result)


@server.tool("inspect_db")
def inspect_db_tool(repo: str | None = None, show: int = 0) -> Any:
    result = _with_services(
        lambda services: inspect_database(services, InspectDBParams(repo=repo, show=show))
    )
    return _to_serializable(result)


@server.tool("evaluate")
def evaluate_tool(
    cases: list[dict[str, Any]],
    repo: str | None = None,
    top_k: int = 5,
) -> Any:
    evaluation_cases = [
        EvaluationCase(
            query=item.get("query", ""),
            expected_numbers=[int(value) for value in item.get("expected_numbers", [])],
        )
        for item in cases
    ]
    result = _with_services(
        lambda services: evaluate_repository(
            services, EvaluateParams(cases=evaluation_cases, repo=repo, top_k=top_k)
        )
    )
    return _to_serializable(result)


@server.tool("tokens")
def tokens_tool(lifetime: bool = False) -> Any:
    config = load_config()
    tracker = TokenTracker(config.token_usage_path, config.cost.per_1k_tokens)
    summary = summarize_token_usage(tracker, lifetime=lifetime)
    if summary is None:
        summary = TokenUsageSummary(scope="lifetime" if lifetime else "session", models=[])
    return _to_serializable(summary)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    server.run()
