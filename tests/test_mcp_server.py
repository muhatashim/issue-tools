"""Tests for the FastMCP server wrappers."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from issue_tools.commands import (
    CommandError,
    FilterItemSummary,
    FilterParams,
    FilterResponse,
    IndexParams,
    IndexResponse,
    TokenStats,
    TokenUsageSummary,
)
from issue_tools.runtime import ConfigurationError
from issue_tools.mcp_server import (
    _with_services,
    filter_tool,
    index_tool,
    tokens_tool,
)


def test_index_tool_serializes_response(monkeypatch) -> None:
    fake_services = object()
    captured: dict[str, object] = {}

    @contextmanager
    def fake_application_services(**_kwargs):
        yield fake_services

    def fake_index_repository(services, params: IndexParams):
        captured["services"] = services
        captured["params"] = params
        return IndexResponse(
            repo=params.repo or "owner/repo",
            indexed=1,
            fetched=1,
            skipped=False,
            message="ok",
            last_indexed_at="2024-01-01T00:00:00Z",
            token_usage=TokenUsageSummary(
                scope="session",
                models=[TokenStats(model="model", prompt_tokens=1, response_tokens=1, total_tokens=2, cost=0.01)],
            ),
        )

    monkeypatch.setattr("issue_tools.mcp_server.application_services", fake_application_services)
    monkeypatch.setattr("issue_tools.mcp_server.index_repository", fake_index_repository)

    result = asyncio.run(
        index_tool.run(
            {
                "repo": "explicit/repo",
                "force": True,
                "limit": 2,
                "include_pulls": False,
                "filter_query": "state:open",
            }
        )
    )

    assert captured["services"] is fake_services
    assert captured["params"].force is True
    assert captured["params"].filter_query == "state:open"
    payload = result.structured_content
    assert payload is not None
    assert payload["repo"] == "explicit/repo"
    assert payload["token_usage"]["models"][0]["total_tokens"] == 2


def test_filter_tool_serializes_response(monkeypatch) -> None:
    fake_services = object()
    captured: dict[str, object] = {}

    @contextmanager
    def fake_application_services(**_kwargs):
        yield fake_services

    response = FilterResponse(
        repo="owner/repo",
        query="label:bug repo:owner/repo",
        items=[
            FilterItemSummary(
                repo="owner/repo",
                number=1,
                title="Preview",
                doc_type="issue",
                state="open",
                html_url="https://example.test/1",
                updated_at="2024-01-01T00:00:00Z",
                created_at="2023-12-01T00:00:00Z",
                author="alice",
                labels=["bug"],
            )
        ],
    )

    def fake_filter_repository(services, params: FilterParams):
        captured["services"] = services
        captured["params"] = params
        return response

    monkeypatch.setattr("issue_tools.mcp_server.application_services", fake_application_services)
    monkeypatch.setattr("issue_tools.mcp_server.filter_repository", fake_filter_repository)

    result = asyncio.run(filter_tool.run({"query": "label:bug", "limit": 2}))

    payload = result.structured_content
    assert payload is not None
    assert payload["items"][0]["number"] == 1
    assert captured["services"] is fake_services
    assert captured["params"].limit == 2


def test_index_tool_wraps_command_errors(monkeypatch) -> None:
    @contextmanager
    def fake_application_services(**_kwargs):
        yield object()

    def failing_index(*_args, **_kwargs):
        raise CommandError("bad")

    monkeypatch.setattr("issue_tools.mcp_server.application_services", fake_application_services)
    monkeypatch.setattr("issue_tools.mcp_server.index_repository", failing_index)

    with pytest.raises(ValueError) as exc:
        asyncio.run(index_tool.run({}))
    assert "bad" in str(exc.value)


def test_tokens_tool_uses_summary(monkeypatch, tmp_path) -> None:
    summary = TokenUsageSummary(
        scope="lifetime",
        models=[TokenStats(model="model", prompt_tokens=1, response_tokens=2, total_tokens=3, cost=0.05)],
    )

    fake_config = SimpleNamespace(
        token_usage_path=tmp_path / "tokens.json",
        cost=SimpleNamespace(per_1k_tokens={"model": 0.0}),
    )

    class FakeTracker:
        def __init__(self, path, cost):  # noqa: D401 - simple stub
            self.path = path
            self.cost = cost

    monkeypatch.setattr("issue_tools.mcp_server.load_config", lambda: fake_config)
    monkeypatch.setattr("issue_tools.mcp_server.TokenTracker", FakeTracker)
    monkeypatch.setattr("issue_tools.mcp_server.summarize_token_usage", lambda tracker, lifetime=False: summary)

    result = asyncio.run(tokens_tool.run({"lifetime": True}))

    payload = result.structured_content
    assert payload is not None
    assert payload["scope"] == "lifetime"
    assert payload["models"][0]["total_tokens"] == 3


def test_with_services_converts_configuration_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "issue_tools.mcp_server.application_services",
        lambda **kwargs: (_ for _ in ()).throw(ConfigurationError("missing key")),
    )

    with pytest.raises(RuntimeError) as exc:
        _with_services(lambda services: services)
    assert "missing key" in str(exc.value)
