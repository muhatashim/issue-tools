"""Tests for the FastAPI application."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from issue_tools.api import create_app, get_services, get_services_without_gemini
from issue_tools.commands import (
    ClusterDocumentSummary,
    ClusterFiltersSummary,
    ClusterResponse,
    ClusterSummary,
    EmbeddingModelInfo,
    FilterItemSummary,
    FilterParams,
    FilterResponse,
    EvaluationResponse,
    IndexParams,
    IndexResponse,
    InspectDBResponse,
    RepoSummary,
    SampleDocumentInfo,
    SearchParams,
    SearchResponse,
    StreamResponse,
    StreamItemSummary,
    TokenStats,
    TokenUsageSummary,
)
from issue_tools.search import SearchMatch


def make_client() -> tuple[TestClient, object, dict[str, object]]:
    captured: dict[str, object] = {}
    services = object()
    app = create_app()
    app.dependency_overrides[get_services] = lambda: services
    app.dependency_overrides[get_services_without_gemini] = lambda: services
    client = TestClient(app)
    return client, services, captured


def test_index_endpoint_returns_response(monkeypatch) -> None:
    client, services, captured = make_client()

    token_usage = TokenUsageSummary(scope="session", models=[TokenStats("model", 1, 2, 3, 0.1)])
    response = IndexResponse(
        repo="owner/repo",
        indexed=1,
        fetched=1,
        skipped=False,
        message="ok",
        last_indexed_at="2024-01-01T00:00:00Z",
        token_usage=token_usage,
    )

    def fake_index_repository(passed_services, params: IndexParams):
        captured["services"] = passed_services
        captured["params"] = params
        return response

    monkeypatch.setattr("issue_tools.api.index_repository", fake_index_repository)

    payload = {
        "repo": "explicit/repo",
        "force": True,
        "limit": 2,
        "include_pulls": False,
        "filter_query": "state:open",
    }
    result = client.post("/index", json=payload)

    assert result.status_code == 200
    body = result.json()
    assert body["repo"] == "owner/repo"
    assert captured["services"] is services
    assert captured["params"].force is True
    assert captured["params"].filter_query == "state:open"


def test_search_endpoint_uses_params(monkeypatch) -> None:
    client, services, captured = make_client()

    match = SearchMatch(
        doc_id="id-1",
        number=5,
        title="Title",
        url="https://example.test/5",
        score=0.75,
        state="open",
        doc_type="issue",
        labels=["bug"],
        updated_at="2024-01-01T00:00:00Z",
        created_at="2023-12-01T00:00:00Z",
        author="alice",
    )
    response = SearchResponse(repo="owner/repo", query="test", matches=[match], token_usage=None)

    def fake_search_repository(passed_services, params: SearchParams):
        captured["services"] = passed_services
        captured["params"] = params
        return response

    monkeypatch.setattr("issue_tools.api.search_repository", fake_search_repository)

    result = client.post("/search", json={"query": "test", "limit": 3})

    assert result.status_code == 200
    assert captured["services"] is services
    assert captured["params"].limit == 3
    assert result.json()["matches"][0]["number"] == 5


def test_filter_endpoint_returns_items(monkeypatch) -> None:
    client, services, captured = make_client()

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

    def fake_filter_repository(passed_services, params: FilterParams):
        captured["services"] = passed_services
        captured["params"] = params
        return response

    monkeypatch.setattr("issue_tools.api.filter_repository", fake_filter_repository)

    result = client.post("/filter", json={"query": "label:bug", "limit": 2})

    assert result.status_code == 200
    assert captured["services"] is services
    assert captured["params"].limit == 2
    body = result.json()
    assert body["query"] == "label:bug repo:owner/repo"
    assert body["items"][0]["number"] == 1


def test_cluster_endpoint_serializes_nested(monkeypatch) -> None:
    client, services, captured = make_client()

    cluster = ClusterSummary(
        cluster_id=1,
        size=2,
        centroid=[0.1, 0.2],
        top_labels=["bug"],
        documents=[
            ClusterDocumentSummary(
                number=7,
                title="Cluster doc",
                doc_type="issue",
                state="open",
                html_url="https://example.test/7",
                labels=["bug"],
            )
        ],
    )
    response = ClusterResponse(
        repo="owner/repo",
        filters=ClusterFiltersSummary(
            labels=["bug"],
            states=["open"],
            types=["issue"],
            numbers=[7],
            updated=(">=", "2024-01-01T00:00:00Z"),
            created=None,
            author="alice",
        ),
        clusters=[cluster],
        token_usage=None,
    )

    def fake_cluster_repository(passed_services, params):
        captured["services"] = passed_services
        captured["params"] = params
        return response

    monkeypatch.setattr("issue_tools.api.cluster_repository", fake_cluster_repository)

    result = client.post("/cluster", json={"repo": "repo/name", "k": 1})

    assert result.status_code == 200
    assert captured["services"] is services
    assert result.json()["clusters"][0]["documents"][0]["number"] == 7


def test_stream_endpoint_returns_items(monkeypatch) -> None:
    client, services, captured = make_client()

    response = StreamResponse(
        repo="owner/repo",
        items=[
            StreamItemSummary(
                number=8,
                title="Updated",
                doc_type="issue",
                state="closed",
                html_url="https://example.test/8",
                updated_at="2024-01-02T00:00:00Z",
                created_at="2024-01-01T00:00:00Z",
                author="bob",
            )
        ],
    )

    def fake_stream_repository(passed_services, params):
        captured["services"] = passed_services
        captured["params"] = params
        return response

    monkeypatch.setattr("issue_tools.api.stream_repository", fake_stream_repository)

    result = client.get("/stream")

    assert result.status_code == 200
    assert captured["services"] is services
    assert result.json()["items"][0]["number"] == 8


def test_inspect_endpoint_serializes_database(monkeypatch) -> None:
    client, services, captured = make_client()

    response = InspectDBResponse(
        database_path="/tmp/db.sqlite",
        summaries=[
            RepoSummary(
                repo="owner/repo",
                total=2,
                issues=2,
                pulls=0,
                first_created="2024-01-01T00:00:00Z",
                last_created="2024-01-02T00:00:00Z",
                first_updated="2024-01-01T00:00:00Z",
                last_updated="2024-01-03T00:00:00Z",
                last_indexed="2024-01-04T00:00:00Z",
            )
        ],
        embedding_models=[EmbeddingModelInfo(model="model", dimensions=2, documents=2)],
        samples=[
            SampleDocumentInfo(
                repo="owner/repo",
                number=9,
                doc_type="issue",
                state="open",
                updated_at="2024-01-03T00:00:00Z",
                title="Sample",
            )
        ],
    )

    def fake_inspect_database(passed_services, params):
        captured["services"] = passed_services
        captured["params"] = params
        return response

    monkeypatch.setattr("issue_tools.api.inspect_database", fake_inspect_database)

    result = client.get("/inspect-db", params={"show": 1})

    assert result.status_code == 200
    assert captured["services"] is services
    assert result.json()["embedding_models"][0]["model"] == "model"


def test_evaluate_endpoint_handles_cases(monkeypatch) -> None:
    client, services, captured = make_client()

    summary = SimpleNamespace(
        results=[],
        mean_precision=0.1,
        mean_reciprocal_rank=0.2,
    )
    response = EvaluationResponse(
        repo="owner/repo",
        top_k=5,
        summary=summary,
        token_usage=None,
    )

    def fake_evaluate_repository(passed_services, params):
        captured["services"] = passed_services
        captured["params"] = params
        return response

    monkeypatch.setattr("issue_tools.api.evaluate_repository", fake_evaluate_repository)

    payload = {
        "repo": "owner/repo",
        "top_k": 5,
        "cases": [{"query": "test", "expected_numbers": [1, 2]}],
    }
    result = client.post("/evaluate", json=payload)

    assert result.status_code == 200
    assert captured["services"] is services
    assert len(captured["params"].cases) == 1


def test_index_endpoint_returns_400_on_command_error(monkeypatch) -> None:
    client, _, _ = make_client()

    def fake_index_repository(*_args, **_kwargs):
        from issue_tools.commands import CommandError

        raise CommandError("bad request")

    monkeypatch.setattr("issue_tools.api.index_repository", fake_index_repository)

    result = client.post("/index", json={})
    assert result.status_code == 400
    assert result.json()["detail"] == "bad request"


def test_tokens_endpoint_reads_lifetime(tmp_path: Path, monkeypatch) -> None:
    app = create_app()
    client = TestClient(app)

    token_file = tmp_path / "token_usage.json"
    token_file.write_text(
        json.dumps(
            {
                "model": {
                    "prompt_tokens": 2,
                    "response_tokens": 3,
                    "total_tokens": 5,
                    "cost": 0.4,
                }
            }
        )
    )

    fake_config = SimpleNamespace(
        token_usage_path=token_file,
        cost=SimpleNamespace(per_1k_tokens={"model": 0.0}),
    )

    monkeypatch.setattr("issue_tools.api.load_config", lambda: fake_config)

    result = client.get("/tokens", params={"lifetime": "true"})

    assert result.status_code == 200
    body = result.json()
    assert body["scope"] == "lifetime"
    assert body["models"][0]["total_tokens"] == 5

