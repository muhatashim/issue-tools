"""Tests for the shared command implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional

import pytest

from issue_tools.clustering import Cluster as RawCluster
from issue_tools.commands import (
    ClusterParams,
    CommandError,
    EvaluateParams,
    EvaluationResponse,
    FilterParams,
    IndexParams,
    InspectDBParams,
    SearchParams,
    StreamParams,
    cluster_repository,
    evaluate_repository,
    filter_repository,
    index_repository,
    inspect_database,
    search_repository,
    stream_repository,
    summarize_token_usage,
)
from issue_tools.config import Config
from issue_tools.evaluation import EvaluationCase, EvaluationSummary
from issue_tools.github_client import GitHubItem
from issue_tools.runtime import Services
from issue_tools.search import SearchMatch
from issue_tools.token_tracker import TokenTracker, TokenUsage
from issue_tools.vector_store import SQLiteVectorStore, StoredDocument


@dataclass
class FakeIndexResult:
    indexed: int = 2
    fetched: int = 3
    skipped: bool = False
    message: str = "done"
    last_indexed_at: Optional[str] = "2024-01-01T00:00:00Z"


class FakeIndexer:
    def __init__(self, stream_items: Optional[List[SimpleNamespace]] = None) -> None:
        self.last_call: dict[str, object] | None = None
        self.stream_items = stream_items or []

    def run(
        self,
        repo: str,
        *,
        force: bool = False,
        limit: Optional[int] = None,
        include_pulls: bool = True,
        filter_query: Optional[str] = None,
    ) -> FakeIndexResult:
        self.last_call = {
            "repo": repo,
            "force": force,
            "limit": limit,
            "include_pulls": include_pulls,
            "filter_query": filter_query,
        }
        return FakeIndexResult()

    def stream_new(self, repo: str) -> List[SimpleNamespace]:
        self.last_stream_repo = repo  # type: ignore[attr-defined]
        return self.stream_items


class FakeSearchService:
    def __init__(self, matches: Optional[List[SearchMatch]] = None) -> None:
        self.matches = matches or []
        self.calls: list[tuple[Optional[str], str, int]] = []

    def search(self, repo: Optional[str], query: str, *, limit: int = 5) -> List[SearchMatch]:
        self.calls.append((repo, query, limit))
        return list(self.matches)


def make_services(
    tmp_path: Path,
    *,
    repo: Optional[str] = "owner/repo",
    search: Optional[FakeSearchService] = None,
    indexer: Optional[FakeIndexer] = None,
    vector_store: Optional[SQLiteVectorStore | SimpleNamespace] = None,
    github: Optional[object] = None,
) -> Services:
    config = Config(repository=repo)
    config.token_usage_path = tmp_path / "tokens.json"
    if isinstance(vector_store, SQLiteVectorStore):
        config.vector_db_path = vector_store.path
    tracker = TokenTracker(config.token_usage_path, config.cost.per_1k_tokens)
    services = Services(
        config=config,
        github=github
        if github is not None
        else SimpleNamespace(search_issues=lambda query, limit: []),
        vector_store=vector_store
        if vector_store is not None
        else SimpleNamespace(
            path=tmp_path / "vector.db", 
            connection=None, 
            all_metadata=lambda: {},
            all_embeddings=lambda repo, filters, limit: [],
        ),
        embeddings=SimpleNamespace(
            embed_queries=lambda texts: SimpleNamespace(embeddings=[[0.1, 0.2] for _ in texts])
        ),
        search=search or FakeSearchService(),
        indexer=indexer or FakeIndexer(),
        token_tracker=tracker,
    )
    return services


def record_usage(services: Services) -> None:
    services.token_tracker.record(
        TokenUsage(model="gemini-embedding-001", prompt_tokens=3, response_tokens=2)
    )


def test_resolve_repo_raises_when_missing(tmp_path: Path) -> None:
    services = make_services(tmp_path, repo=None)
    params = SearchParams(query="hello")
    with pytest.raises(CommandError):
        search_repository(services, params)


def test_index_repository_uses_config_repo(tmp_path: Path) -> None:
    indexer = FakeIndexer()
    services = make_services(tmp_path, repo="config/repo", indexer=indexer)
    record_usage(services)

    result = index_repository(
        services,
        IndexParams(repo=None, force=True, limit=10, include_pulls=False),
    )

    assert result.repo == "config/repo"
    assert indexer.last_call == {
        "repo": "config/repo",
        "force": True,
        "limit": 10,
        "include_pulls": False,
        "filter_query": None,
    }
    assert result.token_usage is not None
    assert result.token_usage.scope == "session"


def test_index_repository_passes_filter_query(tmp_path: Path) -> None:
    indexer = FakeIndexer()
    services = make_services(tmp_path, repo="config/repo", indexer=indexer)
    record_usage(services)

    index_repository(
        services,
        IndexParams(filter_query="state:open"),
    )

    assert indexer.last_call is not None
    assert indexer.last_call["filter_query"] == "state:open"


def test_search_repository_prefers_param_repo(tmp_path: Path) -> None:
    match = SearchMatch(
        doc_id="id-1",
        number=42,
        title="Found",
        url="https://example.test/42",
        score=0.9,
        state="open",
        doc_type="issue",
        labels=["bug"],
        label_details=[],
        updated_at="2024-01-01T00:00:00Z",
        created_at="2023-12-01T00:00:00Z",
        author="octocat",
    )
    search = FakeSearchService(matches=[match])
    services = make_services(tmp_path, repo="config/repo", search=search)
    record_usage(services)

    result = search_repository(
        services,
        SearchParams(query="label:bug", repo="param/repo", limit=3),
    )

    assert result.repo == "param/repo"
    assert search.calls == [("param/repo", "label:bug", 3)]
    assert result.matches[0].number == 42
    assert result.token_usage is not None


def test_filter_repository_appends_repo_filter(tmp_path: Path) -> None:
    class FakeGitHub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def search_issues(self, query: str, limit: int) -> List[GitHubItem]:
            self.calls.append((query, limit))
            return [
                GitHubItem(
                    repo="owner/repo",
                    number=10,
                    title="Preview",
                    body="",
                    labels=["bug"],
                    state="open",
                    html_url="https://example.test/10",
                    updated_at="2024-01-01T00:00:00Z",
                    created_at="2023-12-01T00:00:00Z",
                    is_pull_request=False,
                    author="alice",
                )
            ]

    github = FakeGitHub()
    services = make_services(tmp_path, repo="owner/repo", github=github)

    result = filter_repository(
        services, FilterParams(query="label:bug", repo=None, limit=2)
    )

    assert github.calls == [("label:bug repo:owner/repo", 2)]
    assert result.repo == "owner/repo"
    assert result.items[0].doc_type == "issue"
    assert result.items[0].labels == ["bug"]


def test_filter_repository_respects_query_repo(tmp_path: Path) -> None:
    class FakeGitHub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def search_issues(self, query: str, limit: int) -> List[GitHubItem]:
            self.calls.append((query, limit))
            return []

    github = FakeGitHub()
    services = make_services(tmp_path, repo="config/repo", github=github)

    result = filter_repository(
        services, FilterParams(query=" repo:other/repo state:open ", limit=1)
    )

    assert github.calls == [("repo:other/repo state:open", 1)]
    assert result.repo == "other/repo"


def test_filter_repository_requires_repo(tmp_path: Path) -> None:
    services = make_services(tmp_path, repo=None)

    with pytest.raises(CommandError):
        filter_repository(services, FilterParams(query="label:bug", limit=3))


def test_cluster_repository_summarizes_filters(monkeypatch, tmp_path: Path) -> None:
    doc = SimpleNamespace(
        number=101,
        title="Clustered",
        doc_type="issue",
        state="open",
        html_url="https://example.test/101",
        labels=["bug", "regression"],
    )
    fake_clusters = [RawCluster(cluster_id=0, centroid=[0.1, 0.2], documents=[doc])]
    monkeypatch.setattr(
        "issue_tools.commands.gather_documents_for_clustering",
        lambda vector_store, repo, filters, query_embedding, anchor_documents, limit, allow_repo_fallback: [doc],
    )
    monkeypatch.setattr(
        "issue_tools.commands.cluster_documents",
        lambda documents, k, max_k: fake_clusters,
    )
    services = make_services(tmp_path, repo="owner/repo")
    record_usage(services)

    result = cluster_repository(
        services,
        ClusterParams(query="label:bug state:open author:alice", k=1),
    )

    assert result.repo == "owner/repo"
    assert result.filters is not None
    assert result.filters.labels == ["bug"]
    assert result.filters.states == ["open"]
    assert result.filters.author == "alice"
    assert len(result.clusters) == 1
    assert result.clusters[0].documents[0].number == 101
    assert result.token_usage is not None


def test_cluster_repository_wraps_value_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "issue_tools.commands.cluster_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    services = make_services(tmp_path)

    with pytest.raises(CommandError) as exc:
        cluster_repository(services, ClusterParams(k=2))
    assert "bad" in str(exc.value)


def test_stream_repository_returns_items(tmp_path: Path) -> None:
    items = [
        SimpleNamespace(
            number=5,
            title="Fix bug",
            is_pull_request=False,
            state="open",
            html_url="https://example.test/5",
            updated_at="2024-01-02T00:00:00Z",
            created_at="2024-01-01T00:00:00Z",
            author="alice",
            labels=[],
        )
    ]
    indexer = FakeIndexer(stream_items=items)
    services = make_services(tmp_path, indexer=indexer)

    result = stream_repository(services, StreamParams())

    assert result.repo == "owner/repo"
    assert result.items[0].doc_type == "issue"
    assert result.items[0].title == "Fix bug"


def test_inspect_database_reports_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "vector.db"
    vector_store = SQLiteVectorStore(db_path)
    doc_common = {
        "repo": "owner/repo",
        "doc_type": "issue",
        "state": "open",
        "html_url": "https://example.test/1",
        "updated_at": "2024-01-03T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
        "author": "alice",
        "embedding": [0.1, 0.2],
        "embedding_model": "model-A",
        "embedding_dimensions": 2,
        "metadata": {},
    }
    vector_store.upsert_document(
        StoredDocument(
            doc_id="owner/repo#1",
            number=1,
            title="Issue one",
            body="",
            labels=["bug"],
            **doc_common,
        )
    )
    vector_store.upsert_document(
        StoredDocument(
            doc_id="owner/repo#2",
            number=2,
            title="Issue two",
            body="",
            labels=["enhancement"],
            **doc_common,
        )
    )
    vector_store.set_metadata("last_indexed_at:owner/repo", "2024-01-04T00:00:00Z")

    services = make_services(tmp_path, vector_store=vector_store)

    response = inspect_database(services, InspectDBParams(show=2))

    assert response.database_path == str(db_path)
    assert response.summaries[0].total == 2
    assert response.summaries[0].last_indexed == "2024-01-04T00:00:00Z"
    assert len(response.samples) == 2

    vector_store.close()


def test_evaluate_repository_uses_shared_runner(monkeypatch, tmp_path: Path) -> None:
    summary = EvaluationSummary(results=[], mean_precision=0.5, mean_reciprocal_rank=0.7)

    captured: dict[str, object] = {}

    def fake_run(search_service, repo, cases, *, top_k):
        captured["repo"] = repo
        captured["cases"] = list(cases)
        captured["top_k"] = top_k
        return summary

    monkeypatch.setattr("issue_tools.commands.run_evaluation", fake_run)

    services = make_services(tmp_path)
    record_usage(services)
    cases: Iterable[EvaluationCase] = [EvaluationCase(query="foo", expected_numbers=[1, 2])]

    result = evaluate_repository(services, EvaluateParams(cases=cases, repo=None, top_k=7))

    assert captured["repo"] == "owner/repo"
    assert isinstance(captured["cases"], list)
    assert captured["top_k"] == 7
    assert isinstance(result, EvaluationResponse)
    assert result.summary.mean_precision == pytest.approx(0.5)
    assert result.token_usage is not None


def test_summarize_token_usage_handles_empty(tmp_path: Path) -> None:
    tracker = TokenTracker(tmp_path / "tokens.json", {"model": 0.1})
    assert summarize_token_usage(tracker) is None


def test_summarize_token_usage_sorts_models(tmp_path: Path) -> None:
    tracker = TokenTracker(tmp_path / "tokens.json", {"b": 0.0, "a": 0.0})
    tracker.record(TokenUsage(model="b", prompt_tokens=1, response_tokens=2))
    tracker.record(TokenUsage(model="a", prompt_tokens=1, response_tokens=1))

    summary = summarize_token_usage(tracker, lifetime=True)

    assert summary is not None
    assert summary.scope == "lifetime"
    assert [stats.model for stats in summary.models] == ["a", "b"]

