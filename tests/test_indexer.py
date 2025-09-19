from datetime import datetime, timezone
import json

from rich.console import Console

from issue_tools.config import Config, EmbeddingConfig
from issue_tools.embeddings import EmbeddingResult, FakeEmbeddingsClient
from issue_tools.github_client import GitHubItem, GitHubLabel
from issue_tools.indexer import Indexer
from issue_tools import indexer as indexer_module
from issue_tools.vector_store import SQLiteVectorStore, StoredDocument
from issue_tools.token_tracker import TokenUsage


class DummyGitHubClient:
    def __init__(self, items):
        self.items = list(items)
        self.search_calls = []
        self.hydrate_calls = []

    def fetch_items(self, repo, since=None, limit=None, include_pulls=True):
        results = [
            item
            for item in self.items
            if since is None or item.updated_at > since
        ]
        if limit is not None:
            results = results[:limit]
        return results

    def search_issues(self, query, limit):
        self.search_calls.append((query, limit))
        return []

    def hydrate_search_results(self, repo, items, include_pulls=True):
        payload = list(items)
        self.hydrate_calls.append((repo, [item.number for item in payload], include_pulls))
        return payload


def make_item(number: int, updated: str) -> GitHubItem:
    return GitHubItem(
        repo="org/repo",
        number=number,
        title=f"Issue {number}",
        body="Test body",
        labels=[
            GitHubLabel(
                name="bug" if number % 2 else "docs",
                color="#ff0000" if number % 2 else "#00ff00",
            )
        ],
        state="open",
        html_url=f"https://github.com/org/repo/issues/{number}",
        updated_at=updated,
        created_at="2024-01-01T00:00:00Z",
        is_pull_request=False,
        author="tester",
        comments=[],
    )


def test_indexer_respects_daily_interval(tmp_path, monkeypatch):
    config = Config(
        data_dir=tmp_path,
        vector_db_path=tmp_path / "store.db",
        metadata_path=tmp_path / "metadata.json",
        token_usage_path=tmp_path / "tokens.json",
        initial_index_limit=10,
        embedding=EmbeddingConfig(batch_size=2),
    )
    config.ensure_data_dir()

    items = [
        make_item(1, "2024-01-01T10:00:00Z"),
        make_item(2, "2024-01-01T11:00:00Z"),
    ]
    github = DummyGitHubClient(items)
    vector_store = SQLiteVectorStore(config.vector_db_path)
    embeddings = FakeEmbeddingsClient(dimension=8)
    indexer = Indexer(config, github, vector_store, embeddings, console=Console(record=True))

    monkeypatch.setattr(
        indexer_module,
        "_utcnow",
        lambda: datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc),
    )
    result = indexer.run("org/repo")
    assert not result.skipped
    assert result.indexed == 2

    monkeypatch.setattr(
        indexer_module,
        "_utcnow",
        lambda: datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )
    skipped = indexer.run("org/repo")
    assert skipped.skipped

    github.items = [make_item(3, "2024-01-03T12:00:00Z")]
    monkeypatch.setattr(
        indexer_module,
        "_utcnow",
        lambda: datetime(2024, 1, 3, 13, 0, tzinfo=timezone.utc),
    )
    second = indexer.run("org/repo")
    assert not second.skipped
    assert second.fetched == 1

    vector_store.close()


def test_force_run_reindexes_full_history(tmp_path, monkeypatch):
    config = Config(
        data_dir=tmp_path,
        vector_db_path=tmp_path / "store.db",
        metadata_path=tmp_path / "metadata.json",
        token_usage_path=tmp_path / "tokens.json",
        initial_index_limit=1,
        embedding=EmbeddingConfig(batch_size=2),
    )
    config.ensure_data_dir()

    items = [
        make_item(3, "2024-01-03T00:00:00Z"),
        make_item(2, "2024-01-02T00:00:00Z"),
        make_item(1, "2024-01-01T00:00:00Z"),
    ]
    github = DummyGitHubClient(items)
    vector_store = SQLiteVectorStore(config.vector_db_path)
    embeddings = FakeEmbeddingsClient(dimension=8)
    indexer = Indexer(config, github, vector_store, embeddings, console=Console(record=True))

    monkeypatch.setattr(
        indexer_module,
        "_utcnow",
        lambda: datetime(2024, 1, 4, tzinfo=timezone.utc),
    )
    initial = indexer.run("org/repo")
    assert initial.indexed == 1
    assert initial.fetched == 1

    monkeypatch.setattr(
        indexer_module,
        "_utcnow",
        lambda: datetime(2024, 1, 5, tzinfo=timezone.utc),
    )
    forced = indexer.run("org/repo", force=True, limit=3)
    assert not forced.skipped
    assert forced.fetched == 3
    assert forced.indexed == 3

    cursor = vector_store.connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    assert cursor.fetchone()[0] == 3

    vector_store.close()


class RecordingEmbeddingsClient:
    def __init__(self):
        self.document_calls = []

    def embed_documents(self, texts):
        batch = list(texts)
        self.document_calls.append(batch)
        embeddings = [[0.5] for _ in batch]
        usage = [TokenUsage(model="recorder", total_tokens=len(text)) for text in batch]
        return EmbeddingResult(model="recorder", embeddings=embeddings, usage=usage)

    def embed_queries(self, texts):
        raise NotImplementedError


def test_indexer_includes_comments_in_embedding_input(tmp_path, monkeypatch):
    config = Config(
        data_dir=tmp_path,
        vector_db_path=tmp_path / "store.db",
        metadata_path=tmp_path / "metadata.json",
        token_usage_path=tmp_path / "tokens.json",
        embedding=EmbeddingConfig(batch_size=2),
    )
    config.ensure_data_dir()

    item = make_item(1, "2024-01-01T10:00:00Z")
    item.comments = ["First comment body", "Second comment body"]
    github = DummyGitHubClient([item])
    vector_store = SQLiteVectorStore(config.vector_db_path)
    embeddings = RecordingEmbeddingsClient()
    indexer = Indexer(config, github, vector_store, embeddings, console=Console(record=True))

    monkeypatch.setattr(
        indexer_module,
        "_utcnow",
        lambda: datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    result = indexer.run("org/repo", force=True)
    assert result.indexed == 1
    assert embeddings.document_calls, "embed_documents should be invoked"
    non_empty_calls = [call for call in embeddings.document_calls if call]
    assert non_empty_calls, "Expected at least one non-empty embedding batch"
    embedded_text = non_empty_calls[0][0]
    assert "First comment body" in embedded_text
    assert "Second comment body" in embedded_text

    vector_store.close()


def test_indexer_stores_label_metadata(tmp_path, monkeypatch):
    config = Config(
        data_dir=tmp_path,
        vector_db_path=tmp_path / "store.db",
        metadata_path=tmp_path / "metadata.json",
        token_usage_path=tmp_path / "tokens.json",
        embedding=EmbeddingConfig(batch_size=1),
    )
    config.ensure_data_dir()

    item = GitHubItem(
        repo="org/repo",
        number=1,
        title="Label test",
        body="Body",
        labels=[GitHubLabel(name="bug", color="#123456")],
        state="open",
        html_url="https://github.com/org/repo/issues/1",
        updated_at="2024-01-01T01:00:00Z",
        created_at="2024-01-01T00:00:00Z",
        is_pull_request=False,
        author="tester",
        comments=[],
    )

    github = DummyGitHubClient([item])
    vector_store = SQLiteVectorStore(config.vector_db_path)
    embeddings = FakeEmbeddingsClient(dimension=4)
    indexer = Indexer(config, github, vector_store, embeddings, console=Console(record=True))

    monkeypatch.setattr(
        indexer_module,
        "_utcnow",
        lambda: datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    result = indexer.run("org/repo", force=True)
    assert result.indexed == 1

    cursor = vector_store.connection.cursor()
    cursor.execute("SELECT labels, metadata FROM documents WHERE number = 1")
    row = cursor.fetchone()
    assert json.loads(row["labels"]) == ["bug"]
    metadata = json.loads(row["metadata"])
    assert metadata.get("label_details") == [{"name": "bug", "color": "#123456"}]

    vector_store.close()
