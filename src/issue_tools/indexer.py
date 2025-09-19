"""Indexing pipeline for GitHub issues and pull requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import Config
from .embeddings import EmbeddingsClient
from .filters import parse_query_with_filters
from .github_client import GitHubClient, GitHubItem
from .labels import label_details_from_github
from .vector_store import SQLiteVectorStore, StoredDocument


@dataclass(slots=True)
class IndexResult:
    indexed: int
    fetched: int
    skipped: bool
    message: str
    last_indexed_at: Optional[str]


class Indexer:
    """Coordinates GitHub fetches and vector database updates."""

    def __init__(
        self,
        config: Config,
        github: GitHubClient,
        vector_store: SQLiteVectorStore,
        embeddings: EmbeddingsClient,
        console: Optional[Console] = None,
    ) -> None:
        self.config = config
        self.github = github
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.console = console or Console()

    def run(
        self,
        repo: str,
        *,
        force: bool = False,
        limit: Optional[int] = None,
        include_pulls: bool = True,
        filter_query: Optional[str] = None,
    ) -> IndexResult:
        """Run the indexing pipeline."""

        now = _utcnow()
        now_iso = _isoformat(now)
        metadata_key = self._metadata_key(repo)
        last_indexed_raw = self.vector_store.get_metadata(metadata_key)
        initial_run = last_indexed_raw is None
        last_indexed = _parse_iso(last_indexed_raw) if last_indexed_raw else None

        filter_text = (filter_query or "").strip()
        using_filter = bool(filter_text)

        if not using_filter and not force and last_indexed is not None:
            interval = timedelta(hours=self.config.index_interval_hours)
            if now - last_indexed < interval:
                message = (
                    "Indexing skipped; next run in "
                    f"{_format_timedelta(interval - (now - last_indexed))}."
                )
                return IndexResult(
                    indexed=0,
                    fetched=0,
                    skipped=True,
                    message=message,
                    last_indexed_at=last_indexed_raw,
                )

        since = None
        if not using_filter and not force and not initial_run and last_indexed is not None:
            since = _isoformat(last_indexed)
        fetch_limit = limit
        if not using_filter and initial_run and fetch_limit is None:
            fetch_limit = self.config.initial_index_limit

        embedding_model = self._current_embedding_model()
        skip_unchanged = not force
        fetched_count = 0
        items: List[GitHubItem] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
            console=self.console,
        ) as progress:
            description = (
                "Searching GitHub..." if using_filter else "Fetching issues from GitHub..."
            )
            task_id = progress.add_task(description, total=None)
            if using_filter:
                query = self._prepare_filter_query(repo, filter_text)
                search_limit = fetch_limit or self.config.initial_index_limit
                search_results = self.github.search_issues(query, limit=search_limit)
                candidate_results = [
                    item
                    for item in search_results
                    if (not item.repo or item.repo == repo)
                    and (include_pulls or not item.is_pull_request)
                ]
                fetched_count = len(candidate_results)
                changed_candidates = self._filter_changed_items(
                    repo,
                    candidate_results,
                    embedding_model,
                    skip_unchanged=skip_unchanged,
                )
                progress.update(
                    task_id,
                    description=f"Hydrating {len(changed_candidates)} documents...",
                )
                if changed_candidates:
                    hydrated = self.github.hydrate_search_results(
                        repo, changed_candidates, include_pulls=include_pulls
                    )
                    items = self._filter_changed_items(
                        repo,
                        hydrated,
                        embedding_model,
                        skip_unchanged=skip_unchanged,
                    )
                else:
                    items = []
            else:
                fetched_items = self.github.fetch_items(
                    repo,
                    since=since,
                    limit=fetch_limit,
                    include_pulls=include_pulls,
                )
                fetched_count = len(fetched_items)
                items = self._filter_changed_items(
                    repo,
                    fetched_items,
                    embedding_model,
                    skip_unchanged=skip_unchanged,
                )
            progress.update(task_id, description=f"Embedding {len(items)} documents...")
            documents = self._build_documents(repo, items, now_iso)
            progress.update(task_id, description="Persisting embeddings...")
            if documents:
                self.vector_store.bulk_upsert(documents)
            progress.update(task_id, description="Indexing complete")

        last_indexed_value = last_indexed_raw
        if not using_filter:
            self.vector_store.set_metadata(metadata_key, now_iso)
            last_indexed_value = now_iso

        indexed_count = len(documents)
        unchanged_count = max(fetched_count - len(items), 0)
        message = f"Indexed {indexed_count} documents"
        if unchanged_count:
            message += f" ({unchanged_count} unchanged skipped)"
        return IndexResult(
            indexed=indexed_count,
            fetched=fetched_count,
            skipped=False,
            message=message,
            last_indexed_at=last_indexed_value,
        )

    def stream_new(self, repo: str) -> List[GitHubItem]:
        """Return items updated since the last indexing run."""

        last_indexed_raw = self.vector_store.get_metadata(self._metadata_key(repo))
        if not last_indexed_raw:
            return []
        items = self.github.fetch_items(repo, since=last_indexed_raw, include_pulls=True)
        return [item for item in items if item.updated_at > last_indexed_raw]

    def _build_documents(
        self,
        repo: str,
        items: List[GitHubItem],
        indexed_at: str,
    ) -> List[StoredDocument]:
        documents: List[StoredDocument] = []
        if not items:
            return documents

        batch_size = self.config.embedding.batch_size
        texts = [item.to_document(self.config.embedding.max_chars) for item in items]
        for start in range(0, len(items), batch_size):
            end = start + batch_size
            batch_items = items[start:end]
            batch_texts = texts[start:end]
            result = self.embeddings.embed_documents(batch_texts)
            if len(result.embeddings) != len(batch_items):
                raise RuntimeError("Embedding count does not match item count")
            for issue, embedding in zip(batch_items, result.embeddings):
                details = label_details_from_github(issue.labels)
                label_names = [detail.name for detail in details]
                metadata = {
                    "source": "github",
                    "task_type": self.config.embedding.document_task_type,
                    "indexed_at": indexed_at,
                    "model": result.model,
                }
                if details:
                    metadata["label_details"] = [
                        {"name": detail.name, **({"color": detail.color} if detail.color else {})}
                        for detail in details
                    ]
                doc = StoredDocument(
                    doc_id=_doc_id(repo, issue),
                    repo=repo,
                    number=issue.number,
                    doc_type="pull_request" if issue.is_pull_request else "issue",
                    title=issue.title,
                    body=issue.body,
                    labels=label_names,
                    state=issue.state.lower(),
                    html_url=issue.html_url,
                    updated_at=issue.updated_at,
                    created_at=issue.created_at,
                    author=issue.author,
                    embedding=embedding,
                    embedding_model=result.model,
                    embedding_dimensions=len(embedding),
                    metadata=metadata,
                )
                documents.append(doc)
        return documents

    def _current_embedding_model(self) -> Optional[str]:
        try:
            result = self.embeddings.embed_documents([])
        except Exception:
            return None
        model = getattr(result, "model", None)
        if isinstance(model, str) and model:
            return model
        return None

    def _filter_changed_items(
        self,
        repo: str,
        items: Iterable[GitHubItem],
        embedding_model: Optional[str],
        *,
        skip_unchanged: bool,
    ) -> List[GitHubItem]:
        if not skip_unchanged:
            return list(items)

        changed: List[GitHubItem] = []
        for item in items:
            doc_id = _doc_id(repo, item)
            if embedding_model:
                existing = self.vector_store.get_document(doc_id, embedding_model=embedding_model)
            else:
                existing = self.vector_store.get_document(doc_id)
            if existing:
                existing_updated = (existing.updated_at or "").strip()
                item_updated = (item.updated_at or "").strip()
                if existing_updated and item_updated and existing_updated >= item_updated:
                    continue
            changed.append(item)
        return changed

    def _prepare_filter_query(self, repo: str, query: str) -> str:
        trimmed = query.strip()
        if not trimmed:
            return f"repo:{repo}"
        try:
            _, filters = parse_query_with_filters(trimmed)
        except Exception:
            filters = None
        repo_filter = filters.repo if filters else None
        if not repo_filter or repo_filter.lower() != repo.lower():
            trimmed = f"{trimmed} repo:{repo}".strip()
        return trimmed

    def _metadata_key(self, repo: str) -> str:
        return f"last_indexed_at:{repo}"


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_timedelta(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append("<1m")
    return " ".join(parts)


def _doc_id(repo: str, item: GitHubItem) -> str:
    kind = "PR" if item.is_pull_request else "ISSUE"
    return f"{repo}#{kind}#{item.number}"
