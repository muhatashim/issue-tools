"""Semantic search helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .embeddings import EmbeddingsClient
from .filters import parse_query_with_filters
from .labels import LabelDetail, extract_label_details
from .vector_store import SQLiteVectorStore, SearchResult as VectorSearchResult


@dataclass(slots=True)
class SearchMatch:
    doc_id: str
    number: int
    title: str
    url: str
    score: float
    state: str
    doc_type: str
    labels: List[str]
    updated_at: str
    created_at: str
    author: Optional[str]
    label_details: List[LabelDetail] = field(default_factory=list)


class SearchService:
    """High level API for semantic search over the vector store."""

    def __init__(
        self,
        vector_store: SQLiteVectorStore,
        embeddings: EmbeddingsClient,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings = embeddings

    def search(
        self,
        repo: Optional[str],
        raw_query: str,
        *,
        limit: int = 5,
    ) -> List[SearchMatch]:
        """Perform semantic search with optional GitHub-like filters."""

        query_text, filters = parse_query_with_filters(raw_query)
        filters = filters.with_repo(repo)
        query_text = query_text or ""
        embedding_result = self.embeddings.embed_queries([query_text])
        if not embedding_result.embeddings:
            return []
        query_embedding = embedding_result.embeddings[0]
        matches = self.vector_store.search(repo, query_embedding, filters, limit=limit)
        return [self._to_match(result) for result in matches]

    def _to_match(self, result: VectorSearchResult) -> SearchMatch:
        doc = result.document
        return SearchMatch(
            doc_id=doc.doc_id,
            number=doc.number,
            title=doc.title,
            url=doc.html_url,
            score=result.score,
            state=doc.state,
            doc_type=doc.doc_type,
            labels=doc.labels,
            label_details=extract_label_details(
                doc.metadata,
                fallback_labels=doc.labels,
            ),
            updated_at=doc.updated_at,
            created_at=doc.created_at,
            author=doc.author,
        )

