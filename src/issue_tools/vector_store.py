"""SQLite-backed vector store for GitHub documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence
from pathlib import Path
import json
import sqlite3
from array import array

import numpy as np

from .filters import FilterCriteria


@dataclass(slots=True)
class StoredDocument:
    """Representation of a document stored in the vector database."""

    doc_id: str
    repo: str
    number: int
    doc_type: str
    title: str
    body: str
    labels: List[str]
    state: str
    html_url: str
    updated_at: str
    created_at: str
    author: Optional[str]
    embedding: List[float]
    embedding_model: str
    embedding_dimensions: int
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    document: StoredDocument
    score: float


class SQLiteVectorStore:
    """Persist embeddings to SQLite and perform similarity search."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT NOT NULL,
                repo TEXT NOT NULL,
                number INTEGER NOT NULL,
                type TEXT NOT NULL,
                title TEXT,
                body TEXT,
                labels TEXT,
                state TEXT,
                html_url TEXT,
                updated_at TEXT,
                created_at TEXT,
                author TEXT,
                embedding BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dimensions INTEGER NOT NULL,
                metadata TEXT,
                PRIMARY KEY (doc_id, embedding_model)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_repo ON documents(repo)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_state ON documents(state)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)"
        )
        self.connection.commit()

    @staticmethod
    def _serialize_embedding(embedding: Sequence[float]) -> bytes:
        arr = array("f", embedding)
        return arr.tobytes()

    @staticmethod
    def _deserialize_embedding(blob: bytes, dimensions: int) -> List[float]:
        arr = array("f")
        arr.frombytes(blob)
        return list(arr[:dimensions])

    def upsert_document(
        self,
        document: StoredDocument,
    ) -> None:
        metadata_json = json.dumps(document.metadata or {}, sort_keys=True)
        payload = (
            document.doc_id,
            document.repo,
            document.number,
            document.doc_type,
            document.title,
            document.body,
            json.dumps(document.labels),
            document.state,
            document.html_url,
            document.updated_at,
            document.created_at,
            document.author,
            self._serialize_embedding(document.embedding),
            document.embedding_model,
            document.embedding_dimensions,
            metadata_json,
        )
        query = (
            "INSERT INTO documents (doc_id, repo, number, type, title, body, labels, state, html_url, "
            "updated_at, created_at, author, embedding, embedding_model, embedding_dimensions, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(doc_id, embedding_model) DO UPDATE SET "
            "title=excluded.title, body=excluded.body, labels=excluded.labels, state=excluded.state, "
            "html_url=excluded.html_url, updated_at=excluded.updated_at, created_at=excluded.created_at, "
            "author=excluded.author, embedding=excluded.embedding, embedding_dimensions=excluded.embedding_dimensions, "
            "metadata=excluded.metadata"
        )
        with self.connection:
            self.connection.execute(query, payload)

    def bulk_upsert(self, documents: Iterable[StoredDocument]) -> None:
        for document in documents:
            self.upsert_document(document)

    def _row_to_document(self, row: sqlite3.Row) -> StoredDocument:
        return StoredDocument(
            doc_id=row["doc_id"],
            repo=row["repo"],
            number=row["number"],
            doc_type=row["type"],
            title=row["title"],
            body=row["body"],
            labels=list(json.loads(row["labels"])) if row["labels"] else [],
            state=row["state"],
            html_url=row["html_url"],
            updated_at=row["updated_at"],
            created_at=row["created_at"],
            author=row["author"],
            embedding=self._deserialize_embedding(row["embedding"], row["embedding_dimensions"]),
            embedding_model=row["embedding_model"],
            embedding_dimensions=row["embedding_dimensions"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def get_document(
        self, doc_id: str, *, embedding_model: Optional[str] = None
    ) -> Optional[StoredDocument]:
        query = "SELECT * FROM documents WHERE doc_id = ?"
        params: List[object] = [doc_id]
        if embedding_model:
            query += " AND embedding_model = ?"
            params.append(embedding_model)
        query += " LIMIT 1"
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        if row:
            return self._row_to_document(row)
        return None

    def search(
        self,
        repo: Optional[str],
        query_embedding: Sequence[float],
        filters: FilterCriteria,
        *,
        limit: Optional[int] = 10,
    ) -> List[SearchResult]:
        """Search for documents using cosine similarity after applying filters."""

        where_clauses = []
        params: List[object] = []
        if repo:
            where_clauses.append("repo = ?")
            params.append(repo)
        if filters.types:
            placeholders = ",".join(["?"] * len(filters.types))
            where_clauses.append(f"type IN ({placeholders})")
            params.extend(filters.types)
        if filters.states:
            placeholders = ",".join(["?"] * len(filters.states))
            where_clauses.append(f"state IN ({placeholders})")
            params.extend(filters.states)
        if filters.numbers:
            placeholders = ",".join(["?"] * len(filters.numbers))
            where_clauses.append(f"number IN ({placeholders})")
            params.extend(filters.numbers)
        if filters.updated_filter:
            op, value = filters.updated_filter
            where_clauses.append(f"updated_at {op} ?")
            params.append(value)
        if filters.created_filter:
            op, value = filters.created_filter
            where_clauses.append(f"created_at {op} ?")
            params.append(value)
        if filters.author:
            where_clauses.append("author = ?")
            params.append(filters.author)

        where_sql = " AND ".join(where_clauses)
        if where_sql:
            where_sql = "WHERE " + where_sql
        query = f"SELECT * FROM documents {where_sql}"
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        documents = [self._row_to_document(row) for row in rows]
        documents = [doc for doc in documents if filters.matches(doc)]
        if not documents:
            return []

        matrix = np.array([doc.embedding for doc in documents], dtype=np.float32)
        query_vec = np.array(query_embedding, dtype=np.float32)
        if np.linalg.norm(query_vec) == 0:
            scored = [SearchResult(document=doc, score=0.0) for doc in documents]
            return self._limit_results(scored, limit)

        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        # Avoid division by zero
        matrix_norms[matrix_norms == 0] = 1e-12
        similarities = matrix @ query_vec / (matrix_norms * query_norm)

        scored = [
            SearchResult(document=doc, score=float(score))
            for doc, score in zip(documents, similarities)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return self._limit_results(scored, limit)

    def all_embeddings(
        self,
        repo: Optional[str] = None,
        filters: Optional[FilterCriteria] = None,
        limit: Optional[int] = None,
    ) -> List[StoredDocument]:
        where_clauses = []
        params: List[object] = []
        if repo:
            where_clauses.append("repo = ?")
            params.append(repo)
        if filters and filters.types:
            placeholders = ",".join(["?"] * len(filters.types))
            where_clauses.append(f"type IN ({placeholders})")
            params.extend(filters.types)
        where_sql = " AND ".join(where_clauses)
        if where_sql:
            where_sql = "WHERE " + where_sql
        order_clause = "ORDER BY updated_at DESC"
        query_parts = ["SELECT * FROM documents"]
        if where_sql:
            query_parts.append(where_sql)
        query_parts.append(order_clause)
        query = " ".join(query_parts)
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(int(limit))
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        documents = [self._row_to_document(row) for row in rows]
        if filters:
            documents = [doc for doc in documents if filters.matches(doc)]
        return documents

    def get_by_numbers(
        self,
        repo: str,
        numbers: Sequence[int],
        *,
        types: Optional[Sequence[str]] = None,
    ) -> List[StoredDocument]:
        if not numbers:
            return []

        cursor = self.connection.cursor()
        seen_ids: set[str] = set()
        ordered: List[StoredDocument] = []

        for number in numbers:
            params: List[object] = [repo, number]
            query = "SELECT * FROM documents WHERE repo = ? AND number = ?"
            if types:
                placeholders = ",".join(["?"] * len(types))
                query += f" AND type IN ({placeholders})"
                params.extend(types)
            query += " ORDER BY updated_at DESC"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            for row in rows:
                document = self._row_to_document(row)
                if document.doc_id in seen_ids:
                    continue
                ordered.append(document)
                seen_ids.add(document.doc_id)

        return ordered

    @staticmethod
    def _limit_results(
        results: List[SearchResult],
        limit: Optional[int],
    ) -> List[SearchResult]:
        if limit is None or limit <= 0:
            return results
        return results[:limit]

    def get_metadata(self, key: str) -> Optional[str]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return row[0]
        return None

    def all_metadata(self) -> Dict[str, str]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT key, value FROM metadata")
        rows = cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}

    def set_metadata(self, key: str, value: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def delete_documents_for_model(self, embedding_model: str) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM documents WHERE embedding_model = ?",
                (embedding_model,),
            )
