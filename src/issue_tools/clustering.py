"""K-means clustering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .filters import FilterCriteria
from .vector_store import SQLiteVectorStore, StoredDocument


@dataclass(slots=True)
class Cluster:
    cluster_id: int
    centroid: List[float]
    documents: List[StoredDocument]

    def top_labels(self, limit: int = 3) -> List[str]:
        label_counts: dict[str, int] = {}
        for doc in self.documents:
            for label in doc.labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        return [
            label
            for label, _ in sorted(label_counts.items(), key=lambda item: item[1], reverse=True)[
                :limit
            ]
        ]


def gather_documents_for_clustering(
    vector_store: SQLiteVectorStore,
    repo: str,
    filters: FilterCriteria,
    *,
    query_embedding: Sequence[float] | None,
    anchor_documents: Sequence[StoredDocument] | None = None,
    limit: int | None = None,
    allow_repo_fallback: bool = True,
) -> List[StoredDocument]:
    """Collect documents to cluster based on query and/or anchor issues."""

    anchor_documents = anchor_documents or []
    selected: Dict[str, tuple[StoredDocument, float]] = {}
    order: List[str] = []

    def add_document(document: StoredDocument, score: float) -> None:
        existing = selected.get(document.doc_id)
        if existing is None:
            selected[document.doc_id] = (document, score)
            order.append(document.doc_id)
            return
        if score > existing[1]:
            selected[document.doc_id] = (document, score)

    for index, anchor in enumerate(anchor_documents):
        # Prioritise anchor documents by giving them the highest scores.
        add_document(anchor, float("inf") - index)
        search_results = vector_store.search(
            repo,
            anchor.embedding,
            filters,
            limit=limit,
        )
        for result in search_results:
            add_document(result.document, result.score)

    if query_embedding is not None:
        search_results = vector_store.search(
            repo,
            query_embedding,
            filters,
            limit=limit,
        )
        for result in search_results:
            add_document(result.document, result.score)

    if not order:
        if allow_repo_fallback:
            return vector_store.all_embeddings(repo, filters, limit=limit)
        return []

    documents = [selected[doc_id][0] for doc_id in order]
    if limit is not None and limit > 0:
        return documents[:limit]
    return documents


def _run_kmeans(
    data: np.ndarray,
    k: int,
    max_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(data):
        raise ValueError("k cannot exceed number of samples")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=k, replace=False)
    centroids = data[indices]

    for _ in range(max_iterations):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        converged = True
        for idx in range(k):
            members = data[assignments == idx]
            if len(members) == 0:
                # Re-initialise empty clusters to a random data point to avoid stagnation.
                new_centroids[idx] = data[rng.integers(len(data))]
                converged = False
                continue
            centroid = members.mean(axis=0)
            if not np.allclose(centroid, new_centroids[idx]):
                converged = False
            new_centroids[idx] = centroid
        centroids = new_centroids
        if converged:
            break

    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    return centroids, assignments


def _silhouette_score(distance_matrix: np.ndarray, assignments: np.ndarray) -> float:
    unique_clusters = np.unique(assignments)
    if len(unique_clusters) < 2:
        return 0.0

    cluster_members = {cluster: np.flatnonzero(assignments == cluster) for cluster in unique_clusters}
    scores: List[float] = []
    for index, cluster in enumerate(assignments):
        members = cluster_members[cluster]
        if len(members) <= 1:
            scores.append(0.0)
            continue

        # Mean intra-cluster distance (excluding the point itself).
        a = float(
            distance_matrix[index, members].sum() / (len(members) - 1)
        )

        # Smallest mean distance to points in any other cluster.
        other_distances = [
            float(distance_matrix[index, other_members].mean())
            for other_cluster, other_members in cluster_members.items()
            if other_cluster != cluster and len(other_members) > 0
        ]
        if not other_distances:
            scores.append(0.0)
            continue
        b = min(other_distances)

        denominator = max(a, b)
        if denominator == 0:
            scores.append(0.0)
            continue
        scores.append((b - a) / denominator)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def _determine_optimal_k(
    data: np.ndarray,
    *,
    max_iterations: int,
    seed: int,
    max_clusters: int,
) -> int:
    sample_count = len(data)
    if sample_count <= 1:
        return sample_count

    candidate_max = min(max_clusters, sample_count)
    if candidate_max <= 1:
        return 1

    candidate_max = max(2, candidate_max)

    distance_matrix = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
    best_k = 2
    best_score = -1.0
    for candidate in range(2, candidate_max + 1):
        centroids, assignments = _run_kmeans(
            data,
            candidate,
            max_iterations,
            seed + candidate,
        )
        score = _silhouette_score(distance_matrix, assignments)
        if score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and candidate < best_k):
            best_k = candidate
            best_score = score

    if best_score <= 0:
        return 1
    return best_k


def cluster_documents(
    documents: Sequence[StoredDocument],
    *,
    k: int | None = None,
    max_iterations: int = 50,
    seed: int = 13,
    max_k: int | None = None,
) -> List[Cluster]:
    if not documents:
        return []

    data = np.array([doc.embedding for doc in documents], dtype=np.float32)
    if k is None:
        max_clusters = max_k if max_k is not None else min(len(documents), 8)
        k = _determine_optimal_k(
            data,
            max_iterations=max_iterations,
            seed=seed,
            max_clusters=max_clusters,
        )
    else:
        if k <= 0:
            raise ValueError("k must be positive")
        if k > len(documents):
            raise ValueError("k cannot exceed number of documents")

    centroids, assignments = _run_kmeans(data, k, max_iterations, seed)
    clusters: List[Cluster] = []
    for idx in range(k):
        members = [
            doc
            for doc, assignment in zip(documents, assignments)
            if assignment == idx
        ]
        clusters.append(
            Cluster(
                cluster_id=idx,
                centroid=centroids[idx].tolist(),
                documents=members,
            )
        )
    return clusters


def kmeans(
    vector_store: SQLiteVectorStore,
    repo: str | None,
    filters: FilterCriteria | None,
    *,
    k: int,
    max_iterations: int = 50,
    seed: int = 13,
) -> List[Cluster]:
    documents = vector_store.all_embeddings(repo, filters)
    return cluster_documents(
        documents,
        k=k,
        max_iterations=max_iterations,
        seed=seed,
    )

