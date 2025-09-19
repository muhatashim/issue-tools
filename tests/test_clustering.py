from issue_tools.clustering import cluster_documents, gather_documents_for_clustering
from issue_tools.filters import FilterCriteria
from issue_tools.vector_store import SQLiteVectorStore, StoredDocument


def _make_document(doc_id: str, embedding: list[float]) -> StoredDocument:
    return StoredDocument(
        doc_id=doc_id,
        repo="acme/repo",
        number=int(doc_id.split("-")[-1]),
        doc_type="issue",
        title=f"Doc {doc_id}",
        body="",
        labels=[],
        state="open",
        html_url="https://example.com",
        updated_at="2024-01-01T00:00:00Z",
        created_at="2024-01-01T00:00:00Z",
        author=None,
        embedding=embedding,
        embedding_model="test-model",
        embedding_dimensions=len(embedding),
        metadata={},
    )


def test_cluster_documents_auto_selects_k() -> None:
    documents = [
        _make_document("doc-1", [0.0, 0.0]),
        _make_document("doc-2", [0.1, 0.0]),
        _make_document("doc-3", [5.0, 5.0]),
        _make_document("doc-4", [5.1, 5.1]),
        _make_document("doc-5", [10.0, 10.0]),
        _make_document("doc-6", [10.2, 10.1]),
    ]

    clusters = cluster_documents(documents, max_iterations=200, seed=42, max_k=6)

    assert len(clusters) == 3
    sizes = sorted(len(cluster.documents) for cluster in clusters)
    assert sizes == [2, 2, 2]


def test_cluster_documents_respects_manual_k() -> None:
    documents = [
        _make_document("doc-1", [0.0, 0.0]),
        _make_document("doc-2", [0.0, 0.1]),
        _make_document("doc-3", [10.0, 10.0]),
        _make_document("doc-4", [10.0, 10.1]),
    ]

    clusters = cluster_documents(documents, k=2, max_iterations=100, seed=7)

    assert len(clusters) == 2
    sizes = sorted(len(cluster.documents) for cluster in clusters)
    assert sizes == [2, 2]


def test_cluster_documents_handles_single_document() -> None:
    document = _make_document("doc-1", [1.0, 1.0])

    clusters = cluster_documents([document], max_iterations=50, seed=1)

    assert len(clusters) == 1
    assert clusters[0].documents == [document]


def test_gather_documents_for_clustering_with_anchor(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "cluster.db")
    anchor = _make_document("doc-1", [1.0, 0.0])
    similar = _make_document("doc-2", [0.9, 0.1])
    distant = _make_document("doc-3", [-1.0, 0.0])

    store.upsert_document(anchor)
    store.upsert_document(similar)
    store.upsert_document(distant)

    anchor_docs = store.get_by_numbers("acme/repo", [1])
    filters = FilterCriteria()

    documents = gather_documents_for_clustering(
        store,
        "acme/repo",
        filters,
        query_embedding=None,
        anchor_documents=anchor_docs,
        limit=2,
        allow_repo_fallback=False,
    )

    assert [doc.doc_id for doc in documents] == [anchor.doc_id, similar.doc_id]


def test_gather_documents_for_clustering_without_matches(tmp_path) -> None:
    store = SQLiteVectorStore(tmp_path / "cluster-empty.db")
    document = _make_document("doc-1", [1.0, 0.0])
    store.upsert_document(document)

    filters = FilterCriteria(labels={"missing"})

    documents = gather_documents_for_clustering(
        store,
        "acme/repo",
        filters,
        query_embedding=[0.5, 0.5],
        anchor_documents=[],
        limit=5,
        allow_repo_fallback=False,
    )

    assert documents == []
