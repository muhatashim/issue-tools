from issue_tools.embeddings import FakeEmbeddingsClient
from issue_tools.search import SearchService
from issue_tools.vector_store import SQLiteVectorStore, StoredDocument


def make_doc(doc_id: str, number: int, title: str, labels, embedding, metadata=None):
    return StoredDocument(
        doc_id=doc_id,
        repo="org/repo",
        number=number,
        doc_type="issue",
        title=title,
        body=title,
        labels=labels,
        state="open",
        html_url=f"https://github.com/org/repo/issues/{number}",
        updated_at="2024-01-02T00:00:00Z",
        created_at="2024-01-01T00:00:00Z",
        author="alice",
        embedding=embedding,
        embedding_model="fake-document",
        embedding_dimensions=len(embedding),
        metadata=metadata or {"model": "fake-document"},
    )


def test_search_service_respects_filters(tmp_path):
    store = SQLiteVectorStore(tmp_path / "store.db")
    embeddings = FakeEmbeddingsClient(dimension=8)

    doc_a = embeddings.embed_documents(["fix crash bug"]).embeddings[0]
    doc_b = embeddings.embed_documents(["update docs guide"]).embeddings[0]

    store.upsert_document(make_doc("org/repo#ISSUE#1", 1, "Crash bug", ["bug"], doc_a))
    store.upsert_document(make_doc("org/repo#ISSUE#2", 2, "Docs", ["docs"], doc_b))

    service = SearchService(store, embeddings)

    results = service.search("org/repo", "crash label:bug", limit=5)
    assert results
    assert results[0].number == 1
    assert results[0].label_details
    assert results[0].label_details[0].name == "bug"
    assert results[0].label_details[0].color is None

    results = service.search("org/repo", "docs label:docs", limit=5)
    assert results
    assert results[0].number == 2
    assert results[0].label_details
    assert results[0].label_details[0].name == "docs"
