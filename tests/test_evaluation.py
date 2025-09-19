from issue_tools.embeddings import FakeEmbeddingsClient
from issue_tools.evaluation import EvaluationCase, evaluate
from issue_tools.search import SearchService
from issue_tools.vector_store import SQLiteVectorStore, StoredDocument


def add_doc(store, embeddings, text, number):
    embedding = embeddings.embed_documents([text]).embeddings[0]
    store.upsert_document(
        StoredDocument(
            doc_id=f"org/repo#ISSUE#{number}",
            repo="org/repo",
            number=number,
            doc_type="issue",
            title=text,
            body=text,
            labels=["bug" if "bug" in text else "docs"],
            state="open",
            html_url=f"https://github.com/org/repo/issues/{number}",
            updated_at="2024-01-02T00:00:00Z",
            created_at="2024-01-01T00:00:00Z",
            author="alice",
            embedding=embedding,
            embedding_model="fake-document",
            embedding_dimensions=len(embedding),
            metadata={"model": "fake-document"},
        )
    )


def test_evaluation_metrics(tmp_path):
    store = SQLiteVectorStore(tmp_path / "store.db")
    embeddings = FakeEmbeddingsClient(dimension=8)
    add_doc(store, embeddings, "crash bug in login", 1)
    add_doc(store, embeddings, "improve docs", 2)

    service = SearchService(store, embeddings)
    cases = [
        EvaluationCase(query="login crash", expected_numbers=[1]),
        EvaluationCase(query="docs", expected_numbers=[2]),
    ]

    summary = evaluate(service, "org/repo", cases, top_k=3)
    assert summary.mean_precision > 0.0
    assert summary.mean_reciprocal_rank > 0.0
