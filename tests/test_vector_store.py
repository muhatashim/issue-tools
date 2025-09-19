from issue_tools.embeddings import FakeEmbeddingsClient
from issue_tools.filters import FilterCriteria
from issue_tools.vector_store import SQLiteVectorStore, StoredDocument


def test_vector_store_search_with_filters(tmp_path):
    store = SQLiteVectorStore(tmp_path / "store.db")
    embeddings = FakeEmbeddingsClient(dimension=8)

    doc_embedding = embeddings.embed_documents(["Login bug in API"]).embeddings[0]
    other_embedding = embeddings.embed_documents(["Documentation update"]).embeddings[0]

    store.upsert_document(
        StoredDocument(
            doc_id="org/repo#ISSUE#1",
            repo="org/repo",
            number=1,
            doc_type="issue",
            title="Login bug",
            body="Fails when token expired",
            labels=["bug"],
            state="open",
            html_url="https://github.com/org/repo/issues/1",
            updated_at="2024-01-02T00:00:00Z",
            created_at="2024-01-01T00:00:00Z",
            author="alice",
            embedding=doc_embedding,
            embedding_model="fake-document",
            embedding_dimensions=len(doc_embedding),
            metadata={"model": "fake-document"},
        )
    )
    store.upsert_document(
        StoredDocument(
            doc_id="org/repo#ISSUE#2",
            repo="org/repo",
            number=2,
            doc_type="issue",
            title="Docs update",
            body="Revise README",
            labels=["docs"],
            state="open",
            html_url="https://github.com/org/repo/issues/2",
            updated_at="2024-01-03T00:00:00Z",
            created_at="2024-01-02T00:00:00Z",
            author="bob",
            embedding=other_embedding,
            embedding_model="fake-document",
            embedding_dimensions=len(other_embedding),
            metadata={"model": "fake-document"},
        )
    )

    query_embedding = embeddings.embed_queries(["login failure bug"]).embeddings[0]
    filters = FilterCriteria()
    results = store.search("org/repo", query_embedding, filters, limit=5)
    assert results
    assert results[0].document.number == 1

    filters = FilterCriteria(labels={"docs"})
    query_embedding = embeddings.embed_queries(["documentation"]).embeddings[0]
    filtered = store.search("org/repo", query_embedding, filters, limit=5)
    assert filtered
    assert filtered[0].document.number == 2


def test_all_embeddings_limit(tmp_path):
    store = SQLiteVectorStore(tmp_path / "store-limit.db")
    embeddings = FakeEmbeddingsClient(dimension=4)

    for index in range(3):
        vector = embeddings.embed_documents([f"Doc {index}"]).embeddings[0]
        store.upsert_document(
            StoredDocument(
                doc_id=f"org/repo#ISSUE#{index}",
                repo="org/repo",
                number=index,
                doc_type="issue",
                title=f"Doc {index}",
                body="",
                labels=[],
                state="open",
                html_url=f"https://github.com/org/repo/issues/{index}",
                updated_at=f"2024-01-0{index + 1}T00:00:00Z",
                created_at="2024-01-01T00:00:00Z",
                author=None,
                embedding=vector,
                embedding_model="fake-document",
                embedding_dimensions=len(vector),
                metadata={},
            )
        )

    all_docs = store.all_embeddings("org/repo")
    assert len(all_docs) == 3
    assert [doc.number for doc in all_docs] == [2, 1, 0]

    limited = store.all_embeddings("org/repo", limit=2)
    assert len(limited) == 2
    assert [doc.number for doc in limited] == [2, 1]


def test_search_without_limit_returns_all(tmp_path):
    store = SQLiteVectorStore(tmp_path / "store-all.db")
    embeddings = FakeEmbeddingsClient(dimension=4)

    for index in range(3):
        vector = embeddings.embed_documents([f"Issue {index}"]).embeddings[0]
        store.upsert_document(
            StoredDocument(
                doc_id=f"org/repo#ISSUE#{index}",
                repo="org/repo",
                number=index,
                doc_type="issue",
                title=f"Issue {index}",
                body="",
                labels=[],
                state="open",
                html_url=f"https://github.com/org/repo/issues/{index}",
                updated_at=f"2024-02-0{index + 1}T00:00:00Z",
                created_at="2024-02-0{index + 1}T00:00:00Z",
                author=None,
                embedding=vector,
                embedding_model="fake-document",
                embedding_dimensions=len(vector),
                metadata={},
            )
        )

    query_embedding = embeddings.embed_queries(["Issue"]).embeddings[0]
    filters = FilterCriteria()

    unlimited = store.search("org/repo", query_embedding, filters, limit=None)
    assert {result.document.number for result in unlimited} == {0, 1, 2}

    zero_limit = store.search("org/repo", query_embedding, filters, limit=0)
    assert {result.document.number for result in zero_limit} == {0, 1, 2}


def test_get_by_numbers_respects_types(tmp_path):
    store = SQLiteVectorStore(tmp_path / "store-numbers.db")
    embeddings = FakeEmbeddingsClient(dimension=4)

    issue_embedding = embeddings.embed_documents(["Issue"]).embeddings[0]
    pr_embedding = embeddings.embed_documents(["Pull request"]).embeddings[0]

    store.upsert_document(
        StoredDocument(
            doc_id="org/repo#ISSUE#5",
            repo="org/repo",
            number=5,
            doc_type="issue",
            title="Issue",
            body="",
            labels=[],
            state="open",
            html_url="https://github.com/org/repo/issues/5",
            updated_at="2024-03-01T00:00:00Z",
            created_at="2024-03-01T00:00:00Z",
            author=None,
            embedding=issue_embedding,
            embedding_model="fake-document",
            embedding_dimensions=len(issue_embedding),
            metadata={},
        )
    )

    store.upsert_document(
        StoredDocument(
            doc_id="org/repo#PULL_REQUEST#5",
            repo="org/repo",
            number=5,
            doc_type="pull_request",
            title="PR",
            body="",
            labels=[],
            state="open",
            html_url="https://github.com/org/repo/pull/5",
            updated_at="2024-03-02T00:00:00Z",
            created_at="2024-03-02T00:00:00Z",
            author=None,
            embedding=pr_embedding,
            embedding_model="fake-document",
            embedding_dimensions=len(pr_embedding),
            metadata={},
        )
    )

    both = store.get_by_numbers("org/repo", [5])
    assert {doc.doc_type for doc in both} == {"issue", "pull_request"}

    issues_only = store.get_by_numbers("org/repo", [5], types=["issue"])
    assert [doc.doc_type for doc in issues_only] == ["issue"]
