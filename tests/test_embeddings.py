"""Tests for the Gemini embeddings client."""

from __future__ import annotations

from typing import Dict

import pytest

from issue_tools.embeddings import GeminiEmbeddingsClient


class _DummyResponse:
    def __init__(self, payload: Dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - no-op for tests
        return None

    def json(self) -> Dict[str, object]:
        return self._payload


@pytest.mark.parametrize(
    "model, expected",
    [
        ("models/gemini-embedding-001", "models/gemini-embedding-001"),
        ("gemini-embedding-001", "models/gemini-embedding-001"),
    ],
)
def test_embed_documents_normalises_model_path(monkeypatch: pytest.MonkeyPatch, model: str, expected: str) -> None:
    """The Gemini client should only include a single models/ prefix in the URL."""

    captured: Dict[str, object] = {}

    def fake_post(url: str, json: Dict[str, object], timeout: int) -> _DummyResponse:
        captured["url"] = url
        captured["payload"] = json
        assert timeout == 60
        response = {
            "responses": [
                {
                    "embedding": {"values": [0.1, 0.2, 0.3]},
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 0,
                        "totalTokenCount": 3,
                    },
                }
            ]
        }
        return _DummyResponse(response)

    monkeypatch.setattr("issue_tools.embeddings.requests.post", fake_post)

    client = GeminiEmbeddingsClient(
        api_key="secret",
        document_model=model,
        document_task_type="RETRIEVAL_DOCUMENT",
        query_model=model,
        query_task_type="RETRIEVAL_QUERY",
    )

    result = client.embed_documents(["hello world"])

    expected_url = f"{client.api_root}/{expected}:batchEmbedContents?key=secret"
    assert captured["url"] == expected_url
    payload = captured["payload"]
    assert payload["requests"][0]["model"] == expected
    assert result.model == expected
    assert result.embeddings[0] == [0.1, 0.2, 0.3]
    assert result.usage[0].model == expected


def test_embed_documents_with_no_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty request should avoid hitting the network and return no embeddings."""

    called = False

    def fake_post(*args, **kwargs):  # pragma: no cover - defensive
        nonlocal called
        called = True
        return _DummyResponse({"responses": []})

    monkeypatch.setattr("issue_tools.embeddings.requests.post", fake_post)

    client = GeminiEmbeddingsClient(
        api_key="secret",
        document_model="gemini-embedding-001",
        document_task_type="RETRIEVAL_DOCUMENT",
        query_model="gemini-embedding-001",
        query_task_type="RETRIEVAL_QUERY",
    )

    result = client.embed_documents([])
    assert not called
    assert result.model == "models/gemini-embedding-001"
    assert result.embeddings == []
    assert result.usage == []
