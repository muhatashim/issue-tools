"""Embedding client abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol
import hashlib

import numpy as np
import requests

from .token_tracker import TokenTracker, TokenUsage


@dataclass(slots=True)
class EmbeddingResult:
    """Container for embedding responses."""

    model: str
    embeddings: List[List[float]]
    usage: List[TokenUsage]


class EmbeddingsClient(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: Iterable[str]) -> EmbeddingResult:
        ...

    def embed_queries(self, texts: Iterable[str]) -> EmbeddingResult:
        ...


class GeminiEmbeddingsClient:
    """Client for Google AI Studio Gemini embeddings."""

    api_root = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        document_model: str,
        document_task_type: str,
        query_model: str,
        query_task_type: str,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.api_key = api_key
        self.document_model = document_model
        self.document_task_type = document_task_type
        self.query_model = query_model
        self.query_task_type = query_task_type
        self.token_tracker = token_tracker

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        """Return the fully-qualified Gemini model identifier."""

        if not model:
            return model
        if "/" in model and not model.startswith("models/"):
            # Assume the caller supplied a fully-qualified resource path.
            return model
        if model.startswith("models/"):
            return model
        return f"models/{model}"

    def _embed(
        self,
        texts: Iterable[str],
        model: str,
        task_type: str,
    ) -> EmbeddingResult:
        texts_list = list(texts)
        model_name = self._normalize_model_name(model)
        if not texts_list:
            return EmbeddingResult(model=model_name, embeddings=[], usage=[])

        url = f"{self.api_root}/{model_name}:batchEmbedContents?key={self.api_key}"
        payload = {
            "requests": [
                {
                    "model": model_name,
                    "taskType": task_type,
                    "content": {"parts": [{"text": text}]},
                }
                for text in texts_list
            ]
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        body = response.json()
        embeddings: List[List[float]] = []
        usages: List[TokenUsage] = []

        response_items = body.get("responses")
        if response_items is None and "embeddings" in body:
            response_items = body.get("embeddings")

        if not response_items:
            raise RuntimeError("No embeddings returned from Gemini response")

        def _coerce_int(value: object) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        def _build_usage(metadata: dict | None) -> TokenUsage:
            metadata = metadata or {}
            total_raw = metadata.get("totalTokenCount")
            total_tokens = None
            if total_raw is not None:
                try:
                    total_tokens = int(total_raw)
                except (TypeError, ValueError):
                    total_tokens = None
            return TokenUsage(
                model=model_name,
                prompt_tokens=_coerce_int(metadata.get("promptTokenCount")),
                response_tokens=_coerce_int(metadata.get("candidatesTokenCount")),
                total_tokens=total_tokens,
            )

        for item in response_items:
            if isinstance(item, dict) and "embedding" in item:
                embedding_values = item.get("embedding", {}).get("values")
                usage_metadata = item.get("usageMetadata")
            else:
                embedding_values = item.get("values") if isinstance(item, dict) else None
                usage_metadata = None

            if embedding_values is None:
                raise RuntimeError("Missing embedding values from Gemini response")

            embeddings.append([float(value) for value in embedding_values])
            usages.append(_build_usage(usage_metadata))

        batch_usage_raw = body.get("usageMetadata") if isinstance(body, dict) else None
        batch_usage = _build_usage(batch_usage_raw) if isinstance(batch_usage_raw, dict) else None

        if batch_usage and usages:
            def _distribute(value: int) -> List[int]:
                base, remainder = divmod(value, len(usages))
                return [base + (1 if index < remainder else 0) for index in range(len(usages))]

            if batch_usage.prompt_tokens and all(usage.prompt_tokens == 0 for usage in usages):
                for usage, share in zip(usages, _distribute(batch_usage.prompt_tokens)):
                    usage.prompt_tokens = share

            if batch_usage.response_tokens and all(usage.response_tokens == 0 for usage in usages):
                for usage, share in zip(usages, _distribute(batch_usage.response_tokens)):
                    usage.response_tokens = share

            if (
                batch_usage.total_tokens
                and all(usage.total_tokens in (None, 0) for usage in usages)
            ):
                for usage, share in zip(usages, _distribute(batch_usage.total_tokens)):
                    usage.total_tokens = share

        if usages and all(usage.combined() == 0 for usage in usages):
            estimated_tokens = self._estimate_prompt_tokens(texts_list, model_name)
            if estimated_tokens:
                for usage, tokens in zip(usages, estimated_tokens):
                    usage.prompt_tokens = tokens
                    usage.response_tokens = 0
                    usage.total_tokens = tokens

        if self.token_tracker is not None:
            self.token_tracker.extend(usages)

        return EmbeddingResult(model=model_name, embeddings=embeddings, usage=usages)

    def embed_documents(self, texts: Iterable[str]) -> EmbeddingResult:
        return self._embed(texts, self.document_model, self.document_task_type)

    def embed_queries(self, texts: Iterable[str]) -> EmbeddingResult:
        return self._embed(texts, self.query_model, self.query_task_type)

    def _estimate_prompt_tokens(self, texts: List[str], model_name: str) -> List[int]:
        if not texts:
            return []

        url = f"{self.api_root}/{model_name}:countTokens?key={self.api_key}"
        payload = {
            "contents": [
                {"parts": [{"text": text}]}
                for text in texts
            ]
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.RequestException:
            return []

        data = response.json()
        total_tokens = data.get("totalTokens")
        try:
            total_tokens = int(total_tokens)
        except (TypeError, ValueError):
            return []

        if total_tokens <= 0:
            return []

        weights = [max(len(text), 1) for text in texts]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            share = total_tokens // len(texts)
            remainder = total_tokens % len(texts)
            return [share + (1 if index < remainder else 0) for index in range(len(texts))]

        allocations: List[int] = []
        allocated = 0
        for weight in weights:
            portion = int((total_tokens * weight) / weight_sum)
            allocations.append(portion)
            allocated += portion

        remainder = total_tokens - allocated
        index = 0
        while remainder > 0 and allocations:
            allocations[index % len(allocations)] += 1
            remainder -= 1
            index += 1

        return allocations


class FakeEmbeddingsClient:
    """Deterministic embedding generator for tests."""

    def __init__(self, dimension: int = 16, seed: int = 13) -> None:
        self.dimension = dimension
        self.seed = seed

    def _vectorize(self, text: str) -> List[float]:
        values = []
        for index in range(self.dimension):
            hasher = hashlib.sha256()
            hasher.update(text.encode("utf-8"))
            hasher.update(str(self.seed).encode("utf-8"))
            hasher.update(index.to_bytes(2, "little", signed=False))
            digest = hasher.digest()
            values.append(int.from_bytes(digest[:4], "little", signed=False) / 2**32)
        # Normalize to unit length for cosine similarity
        arr = np.array(values[: self.dimension], dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()

    def embed_documents(self, texts: Iterable[str]) -> EmbeddingResult:
        vectors = [self._vectorize(text) for text in texts]
        return EmbeddingResult(
            model="fake-document",
            embeddings=vectors,
            usage=[TokenUsage(model="fake-document", total_tokens=len(text)) for text in texts],
        )

    def embed_queries(self, texts: Iterable[str]) -> EmbeddingResult:
        vectors = [self._vectorize(text) for text in texts]
        return EmbeddingResult(
            model="fake-query",
            embeddings=vectors,
            usage=[TokenUsage(model="fake-query", total_tokens=len(text)) for text in texts],
        )
