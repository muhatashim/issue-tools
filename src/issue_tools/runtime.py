"""Runtime helpers for constructing shared application services."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional
import logging

from rich.console import Console

from .config import Config, load_config
from .embeddings import GeminiEmbeddingsClient
from .github_client import GitHubClient
from .indexer import Indexer
from .search import SearchService
from .token_tracker import TokenTracker
from .vector_store import SQLiteVectorStore

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(slots=True)
class Services:
    """Bundle of long-lived services used across entry points."""

    config: Config
    github: GitHubClient
    vector_store: SQLiteVectorStore
    embeddings: GeminiEmbeddingsClient
    search: SearchService
    indexer: Indexer
    token_tracker: TokenTracker


@contextmanager
def application_services(
    *, console: Optional[Console] = None, require_gemini_api_key: bool = True
) -> Iterator[Services]:
    """Yield initialized services for a single command execution."""

    config = load_config()
    token_tracker = TokenTracker(config.token_usage_path, config.cost.per_1k_tokens)

    github_token = config.get_github_token()
    github = GitHubClient(github_token)

    if not github_token:
        message = (
            "GitHub token not set; unauthenticated requests are heavily rate limited."
        )
        if console is not None:
            console.print(f"[yellow]Warning: {message}[/yellow]")
        else:
            logger.warning(message)

    api_key = config.get_gemini_api_key()
    if require_gemini_api_key and not api_key:
        raise ConfigurationError(
            "Gemini API key not configured. Set the environment variable "
            f"{config.gemini_api_key_env}."
        )

    vector_store = SQLiteVectorStore(config.vector_db_path)
    embeddings = GeminiEmbeddingsClient(
        api_key=api_key or "",
        document_model=config.embedding.document_model,
        document_task_type=config.embedding.document_task_type,
        query_model=config.embedding.query_model,
        query_task_type=config.embedding.query_task_type,
        token_tracker=token_tracker,
    )
    search = SearchService(vector_store, embeddings)
    indexer = Indexer(config, github, vector_store, embeddings, console=console)

    try:
        yield Services(
            config=config,
            github=github,
            vector_store=vector_store,
            embeddings=embeddings,
            search=search,
            indexer=indexer,
            token_tracker=token_tracker,
        )
    finally:
        vector_store.close()
        token_tracker.save()
