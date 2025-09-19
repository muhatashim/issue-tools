"""Configuration helpers for issue tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import json
import os
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[assignment]

from dotenv import load_dotenv


DEFAULT_CONFIG_PATH = Path("config.toml")
DEFAULT_ENV_PATH = Path(".env")


@dataclass(slots=True)
class EmbeddingConfig:
    """Configuration for embedding models."""

    document_model: str = "gemini-embedding-001"
    document_task_type: str = "RETRIEVAL_DOCUMENT"
    query_model: str = "gemini-embedding-001"
    query_task_type: str = "RETRIEVAL_QUERY"
    max_chars: int = 6000
    batch_size: int = 8


@dataclass(slots=True)
class CostConfig:
    """Model pricing configuration."""

    per_1k_tokens: Dict[str, float] = field(
        default_factory=lambda: {
            "models/gemini-embedding-001": 0.00013,
            "gemini-embedding-001": 0.00013,
            "models/gemini-2.5-flash-lite": 0.10,
        }
    )


@dataclass(slots=True)
class Config:
    """Application configuration."""

    github_token_env: str = "GITHUB_TOKEN"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    repository: Optional[str] = None
    data_dir: Path = Path("data")
    vector_db_path: Path = Path("data") / "vector_store.db"
    metadata_path: Path = Path("data") / "metadata.json"
    token_usage_path: Path = Path("data") / "token_usage.json"
    index_interval_hours: int = 24
    initial_index_limit: int = 10
    llm_model: str = "models/gemini-2.5-flash-lite"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    @property
    def data_directory(self) -> Path:
        return Path(self.data_dir)

    def ensure_data_dir(self) -> None:
        self.data_directory.mkdir(parents=True, exist_ok=True)

    def get_github_token(self) -> Optional[str]:
        return os.environ.get(self.github_token_env)

    def get_gemini_api_key(self) -> Optional[str]:
        return os.environ.get(self.gemini_api_key_env)


def _load_dict(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(path: Optional[Path] = None, env_path: Optional[Path] = None) -> Config:
    """Load configuration from disk, falling back to defaults."""

    env_file = env_path or DEFAULT_ENV_PATH
    if env_file:
        load_dotenv(env_file)

    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_dict(config_path)

    embedding_raw = raw.get("embedding", {}) if isinstance(raw, dict) else {}
    cost_raw = raw.get("cost", {}) if isinstance(raw, dict) else {}

    default_embedding = EmbeddingConfig()
    embedding = EmbeddingConfig(
        document_model=embedding_raw.get("document_model", default_embedding.document_model),
        document_task_type=embedding_raw.get("document_task_type", default_embedding.document_task_type),
        query_model=embedding_raw.get("query_model", default_embedding.query_model),
        query_task_type=embedding_raw.get("query_task_type", default_embedding.query_task_type),
        max_chars=int(embedding_raw.get("max_chars", default_embedding.max_chars)),
        batch_size=int(embedding_raw.get("batch_size", default_embedding.batch_size)),
    )

    default_cost = CostConfig()
    cost = CostConfig(
        per_1k_tokens={
            **default_cost.per_1k_tokens,
            **{str(k): float(v) for k, v in cost_raw.get("per_1k_tokens", {}).items()},
        }
    )

    defaults = Config()

    cfg = Config(
        github_token_env=str(raw.get("github_token_env", defaults.github_token_env)),
        gemini_api_key_env=str(raw.get("gemini_api_key_env", defaults.gemini_api_key_env)),
        repository=raw.get("repository"),
        data_dir=Path(raw.get("data_dir", defaults.data_dir)),
        vector_db_path=Path(raw.get("vector_db_path", defaults.vector_db_path)),
        metadata_path=Path(raw.get("metadata_path", defaults.metadata_path)),
        token_usage_path=Path(raw.get("token_usage_path", defaults.token_usage_path)),
        index_interval_hours=int(raw.get("index_interval_hours", defaults.index_interval_hours)),
        initial_index_limit=int(raw.get("initial_index_limit", defaults.initial_index_limit)),
        llm_model=str(raw.get("llm_model", defaults.llm_model)),
        embedding=embedding,
        cost=cost,
    )

    cfg.ensure_data_dir()
    return cfg


def save_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_metadata(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
