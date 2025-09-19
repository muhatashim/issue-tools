"""Shared command implementations used by the CLI, API, and MCP layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Literal

from .clustering import Cluster as RawCluster, kmeans, cluster_documents, gather_documents_for_clustering
from .evaluation import EvaluationCase, EvaluationSummary, evaluate as run_evaluation
from .filters import FilterCriteria, parse_query_with_filters
from .runtime import Services
from .search import SearchMatch
from .token_tracker import TokenTracker


class CommandError(RuntimeError):
    """Raised when command parameters are invalid."""


@dataclass(slots=True)
class IndexParams:
    repo: Optional[str] = None
    force: bool = False
    limit: Optional[int] = None
    include_pulls: bool = True
    filter_query: Optional[str] = None


@dataclass(slots=True)
class IndexResponse:
    repo: str
    indexed: int
    fetched: int
    skipped: bool
    message: str
    last_indexed_at: Optional[str]
    token_usage: "TokenUsageSummary | None"


@dataclass(slots=True)
class SearchParams:
    query: str
    repo: Optional[str] = None
    limit: int = 5


@dataclass(slots=True)
class SearchResponse:
    repo: str
    query: str
    matches: List[SearchMatch]
    token_usage: "TokenUsageSummary | None"


@dataclass(slots=True)
class FilterParams:
    query: str
    repo: Optional[str] = None
    limit: int = 10


@dataclass(slots=True)
class FilterItemSummary:
    repo: str
    number: int
    title: str
    doc_type: str
    state: str
    html_url: str
    updated_at: str
    created_at: str
    author: Optional[str]
    labels: List[str]


@dataclass(slots=True)
class FilterResponse:
    repo: Optional[str]
    query: str
    items: List[FilterItemSummary]


@dataclass(slots=True)
class ClusterParams:
    repo: Optional[str] = None
    query: Optional[str] = None
    k: Optional[int] = None
    limit: int = 50
    max_clusters: int = 8


@dataclass(slots=True)
class ClusterFiltersSummary:
    labels: List[str]
    states: List[str]
    types: List[str]
    numbers: List[int]
    updated: Optional[tuple[str, str]]
    created: Optional[tuple[str, str]]
    author: Optional[str]


@dataclass(slots=True)
class ClusterDocumentSummary:
    number: int
    title: str
    doc_type: str
    state: str
    html_url: str
    labels: List[str]


@dataclass(slots=True)
class ClusterSummary:
    cluster_id: int
    size: int
    centroid: List[float]
    top_labels: List[str]
    documents: List[ClusterDocumentSummary]


@dataclass(slots=True)
class ClusterResponse:
    repo: str
    filters: Optional[ClusterFiltersSummary]
    clusters: List[ClusterSummary]
    token_usage: "TokenUsageSummary | None"


@dataclass(slots=True)
class StreamParams:
    repo: Optional[str] = None


@dataclass(slots=True)
class StreamItemSummary:
    number: int
    title: str
    doc_type: str
    state: str
    html_url: str
    updated_at: str
    created_at: str
    author: Optional[str]
    labels: List[str]


@dataclass(slots=True)
class StreamResponse:
    repo: str
    items: List[StreamItemSummary]


@dataclass(slots=True)
class InspectDBParams:
    repo: Optional[str] = None
    show: int = 0


@dataclass(slots=True)
class RepoSummary:
    repo: str
    total: int
    issues: int
    pulls: int
    first_created: Optional[str]
    last_created: Optional[str]
    first_updated: Optional[str]
    last_updated: Optional[str]
    last_indexed: Optional[str]


@dataclass(slots=True)
class EmbeddingModelInfo:
    model: str
    dimensions: int
    documents: int


@dataclass(slots=True)
class SampleDocumentInfo:
    repo: Optional[str]
    number: int
    doc_type: str
    state: str
    updated_at: str
    title: str


@dataclass(slots=True)
class InspectDBResponse:
    database_path: str
    summaries: List[RepoSummary]
    embedding_models: List[EmbeddingModelInfo]
    samples: List[SampleDocumentInfo]


@dataclass(slots=True)
class EvaluateParams:
    cases: Iterable[EvaluationCase]
    repo: Optional[str] = None
    top_k: int = 5


@dataclass(slots=True)
class EvaluationResponse:
    repo: str
    top_k: int
    summary: EvaluationSummary
    token_usage: "TokenUsageSummary | None"


@dataclass(slots=True)
class TokenStats:
    model: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    cost: float


@dataclass(slots=True)
class TokenUsageSummary:
    scope: Literal["session", "lifetime"]
    models: List[TokenStats]


def resolve_repo(config, repo: Optional[str]) -> str:
    if repo:
        return repo
    if config.repository:
        return config.repository
    raise CommandError("Repository must be supplied via parameter or config file.")


def index_repository(services: Services, params: IndexParams) -> IndexResponse:
    repo_name = resolve_repo(services.config, params.repo)
    result = services.indexer.run(
        repo_name,
        force=params.force,
        limit=params.limit,
        include_pulls=params.include_pulls,
        filter_query=params.filter_query,
    )
    return IndexResponse(
        repo=repo_name,
        indexed=result.indexed,
        fetched=result.fetched,
        skipped=result.skipped,
        message=result.message,
        last_indexed_at=result.last_indexed_at,
        token_usage=summarize_token_usage(services.token_tracker),
    )


def search_repository(services: Services, params: SearchParams) -> SearchResponse:
    repo_name = resolve_repo(services.config, params.repo)
    matches = services.search.search(repo_name, params.query, limit=params.limit)
    return SearchResponse(
        repo=repo_name,
        query=params.query,
        matches=matches,
        token_usage=summarize_token_usage(services.token_tracker),
    )


def filter_repository(services: Services, params: FilterParams) -> FilterResponse:
    limit = params.limit
    if limit <= 0:
        raise CommandError("Limit must be greater than zero.")

    query = (params.query or "").strip()
    query_lower = query.lower()
    query_repo: Optional[str] = None
    if "repo:" in query_lower:
        _, parsed_filters = parse_query_with_filters(query)
        query_repo = parsed_filters.repo

    repo_name = params.repo or query_repo or services.config.repository

    if repo_name:
        if "repo:" not in query_lower:
            repo_filter = f"repo:{repo_name}"
            query = f"{query} {repo_filter}".strip()
            query_lower = query.lower()
    else:
        raise CommandError(
            "Repository must be supplied via parameter, config file, or included in the query."
        )

    results = services.github.search_issues(query, limit=limit)
    items = [
        FilterItemSummary(
            repo=item.repo or repo_name or "",
            number=item.number,
            title=item.title,
            doc_type="pull_request" if item.is_pull_request else "issue",
            state=item.state,
            html_url=item.html_url,
            updated_at=item.updated_at,
            created_at=item.created_at,
            author=item.author,
            labels=list(item.labels),
        )
        for item in results
    ]
    return FilterResponse(repo=repo_name, query=query, items=items)


def cluster_repository(services: Services, params: ClusterParams) -> ClusterResponse:
    repo_name = resolve_repo(services.config, params.repo)
    filters: Optional[FilterCriteria] = None
    filters_summary: Optional[ClusterFiltersSummary] = None
    
    try:
        if params.query:
            query_text, filters = parse_query_with_filters(params.query)
            filters = filters.with_repo(repo_name)
            filters_summary = _summarize_filters(filters)
            
            # Get query embedding for semantic search
            query_embedding_result = services.embeddings.embed_queries([query_text])
            query_embedding = query_embedding_result.embeddings[0]
            
            # Handle anchor documents if any number filters are present
            anchor_documents = []
            if filters.numbers:
                anchor_documents = services.vector_store.get_by_numbers(repo_name, filters.numbers)
            
            # Gather documents for clustering using semantic search
            documents = gather_documents_for_clustering(
                services.vector_store,
                repo_name,
                filters,
                query_embedding=query_embedding,
                anchor_documents=anchor_documents,
                limit=params.limit if params.limit > 0 else None,
                allow_repo_fallback=True,
            )
            
            # Cluster the gathered documents
            clusters = cluster_documents(
                documents,
                k=params.k,
                max_k=params.max_clusters,
            )
        else:
            # Fallback to simple clustering without semantic search
            documents = services.vector_store.all_embeddings(repo_name, filters, limit=params.limit if params.limit > 0 else None)
            clusters = cluster_documents(
                documents,
                k=params.k,
                max_k=params.max_clusters,
            )
        
        summaries = [_summarize_cluster(cluster) for cluster in clusters]
        return ClusterResponse(
            repo=repo_name,
            filters=filters_summary,
            clusters=summaries,
            token_usage=summarize_token_usage(services.token_tracker),
        )
    except ValueError as exc:
        raise CommandError(str(exc)) from exc


def stream_repository(services: Services, params: StreamParams) -> StreamResponse:
    repo_name = resolve_repo(services.config, params.repo)
    items = services.indexer.stream_new(repo_name)
    return StreamResponse(
        repo=repo_name,
        items=[
            StreamItemSummary(
                number=item.number,
                title=item.title,
                doc_type="pull_request" if item.is_pull_request else "issue",
                state=item.state,
                html_url=item.html_url,
                updated_at=item.updated_at,
                created_at=item.created_at,
                author=item.author,
                labels=[label.name for label in item.labels],
            )
            for item in items
        ],
    )


def inspect_database(services: Services, params: InspectDBParams) -> InspectDBResponse:
    repo_name = params.repo
    where_clause = ""
    query_params: List[object] = []
    if repo_name:
        where_clause = "WHERE repo = ?"
        query_params.append(repo_name)

    cursor = services.vector_store.connection.cursor()
    summary_query = (
        "SELECT repo, COUNT(*) AS total, "
        "SUM(CASE WHEN type = 'issue' THEN 1 ELSE 0 END) AS issues, "
        "SUM(CASE WHEN type = 'pull_request' THEN 1 ELSE 0 END) AS pulls, "
        "MIN(created_at) AS first_created, "
        "MAX(created_at) AS last_created, "
        "MIN(updated_at) AS first_updated, "
        "MAX(updated_at) AS last_updated "
        "FROM documents "
        f"{where_clause} "
        "GROUP BY repo "
        "ORDER BY repo"
    )
    cursor.execute(summary_query, query_params)
    rows = cursor.fetchall()

    summaries: List[RepoSummary] = []
    metadata = services.vector_store.all_metadata()
    for row in rows:
        repo_key = row["repo"]
        summaries.append(
            RepoSummary(
                repo=repo_key,
                total=int(row["total"] or 0),
                issues=int(row["issues"] or 0),
                pulls=int(row["pulls"] or 0),
                first_created=row["first_created"],
                last_created=row["last_created"],
                first_updated=row["first_updated"],
                last_updated=row["last_updated"],
                last_indexed=metadata.get(f"last_indexed_at:{repo_key}"),
            )
        )

    model_query = (
        "SELECT embedding_model, embedding_dimensions, COUNT(*) AS count "
        "FROM documents "
        f"{where_clause} "
        "GROUP BY embedding_model, embedding_dimensions "
        "ORDER BY embedding_model"
    )
    cursor.execute(model_query, query_params)
    model_rows = cursor.fetchall()
    embedding_models = [
        EmbeddingModelInfo(
            model=row["embedding_model"],
            dimensions=int(row["embedding_dimensions"] or 0),
            documents=int(row["count"] or 0),
        )
        for row in model_rows
    ]

    samples: List[SampleDocumentInfo] = []
    if params.show:
        sample_params = list(query_params)
        sample_query = (
            "SELECT repo, number, type, state, updated_at, title "
            "FROM documents "
            f"{where_clause} "
            "ORDER BY updated_at DESC "
            "LIMIT ?"
        )
        sample_params.append(params.show)
        cursor.execute(sample_query, sample_params)
        sample_rows = cursor.fetchall()
        for row in sample_rows:
            samples.append(
                SampleDocumentInfo(
                    repo=row["repo"],
                    number=int(row["number"] or 0),
                    doc_type=row["type"],
                    state=row["state"],
                    updated_at=row["updated_at"],
                    title=row["title"] or "",
                )
            )

    return InspectDBResponse(
        database_path=str(services.vector_store.path),
        summaries=summaries,
        embedding_models=embedding_models,
        samples=samples,
    )


def evaluate_repository(services: Services, params: EvaluateParams) -> EvaluationResponse:
    repo_name = resolve_repo(services.config, params.repo)
    summary = run_evaluation(services.search, repo_name, params.cases, top_k=params.top_k)
    return EvaluationResponse(
        repo=repo_name,
        top_k=params.top_k,
        summary=summary,
        token_usage=summarize_token_usage(services.token_tracker),
    )


def summarize_token_usage(
    tracker: TokenTracker, *, lifetime: bool = False
) -> TokenUsageSummary | None:
    bucket = (
        tracker.lifetime_summary() if lifetime else tracker.session_summary()
    )
    if not bucket:
        return None
    stats = [
        TokenStats(
            model=model,
            prompt_tokens=int(values.get("prompt_tokens", 0)),
            response_tokens=int(values.get("response_tokens", 0)),
            total_tokens=int(values.get("total_tokens", 0)),
            cost=float(values.get("cost", 0.0)),
        )
        for model, values in bucket.items()
    ]
    stats.sort(key=lambda item: item.model)
    scope: Literal["session", "lifetime"] = "lifetime" if lifetime else "session"
    return TokenUsageSummary(scope=scope, models=stats)


def _summarize_filters(filters: FilterCriteria) -> ClusterFiltersSummary:
    return ClusterFiltersSummary(
        labels=sorted(filters.labels),
        states=sorted(filters.states),
        types=sorted(filters.types),
        numbers=sorted(filters.numbers),
        updated=filters.updated_filter,
        created=filters.created_filter,
        author=filters.author,
    )


def _summarize_cluster(cluster: RawCluster) -> ClusterSummary:
    documents = [
        ClusterDocumentSummary(
            number=doc.number,
            title=doc.title,
            doc_type=doc.doc_type,
            state=doc.state,
            html_url=doc.html_url,
            labels=list(doc.labels),
        )
        for doc in cluster.documents
    ]
    return ClusterSummary(
        cluster_id=cluster.cluster_id,
        size=len(cluster.documents),
        centroid=list(cluster.centroid),
        top_labels=cluster.top_labels(),
        documents=documents,
    )
