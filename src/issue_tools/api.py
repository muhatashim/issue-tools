"""HTTP API for interacting with issue-tools functionality."""

from __future__ import annotations

from typing import Iterable, List, Optional, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from .commands import (
    ClusterParams,
    ClusterResponse,
    CommandError,
    EvaluateParams,
    EvaluationResponse,
    FilterParams,
    FilterResponse,
    IndexParams,
    IndexResponse,
    InspectDBParams,
    InspectDBResponse,
    SearchParams,
    SearchResponse,
    StreamParams,
    StreamResponse,
    TokenUsageSummary,
    cluster_repository,
    evaluate_repository,
    filter_repository,
    index_repository,
    inspect_database,
    search_repository,
    stream_repository,
    summarize_token_usage,
)
from .config import load_config
from .evaluation import EvaluationCase
from .runtime import ConfigurationError, application_services
from .token_tracker import TokenTracker


class IndexRequest(BaseModel):
    repo: Optional[str] = None
    force: bool = False
    limit: Optional[int] = None
    include_pulls: bool = True
    filter_query: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    repo: Optional[str] = None
    limit: int = 5


class FilterRequest(BaseModel):
    query: str
    repo: Optional[str] = None
    limit: int = 10


class ClusterRequest(BaseModel):
    repo: Optional[str] = None
    query: Optional[str] = None
    k: int = 3


class EvaluationCaseInput(BaseModel):
    query: str
    expected_numbers: List[int]


class EvaluationRequest(BaseModel):
    repo: Optional[str] = None
    top_k: int = 5
    cases: List[EvaluationCaseInput]


class TokenStatsModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    cost: float


class TokenUsageModel(BaseModel):
    scope: Literal["session", "lifetime"]
    models: List[TokenStatsModel]

    @classmethod
    def from_summary(cls, summary: TokenUsageSummary | None) -> "TokenUsageModel | None":
        if summary is None:
            return None
        return cls(
            scope=summary.scope,
            models=[TokenStatsModel.model_validate(item) for item in summary.models],
        )


class IndexResponseModel(BaseModel):
    repo: str
    indexed: int
    fetched: int
    skipped: bool
    message: str
    last_indexed_at: Optional[str] = None
    token_usage: Optional[TokenUsageModel] = None

    @classmethod
    def from_result(cls, result: IndexResponse) -> "IndexResponseModel":
        return cls(
            repo=result.repo,
            indexed=result.indexed,
            fetched=result.fetched,
            skipped=result.skipped,
            message=result.message,
            last_indexed_at=result.last_indexed_at,
            token_usage=TokenUsageModel.from_summary(result.token_usage),
        )


class SearchMatchModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    doc_id: str
    number: int
    title: str
    url: str
    score: float
    state: str
    doc_type: str
    labels: List[str]
    updated_at: str
    created_at: str
    author: Optional[str]


class SearchResponseModel(BaseModel):
    repo: str
    query: str
    matches: List[SearchMatchModel]
    token_usage: Optional[TokenUsageModel] = None

    @classmethod
    def from_result(cls, result: SearchResponse) -> "SearchResponseModel":
        return cls(
            repo=result.repo,
            query=result.query,
            matches=[SearchMatchModel.model_validate(match) for match in result.matches],
            token_usage=TokenUsageModel.from_summary(result.token_usage),
        )


class FilterItemModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class FilterResponseModel(BaseModel):
    repo: Optional[str]
    query: str
    items: List[FilterItemModel]

    @classmethod
    def from_result(cls, result: FilterResponse) -> "FilterResponseModel":
        return cls(
            repo=result.repo,
            query=result.query,
            items=[FilterItemModel.model_validate(item) for item in result.items],
        )


class ClusterFiltersModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    labels: List[str]
    states: List[str]
    types: List[str]
    numbers: List[int]
    updated: Optional[tuple[str, str]] = None
    created: Optional[tuple[str, str]] = None
    author: Optional[str] = None


class ClusterDocumentModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    number: int
    title: str
    doc_type: str
    state: str
    html_url: str
    labels: List[str]


class ClusterSummaryModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    cluster_id: int
    size: int
    centroid: List[float]
    top_labels: List[str]
    documents: List[ClusterDocumentModel]


class ClusterResponseModel(BaseModel):
    repo: str
    filters: Optional[ClusterFiltersModel]
    clusters: List[ClusterSummaryModel]
    token_usage: Optional[TokenUsageModel] = None

    @classmethod
    def from_result(cls, result: ClusterResponse) -> "ClusterResponseModel":
        return cls(
            repo=result.repo,
            filters=ClusterFiltersModel.model_validate(result.filters)
            if result.filters
            else None,
            clusters=[
                ClusterSummaryModel.model_validate(cluster)
                for cluster in result.clusters
            ],
            token_usage=TokenUsageModel.from_summary(result.token_usage),
        )


class StreamItemModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    number: int
    title: str
    doc_type: str
    state: str
    html_url: str
    updated_at: str
    created_at: str
    author: Optional[str]


class StreamResponseModel(BaseModel):
    repo: str
    items: List[StreamItemModel]

    @classmethod
    def from_result(cls, result: StreamResponse) -> "StreamResponseModel":
        return cls(
            repo=result.repo,
            items=[StreamItemModel.model_validate(item) for item in result.items],
        )


class RepoSummaryModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    repo: str
    total: int
    issues: int
    pulls: int
    first_created: Optional[str]
    last_created: Optional[str]
    first_updated: Optional[str]
    last_updated: Optional[str]
    last_indexed: Optional[str]


class EmbeddingModelInfoModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: str
    dimensions: int
    documents: int


class SampleDocumentModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    repo: Optional[str]
    number: int
    doc_type: str
    state: str
    updated_at: str
    title: str


class InspectDBResponseModel(BaseModel):
    database_path: str
    summaries: List[RepoSummaryModel]
    embedding_models: List[EmbeddingModelInfoModel]
    samples: List[SampleDocumentModel]

    @classmethod
    def from_result(cls, result: InspectDBResponse) -> "InspectDBResponseModel":
        return cls(
            database_path=result.database_path,
            summaries=[RepoSummaryModel.model_validate(item) for item in result.summaries],
            embedding_models=[
                EmbeddingModelInfoModel.model_validate(item)
                for item in result.embedding_models
            ],
            samples=[SampleDocumentModel.model_validate(item) for item in result.samples],
        )


class EvaluationCaseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    expected_numbers: List[int]


class EvaluationResultModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    case: EvaluationCaseModel
    matches: List[int]
    precision_at_k: float
    reciprocal_rank: float


class EvaluationSummaryModel(BaseModel):
    results: List[EvaluationResultModel]
    mean_precision: float
    mean_reciprocal_rank: float


class EvaluationResponseModel(BaseModel):
    repo: str
    top_k: int
    summary: EvaluationSummaryModel
    token_usage: Optional[TokenUsageModel] = None

    @classmethod
    def from_result(cls, result: EvaluationResponse) -> "EvaluationResponseModel":
        summary = EvaluationSummaryModel(
            results=[
                EvaluationResultModel(
                    case=EvaluationCaseModel.model_validate(item.case),
                    matches=item.matches,
                    precision_at_k=item.precision_at_k,
                    reciprocal_rank=item.reciprocal_rank,
                )
                for item in result.summary.results
            ],
            mean_precision=result.summary.mean_precision,
            mean_reciprocal_rank=result.summary.mean_reciprocal_rank,
        )
        return cls(
            repo=result.repo,
            top_k=result.top_k,
            summary=summary,
            token_usage=TokenUsageModel.from_summary(result.token_usage),
        )


def _yield_services(*, require_gemini_api_key: bool):
    try:
        with application_services(
            console=None, require_gemini_api_key=require_gemini_api_key
        ) as services:
            yield services
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def get_services():
    yield from _yield_services(require_gemini_api_key=True)


def get_services_without_gemini():
    yield from _yield_services(require_gemini_api_key=False)


def create_app() -> FastAPI:
    app = FastAPI(title="Issue Tools API")

    @app.post("/index", response_model=IndexResponseModel)
    def run_index(
        request: IndexRequest, services=Depends(get_services)
    ) -> IndexResponseModel:
        try:
            result = index_repository(
                services,
                IndexParams(
                    repo=request.repo,
                    force=request.force,
                    limit=request.limit,
                    include_pulls=request.include_pulls,
                    filter_query=request.filter_query,
                ),
            )
        except CommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return IndexResponseModel.from_result(result)

    @app.post("/filter", response_model=FilterResponseModel)
    def run_filter(
        request: FilterRequest, services=Depends(get_services_without_gemini)
    ) -> FilterResponseModel:
        try:
            result = filter_repository(
                services,
                FilterParams(
                    query=request.query,
                    repo=request.repo,
                    limit=request.limit,
                ),
            )
        except CommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FilterResponseModel.from_result(result)

    @app.post("/search", response_model=SearchResponseModel)
    def run_search(
        request: SearchRequest, services=Depends(get_services)
    ) -> SearchResponseModel:
        try:
            result = search_repository(
                services,
                SearchParams(query=request.query, repo=request.repo, limit=request.limit),
            )
        except CommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SearchResponseModel.from_result(result)

    @app.post("/cluster", response_model=ClusterResponseModel)
    def run_cluster(
        request: ClusterRequest, services=Depends(get_services)
    ) -> ClusterResponseModel:
        try:
            result = cluster_repository(
                services,
                ClusterParams(repo=request.repo, query=request.query, k=request.k),
            )
        except CommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ClusterResponseModel.from_result(result)

    @app.get("/stream", response_model=StreamResponseModel)
    def run_stream(
        repo: Optional[str] = Query(None, description="Repository to query"),
        services=Depends(get_services),
    ) -> StreamResponseModel:
        try:
            result = stream_repository(services, StreamParams(repo=repo))
        except CommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamResponseModel.from_result(result)

    @app.get("/inspect-db", response_model=InspectDBResponseModel)
    def run_inspect(
        repo: Optional[str] = Query(None, description="Repository to inspect"),
        show: int = Query(0, ge=0, description="Number of recent documents to include"),
        services=Depends(get_services),
    ) -> InspectDBResponseModel:
        try:
            result = inspect_database(services, InspectDBParams(repo=repo, show=show))
        except CommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return InspectDBResponseModel.from_result(result)

    @app.post("/evaluate", response_model=EvaluationResponseModel)
    def run_evaluate(
        request: EvaluationRequest, services=Depends(get_services)
    ) -> EvaluationResponseModel:
        cases: Iterable[EvaluationCase] = [
            EvaluationCase(
                query=case.query,
                expected_numbers=[int(value) for value in case.expected_numbers],
            )
            for case in request.cases
        ]
        try:
            result = evaluate_repository(
                services,
                EvaluateParams(cases=cases, repo=request.repo, top_k=request.top_k),
            )
        except CommandError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return EvaluationResponseModel.from_result(result)

    @app.get("/tokens", response_model=TokenUsageModel)
    def run_tokens(
        lifetime: bool = Query(False, description="Show lifetime totals instead of session"),
    ) -> TokenUsageModel:
        config = load_config()
        tracker = TokenTracker(config.token_usage_path, config.cost.per_1k_tokens)
        summary = summarize_token_usage(tracker, lifetime=lifetime)
        if summary is None:
            summary = TokenUsageSummary(scope="lifetime" if lifetime else "session", models=[])
        model = TokenUsageModel.from_summary(summary)
        assert model is not None  # for type checkers
        return model

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import uvicorn

    uvicorn.run("issue_tools.api:app", host="0.0.0.0", port=8000, reload=False)
