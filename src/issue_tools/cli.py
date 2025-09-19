"""Command line interface for issue tools."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, TypeVar

import typer
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .commands import (
    CommandError,
    ClusterParams,
    EvaluateParams,
    FilterParams,
    IndexParams,
    InspectDBParams,
    SearchParams,
    StreamParams,
    TokenUsageSummary,
    cluster_repository,
    evaluate_repository,
    filter_repository,
    index_repository,
    inspect_database,
    resolve_repo,
    search_repository,
    stream_repository,
    summarize_token_usage,
)
from .colors import normalize_hex_color, pick_contrast_color
from .labels import LabelDetail, extract_label_details, label_details_from_github
from .token_tracker import TokenTracker
from .config import load_config
from .evaluation import load_cases
from .runtime import ConfigurationError, application_services

app = typer.Typer(add_completion=True)
console = Console()

T = TypeVar("T")


def _invoke(callback: Callable[[object], T], *, require_gemini_api_key: bool = True) -> T:
    try:
        with application_services(
            console=console, require_gemini_api_key=require_gemini_api_key
        ) as services:
            return callback(services)
    except ConfigurationError as exc:  # pragma: no cover - exercised via CLI usage
        raise typer.BadParameter(str(exc)) from exc
    except CommandError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def index(
    repo: Optional[str] = typer.Option(None, "--repo", help="owner/repo to index"),
    force: bool = typer.Option(False, "--force", help="Force indexing regardless of schedule"),
    limit: Optional[int] = typer.Option(None, help="Limit the number of items fetched"),
    include_pulls: bool = typer.Option(True, help="Include pull requests during indexing"),
    filter_query: Optional[str] = typer.Option(
        None,
        "--filter",
        "--filter-query",
        help="GitHub search filter applied before indexing",
    ),
) -> None:
    """Index GitHub issues and pull requests."""

    result = _invoke(
        lambda services: index_repository(
            services,
            IndexParams(
                repo=repo,
                force=force,
                limit=limit,
                include_pulls=include_pulls,
                filter_query=filter_query,
            ),
        )
    )
    if result.skipped:
        console.print(f"[yellow]{result.message}[/yellow]")
    else:
        console.print(f"[green]{result.message}[/green]")
        console.print(f"Last indexed at {result.last_indexed_at}")
    _print_token_summary(result.token_usage, title="Token usage (this run)")


@app.command()
def filter(
    query: str = typer.Argument(..., help="GitHub search filters or keywords"),
    repo: Optional[str] = typer.Option(None, "--repo", help="Repository to search"),
    limit: int = typer.Option(10, help="Number of items to display"),
) -> None:
    """Fetch GitHub issues or pull requests that match a filter query."""

    result = _invoke(
        lambda services: filter_repository(
            services, FilterParams(query=query, repo=repo, limit=limit)
        ),
        require_gemini_api_key=False,
    )
    if not result.items:
        console.print("[yellow]No matching issues or pull requests found.[/yellow]")
        return

    table = Table(show_lines=False)
    table.add_column("Repo")
    table.add_column("Number")
    table.add_column("Type")
    table.add_column("State")
    table.add_column("Title")
    table.add_column("URL")
    for item in result.items:
        table.add_row(
            item.repo,
            f"#{item.number}",
            item.doc_type,
            item.state,
            item.title,
            f"[link={item.html_url}]{item.html_url}[/link]",
        )
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query with optional GitHub-style filters"),
    repo: Optional[str] = typer.Option(None, "--repo", help="Repository to search"),
    limit: int = typer.Option(5, help="Number of results to display"),
) -> None:
    """Search indexed issues and pull requests."""

    result = _invoke(
        lambda services: search_repository(
            services, SearchParams(query=query, repo=repo, limit=limit)
        )
    )
    if not result.matches:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(show_lines=False)
    table.add_column("Score")
    table.add_column("Number")
    table.add_column("Type")
    table.add_column("State")
    table.add_column("Title")
    table.add_column("Labels")
    table.add_column("URL")
    for match in result.matches:
        table.add_row(
            f"{match.score:.3f}",
            f"#{match.number}",
            match.doc_type,
            match.state,
            match.title,
            _render_labels(match.label_details),
            f"[link={match.url}]{match.url}[/link]",
        )
    console.print(table)
    _print_token_summary(result.token_usage, title="Token usage (this run)")


@app.command()
def cluster(
    repo: Optional[str] = typer.Option(None, "--repo", help="Repository to cluster"),
    query: Optional[str] = typer.Option(
        None,
        help="Search query and/or filters applied before clustering",
    ),
    k: Optional[int] = typer.Option(
        None,
        help="Number of clusters (omit to auto-detect)",
    ),
    limit: int = typer.Option(
        50,
        help="Maximum number of documents to cluster (set to 0 for no limit)",
    ),
    max_clusters: int = typer.Option(
        8,
        help="Maximum clusters to consider when auto-detecting",
    ),
) -> None:
    """Cluster documents, optionally based on semantic search results."""

    if max_clusters <= 0:
        raise typer.BadParameter("max-clusters must be positive.")

    result = _invoke(
        lambda services: cluster_repository(
            services, ClusterParams(repo=repo, query=query, k=k, limit=limit, max_clusters=max_clusters)
        )
    )
    if not result.clusters:
        console.print("[yellow]No documents available for clustering.[/yellow]")
        return

    for cluster_info in result.clusters:
        console.print(
            f"[bold]Cluster {cluster_info.cluster_id}[/bold] "
            f"({cluster_info.size} docs, top labels: {', '.join(cluster_info.top_labels) or 'n/a'})"
        )
        for doc in cluster_info.documents[:5]:
            entry = Text("  - ")
            entry.append(
                f"#{doc.number} {doc.title}",
                style=f"link {doc.html_url}",
            )
            entry.append(f" ({doc.doc_type}, {doc.state})")
            details = extract_label_details(
                None,  # ClusterDocumentSummary doesn't have metadata
                fallback_labels=doc.labels,
            )
            if details:
                entry.append(" ")
                entry.append_text(_render_labels(details))
            console.print(entry)
    _print_token_summary(result.token_usage, title="Token usage (this run)")


@app.command()
def stream(
    repo: Optional[str] = typer.Option(None, "--repo", help="Repository to check for updates"),
) -> None:
    """Show issues updated since the last indexing run."""

    result = _invoke(lambda services: stream_repository(services, StreamParams(repo=repo)))
    if not result.items:
        console.print("[yellow]No new updates since the last index.[/yellow]")
        return
    for item in result.items:
        entry = Text()
        entry.append(f"#{item.number} {item.title}", style=f"link {item.html_url}")
        entry.append(" ")
        entry.append(f"({item.state}, updated {item.updated_at})")
        if item.labels:
            details = extract_label_details(
                None,  # StreamItemSummary doesn't have metadata
                fallback_labels=item.labels,
            )
            if details:
                entry.append(" ")
                entry.append_text(_render_labels(details))
        console.print(entry)


@app.command("inspect-db")
def inspect_db(
    repo: Optional[str] = typer.Option(None, "--repo", help="Repository to inspect"),
    show: int = typer.Option(0, "--show", min=0, help="Number of recent documents to display"),
) -> None:
    """Inspect the vector database for debugging."""

    result = _invoke(
        lambda services: inspect_database(services, InspectDBParams(repo=repo, show=show))
    )
    if not result.summaries:
        if repo:
            console.print(f"[yellow]No documents found for repo {repo}.[/yellow]")
        else:
            console.print("[yellow]Vector store is empty.[/yellow]")
        return

    console.print(f"Database path: {result.database_path}")

    summary_table = Table(title="Repository summary", show_lines=False)
    summary_table.add_column("Repo")
    summary_table.add_column("Total")
    summary_table.add_column("Issues")
    summary_table.add_column("Pulls")
    summary_table.add_column("Created range")
    summary_table.add_column("Updated range")
    summary_table.add_column("Last indexed")
    for summary in result.summaries:
        summary_table.add_row(
            summary.repo,
            str(summary.total),
            str(summary.issues),
            str(summary.pulls),
            _format_range(summary.first_created, summary.last_created),
            _format_range(summary.first_updated, summary.last_updated),
            summary.last_indexed or "-",
        )
    console.print(summary_table)

    if result.embedding_models:
        model_table = Table(title="Embedding models", show_lines=False)
        model_table.add_column("Model")
        model_table.add_column("Dimensions")
        model_table.add_column("Documents")
        for info in result.embedding_models:
            model_table.add_row(info.model, str(info.dimensions), str(info.documents))
        console.print(model_table)

    if result.samples:
        sample_table = Table(
            title=f"Most recent {len(result.samples)} documents", show_lines=False
        )
        if not repo:
            sample_table.add_column("Repo")
        sample_table.add_column("Number")
        sample_table.add_column("Type")
        sample_table.add_column("State")
        sample_table.add_column("Updated")
        sample_table.add_column("Title")
        for doc in result.samples:
            values = []
            if not repo:
                values.append(doc.repo or "")
            values.extend(
                [
                    f"#{doc.number}",
                    doc.doc_type,
                    doc.state,
                    doc.updated_at,
                    _shorten(doc.title or ""),
                ]
            )
            sample_table.add_row(*values)
        console.print(sample_table)


@app.command()
def evaluate(
    evaluation_file: Path = typer.Argument(..., exists=True, help="Path to evaluation cases JSON"),
    repo: Optional[str] = typer.Option(None, "--repo", help="Repository to evaluate"),
    top_k: int = typer.Option(5, help="Rank cutoff for precision metrics"),
) -> None:
    """Run offline evaluations against the stored embeddings."""

    cases = load_cases(evaluation_file)
    result = _invoke(
        lambda services: evaluate_repository(
            services, EvaluateParams(cases=cases, repo=repo, top_k=top_k)
        )
    )

    table = Table(show_lines=False)
    table.add_column("Query")
    table.add_column("Expected")
    table.add_column("Matches")
    table.add_column("P@k")
    table.add_column("MRR")
    for item in result.summary.results:
        table.add_row(
            item.case.query,
            ", ".join(map(str, item.case.expected_numbers)),
            ", ".join(map(str, item.matches)) or "-",
            f"{item.precision_at_k:.2f}",
            f"{item.reciprocal_rank:.2f}",
        )
    console.print(table)
    console.print(
        f"Mean precision@{top_k}: {result.summary.mean_precision:.2f}, "
        f"MRR: {result.summary.mean_reciprocal_rank:.2f}"
    )
    _print_token_summary(result.token_usage, title="Token usage (this run)")


@app.command()
def tokens(
    show_lifetime: bool = typer.Option(
        False, help="Show lifetime totals instead of current session"
    ),
) -> None:
    """Display token usage and cost estimates."""

    config = load_config()
    tracker = TokenTracker(config.token_usage_path, config.cost.per_1k_tokens)
    summary = summarize_token_usage(tracker, lifetime=show_lifetime)
    if not summary or not summary.models:
        console.print("[yellow]No token usage recorded yet.[/yellow]")
        return
    _print_token_summary(summary)


def _render_labels(label_details: list["LabelDetail"]) -> Text:
    if not label_details:
        return Text("-")

    text = Text()
    for index, detail in enumerate(label_details):
        if index:
            text.append(" ")
        text.append(_format_label(detail))
    return text


def _format_label(detail: "LabelDetail") -> Text:
    normalized = normalize_hex_color(detail.color) if detail.color else None
    if normalized:
        text_color = pick_contrast_color(normalized)
        style = Style(color=text_color, bgcolor=normalized, bold=True)
    else:
        style = Style(color="white", bgcolor="grey27", bold=True)
    return Text(f" {detail.name} ", style=style)


def _print_token_summary(summary: TokenUsageSummary | None, *, title: Optional[str] = None) -> None:
    if not summary or not summary.models:
        return
    table = Table(
        title=title or f"Token usage ({summary.scope})",
        show_lines=False,
    )
    table.add_column("Model")
    table.add_column("Prompt tokens")
    table.add_column("Response tokens")
    table.add_column("Total tokens")
    table.add_column("Cost ($)")
    for stats in summary.models:
        table.add_row(
            stats.model,
            str(stats.prompt_tokens),
            str(stats.response_tokens),
            str(stats.total_tokens),
            f"{stats.cost:.4f}",
        )
    console.print(table)


def _format_range(start: Optional[str], end: Optional[str]) -> str:
    if start and end:
        if start == end:
            return start
        return f"{start} -> {end}"
    if end:
        return f"n/a -> {end}"
    if start:
        return f"{start} -> n/a"
    return "n/a"


def _shorten(text: str, limit: int = 80) -> str:
    if limit <= 3:
        return text[:limit]
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


if __name__ == "__main__":
    app()
