"""GitHub API client helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
import logging

import requests

from .colors import normalize_hex_color


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GitHubLabel:
    """Representation of a GitHub label."""

    name: str
    color: Optional[str] = None


@dataclass(slots=True)
class GitHubItem:
    """Lightweight representation of an issue or pull request."""

    repo: str
    number: int
    title: str
    body: str
    labels: List[GitHubLabel]
    state: str
    html_url: str
    updated_at: str
    created_at: str
    is_pull_request: bool
    author: Optional[str]
    comments: List[str] = field(default_factory=list)

    def to_document(self, char_limit: Optional[int] = None) -> str:
        """Convert the issue to a text document suitable for embeddings."""

        parts: List[str] = []
        title = (self.title or "").strip()
        if title:
            parts.append(title)

        body = (self.body or "").strip()
        if body:
            parts.append(body)

        for comment in self.comments:
            comment_text = (comment or "").strip()
            if comment_text:
                parts.append(comment_text)

        text = "\n\n".join(parts)
        if char_limit is not None and len(text) > char_limit:
            return text[:char_limit]
        return text


class GitHubClient:
    """Simple REST client for fetching GitHub issues and pull requests."""

    base_url = "https://api.github.com"

    def __init__(self, token: Optional[str] = None) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "User-Agent": "issue-tools/0.1.0",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"token {token}"

    def fetch_items(
        self,
        repo: str,
        *,
        since: Optional[str] = None,
        limit: Optional[int] = None,
        include_pulls: bool = True,
    ) -> List[GitHubItem]:
        """Fetch issues (and optionally pull requests) for a repository."""

        per_page = 100
        url = f"{self.base_url}/repos/{repo}/issues"
        params = {
            "state": "all",
            "per_page": per_page,
            "sort": "updated",
            "direction": "desc",
        }
        if since:
            params["since"] = since

        items: List[GitHubItem] = []
        page = 1
        while True:
            params["page"] = page
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if not payload:
                break

            for issue in payload:
                item = self._build_item(repo, issue, include_pulls=include_pulls)
                if item is None:
                    continue
                items.append(item)
                if limit and len(items) >= limit:
                    return items

            if limit and len(items) >= limit:
                break

            page += 1

        return items

    def fetch_item(
        self,
        repo: str,
        number: int,
        *,
        include_pulls: bool = True,
    ) -> Optional[GitHubItem]:
        """Fetch a single issue or pull request including comments."""

        url = f"{self.base_url}/repos/{repo}/issues/{number}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s#%s: %s", repo, number, exc)
            return None

        payload = response.json()
        if not isinstance(payload, dict):
            return None
        return self._build_item(repo, payload, include_pulls=include_pulls)

    def hydrate_search_results(
        self,
        repo: str,
        items: Iterable[GitHubItem],
        *,
        include_pulls: bool = True,
    ) -> List[GitHubItem]:
        """Fetch full issue details (including comments) for search results."""

        hydrated: List[GitHubItem] = []
        seen_numbers: set[int] = set()
        for item in items:
            if item.repo and item.repo != repo:
                continue
            if item.is_pull_request and not include_pulls:
                continue
            number = int(item.number or 0)
            if number <= 0 or number in seen_numbers:
                continue
            seen_numbers.add(number)
            hydrated_item = self.fetch_item(repo, number, include_pulls=include_pulls)
            if hydrated_item is not None:
                hydrated.append(hydrated_item)
        return hydrated

    def _build_item(
        self,
        repo: str,
        issue: dict,
        *,
        include_pulls: bool,
        fetch_comments: bool = True,
    ) -> Optional[GitHubItem]:
        if not isinstance(issue, dict):
            return None
        is_pr = "pull_request" in issue
        try:
            number = int(issue.get("number") or 0)
        except (TypeError, ValueError):
            number = 0
        if number <= 0:
            return None
        if is_pr and not include_pulls:
            return None

        labels: List[GitHubLabel] = []
        for raw_label in issue.get("labels", []):
            name = str(raw_label.get("name", "")).strip()
            if not name:
                continue
            labels.append(
                GitHubLabel(
                    name=name,
                    color=normalize_hex_color(raw_label.get("color")),
                )
            )
        comments: List[str] = []
        if fetch_comments:
            issue_comment_count = int(issue.get("comments", 0) or 0)
            if issue_comment_count:
                comments.extend(
                    self._fetch_issue_comments(repo, number, issue_comment_count)
                )
            if is_pr:
                review_comment_count = int(issue.get("review_comments", 0) or 0)
                if review_comment_count:
                    comments.extend(
                        self._fetch_review_comments(repo, number, review_comment_count)
                    )
                comments.extend(self._fetch_pull_reviews(repo, number))

        return GitHubItem(
            repo=repo,
            number=number,
            title=str(issue.get("title", "") or ""),
            body=str(issue.get("body") or ""),
            labels=labels,
            state=str(issue.get("state", "open") or "open"),
            html_url=str(issue.get("html_url", "") or ""),
            updated_at=str(issue.get("updated_at", "") or ""),
            created_at=str(issue.get("created_at", "") or ""),
            is_pull_request=is_pr,
            author=(issue.get("user") or {}).get("login") if isinstance(issue.get("user"), dict) else None,
            comments=comments,
        )

    def search_issues(self, query: str, *, limit: int = 10) -> List[GitHubItem]:
        """Search GitHub issues and pull requests using the REST API."""

        if limit <= 0:
            return []

        per_page = min(100, max(1, limit))
        url = f"{self.base_url}/search/issues"
        params: Dict[str, object] = {
            "q": query,
            "per_page": per_page,
            "page": 1,
            "sort": "updated",
            "order": "desc",
        }

        items: List[GitHubItem] = []
        while True:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json() or {}
            raw_items = payload.get("items", []) if isinstance(payload, dict) else []
            if not raw_items:
                break

            for issue in raw_items:
                repo_name = self._parse_repo_from_url(issue.get("repository_url", ""))
                labels = [label.get("name", "") for label in issue.get("labels", [])]
                item = GitHubItem(
                    repo=repo_name,
                    number=int(issue.get("number") or 0),
                    title=issue.get("title", ""),
                    body=issue.get("body") or "",
                    labels=[label for label in labels if label],
                    state=issue.get("state", "open"),
                    html_url=issue.get("html_url", ""),
                    updated_at=issue.get("updated_at", ""),
                    created_at=issue.get("created_at", ""),
                    is_pull_request="pull_request" in issue,
                    author=(issue.get("user") or {}).get("login"),
                    comments=[],
                )
                items.append(item)
                if len(items) >= limit:
                    return items

            if len(raw_items) < per_page:
                break

            params["page"] = params.get("page", 1) + 1  # type: ignore[assignment]

        return items

    @staticmethod
    def _parse_repo_from_url(url: str) -> str:
        url = url.rstrip("/")
        if not url:
            return ""
        parts = url.split("/")
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return url

    def _fetch_issue_comments(
        self,
        repo: str,
        number: int,
        expected_count: int,
    ) -> List[str]:
        if expected_count <= 0:
            return []
        url = f"{self.base_url}/repos/{repo}/issues/{number}/comments"
        return self._fetch_comment_bodies(url)

    def _fetch_review_comments(
        self,
        repo: str,
        number: int,
        expected_count: int,
    ) -> List[str]:
        if expected_count <= 0:
            return []
        url = f"{self.base_url}/repos/{repo}/pulls/{number}/comments"
        return self._fetch_comment_bodies(url)

    def _fetch_pull_reviews(self, repo: str, number: int) -> List[str]:
        url = f"{self.base_url}/repos/{repo}/pulls/{number}/reviews"
        try:
            reviews = self._paginate(url)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch pull request reviews for %s#%s: %s", repo, number, exc)
            return []

        bodies: List[str] = []
        for review in reviews:
            body = (review or {}).get("body")
            if body:
                bodies.append(str(body))
        return bodies

    def _fetch_comment_bodies(self, url: str) -> List[str]:
        try:
            payload = self._paginate(url)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch comments from %s: %s", url, exc)
            return []

        comments: List[str] = []
        for comment in payload:
            body = (comment or {}).get("body")
            if body:
                comments.append(str(body))
        return comments

    def _paginate(
        self,
        url: str,
        params: Optional[Dict[str, object]] = None,
    ) -> List[object]:
        per_page = 100
        page = 1
        results: List[object] = []
        while True:
            page_params: Dict[str, object] = {"per_page": per_page, "page": page}
            if params:
                page_params.update(params)
            response = self.session.get(url, params=page_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            if isinstance(data, list):
                results.extend(data)
                if len(data) < per_page:
                    break
            else:
                # For endpoints that do not return lists, stop after first response.
                results.append(data)
                break
            page += 1
        return results
