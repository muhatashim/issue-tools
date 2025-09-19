"""Helpers for working with GitHub label metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING

from .colors import normalize_hex_color

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .github_client import GitHubLabel


@dataclass(slots=True)
class LabelDetail:
    """Normalized representation of a label for display."""

    name: str
    color: Optional[str] = None


def _clean_name(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
    else:
        cleaned = str(value).strip()
    if not cleaned:
        return None
    return cleaned


def extract_label_details(
    metadata: Mapping[str, object] | None,
    *,
    fallback_labels: Iterable[str] | None = None,
) -> List[LabelDetail]:
    """Extract label metadata from stored document metadata.

    When explicit label metadata is present it takes precedence. Otherwise the
    provided ``fallback_labels`` are used to build simple label details.
    """

    details: List[LabelDetail] = []
    if metadata:
        raw_details = metadata.get("label_details")
        if isinstance(raw_details, Sequence):
            for entry in raw_details:
                if not isinstance(entry, Mapping):
                    continue
                name = _clean_name(entry.get("name"))
                if not name:
                    continue
                color_value = entry.get("color")
                color = normalize_hex_color(color_value) if isinstance(color_value, str) else None
                details.append(LabelDetail(name=name, color=color))
    if details:
        return details
    fallback_details: List[LabelDetail] = []
    if fallback_labels:
        for label in fallback_labels:
            name = _clean_name(label)
            if name:
                fallback_details.append(LabelDetail(name=name))
    return fallback_details


def label_details_from_github(labels: Iterable["GitHubLabel"]) -> List[LabelDetail]:
    """Build label details directly from GitHub label objects."""

    details: List[LabelDetail] = []
    for label in labels:
        if label is None:
            continue
        name = _clean_name(getattr(label, "name", ""))
        if not name:
            continue
        color_value = getattr(label, "color", None)
        color = normalize_hex_color(color_value) if color_value else None
        details.append(LabelDetail(name=name, color=color))
    return details
