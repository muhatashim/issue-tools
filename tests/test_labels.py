from issue_tools.github_client import GitHubLabel
from issue_tools.labels import (
    LabelDetail,
    extract_label_details,
    label_details_from_github,
)


def test_extract_label_details_prefers_metadata_entries():
    metadata = {
        "label_details": [
            {"name": " Bug ", "color": "  #ABCDEF  "},
            {"name": "", "color": "#ffffff"},
            {"foo": "bar"},
            "invalid",
            {"name": "Needs info", "color": ""},
        ]
    }

    details = extract_label_details(metadata, fallback_labels=["Bug", "Needs info"])
    assert details == [
        LabelDetail(name="Bug", color="#abcdef"),
        LabelDetail(name="Needs info", color=None),
    ]


def test_extract_label_details_falls_back_to_label_names():
    metadata = {"model": "fake-document"}

    details = extract_label_details(metadata, fallback_labels=["bug", " docs ", ""])
    assert details == [
        LabelDetail(name="bug", color=None),
        LabelDetail(name="docs", color=None),
    ]


def test_label_details_from_github_normalizes_and_skips_invalid():
    labels = [
        GitHubLabel(name="bug", color="#ABCDEF"),
        GitHubLabel(name="", color="123456"),
        GitHubLabel(name="Docs", color=None),
    ]

    details = label_details_from_github(labels)
    assert details == [
        LabelDetail(name="bug", color="#abcdef"),
        LabelDetail(name="Docs", color=None),
    ]
