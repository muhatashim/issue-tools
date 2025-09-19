from issue_tools.filters import FilterCriteria, parse_query_with_filters


def test_parse_filters_with_labels_and_dates():
    query, filters = parse_query_with_filters(
        "crash label:bug tag:frontend state:open date:>=2024-01-01 is:issue"
    )
    assert query == "crash"
    assert filters.labels == {"bug", "frontend"}
    assert filters.states == {"open"}
    assert filters.types == {"issue"}
    assert filters.updated_filter == (">=", "2024-01-01T00:00:00Z")


def test_parse_filters_numbers_and_repo():
    query, filters = parse_query_with_filters("fix auth repo:org/example number:42")
    assert query == "fix auth"
    assert filters.repo == "org/example"
    assert filters.numbers == {42}
