from issue_tools.github_client import GitHubItem, GitHubLabel


def _make_item(**overrides):
    defaults = dict(
        repo="org/repo",
        number=1,
        title="Sample issue",
        body="Issue body",
        labels=[GitHubLabel(name="bug", color="#ff0000")],
        state="open",
        html_url="https://example.com",
        updated_at="2024-01-01T00:00:00Z",
        created_at="2024-01-01T00:00:00Z",
        is_pull_request=False,
        author="octocat",
        comments=["First comment", "Second comment"],
    )
    defaults.update(overrides)
    return GitHubItem(**defaults)


def test_to_document_includes_comments():
    item = _make_item()
    doc = item.to_document()
    assert "Issue body" in doc
    assert "First comment" in doc
    assert "Second comment" in doc
    assert doc.startswith("Sample issue")


def test_to_document_respects_char_limit():
    item = _make_item(body="A" * 100, comments=["B" * 200])
    doc = item.to_document(char_limit=50)
    assert len(doc) == 50
    assert doc == doc.strip()
