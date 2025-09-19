from rich.style import Style

from issue_tools.cli import _format_label
from issue_tools.labels import LabelDetail


def test_format_label_uses_normalized_color_and_contrast():
    text = _format_label(LabelDetail(name="bug", color="#ABCDEF"))
    assert text.plain == " bug "
    assert text.style == Style(color="black", bgcolor="#abcdef", bold=True)


def test_format_label_without_color_uses_default_style():
    text = _format_label(LabelDetail(name="docs", color=None))
    assert text.plain == " docs "
    assert text.style == Style(color="white", bgcolor="grey27", bold=True)


def test_format_label_invalid_color_falls_back_to_default():
    text = _format_label(LabelDetail(name="needs info", color="invalid"))
    assert text.plain == " needs info "
    assert text.style == Style(color="white", bgcolor="grey27", bold=True)
