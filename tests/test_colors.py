import pytest

from issue_tools.colors import normalize_hex_color, pick_contrast_color


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("#ABCDEF", "#abcdef"),
        ("ABCDEF", "#abcdef"),
        (" abcdef ", "#abcdef"),
        ("abc", "#aabbcc"),
        ("#123456", "#123456"),
    ],
)
def test_normalize_hex_color_valid_values(raw, expected):
    assert normalize_hex_color(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [None, "", "  ", "xyz", "#12", "1234", "12345g"],
)
def test_normalize_hex_color_invalid_values(raw):
    assert normalize_hex_color(raw) is None


@pytest.mark.parametrize(
    "color, expected",
    [
        ("#000000", "white"),
        ("#ffffff", "black"),
        ("#00ff00", "black"),
        ("abc", "white"),
        ("not-a-color", "white"),
    ],
)
def test_pick_contrast_color(color, expected):
    assert pick_contrast_color(color) == expected
