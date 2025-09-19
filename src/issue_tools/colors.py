"""Utility helpers for working with color values."""

from __future__ import annotations

from typing import Optional

import string

_HEX_DIGITS = set(string.hexdigits)


def normalize_hex_color(color: Optional[str]) -> Optional[str]:
    """Return a normalized ``#rrggbb`` color string or ``None`` if invalid."""

    if color is None:
        return None
    value = str(color).strip()
    if not value:
        return None
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return None
    if any(ch not in _HEX_DIGITS for ch in value):
        return None
    return f"#{value.lower()}"


def pick_contrast_color(color: str) -> str:
    """Return ``"black"`` or ``"white"`` depending on background luminance."""

    normalized = normalize_hex_color(color)
    if not normalized:
        return "white"
    r = int(normalized[1:3], 16) / 255.0
    g = int(normalized[3:5], 16) / 255.0
    b = int(normalized[5:7], 16) / 255.0

    def _linearize(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    r_lin = _linearize(r)
    g_lin = _linearize(g)
    b_lin = _linearize(b)
    luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    return "black" if luminance > 0.5 else "white"
