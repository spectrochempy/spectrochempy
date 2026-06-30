# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Internal plotting kwargs normalization helpers."""

from __future__ import annotations

from typing import Any

_PLOTTING_KWARG_ALIASES = {
    "c": "color",
    "lw": "linewidth",
    "ls": "linestyle",
    "m": "marker",
    "ms": "markersize",
    "mew": "markeredgewidth",
    "mec": "markeredgecolor",
    "mfc": "markerfacecolor",
    "colormap": "cmap",
    "nc": "nlevels",
    "calpha": "alpha",
}


def normalize_plot_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of *kwargs* with plotting aliases normalized."""
    if kwargs is None:
        return {}

    normalized = dict(kwargs)
    for alias, canonical in _PLOTTING_KWARG_ALIASES.items():
        if canonical not in normalized and alias in normalized:
            normalized[canonical] = normalized[alias]
        normalized.pop(alias, None)
    return normalized


def normalize_style_argument(
    style: Any,
    *,
    default: Any,
) -> Any:
    """Normalize a style argument to the local plotting convention."""
    if style is None:
        style = default
    if isinstance(style, str):
        return [style]
    return style
