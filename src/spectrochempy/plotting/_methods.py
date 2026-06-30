# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Internal plotting method normalization helpers."""

from __future__ import annotations

import warnings

CANONICAL_1D_METHODS = frozenset(
    {
        "pen",
        "bar",
        "scatter",
        "scatter_pen",
        "scatter+pen",
    }
)

CANONICAL_2D_PLUS_METHODS = frozenset(
    {
        "lines",
        "stack",
        "contour",
        "map",
        "contourf",
        "image",
        "surface",
        "waterfall",
    }
)

_DEFAULT_METHODS_BY_NDIM = {
    1: "pen",
    2: "lines",
    3: "surface",
}

_BACKEND_METHOD_ALIASES = {
    "stack": "lines",
    "map": "contour",
}

_SEMANTIC_METHOD_ALIASES = {
    "image": "contourf",
}

_MULTIPLOT_METHOD_ALIASES_BY_NDIM = {
    1: {"lines": "pen"},
}

_FALLBACK_METHOD_ALIASES = {
    "1d": {
        "lines": "pen",
    },
    "2d+": {
        "pen": "lines",
    },
}


def get_default_method_for_ndim(ndim: int) -> str | None:
    """Return the default plotting method for a squeezed dimensionality."""
    return _DEFAULT_METHODS_BY_NDIM.get(ndim)


def get_dispatch_method_key(method: str | None) -> str | None:
    """Normalize a method name for dispatch-table lookup."""
    if method is None:
        return None
    return method.replace("+", "_")


def normalize_backend_method(
    method: str | None,
    *,
    warned_aliases: set[str] | None = None,
    stacklevel: int = 3,
) -> str | None:
    """
    Normalize backend-facing method names.

    Preserves current behavior:
    - ``image`` is a silent semantic alias for ``contourf``
    - ``stack`` and ``map`` emit a deprecation warning once per session
    """
    if method is None:
        return None

    if method in _SEMANTIC_METHOD_ALIASES:
        return _SEMANTIC_METHOD_ALIASES[method]

    if method in _BACKEND_METHOD_ALIASES:
        canonical = _BACKEND_METHOD_ALIASES[method]
        if warned_aliases is not None and method not in warned_aliases:
            warned_aliases.add(method)
            warnings.warn(
                f'method="{method}" is deprecated and will be removed in 0.11.0. '
                f'Use method="{canonical}" instead.',
                DeprecationWarning,
                stacklevel=stacklevel,
            )
        return canonical

    return method


def normalize_multiplot_method(method: str | None, ndim: int | None) -> str | None:
    """Normalize multiplot's internal default method vocabulary."""
    if method is None or ndim is None:
        return method
    return _MULTIPLOT_METHOD_ALIASES_BY_NDIM.get(ndim, {}).get(method, method)


def normalize_fallback_method_for_target(
    method: str | None,
    *,
    target: str,
) -> str | None:
    """Normalize public compatibility aliases for cross-dimensional fallbacks."""
    if method is None:
        return None
    return _FALLBACK_METHOD_ALIASES.get(target, {}).get(method, method)


def validate_method_for_target_dimension(
    method: str,
    *,
    target: str,
    source: str,
) -> str:
    """Validate an explicit fallback method against the target dimensional family."""
    method = normalize_fallback_method_for_target(method, target=target)
    if target == "1d" and method in CANONICAL_1D_METHODS:
        return method
    if target == "2d+" and method in CANONICAL_2D_PLUS_METHODS:
        return method
    raise ValueError(
        f"method={method!r} is incompatible with {source}; "
        f"use a {target} plotting method or call dataset.plot() for automatic dispatch."
    )
