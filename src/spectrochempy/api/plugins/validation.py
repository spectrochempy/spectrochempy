# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Plugin compatibility validation.

Used by the plugin-manager during registration to skip incompatible
plugins gracefully.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

from spectrochempy.api.plugins.constants import CORE_PLUGIN_API_VERSION

logger = logging.getLogger(__name__)


def validate_plugin_compatibility(plugin: Any) -> tuple[bool, list[str]]:
    """
    Validate that *plugin* is compatible with the current SpectroChemPy.

    Parameters
    ----------
    plugin : Any
        A plugin instance (duck-typed or inheriting from
        :class:`~spectrochempy.api.plugins.base.SpectroChemPyPlugin`).

    Returns
    -------
    is_compatible : bool
        ``True`` when all checks pass.
    errors : list of str
        Human-readable error messages for every failed check (empty
        when *is_compatible* is ``True``).
    """
    errors: list[str] = []

    metadata = _get_plugin_metadata(plugin)
    name = metadata.get("name", "unknown")

    # 1.  Required metadata fields
    for key in ("name", "version", "plugin_api_version"):
        if not metadata.get(key):
            errors.append(
                f"Plugin '{name}' is missing required metadata field '{key}'. "
                f"Plugin skipped."
            )

    # 2.  Plugin API version compatibility (major version must match)
    plugin_api = metadata.get("plugin_api_version", "")
    if plugin_api and not _is_compatible_api_version(
        plugin_api, CORE_PLUGIN_API_VERSION
    ):
        errors.append(
            f"Plugin '{name}' requires API version {plugin_api}, "
            f"but SpectroChemPy supports {CORE_PLUGIN_API_VERSION}.  "
            f"Plugin skipped."
        )

    # 3.  SpectroChemPy minimum version
    min_version = metadata.get("spectrochempy_min_version", "")
    if min_version and min_version not in ("", "0.0.0"):
        current = importlib.metadata.version("spectrochempy")
        if not _satisfies_min_version(current, min_version):
            errors.append(
                f"Plugin '{name}' requires SpectroChemPy >= {min_version}, "
                f"but the installed version is {current}.  "
                f"Plugin skipped."
            )

    for msg in errors:
        logger.warning(msg)

    return len(errors) == 0, errors


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _get_plugin_metadata(plugin: Any) -> dict[str, Any]:
    """
    Return a metadata dict for *plugin*.

    If the plugin exposes a ``plugin_info()`` method (recommended),
    its return value is used.  Otherwise, metadata is scraped from
    public attributes (backward-compatible fallback).
    """
    if hasattr(plugin, "plugin_info") and callable(plugin.plugin_info):
        try:
            return plugin.plugin_info()
        except Exception as exc:
            logger.debug("plugin_info() raised %s: %s", type(exc).__name__, exc)

    return {
        "name": getattr(plugin, "name", ""),
        "version": getattr(plugin, "version", ""),
        "plugin_api_version": getattr(
            plugin,
            "api_version",
            getattr(plugin, "PLUGIN_API_VERSION", ""),
        ),
        "spectrochempy_min_version": getattr(plugin, "spectrochempy_min_version", ""),
        "description": getattr(plugin, "description", ""),
    }


def _is_compatible_api_version(plugin_api: str, core_api: str) -> bool:
    """Compare major version numbers only."""
    try:
        return plugin_api.split(".")[0] == core_api.split(".")[0]
    except (ValueError, IndexError):
        return False


def _satisfies_min_version(current: str, minimum: str) -> bool:
    """Semver check: *current* >= *minimum*."""
    try:
        _cur = [int(x) for x in current.split(".")]
        _min = [int(x) for x in minimum.split(".")]
        while len(_cur) < len(_min):
            _cur.append(0)
        while len(_min) < len(_cur):
            _min.append(0)
        return _cur >= _min
    except (ValueError, IndexError):
        return False
