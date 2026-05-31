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
import importlib.util
import logging
from typing import Any

from packaging.version import parse

from spectrochempy.api.plugins.constants import CORE_PLUGIN_API_VERSION

logger = logging.getLogger(__name__)


def check_plugin_metadata(plugin: Any) -> list[str]:
    """
    Check plugin metadata completeness and return a list of issues.

    Unlike ``validate_plugin_compatibility``, this function does **not**
    check API version compatibility or minimum SpectroChemPy version.
    It only verifies that required metadata fields exist and have
    non-empty values.

    Parameters
    ----------
    plugin : Any
        A plugin instance.

    Returns
    -------
    list of str
        Human-readable issue descriptions.  An empty list means no
        issues were found.
    """
    issues: list[str] = []
    metadata = _get_plugin_metadata(plugin)
    name = metadata.get("name", "unknown")

    for key in ("name", "version", "plugin_api_version"):
        if not metadata.get(key):
            issues.append(
                f"Plugin '{name}' is missing required metadata field '{key}'."
            )

    if not metadata.get("description"):
        issues.append(f"Plugin '{name}' has no description.")

    return issues


def check_plugin_requires(plugin: Any) -> list[str]:
    """
    Check that optional dependencies declared via ``requires`` are importable.

    Parameters
    ----------
    plugin : Any
        A plugin instance.

    Returns
    -------
    list of str
        List of missing dependencies.  An empty list means all declared
        dependencies are available.
    """
    issues: list[str] = []
    name = getattr(plugin, "name", "unknown")
    requires = getattr(plugin, "requires", [])
    if not requires:
        return issues
    for dep in requires:
        # Extract package name from a pip-style spec (e.g. "cantera>=3.0" -> "cantera")
        pkg_name = dep.split(">=")[0].split("==")[0].split("~=")[0].strip()
        if importlib.util.find_spec(pkg_name) is None:
            issues.append(f"Plugin '{name}' requires '{dep}' which is not installed.")
    return issues


def check_plugin_contributions(plugin: Any) -> list[str]:
    """
    Check plugin contribution declarations for consistency.

    Inspects declarative hooks (``register_readers``,
    ``register_writers``, ``register_processors``,
    ``register_visualizers``, ``register_analyses``,
    ``register_simulations``, ``register_accessors``,
    ``register_unit_contexts``) without executing them. This intentionally
    avoids hook side effects during validation.

    Parameters
    ----------
    plugin : Any
        A plugin instance.

    Returns
    -------
    list of str
        Human-readable issue descriptions.  An empty list means no
        issues were found.
    """
    issues: list[str] = []
    name = getattr(plugin, "name", "unknown")

    for hook_name in (
        "register_readers",
        "register_writers",
        "register_processors",
        "register_visualizers",
        "register_analyses",
        "register_simulations",
        "register_accessors",
        "register_unit_contexts",
    ):
        if not hasattr(plugin, hook_name):
            continue
        method = getattr(plugin, hook_name)
        if not callable(method):
            issues.append(f"Plugin '{name}': '{hook_name}' is not callable.")

    return issues


def check_plugin_compatibility(plugin: Any) -> list[str]:
    """
    Full compatibility check for a plugin.

    Combines ``check_plugin_metadata``, ``check_plugin_contributions``,
    ``check_plugin_requires``, and ``validate_plugin_compatibility``
    into a single function that returns all issues at once.

    Parameters
    ----------
    plugin : Any
        A plugin instance.

    Returns
    -------
    list of str
        All compatibility issues found.  An empty list means the
        plugin is fully compatible.
    """
    issues: list[str] = []
    issues.extend(check_plugin_metadata(plugin))
    issues.extend(check_plugin_contributions(plugin))
    issues.extend(check_plugin_requires(plugin))
    _, compat_errors = validate_plugin_compatibility(plugin)
    issues.extend(compat_errors)
    return issues


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

    plugin_api_version = getattr(
        plugin,
        "PLUGIN_API_VERSION",
        getattr(plugin, "api_version", ""),
    )
    return {
        "name": getattr(plugin, "name", ""),
        "version": getattr(plugin, "version", ""),
        "plugin_api_version": plugin_api_version,
        "spectrochempy_min_version": getattr(plugin, "spectrochempy_min_version", ""),
        "description": getattr(plugin, "description", ""),
    }


def _is_compatible_api_version(plugin_api: str, core_api: str) -> bool:
    """Compare major version numbers only."""
    try:
        return (
            plugin_api.split(".", maxsplit=1)[0] == core_api.split(".", maxsplit=1)[0]
        )
    except (ValueError, IndexError):
        return False


def _satisfies_min_version(current: str, minimum: str) -> bool:
    """
    PEP 440 version check: *current* >= *minimum*.

    Dev versions of a later release are considered sufficient, e.g.
    ``0.9.1.dev0 >= 0.9.0``, so that an editable post-release checkout
    satisfies the ``>=0.9.0`` constraint during development.
    """
    try:
        current_ver = parse(current)
        minimum_ver = parse(minimum)
        if current_ver >= minimum_ver:
            return True
        if current_ver.is_prerelease and parse(current_ver.base_version) > minimum_ver:
            return True
        return False
    except Exception:
        return False
