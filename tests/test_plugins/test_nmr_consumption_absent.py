# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Core-only behavior when the optional spectrochempy-nmr plugin is absent.

This module must run in an environment where the plugin is neither installed nor
importable. It is skipped in plugin-enabled environments and executed by the
dedicated cross-repository consumption workflow.
"""

from __future__ import annotations

import importlib.util

import pytest

import spectrochempy as scp

pytestmark = [
    pytest.mark.plugin,
    pytest.mark.skipif(
        importlib.util.find_spec("spectrochempy_nmr") is not None,
        reason="requires a core-only environment without spectrochempy-nmr",
    ),
]


def test_core_imports_without_the_plugin():
    """The core package remains usable without the optional plugin."""
    assert scp.NDDataset is not None
    assert importlib.util.find_spec("spectrochempy_nmr") is None


def test_nmr_namespace_guides_installation():
    """Namespace access explains how to install the missing plugin."""
    from spectrochempy.plugins.deps import MissingPluginNamespaceError  # noqa: PLC0415

    with pytest.raises(MissingPluginNamespaceError) as excinfo:
        scp.nmr.read("missing")

    message = str(excinfo.value)
    assert "spectrochempy-nmr" in message
    assert "pip install spectrochempy[nmr]" in message


def test_root_alias_guides_installation():
    """The historical read_topspin alias gives the same actionable error."""
    from spectrochempy.plugins.deps import MissingPluginError  # noqa: PLC0415

    with pytest.raises(MissingPluginError) as excinfo:
        scp.read_topspin("missing")

    message = str(excinfo.value)
    assert "spectrochempy-nmr" in message
    assert "pip install spectrochempy[nmr]" in message


def test_plugin_inspection_reports_the_plugin_missing():
    """Plugin inspection does not advertise scp.nmr when it is unavailable."""
    from spectrochempy.plugins.inspection import inspect_plugins  # noqa: PLC0415

    result = inspect_plugins()
    nmr = next(plugin for plugin in result.official if plugin.name == "nmr")

    assert nmr.status == "missing"
    assert "scp.nmr" not in result.namespaces
