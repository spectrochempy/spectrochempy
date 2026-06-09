# ======================================================================================
# Copyright (c) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, PLC0415

"""Core compatibility tests for CP after migration to the tensor plugin."""

from __future__ import annotations

import importlib
import importlib.metadata as im
import importlib.util
import sys
import warnings

import pytest

import spectrochempy as scp
from spectrochempy.plugins.deps import MissingPluginNamespaceError
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry

_TENSOR_PLUGIN_INSTALLED = importlib.util.find_spec("spectrochempy_tensor") is not None


def _simulate_no_tensor_plugin(monkeypatch):
    registry = PluginRegistry()
    manager = PluginManager(registry=registry)
    monkeypatch.setattr(im, "entry_points", lambda group=None: [])
    monkeypatch.setattr(scp, "plugin_manager", manager)
    monkeypatch.setattr(scp, "registry", registry)
    monkeypatch.delitem(scp.__dict__, "CP", raising=False)
    sys.modules.pop("spectrochempy.tensor", None)


def test_core_lazy_imports_do_not_expose_cp_as_core_symbol():
    """CP is no longer a core lazy import."""
    assert "CP" not in scp._LAZY_IMPORTS


def test_missing_tensor_namespace_has_install_hint(monkeypatch):
    """Missing tensor plugin namespace access is actionable."""
    _simulate_no_tensor_plugin(monkeypatch)

    with pytest.raises(MissingPluginNamespaceError) as excinfo:
        _ = scp.tensor.CP

    message = str(excinfo.value)
    assert "spectrochempy-tensor" in message
    assert "spectrochempy[tensor]" in message


def test_missing_cp_root_symbol_has_install_hint(monkeypatch):
    """Missing scp.CP points users to the tensor plugin namespace."""
    _simulate_no_tensor_plugin(monkeypatch)

    with pytest.raises(AttributeError) as excinfo:
        _ = scp.CP

    message = str(excinfo.value)
    assert "module 'spectrochempy' has no attribute 'CP'" in message
    assert "scp.tensor.CP" in message
    assert "spectrochempy-tensor" in message


@pytest.mark.skipif(not _TENSOR_PLUGIN_INSTALLED, reason="tensor plugin not installed")
def test_legacy_cp_module_import_warns():
    """Legacy decomposition import path delegates to the tensor plugin."""
    module = importlib.import_module("spectrochempy.analysis.decomposition.cp")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        CP = module.CP

    assert CP.__name__ == "CP"

    legacy_warnings = [
        warning
        for warning in captured
        if "spectrochempy.analysis.decomposition.cp" in str(warning.message)
    ]

    assert len(legacy_warnings) == 1
    assert "scp.tensor.CP" in str(legacy_warnings[0].message)
