# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the lightweight plugin inspection UX."""

from __future__ import annotations

import importlib.metadata

import pytest


class FakeEntryPoint:
    """Minimal entry-point object that must not be loaded by inspection."""

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value

    def load(self):  # pragma: no cover - should never be called
        raise AssertionError("plugin inspection must not load entry points")


def test_scp_plugins_is_callable_and_keeps_submodule_access():
    import spectrochempy as scp

    assert callable(scp.plugins)
    assert hasattr(scp.plugins, "manager")


def test_plugin_inspection_reports_installed_and_missing(monkeypatch):
    from spectrochempy.plugins import inspection

    monkeypatch.setattr(
        inspection.metadata,
        "entry_points",
        lambda group=None: [FakeEntryPoint("iris", "spectrochempy_iris:IrisPlugin")],
    )

    def fake_version(package):
        if package == "spectrochempy-iris":
            return "1.2.3"
        raise importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(inspection.metadata, "version", fake_version)

    result = inspection.inspect_plugins()
    rendered = str(result)

    import re

    assert re.search(r"IRIS plugin\s+installed", rendered)
    assert re.search(r"NMR plugin\s+missing", rendered)
    assert re.search(r"Cantera plugin\s+missing", rendered)
    assert re.search(r"Hypercomplex plugin\s+missing", rendered)
    assert re.search(r"Carroucell plugin\s+missing", rendered)
    assert "scp.iris" in rendered
    assert "scp.nmr" not in rendered
    assert result.official[0].version == "1.2.3"


def test_plugin_inspection_verbose_lists_other_entry_points(monkeypatch):
    from spectrochempy.plugins import inspection

    monkeypatch.setattr(
        inspection.metadata,
        "entry_points",
        lambda group=None: [FakeEntryPoint("custom", "custom_plugin:Plugin")],
    )
    monkeypatch.setattr(
        inspection.metadata,
        "version",
        lambda package: (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError(package)
        ),
    )

    rendered = str(inspection.inspect_plugins(verbose=True))

    assert "Other discovered plugins" in rendered
    assert "custom  discovered  (custom_plugin:Plugin)" in rendered


def test_plugin_inspection_does_not_trigger_manager_discovery(monkeypatch):
    import spectrochempy as scp

    monkeypatch.setattr(
        scp.plugin_manager,
        "discover",
        lambda: pytest.fail("scp.plugins() must not call plugin_manager.discover()"),
    )

    result = scp.plugins()

    assert "Installed official plugins" in str(result)
