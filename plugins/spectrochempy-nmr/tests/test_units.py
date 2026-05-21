# ruff: noqa: S101
"""Tests for NMR unit-context contributions."""

import numpy as np
import pytest
from spectrochempy_nmr import NMRPlugin
from spectrochempy_nmr.units import set_nmr_context

import spectrochempy as scp
import spectrochempy.plugins.manager as manager_module
from spectrochempy.core.units import set_nmr_context as core_set_nmr_context
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.manager import ENTRY_POINT_GROUP
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry


def _isolated_plugin_manager(monkeypatch, with_nmr: bool) -> PluginManager:
    registry = PluginRegistry()
    plugin_manager = PluginManager(registry=registry)
    monkeypatch.setattr(
        manager_module.importlib.metadata,
        "entry_points",
        lambda group=None: [] if group == ENTRY_POINT_GROUP else [],
    )
    if with_nmr:
        plugin_manager.register(NMRPlugin())
    monkeypatch.setattr(manager_module, "plugin_manager", plugin_manager)
    return plugin_manager


def test_set_nmr_context_converts_frequency_to_ppm():
    set_nmr_context(larmor=104.3 * scp.ur.MHz)
    fhz = 10000 * scp.ur.Hz

    with scp.ur.context("nmr"):
        fppm = fhz.to("ppm")
        roundtrip = fppm.to("Hz")

    assert f"{fppm:~.3f}" == "95.877 ppm"
    assert f"{roundtrip:~.3f}" == "10000.000 Hz"


def test_plugin_declares_nmr_unit_context(monkeypatch):
    plugin_manager = _isolated_plugin_manager(monkeypatch, with_nmr=True)
    setup = plugin_manager.registry.get_unit_context("nmr")

    assert setup is not None
    setup(100 * scp.ur.MHz)
    with scp.ur.context("nmr"):
        assert f"{(10000 * scp.ur.Hz).to('ppm'):~.3f}" == "100.000 ppm"


def test_public_nmr_namespace_exposes_set_nmr_context(monkeypatch):
    _isolated_plugin_manager(monkeypatch, with_nmr=True)

    scp.nmr.set_nmr_context(100 * scp.ur.MHz)
    with scp.ur.context("nmr"):
        assert f"{(10000 * scp.ur.Hz).to('ppm'):~.3f}" == "100.000 ppm"


def test_core_set_nmr_context_shim_warns():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        core_set_nmr_context(100 * scp.ur.MHz)

    with scp.ur.context("nmr"):
        assert f"{(10000 * scp.ur.Hz).to('ppm'):~.3f}" == "100.000 ppm"


def test_coord_larmor_conversion_uses_plugin_unit_context(monkeypatch):
    _isolated_plugin_manager(monkeypatch, with_nmr=True)

    coord = scp.Coord([0.0, 10000.0], units="Hz", larmor=100 * scp.ur.MHz)
    ppm = coord.to("ppm")
    hz = ppm.to("Hz")

    np.testing.assert_allclose(ppm.data, [0.0, 100.0])
    assert ppm.units == scp.ur.ppm
    np.testing.assert_allclose(hz.data, [0.0, 10000.0])
    assert hz.units == scp.ur.Hz


def test_coord_larmor_conversion_requires_nmr_plugin(monkeypatch):
    _isolated_plugin_manager(monkeypatch, with_nmr=False)

    coord = scp.Coord([10000.0], units="Hz", larmor=100 * scp.ur.MHz)
    with pytest.raises(MissingPluginError) as excinfo:
        coord.to("ppm")

    message = str(excinfo.value)
    assert "spectrochempy-nmr" in message
    assert "NMR ppm/frequency unit conversion" in message
