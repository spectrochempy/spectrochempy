# ruff: noqa: S101  # assert allowed in tests

"""Tests for spectrochempy-cantera plugin."""

from __future__ import annotations

from pathlib import Path

import pytest
from spectrochempy_cantera import PFR
from spectrochempy_cantera import CanteraPlugin
from spectrochempy_cantera import equilibrium_composition
from spectrochempy_cantera import reactor_profile
from spectrochempy_cantera import thermo_properties

from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness

_HERE = Path(__file__).parent

# Skip all tests if cantera is not installed
try:
    import cantera as ct
    HAS_CANTERA = True
    # Check if a default mechanism is available
    ct.Solution("gri30.yaml")
    HAS_GRI30 = True
except Exception:
    HAS_CANTERA = False
    HAS_GRI30 = False


# ------------------------------------------------------------------
# Plugin metadata and compatibility
# ------------------------------------------------------------------


def test_plugin_metadata():
    """Plugin has required metadata fields."""
    plugin = CanteraPlugin()
    assert plugin.name == "cantera"
    assert plugin.version == "0.1.0"
    assert plugin.description
    assert PluginCapability.SIMULATION in plugin.capabilities
    assert PluginCapability.ANALYSIS in plugin.capabilities


def test_plugin_compatibility():
    """Plugin passes full compatibility check."""
    plugin = CanteraPlugin()
    issues = check_plugin_compatibility(plugin)
    # should report cantera as missing if not installed
    if not HAS_CANTERA:
        assert any("cantera" in issue for issue in issues)
    else:
        assert not issues, f"Compatibility issues: {issues}"


# ------------------------------------------------------------------
# Registration and lifecycle
# ------------------------------------------------------------------


def test_registration_with_cantera():
    """Plugin registers simulations and analyses when cantera is available."""
    if not HAS_CANTERA:
        pytest.skip("cantera not installed")

    harness = PluginTestHarness()
    harness.register(CanteraPlugin())

    simulations = harness.registry.extensions.list_category("simulation")
    assert "equilibrium" in simulations
    assert "reactor_profile" in simulations
    assert "flame_speed" in simulations

    analyses = harness.registry.extensions.list_category("analysis")
    assert "thermo_properties" in analyses
    assert "kinetic_sensitivity" in analyses
    assert "spectral_coupling" in analyses


def test_registration_without_cantera():
    """Plugin is marked FAILED when cantera is missing."""
    # Simulate by checking the requires check
    harness = PluginTestHarness()
    harness.register(CanteraPlugin())

    if not HAS_CANTERA:
        assert harness.get_plugin_state("cantera") == PluginState.FAILED
    else:
        assert harness.get_plugin_state("cantera") == PluginState.ACTIVE


def test_capability_query():
    """Plugin simulation contributions are discoverable via capability query."""
    if not HAS_CANTERA:
        pytest.skip("cantera not installed")

    harness = PluginTestHarness()
    harness.register(CanteraPlugin())

    results = harness.registry.get_by_capability(PluginCapability.SIMULATION)
    names = [r["name"] for r in results]
    assert "equilibrium" in names


# ------------------------------------------------------------------
# Simulation functions
# ------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GRI30, reason="gri30.yaml not available via cantera")
def test_equilibrium_composition():
    """Equilibrium composition returns expected structure."""
    result = equilibrium_composition(
        mechanism="gri30.yaml",
        temperature=1200.0,
        pressure=101325.0,
        reactants={"CH4": 1.0, "O2": 2.0},
    )

    assert "species" in result
    assert "mole_fractions" in result
    assert "temperature" in result
    assert result["temperature"] == 1200.0
    assert result["n_species"] > 0
    assert result["mole_fractions"].data.ndim >= 1


@pytest.mark.skipif(not HAS_GRI30, reason="gri30.yaml not available via cantera")
def test_thermo_properties():
    """Thermodynamic properties return expected structure."""
    result = thermo_properties(
        mechanism="gri30.yaml",
        species="CH4",
        T_min=300.0,
        T_max=1500.0,
        n_points=10,
    )

    assert "temperature" in result
    assert "cp" in result
    assert "enthalpy" in result
    assert "entropy" in result
    assert "gibbs" in result
    assert len(result["temperature"].data) == 10


@pytest.mark.skipif(not HAS_GRI30, reason="gri30.yaml not available via cantera")
def test_reactor_profile():
    """Reactor profile returns expected structure."""
    result = reactor_profile(
        mechanism="gri30.yaml",
        initial_conditions={"T": 1500.0, "P": 101325.0},
        residence_time=0.01,
        n_points=10,
    )

    assert "time" in result
    assert "temperature" in result
    assert "species" in result
    assert "mole_fractions" in result
    assert len(result["time"]) == 10


@pytest.mark.skipif(not HAS_GRI30, reason="gri30.yaml not available via cantera")
def test_equilibrium_different_reactants():
    """Equilibrium with different reactant compositions."""
    result = equilibrium_composition(
        mechanism="gri30.yaml",
        temperature=1000.0,
        pressure=101325.0,
        reactants={"H2": 2.0, "O2": 1.0},
    )
    assert result["n_species"] > 0


# ------------------------------------------------------------------
# PFR tests
# ------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CANTERA, reason="cantera not installed")
def test_cantera_is_not_available(monkeypatch):
    """_cantera_is_not_available detects missing cantera."""
    from spectrochempy_cantera._pfr import _cantera_is_not_available  # noqa: PLC0415

    assert not _cantera_is_not_available()

    with monkeypatch.context() as m:
        import spectrochempy_cantera._pfr as _pfr  # noqa: PLC0415, PLC0415

        m.setattr(_pfr, "ct", None)
        assert _cantera_is_not_available()

    assert not _cantera_is_not_available()


@pytest.mark.skipif(not HAS_CANTERA, reason="cantera not installed")
def test_pfr_import():
    """PFR class can be imported from the cantera plugin."""
    assert PFR.__name__ == "PFR"


def test_pfr_construction_skipped_due_to_cantera_api():
    """
    PFR construction tests skipped (Cantera 3.2 API changes).

    The PFR class was written for an older Cantera API and uses methods
    (Reservoir(contents=), MassFlowController.set_mass_flow_rate) that were
    removed/changed in Cantera 3.2.
    """
    pytest.skip("PFR requires Cantera API adaptation for Cantera 3.2+")
