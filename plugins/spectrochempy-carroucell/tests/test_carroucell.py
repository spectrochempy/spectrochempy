# ruff: noqa: S101  # assert allowed in tests

"""Tests for spectrochempy-carroucell plugin."""

from importlib.metadata import version

import pytest
from spectrochempy_carroucell import CarroucellPlugin

import spectrochempy as scp
from spectrochempy import NDDataset
from spectrochempy import preferences as prefs
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness

DATADIR = prefs.datadir
CARROUCELL_FOLDER = DATADIR / "irdata/carroucell_samp"


def _has_required_carroucell_data():
    if not CARROUCELL_FOLDER.exists():
        return False

    available = {}
    for filename in CARROUCELL_FOLDER.glob("*.SPA"):
        if "BCKG" in filename.stem:
            continue
        try:
            prefix, spectrum = filename.stem.rsplit("_", maxsplit=1)
            available.setdefault(prefix, set()).add(int(spectrum))
        except ValueError:
            continue

    complete_samples = [
        prefix for prefix, spectra in available.items() if {1, 2, 3} <= spectra
    ]
    return len(complete_samples) >= 11

# ------------------------------------------------------------------
# Plugin lifecycle tests
# ------------------------------------------------------------------


def test_import():
    import spectrochempy_carroucell  # noqa: F401


def test_plugin_metadata():
    """Plugin declares CarrouCELL reader capability."""
    plugin = CarroucellPlugin()
    assert plugin.name == "carroucell"
    assert plugin.version == version("spectrochempy-carroucell")
    assert plugin.description
    assert PluginCapability.READER in plugin.capabilities


def test_plugin_compatibility():
    """Plugin passes full compatibility check."""
    plugin = CarroucellPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues, f"Compatibility issues: {issues}"


def test_registration():
    """Plugin registers the carroucell reader via PluginTestHarness."""
    harness = PluginTestHarness()
    harness.register(CarroucellPlugin())

    reader = harness.get_reader("carroucell")
    assert reader is not None
    assert reader["plugin"] == "carroucell"
    assert reader["description"]
    assert reader["plugin"] == "carroucell"
    assert reader["namespace"] == "carroucell"


def test_lifecycle_state():
    """Plugin transitions to ACTIVE after registration."""
    harness = PluginTestHarness()
    harness.register(CarroucellPlugin())
    assert harness.get_plugin_state("carroucell") == PluginState.ACTIVE


def test_registration_is_idempotent():
    """Registering the same CarrouCELL plugin twice does not duplicate contributions."""
    harness = PluginTestHarness()
    plugin = CarroucellPlugin()

    harness.register(plugin)
    first = harness.get_reader("carroucell")
    harness.register(plugin)

    assert harness.get_plugin_state("carroucell") == PluginState.ACTIVE
    assert list(harness.available_readers).count("carroucell") == 1
    assert harness.get_reader("carroucell") is not None
    assert harness.get_reader("carroucell")["func"] is first["func"]


def test_isolated_harness():
    """Each PluginTestHarness is independent for carroucell."""
    h1 = PluginTestHarness()
    h2 = PluginTestHarness()

    h1.register(CarroucellPlugin())
    assert h1.has_plugin("carroucell")
    assert h2.registry.metadata.get_plugin("carroucell") is None


def test_namespace_exposes_reader():
    """scp.carroucell.read_carroucell is exposed when plugin is registered."""
    assert hasattr(scp, "carroucell")
    assert callable(scp.carroucell.read_carroucell)


# ------------------------------------------------------------------
# Error behaviour when data is unavailable
# ------------------------------------------------------------------


def test_reader_stub_raises_on_nonexistent_path():
    """scp.carroucell.read_carroucell raises FileNotFoundError for non-existent paths."""
    reader = scp.carroucell.read_carroucell
    assert callable(reader)

    with pytest.raises(FileNotFoundError):
        reader("/nonexistent/path")


# ------------------------------------------------------------------
# Data-dependent tests
# ------------------------------------------------------------------


@pytest.mark.data
@pytest.mark.skipif(
    not _has_required_carroucell_data(),
    reason="Complete Carroucell experimental data not available for testing",
)
def test_read_carroucell():
    nd = scp.carroucell.read_carroucell("irdata/carroucell_samp", spectra=(1, 2))
    assert len(nd) == 11
    assert nd[3].shape == (2, 11098)

    nd = scp.carroucell.read_carroucell("irdata/carroucell_samp", spectra=(1, 1))
    assert isinstance(nd, NDDataset)

    nd = scp.carroucell.read_carroucell("irdata/carroucell_samp", spectra=(1, 3))
    assert nd[3].shape == (3, 11098)

    nd = scp.carroucell.read_carroucell(
        "irdata/carroucell_samp", spectra=(2, 3), discardbg=False
    )
    assert nd[3].shape == (2, 11098)
