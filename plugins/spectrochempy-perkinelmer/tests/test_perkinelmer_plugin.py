# ruff: noqa: S101

"""Tests for the spectrochempy-perkinelmer plugin."""

import warnings

import numpy as np
import pytest
from spectrochempy_perkinelmer import PerkinElmerPlugin
from spectrochempy_perkinelmer.read_perkinelmer import _SpFile
from spectrochempy_perkinelmer.read_perkinelmer import read_perkinelmer

import spectrochempy as scp
import spectrochempy.plugins.manager as manager_module
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry
from spectrochempy.testing.plugins import PluginTestHarness

# ------------------------------------------------------------------------------
# Datadir path (for CI integration with spectrochempy_data)
# ------------------------------------------------------------------------------

DATADIR_SP = scp.preferences.datadir / "irdata" / "perkinelmer" / "spectra.sp"

# ------------------------------------------------------------------------------
# Parser unit tests (require spectrochempy_data)
# ------------------------------------------------------------------------------


@pytest.fixture
def sp_content() -> bytes:
    if not DATADIR_SP.exists():
        pytest.skip("PerkinElmer testdata not available (set SCP_TEST_DATA_DOWNLOAD=1)")
    return DATADIR_SP.read_bytes()


def test_sp_parser_signature(sp_content: bytes) -> None:
    spf = _SpFile(sp_content)
    assert spf.signature == b"PEPE"
    assert "DataSet" in spf.description


def test_sp_parser_wavelength(sp_content: bytes) -> None:
    spf = _SpFile(sp_content)
    assert spf.meta["min_wavelength"] == 4000.0
    assert spf.meta["max_wavelength"] == 700.0
    assert spf.meta["n_points"] == 3301
    assert spf.wavelength.size == 3301
    assert spf.wavelength[0] == 4000.0
    assert spf.wavelength[-1] == 700.0


def test_sp_parser_spectrum(sp_content: bytes) -> None:
    spf = _SpFile(sp_content)
    assert spf.spectrum.size == 3301
    np.testing.assert_allclose(
        spf.spectrum[:5], [0.037239, 0.037186, 0.037133, 0.037079, 0.037026], rtol=1e-4
    )


def test_sp_parser_metadata(sp_content: bytes) -> None:
    spf = _SpFile(sp_content)
    assert spf.meta["instrument_model"] == "Spectrum One"
    assert spf.meta["date"] == "Thu Mar 09 09:17:56 2006"
    assert spf.meta["detector"] == "MCT"
    assert spf.meta["source"] == "MIR"
    assert spf.meta["accumulations"] == 32


def test_sp_parser_invalid_signature() -> None:
    with pytest.raises(ValueError, match="PEPE"):
        _SpFile(b"NOT_PEPE" + b"\x00" * 100)


# ------------------------------------------------------------------------------
# Plugin registration tests
# ------------------------------------------------------------------------------


def test_plugin_metadata() -> None:
    plugin = PerkinElmerPlugin()
    assert plugin.name == "perkinelmer"
    assert plugin.version == "0.1.2"
    assert plugin.description
    assert PluginCapability.READER in plugin.capabilities


def test_plugin_compatibility() -> None:
    plugin = PerkinElmerPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues


def test_registration() -> None:
    harness = PluginTestHarness()
    harness.register(PerkinElmerPlugin())

    reader = harness.get_reader("perkinelmer")
    assert reader is not None
    assert reader["description"]
    assert reader["plugin"] == "perkinelmer"
    assert reader["namespace"] == "perkinelmer"

    # Alias read_sp
    reader_sp = harness.get_reader("sp")
    assert reader_sp is not None
    assert reader_sp["func"] is reader["func"]


def test_registration_is_idempotent() -> None:
    harness = PluginTestHarness()
    plugin = PerkinElmerPlugin()

    harness.register(plugin)
    first_reader = harness.get_reader("perkinelmer")
    harness.register(plugin)
    second_reader = harness.get_reader("perkinelmer")

    assert harness.get_plugin_state("perkinelmer") == PluginState.ACTIVE
    assert list(harness.available_readers).count("perkinelmer") == 1
    assert second_reader is not None
    assert first_reader is not None
    assert second_reader["func"] is first_reader["func"]


def test_discovery_is_idempotent(monkeypatch) -> None:
    class PEEntryPoint:
        name = "perkinelmer"
        value = "spectrochempy_perkinelmer:PerkinElmerPlugin"

        @staticmethod
        def load():
            return PerkinElmerPlugin

    def mock_entry_points(group=None):
        if group == manager_module.ENTRY_POINT_GROUP:
            return [PEEntryPoint()]
        return []

    monkeypatch.setattr(
        manager_module.importlib.metadata,
        "entry_points",
        mock_entry_points,
    )

    pm = PluginManager()
    pm.discover()
    pm.discover()

    assert pm.get_plugin_state("perkinelmer") == PluginState.ACTIVE
    assert list(pm.registry.available_readers).count("perkinelmer") == 1


def test_lifecycle_state() -> None:
    harness = PluginTestHarness()
    harness.register(PerkinElmerPlugin())
    assert harness.get_plugin_state("perkinelmer") == PluginState.ACTIVE


# ------------------------------------------------------------------------------
# Integration tests (require SpectroChemPy core)
# ------------------------------------------------------------------------------


@pytest.mark.skipif(not DATADIR_SP.exists(), reason="test .sp file not available")
def test_read_perkinelmer_returns_nddataset() -> None:
    ds = read_perkinelmer(DATADIR_SP)
    assert hasattr(ds, "_implements")
    assert ds._implements() == "NDDataset"


@pytest.mark.skipif(not DATADIR_SP.exists(), reason="test .sp file not available")
def test_read_perkinelmer_dimensions() -> None:
    ds = read_perkinelmer(DATADIR_SP)
    assert ds.ndim == 2
    assert ds.shape == (1, 3301)


@pytest.mark.skipif(not DATADIR_SP.exists(), reason="test .sp file not available")
def test_read_perkinelmer_coordinates() -> None:
    ds = read_perkinelmer(DATADIR_SP)
    x = ds.x
    assert x.size == 3301
    assert x.data[0] == 4000.0
    assert x.data[-1] == 700.0
    assert str(x.units) == "nm"


@pytest.mark.skipif(not DATADIR_SP.exists(), reason="test .sp file not available")
def test_read_perkinelmer_metadata() -> None:
    ds = read_perkinelmer(DATADIR_SP)
    assert ds.origin == "perkinelmer"
    assert ds.meta.instrument_model == "Spectrum One"
    assert ds.meta.date == "Thu Mar 09 09:17:56 2006"


@pytest.mark.skipif(not DATADIR_SP.exists(), reason="test .sp file not available")
def test_read_perkinelmer_invalid_file(tmp_path) -> None:
    bad = tmp_path / "bad.sp"
    bad.write_bytes(b"NOT_PEPE" + b"\x00" * 100)
    # Importer catches parsing errors and logs warnings rather than raising.
    ds = read_perkinelmer(bad)
    assert ds is None


def test_sp_parser_invalid_signature_direct() -> None:
    with pytest.raises(ValueError, match="PEPE"):
        _SpFile(b"NOT_PEPE" + b"\x00" * 100)


@pytest.mark.skipif(not DATADIR_SP.exists(), reason="test .sp file not available")
def test_read_sp_alias() -> None:
    ds = read_perkinelmer(DATADIR_SP)
    ds_alias = scp.read_sp(DATADIR_SP)
    assert ds_alias is not None
    assert ds_alias.shape == ds.shape


@pytest.mark.skipif(not DATADIR_SP.exists(), reason="test .sp file not available")
def test_namespace_read_alias() -> None:
    ds = scp.perkinelmer.read(DATADIR_SP)
    assert ds is not None
    assert ds.shape == (1, 3301)


@pytest.mark.skipif(not DATADIR_SP.exists(), reason="test .sp file not available")
def test_read_perkinelmer_content_kwarg() -> None:
    content = DATADIR_SP.read_bytes()
    ds = read_perkinelmer("dummy.sp", content=content)
    assert ds.shape == (1, 3301)


# ------------------------------------------------------------------------------
# Metadata enrichment tests
# ------------------------------------------------------------------------------


def test_read_perkinelmer_extra_meta_fields(monkeypatch) -> None:
    """New metadata fields are copied when present in the parser output."""
    from spectrochempy_perkinelmer.read_perkinelmer import _read_sp

    class FakeSpFile:
        spectrum = np.array([1.0, 2.0, 3.0])
        wavelength = np.array([4000.0, 3500.0, 3000.0])
        meta = {
            "analyst": "Test Analyst",
            "date": "Mon Jan 01 00:00:00 2024",
            "instrument_model": "Spectrum Two",
            "detector": "MCT",
            "source": "MIR",
            "accumulations": 16,
            "spectrum_type": "Spectrum",
            "instrument_serial_number": "12345",
            "instrument_software_version": "v1.0",
            "ir_accessory": "Universal ATR",
            "image_name": "Sample 001",
        }

    monkeypatch.setattr(
        "spectrochempy_perkinelmer.read_perkinelmer._SpFile",
        lambda content: FakeSpFile(),
    )

    ds = scp.NDDataset()
    result = _read_sp(ds, "dummy.sp", content=b"PEPE")

    assert result.meta.instrument_serial_number == "12345"
    assert result.meta.instrument_software_version == "v1.0"
    assert result.meta.ir_accessory == "Universal ATR"
    assert result.meta.image_name == "Sample 001"
    # Core fields should still be present
    assert result.meta.analyst == "Test Analyst"
    assert result.meta.instrument_model == "Spectrum Two"


def test_read_perkinelmer_image_name_fallback_description(monkeypatch) -> None:
    """image_name is used as dataset.description when description is empty."""
    from spectrochempy_perkinelmer.read_perkinelmer import _read_sp

    class FakeSpFile:
        spectrum = np.array([1.0, 2.0, 3.0])
        wavelength = np.array([4000.0, 3500.0, 3000.0])
        meta = {
            "analyst": "",
            "date": "",
            "instrument_model": "",
            "detector": "",
            "source": "",
            "accumulations": "",
            "spectrum_type": "",
            "image_name": "Sample 001 By Analyst Date Monday",
        }

    monkeypatch.setattr(
        "spectrochempy_perkinelmer.read_perkinelmer._SpFile",
        lambda content: FakeSpFile(),
    )

    ds = scp.NDDataset()
    result = _read_sp(ds, "dummy.sp", content=b"PEPE")
    assert result.description == "Sample 001 By Analyst Date Monday"


def test_read_perkinelmer_description_not_overwritten(monkeypatch) -> None:
    """image_name does NOT overwrite an existing description."""
    from spectrochempy_perkinelmer.read_perkinelmer import _read_sp

    class FakeSpFile:
        spectrum = np.array([1.0, 2.0, 3.0])
        wavelength = np.array([4000.0, 3500.0, 3000.0])
        meta = {
            "analyst": "",
            "date": "",
            "instrument_model": "",
            "detector": "",
            "source": "",
            "accumulations": "",
            "spectrum_type": "",
            "image_name": "Sample 001",
        }

    monkeypatch.setattr(
        "spectrochempy_perkinelmer.read_perkinelmer._SpFile",
        lambda content: FakeSpFile(),
    )

    ds = scp.NDDataset()
    ds.description = "Existing description"
    result = _read_sp(ds, "dummy.sp", content=b"PEPE")
    assert result.description == "Existing description"


def test_read_perkinelmer_empty_extra_fields_ignored(monkeypatch) -> None:
    """Empty or missing extra metadata values are not attached."""
    from spectrochempy_perkinelmer.read_perkinelmer import _read_sp

    class FakeSpFile:
        spectrum = np.array([1.0, 2.0, 3.0])
        wavelength = np.array([4000.0, 3500.0, 3000.0])
        meta = {
            "analyst": "Analyst",
            "date": "Mon Jan 01 00:00:00 2024",
            "instrument_model": "Spectrum Two",
            "detector": "MCT",
            "source": "MIR",
            "accumulations": 16,
            "spectrum_type": "Spectrum",
            "instrument_serial_number": "",
            "instrument_software_version": None,
            # ir_accessory and image_name deliberately absent
        }

    monkeypatch.setattr(
        "spectrochempy_perkinelmer.read_perkinelmer._SpFile",
        lambda content: FakeSpFile(),
    )

    ds = scp.NDDataset()
    result = _read_sp(ds, "dummy.sp", content=b"PEPE")

    assert getattr(result.meta, "instrument_serial_number", None) in (None, "")
    assert getattr(result.meta, "instrument_software_version", None) in (None, "")
    assert getattr(result.meta, "ir_accessory", None) in (None, "")
    assert getattr(result.meta, "image_name", None) in (None, "")
    # Core fields should still be present
    assert result.meta.analyst == "Analyst"


# ------------------------------------------------------------------------------
# Namespace exposure tests
# ------------------------------------------------------------------------------


def test_package_namespace_exposes_reader(monkeypatch) -> None:
    def _isolate_scp_plugins(monkeypatch, scp):
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        monkeypatch.setattr(
            manager_module.importlib.metadata,
            "entry_points",
            lambda group=None: [],
        )
        monkeypatch.setattr(scp, "plugin_manager", pm)
        monkeypatch.setattr(scp, "registry", registry)
        return pm, registry

    pm, registry = _isolate_scp_plugins(monkeypatch, scp)
    pm.register(PerkinElmerPlugin())

    namespaced_reader = scp.perkinelmer.read_perkinelmer
    top_level_reader = scp.read_perkinelmer

    assert callable(namespaced_reader)
    assert callable(top_level_reader)
    assert namespaced_reader.__name__ == top_level_reader.__name__ == "read_perkinelmer"
    assert registry.get_reader("perkinelmer")["namespace"] == "perkinelmer"


def test_top_level_stub_without_plugin(monkeypatch) -> None:
    """Without the plugin registered, scp.read_perkinelmer raises MissingPluginError."""
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    monkeypatch.setattr(
        manager_module.importlib.metadata,
        "entry_points",
        lambda group=None: [],
    )
    monkeypatch.setattr(scp, "plugin_manager", pm)
    monkeypatch.setattr(scp, "registry", registry)

    stub = scp.read_perkinelmer
    assert callable(stub)
    with pytest.raises(scp.plugins.deps.MissingPluginError) as excinfo:
        stub("missing")

    assert "spectrochempy-perkinelmer" in str(excinfo.value)


def test_read_perkinelmer_root_alias_works() -> None:
    """scp.read_perkinelmer remains a working alias with no DeprecationWarning."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        rt = scp.read_perkinelmer
        rt_ns = scp.perkinelmer.read_perkinelmer

    assert callable(rt)
    assert callable(rt_ns)
    deprecation_warnings = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings == [], deprecation_warnings


# ------------------------------------------------------------------------------
# Datadir integration test (uses spectrochempy_data in CI)
# ------------------------------------------------------------------------------


@pytest.mark.data
def test_read_perkinelmer_from_datadir() -> None:
    """
    Read the sample file through the standard datadir path.

    This test verifies that the file published in the spectrochempy_data
    repository is accessible and readable by the plugin.  It is skipped when
    the datadir is not available locally; in CI it runs when
    spectrochempy_data is installed or when SCP_TEST_DATA_DOWNLOAD=1.
    """
    if not DATADIR_SP.exists():
        pytest.skip("PerkinElmer testdata not available (set SCP_TEST_DATA_DOWNLOAD=1)")
    ds = read_perkinelmer(DATADIR_SP)
    assert ds is not None
    assert ds.shape == (1, 3301)
    assert ds.origin == "perkinelmer"
