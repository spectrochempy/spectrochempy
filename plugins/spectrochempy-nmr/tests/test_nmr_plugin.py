# ruff: noqa: S101  # assert allowed in tests

"""Tests for spectrochempy-nmr plugin registration and lifecycle."""

import warnings
from importlib.metadata import version

import pytest
from spectrochempy_nmr import NMRPlugin

import spectrochempy as scp
import spectrochempy.plugins.manager as manager_module
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.core.io_namespaces import _is_io_namespace
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.manager import ENTRY_POINT_GROUP
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry
from spectrochempy.testing.plugins import PluginTestHarness


def _require_reader_dependencies() -> None:
    pytest.importorskip("quaternion")


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


def test_plugin_metadata():
    """Plugin has required metadata fields."""
    plugin = NMRPlugin()
    assert plugin.name == "nmr"
    assert plugin.version == version("spectrochempy-nmr")
    assert plugin.description
    assert PluginCapability.READER in plugin.capabilities


def test_plugin_compatibility():
    """Plugin passes compatibility checks."""
    plugin = NMRPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues, f"Compatibility issues: {issues}"


def test_registration():
    """Plugin registers the topspin reader via declarative hook."""
    _require_reader_dependencies()
    harness = PluginTestHarness()
    harness.register(NMRPlugin())

    reader = harness.get_reader("topspin")
    assert reader is not None
    assert reader["description"]
    assert reader["plugin"] == "nmr"
    assert reader["namespace"] == "nmr"


def test_registration_is_idempotent():
    """Registering the same NMR plugin twice does not duplicate contributions."""
    _require_reader_dependencies()
    harness = PluginTestHarness()
    plugin = NMRPlugin()

    harness.register(plugin)
    first_reader = harness.get_reader("topspin")
    harness.register(plugin)
    second_reader = harness.get_reader("topspin")

    assert harness.get_plugin_state("nmr") == PluginState.ACTIVE
    assert list(harness.available_readers).count("topspin") == 1
    assert second_reader is not None
    assert first_reader is not None
    assert second_reader["func"] is first_reader["func"]


def test_discovery_is_idempotent(monkeypatch):
    """Repeated discovery keeps a single active NMR reader and namespace."""
    _require_reader_dependencies()

    class NMREntryPoint:
        name = "nmr"
        value = "spectrochempy_nmr:NMRPlugin"

        @staticmethod
        def load():
            return NMRPlugin

    def mock_entry_points(group=None):
        if group == ENTRY_POINT_GROUP:
            return [NMREntryPoint()]
        return []

    monkeypatch.setattr(
        manager_module.importlib.metadata,
        "entry_points",
        mock_entry_points,
    )

    pm = PluginManager()
    pm.discover()
    pm.discover()

    assert pm.get_plugin_state("nmr") == PluginState.ACTIVE
    assert list(pm.registry.available_readers).count("topspin") == 1
    assert pm.registry.get_reader("topspin")["namespace"] == "nmr"


def test_lifecycle_state():
    """Plugin transitions to ACTIVE after registration."""
    _require_reader_dependencies()
    harness = PluginTestHarness()
    harness.register(NMRPlugin())
    assert harness.get_plugin_state("nmr") == PluginState.ACTIVE


def test_isolated_harness():
    """Each PluginTestHarness is independent."""
    _require_reader_dependencies()
    h1 = PluginTestHarness()
    h2 = PluginTestHarness()

    h1.register(NMRPlugin())
    assert h1.has_plugin("nmr")
    assert h2.registry.metadata.get_plugin("nmr") is None


def test_package_namespace_exposes_topspin_reader(monkeypatch):
    """scp.nmr.read_topspin exposes the reader while preserving legacy alias."""
    _require_reader_dependencies()

    pm, registry = _isolate_scp_plugins(monkeypatch, scp)
    pm.register(NMRPlugin())

    namespaced_reader = scp.nmr.read_topspin
    top_level_reader = scp.read_topspin

    assert callable(namespaced_reader)
    assert callable(top_level_reader)
    assert namespaced_reader.__name__ == top_level_reader.__name__ == "read_topspin"
    assert namespaced_reader.__wrapped__ is top_level_reader.__wrapped__
    assert registry.get_reader("topspin")["namespace"] == "nmr"
    assert top_level_reader.__module__ == namespaced_reader.__module__
    assert top_level_reader.__module__.startswith("spectrochempy_nmr")


def test_package_namespace_exposes_short_read_alias(monkeypatch):
    """scp.nmr.read is a generic dispatcher for all NMR readers."""
    _require_reader_dependencies()

    pm, _registry = _isolate_scp_plugins(monkeypatch, scp)
    pm.register(NMRPlugin())

    generic_reader = scp.nmr.read
    topspin_reader = scp.nmr.read_topspin
    agilent_reader = scp.nmr.read_agilent

    assert callable(generic_reader)
    assert generic_reader.__name__ == "read"
    # The generic dispatcher delegates to the format-specific readers based on
    # the ``protocol`` keyword or directory contents, so we only verify that
    # the namespaced format readers remain directly accessible.
    assert callable(topspin_reader)
    assert callable(agilent_reader)
    assert topspin_reader.__name__ == "read_topspin"
    assert agilent_reader.__name__ == "read_agilent"


def test_top_level_stub_is_actionable_without_registered_nmr(monkeypatch):
    """scp.read_topspin is a callable MissingPluginError stub without NMR."""
    _isolate_scp_plugins(monkeypatch, scp)

    read_topspin = scp.read_topspin
    assert callable(read_topspin)
    with pytest.raises(MissingPluginError) as excinfo:
        read_topspin("missing")

    message = str(excinfo.value)
    assert "spectrochempy-nmr" in message
    assert "pip install spectrochempy[nmr]" in message


def test_read_topspin_root_alias_works():
    """scp.read_topspin remains a working alias with no DeprecationWarning."""
    _require_reader_dependencies()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        rt = scp.read_topspin
        rt_ns = scp.nmr.read_topspin

    assert callable(rt)
    assert callable(rt_ns)
    assert rt is rt_ns
    deprecation_warnings = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings == [], deprecation_warnings


def test_io_namespace_topspin_and_agilent(monkeypatch):
    """scp.topspin.read and scp.agilent.read expose the namespaced readers."""
    _require_reader_dependencies()

    pm, _registry = _isolate_scp_plugins(monkeypatch, scp)
    pm.register(NMRPlugin())

    assert _is_io_namespace("topspin")
    assert _is_io_namespace("agilent")

    assert callable(scp.topspin.read)
    assert callable(scp.agilent.read)
    assert scp.topspin.read is scp.nmr.read_topspin
    assert scp.agilent.read is scp.nmr.read_agilent


def test_nmr_reader_is_not_dataset_accessor_namespace(monkeypatch):
    """Readers are package-level APIs, not dataset accessor methods."""
    _require_reader_dependencies()

    pm, _registry = _isolate_scp_plugins(monkeypatch, scp)
    pm.register(NMRPlugin())

    dataset = scp.NDDataset([1, 2, 3])
    assert not hasattr(dataset, "nmr")
    assert not hasattr(dataset, "read_topspin")


# ---------------------------------------------------------------------------
# Cross-vendor metadata extraction
# ---------------------------------------------------------------------------


class _MockMeta:
    """Minimal metadata mock for testing extractors."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return 1


class TestExtractJeolMetadata:
    """Tests for extract_jeol_metadata."""

    def test_1d_dataset(self):
        from spectrochempy_nmr.nmr_metadata import extract_jeol_metadata

        meta = _MockMeta(
            td=("1024",),
            isfreq=(True,),
            encoding=("QF",),
            nucleus=("1H",),
            experiment="EXPERIMENT_1D",
            datatype="FID",
            iscomplex=(True,),
            sw_h=(5000.0,),
            sfrq=(400.0,),
        )
        result = extract_jeol_metadata(meta)
        assert result.ndim == 1
        assert result.domains == ("frequency",)
        assert result.encoding == ("QF",)
        assert result.nuclei == ("1H",)
        assert result.pulse_program == "EXPERIMENT_1D"
        assert result.source_kind == "processed_1d"

    def test_none_meta(self):
        from spectrochempy_nmr.nmr_metadata import extract_jeol_metadata

        result = extract_jeol_metadata(None)
        assert result.ndim == 0
        assert result.domains == ()

    def test_empty_meta(self):
        from spectrochempy_nmr.nmr_metadata import extract_jeol_metadata

        meta = _MockMeta()
        result = extract_jeol_metadata(meta)
        assert result.ndim == 0
        assert result.domains == ()


class TestExtractTecMagMetadata:
    """Tests for extract_tecmag_metadata."""

    def test_1d_dataset(self):
        from spectrochempy_nmr.nmr_metadata import extract_tecmag_metadata

        meta = _MockMeta(
            td=("2048",),
            isfreq=(False,),
            encoding=("QSIM",),
            nucleus=("13C",),
            experiment="CPMG",
            iscomplex=(True,),
            sw_h=(10000.0,),
            sfrq=(100.0,),
        )
        result = extract_tecmag_metadata(meta)
        assert result.ndim == 1
        assert result.domains == ("time",)
        assert result.encoding == ("QSIM",)
        assert result.nuclei == ("13C",)
        assert result.pulse_program == "CPMG"
        assert result.source_kind == "fid"

    def test_none_meta(self):
        from spectrochempy_nmr.nmr_metadata import extract_tecmag_metadata

        result = extract_tecmag_metadata(None)
        assert result.ndim == 0


class TestExtractSimpsonMetadata:
    """Tests for extract_simpson_metadata."""

    def test_2d_dataset(self):
        from spectrochempy_nmr.nmr_metadata import extract_simpson_metadata

        meta = _MockMeta(
            td=(256, 256),
            isfreq=(True, True),
            encoding=("States", "States"),
            nucleus=("1H", "13C"),
            iscomplex=(True, True),
            sw_h=(5000.0, 10000.0),
            sfrq=(400.0, 100.0),
        )
        result = extract_simpson_metadata(meta)
        assert result.ndim == 2
        assert result.domains == ("frequency", "frequency")
        assert result.pulse_program is None
        assert result.source_kind == "processed_2d"

    def test_none_meta(self):
        from spectrochempy_nmr.nmr_metadata import extract_simpson_metadata

        result = extract_simpson_metadata(None)
        assert result.ndim == 0


class TestExtractNmrMetadataDispatcher:
    """Tests for the vendor-neutral extract_nmr_metadata dispatcher."""

    def test_dispatches_to_jeol(self):
        from spectrochempy_nmr.nmr_metadata import extract_nmr_metadata

        meta = _MockMeta(
            origin="jeol",
            td=("1024",),
            isfreq=(True,),
            encoding=("QF",),
            nucleus=("1H",),
            experiment="EXPERIMENT_1D",
            datatype="FID",
            iscomplex=(True,),
            sw_h=(5000.0,),
            sfrq=(400.0,),
        )
        result = extract_nmr_metadata(meta)
        assert result.nuclei == ("1H",)
        assert result.pulse_program == "EXPERIMENT_1D"

    def test_dispatches_to_tecmag(self):
        from spectrochempy_nmr.nmr_metadata import extract_nmr_metadata

        meta = _MockMeta(
            origin="tecmag",
            td=("2048",),
            isfreq=(False,),
            encoding=("QSIM",),
            nucleus=("13C",),
            experiment="CPMG",
            iscomplex=(True,),
            sw_h=(10000.0,),
            sfrq=(100.0,),
        )
        result = extract_nmr_metadata(meta)
        assert result.nuclei == ("13C",)
        assert result.pulse_program == "CPMG"

    def test_dispatches_to_simpson(self):
        from spectrochempy_nmr.nmr_metadata import extract_nmr_metadata

        meta = _MockMeta(
            origin="simpson",
            td=(256, 256),
            isfreq=(True, True),
            encoding=("States", "States"),
            nucleus=("1H", "13C"),
            iscomplex=(True, True),
            sw_h=(5000.0, 10000.0),
            sfrq=(400.0, 100.0),
        )
        result = extract_nmr_metadata(meta)
        assert result.pulse_program is None
        assert result.source_kind == "processed_2d"

    def test_dispatches_to_topspin(self):
        from spectrochempy_nmr.nmr_metadata import extract_nmr_metadata

        meta = _MockMeta(
            origin="topspin",
            ndim=1,
            isfreq=(True,),
            encoding=(5,),
            nuc1=("1H",),
            pulprog="zg30",
            datatype="FID",
            iscomplex=(False,),
            sw_h=(5000.0,),
            sfo1=(400.0,),
        )
        result = extract_nmr_metadata(meta)
        assert result.pulse_program == "zg30"
        assert result.encoding == ("STATES-TPPI",)

    def test_falls_back_to_topspin_for_unknown_origin(self):
        from spectrochempy_nmr.nmr_metadata import extract_nmr_metadata

        meta = _MockMeta(
            origin="unknown_vendor",
            ndim=1,
            isfreq=(True,),
            encoding=(4,),
            nuc1=("1H",),
            pulprog="hsqc",
            iscomplex=(False,),
            sw_h=(5000.0,),
            sfo1=(400.0,),
        )
        result = extract_nmr_metadata(meta)
        assert result.nuclei == ("1H",)
        assert result.pulse_program == "hsqc"

    def test_none_meta(self):
        from spectrochempy_nmr.nmr_metadata import extract_nmr_metadata

        result = extract_nmr_metadata(None)
        assert result.ndim == 0
        assert result.domains == ()
