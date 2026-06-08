# ruff: noqa: S101  # assert allowed in tests

"""Reader API exposure policy tests."""

import importlib.metadata
import sys

import pytest

import spectrochempy as scp
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry


def test_core_readers_are_package_level_not_dataset_methods():
    """Core readers create datasets, so they are package-level APIs only."""
    dataset = scp.NDDataset([1, 2, 3])

    for name in (
        "read",
        "read_dir",
        "read_csv",
        "read_omnic",
        "read_spa",
        "read_spg",
        "read_srs",
        "read_matlab",
        "read_mat",
        "read_jcamp",
        "read_labspec",
        "read_opus",
        "read_quadera",
        "read_soc",
        "read_ddr",
        "read_sdr",
        "read_hdr",
        "read_spc",
        "read_wire",
        "read_wdf",
        "read_zip",
        "load_iris",
        "download_nist_ir",
    ):
        assert hasattr(scp, name)
        assert not hasattr(dataset, name)


def test_missing_topspin_top_level_alias_is_actionable(monkeypatch):
    """Without the NMR plugin reader, scp.read_topspin points to a clear stub."""
    isolated_registry = PluginRegistry()
    isolated_manager = PluginManager(registry=isolated_registry)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None: [])
    monkeypatch.setattr(scp, "plugin_manager", isolated_manager)
    monkeypatch.setattr(scp, "registry", isolated_registry)
    monkeypatch.delitem(scp.__dict__, "read_topspin", raising=False)
    for module_name in list(sys.modules):
        if module_name == "spectrochempy_nmr" or module_name.startswith(
            "spectrochempy_nmr."
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    assert not any(name.startswith("spectrochempy_nmr") for name in sys.modules)
    read_topspin = scp.read_topspin
    assert callable(read_topspin)
    assert not any(name.startswith("spectrochempy_nmr") for name in sys.modules)

    with pytest.raises(MissingPluginError, match="spectrochempy-nmr") as excinfo:
        read_topspin("missing")

    assert "pip install spectrochempy[nmr]" in str(excinfo.value)
    assert not any(name.startswith("spectrochempy_nmr") for name in sys.modules)
