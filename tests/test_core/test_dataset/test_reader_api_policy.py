# ruff: noqa: S101  # assert allowed in tests

"""Reader API exposure policy tests."""

import pytest

import spectrochempy as scp
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.registry import registry


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
        "read_carroucell",
        "load_iris",
        "download_nist_ir",
    ):
        assert hasattr(scp, name)
        assert not hasattr(dataset, name)


def test_missing_topspin_top_level_alias_is_actionable(monkeypatch):
    """Without the NMR plugin reader, scp.read_topspin points to a clear stub."""
    scp.plugin_manager.discover()
    monkeypatch.delitem(scp.__dict__, "read_topspin", raising=False)
    monkeypatch.setattr(
        registry.io,
        "_readers",
        {
            name: info
            for name, info in registry.io.available_readers.items()
            if name != "topspin"
        },
    )

    with pytest.raises(MissingPluginError, match="spectrochempy-nmr") as excinfo:
        scp.read_topspin("missing")

    assert "pip install spectrochempy[nmr]" in str(excinfo.value)
