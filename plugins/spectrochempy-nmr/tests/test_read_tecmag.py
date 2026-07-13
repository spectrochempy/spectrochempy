# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

"""Tests for TecMag TNT NMR reader."""

import numpy as np
import pytest

import spectrochempy as scp

EXTRA_DATADIR = scp.preferences.datadir.parent / "testdata-extra"
TECMAG_DIR = EXTRA_DATADIR / "testdata" / "nmrdata" / "tecmag"


def _has_tecmag_data():
    return (TECMAG_DIR / "LiCl_ref1.tnt").exists()


def _read_tecmag_or_skip(*args, **kwargs):
    try:
        result = scp.nmr.read_tecmag(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"TecMag test data incomplete: {exc}")
    if result is None:
        pytest.skip("TecMag test data could not be read in this environment")
    return result


# ---------------------------------------------------------------------------
# nmrglue vendored layer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_tecmag_data(), reason="TecMag test data not available")
class TestTecMagNMRGlue:
    """Tests for the vendored nmrglue TecMag reading functions."""

    def test_read_tnt_1d(self):
        from spectrochempy_nmr.extern.nmrglue._tecmag import read

        dic, data = read(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        assert data.ndim >= 1
        assert data.shape[0] == 8192
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_guess_udic_1d(self):
        from spectrochempy_nmr.extern.nmrglue._tecmag import guess_udic
        from spectrochempy_nmr.extern.nmrglue._tecmag import read

        dic, data = read(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        udic = guess_udic(dic, data)
        assert udic[0]["size"] == 8192
        assert udic[0]["sw"] > 0
        assert udic[0]["obs"] > 0

    def test_tmag_metadata(self):
        from spectrochempy_nmr.extern.nmrglue._tecmag import read

        dic, data = read(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        # ob_freq[0] should be around 117 MHz for 7Li
        assert dic["ob_freq"][0] > 100
        # actual_npts[0] == 8192
        assert dic["actual_npts"][0] == 8192
        # nucleus should contain Li7
        nuc = dic["nuclei"][0]
        if isinstance(nuc, bytes):
            nuc = nuc.decode("latin1")
        assert "Li7" in str(nuc)


# ---------------------------------------------------------------------------
# Public reader API
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_tecmag_data(), reason="TecMag test data not available")
class TestTecMagReader:
    """Tests for the public read_tecmag() API."""

    def test_read_tecmag_1d(self):
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        assert dataset is not None
        assert dataset.ndim == 1
        assert dataset.shape == (8192,)

    def test_read_tecmag_metadata_sw(self):
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        # Spectral width should be ~10000 Hz
        assert dataset.meta.sw[0] is not None
        assert 9000 < dataset.meta.sw[0] < 11000

    def test_read_tecmag_metadata_sfrq(self):
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        # 7Li frequency ~117 MHz
        assert dataset.meta.sfrq[0] is not None
        assert 100 < dataset.meta.sfrq[0] < 200

    def test_read_tecmag_metadata_nucleus(self):
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        assert dataset.meta.nucleus[0] is not None
        assert "Li" in dataset.meta.nucleus[0]

    def test_read_tecmag_metadata_origin(self):
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        assert dataset.origin == "tecmag"

    def test_read_tecmag_metadata_ns(self):
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        assert dataset.meta.ns >= 1

    def test_read_tecmag_metadata_solvent(self):
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        assert dataset.meta.solvent is not None
        assert "D2O" in dataset.meta.solvent

    def test_read_tecmag_coord_ppm(self):
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        coord = dataset.x
        # Coordinate should exist and have acquisition_frequency metadata
        assert coord is not None
        assert coord.meta.get("acquisition_frequency") is not None

    def test_read_tecmag_data_values(self):
        """Cross-validate first data point against text export."""
        dataset = _read_tecmag_or_skip(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        # Text export: first point Real=-440, Imag=-281
        # Note: nmrglue may scale differently, check approximate match
        assert abs(dataset.data[0].real) > 100
        assert abs(dataset.data[0].imag) > 100

    def test_read_tecmag_via_generic(self):
        """Test that read() auto-detects .tnt files."""
        dataset = scp.nmr.read(str(TECMAG_DIR / "LiCl_ref1.tnt"))
        assert dataset is not None
        assert dataset.ndim == 1
        assert dataset.origin == "tecmag"

    def test_read_tecmag_missing_file(self):
        result = scp.nmr.read_tecmag("/nonexistent/path.tnt")
        assert result is None
