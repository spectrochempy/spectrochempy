# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

"""Tests for JEOL JDF NMR reader."""

import numpy as np
import pytest

import spectrochempy as scp

EXTRA_DATADIR = scp.preferences.datadir.parent / "testdata-extra"
JEOL_DIR = EXTRA_DATADIR / "testdata" / "nmrdata" / "jeol"


def _has_jeol_data():
    return (JEOL_DIR / "1H.jdf").exists()


def _read_jeol_or_skip(*args, **kwargs):
    try:
        result = scp.nmr.read_jeol(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"JEOL test data incomplete: {exc}")
    if result is None:
        pytest.skip("JEOL test data could not be read in this environment")
    return result


# ---------------------------------------------------------------------------
# nmrglue vendored layer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jeol_data(), reason="JEOL test data not available")
class TestJeolNMRGlue:
    """Tests for the vendored nmrglue JEOL reading functions."""

    def test_read_jeol_1d(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "1H.jdf"))
        assert data.ndim == 1
        assert data.shape == (32768,)
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_read_jeol_2d(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "COSY.jdf"))
        assert data.ndim == 2
        assert data.shape[0] == 512
        assert data.shape[1] == 1280
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_guess_udic_1d(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import guess_udic
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "1H.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 1
        assert udic[0]["label"] == "1H"
        assert udic[0]["sw"] > 0
        assert udic[0]["obs"] > 0

    def test_guess_udic_2d(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import guess_udic
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "COSY.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 2
        assert udic[0]["label"] == "1H"
        assert udic[1]["label"] == "1H"

    def test_guess_udic_hsqc(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import guess_udic
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "HSQC.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 2
        assert udic[0]["label"] == "1H"
        assert udic[1]["label"] == "13C"

    def test_header_fields(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, _ = read_jeol(str(JEOL_DIR / "1H.jdf"))
        assert "header" in dic
        assert "parameters" in dic
        assert dic["header"]["data_format"] == "one_d"
        assert dic["header"]["endian"] in ("big_endian", "little_endian")


# ---------------------------------------------------------------------------
# Public reader API
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jeol_data(), reason="JEOL test data not available")
class TestReadJeol:
    """Tests for the public read_jeol() function."""

    def test_read_1d(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.shape == (32768,)
        assert ds.origin == "jeol"

    def test_read_13c(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "13C.jdf"))
        assert ds.shape == (32768,)
        assert ds.origin == "jeol"

    def test_read_cosy(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        assert ds.ndim == 2
        assert ds.origin == "jeol"

    def test_read_hsqc(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "HSQC.jdf"))
        assert ds.ndim == 2
        assert ds.origin == "jeol"

    def test_read_hmbc(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "HMBC.jdf"))
        assert ds.ndim == 2
        assert ds.origin == "jeol"

    def test_1d_metadata_sw(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        sw = ds.meta.sw
        assert sw[0] is not None
        assert abs(sw[0] - 10016.0) < 1.0  # approximately 10016 Hz

    def test_1d_metadata_sfrq(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        sfrq = ds.meta.sfrq
        assert sfrq[0] is not None
        assert 399 < sfrq[0] < 401  # 400 MHz spectrometer

    def test_1d_metadata_nucleus(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        nuc = ds.meta.nucleus
        assert nuc[0] == "1H"

    def test_1d_metadata_solvent(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.meta.solvent == "DMSO-D6"

    def test_1d_metadata_ns(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.meta.ns == 128

    def test_1d_complex_data(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert np.issubdtype(ds.data.dtype, np.complexfloating)

    def test_1d_encoding_is_direct_complex(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.meta.encoding == ["QSIM"]

    def test_1d_has_coords(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.x is not None

    def test_2d_shape(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        assert ds.ndim == 2
        assert ds.shape[0] == 512
        assert ds.shape[1] == 640

    def test_2d_encoding(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        enc = ds.meta.encoding
        assert len(enc) == 2
        assert enc[0] is not None
        assert enc[1] is not None

    def test_2d_iscomplex(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        ic = ds.meta.iscomplex
        assert len(ic) == 2

    def test_2d_has_coords(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        assert ds.x is not None
        assert ds.y is not None

    def test_2d_origin(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        assert ds.origin == "jeol"

    def test_meta_readonly(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.meta.readonly is True

    def test_hsqc_different_nuclei(self):
        """HSQC has 1H indirect and 13C direct dimensions."""
        ds = _read_jeol_or_skip(str(JEOL_DIR / "HSQC.jdf"))
        nuc = ds.meta.nucleus
        assert "1H" in nuc
        assert "13C" in nuc

    def test_auto_detect_protocol(self):
        """Auto-detect JEOL from .jdf extension."""
        ds = scp.nmr.read(str(JEOL_DIR / "1H.jdf"))
        assert ds.shape == (32768,)
        assert ds.origin == "jeol"

    def test_auto_detect_generic_read(self):
        """Auto-detect JEOL from .jdf extension via generic read."""
        ds = scp.nmr.read(str(JEOL_DIR / "1H.jdf"))
        assert ds.origin == "jeol"
