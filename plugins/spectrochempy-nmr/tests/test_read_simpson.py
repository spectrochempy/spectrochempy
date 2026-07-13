# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

"""Tests for SIMPSON NMR reader."""

import numpy as np

import spectrochempy as scp

# ---------------------------------------------------------------------------
# Synthetic SIMPSON file fixtures
# ---------------------------------------------------------------------------


def _write_text_1d(path, npts=8, sw=10000.0):
    """Write a synthetic SIMPSON TEXT 1D file."""
    data = np.exp(-np.arange(npts) / 4.0) * np.exp(2j * np.pi * np.arange(npts) / 8)
    with open(path, "w") as f:
        f.write("SIMP\n")
        f.write(f"NP={npts}\n")
        f.write(f"SW={sw}\n")
        f.write("DATA\n")
        for r, i in zip(data.real, data.imag, strict=False):
            f.write(f"{r:.6f} {i:.6f}\n")
        f.write("END\n")
    return data


def _write_text_2d(path, npts=8, ni=4, sw=10000.0, sw1=5000.0):
    """Write a synthetic SIMPSON TEXT 2D file."""
    data = np.empty((ni, npts), dtype="complex64")
    for i in range(ni):
        for j in range(npts):
            data[i, j] = (i + 1) * np.exp(-j / 4.0) * np.exp(1j * np.pi * j / 4)
    with open(path, "w") as f:
        f.write("SIMP\n")
        f.write(f"NP={npts}\n")
        f.write(f"NI={ni}\n")
        f.write(f"SW={sw}\n")
        f.write(f"SW1={sw1}\n")
        f.write("DATA\n")
        for row in data:
            for r, i in zip(row.real, row.imag, strict=False):
                f.write(f"{r:.6f} {i:.6f}\n")
        f.write("END\n")
    return data


def _write_xreim_1d(path, npts=8):
    """Write a synthetic SIMPSON XREIM 1D file."""
    data = np.exp(-np.arange(npts) / 4.0) * np.exp(2j * np.pi * np.arange(npts) / 8)
    with open(path, "w") as f:
        for i, (r, img) in enumerate(zip(data.real, data.imag, strict=False)):
            f.write(f"{i * 50.0:.1f} {r:.6f} {img:.6f}\n")
    return data


def _write_xyreim_2d(path, npts=8, ni=4):
    """Write a synthetic SIMPSON XYREIM 2D file."""
    data = np.empty((ni, npts), dtype="complex64")
    for i in range(ni):
        for j in range(npts):
            data[i, j] = (i + 1) * np.exp(-j / 4.0) * np.exp(1j * np.pi * j / 4)
    with open(path, "w") as f:
        for i in range(ni):
            for j in range(npts):
                f.write(
                    f"{i * 100.0:.1f} {j * 50.0:.1f} {data.real[i, j]:.6f} {data.imag[i, j]:.6f}\n"
                )
            if i < ni - 1:
                f.write("\n")
    return data


def _write_in_file(path):
    """Write a synthetic SIMPSON .in file for metadata parsing."""
    text = """
spinsys {
    channels 13C
    nuclei 13C 13C
    shift 1 0 6000 1 0 0 0
}

par {
    spin_rate 2000
    np 8
    ni 4
    sw 10000
    sw1 5000
    proton_frequency 400e6
}
"""
    path.write_text(text)


# ---------------------------------------------------------------------------
# Parser layer tests
# ---------------------------------------------------------------------------


class TestSimpsonParser:
    """Tests for the custom SIMPSON parser."""

    def test_read_text_1d(self, tmp_path):
        from spectrochempy_nmr.extern.nmrglue._simpson import read

        path = tmp_path / "1d_text.spe"
        expected = _write_text_1d(path)
        dic, data = read(str(path))
        assert data.shape == (8,)
        assert np.allclose(data, expected)
        assert dic["NP"] == 8
        assert dic["SW"] == 10000.0

    def test_read_text_2d(self, tmp_path):
        from spectrochempy_nmr.extern.nmrglue._simpson import read

        path = tmp_path / "2d_text.spe"
        expected = _write_text_2d(path)
        dic, data = read(str(path))
        assert data.shape == (4, 8)
        assert np.allclose(data, expected)
        assert dic["NI"] == 4
        assert dic["SW1"] == 5000.0

    def test_read_xreim_1d(self, tmp_path):
        from spectrochempy_nmr.extern.nmrglue._simpson import read

        path = tmp_path / "1d_xreim.txt"
        expected = _write_xreim_1d(path)
        dic, data = read(str(path))
        assert data.shape == (8,)
        assert np.allclose(data, expected)
        assert "units" in dic

    def test_read_xyreim_2d(self, tmp_path):
        from spectrochempy_nmr.extern.nmrglue._simpson import read

        path = tmp_path / "2d_xyreim.txt"
        expected = _write_xyreim_2d(path)
        dic, data = read(str(path))
        assert data.shape == (4, 8)
        assert np.allclose(data, expected)
        assert "units" in dic

    def test_guess_ftype_text(self, tmp_path):
        from spectrochempy_nmr.extern.nmrglue._simpson import guess_ftype

        path = tmp_path / "1d_text.spe"
        _write_text_1d(path)
        assert guess_ftype(str(path)) == "TEXT"

    def test_guess_ftype_xreim(self, tmp_path):
        from spectrochempy_nmr.extern.nmrglue._simpson import guess_ftype

        path = tmp_path / "1d_xreim.txt"
        _write_xreim_1d(path)
        assert guess_ftype(str(path)) == "XREIM"


# ---------------------------------------------------------------------------
# Public reader API tests
# ---------------------------------------------------------------------------


class TestSimpsonReader:
    """Tests for the public read_simpson() API."""

    def test_read_simpson_text_1d(self, tmp_path):
        path = tmp_path / "1d_text.spe"
        _write_text_1d(path)
        ds = scp.nmr.read_simpson(str(path))
        assert ds is not None
        assert ds.ndim == 1
        assert ds.shape == (8,)
        assert ds.origin == "simpson"

    def test_read_simpson_text_2d(self, tmp_path):
        path = tmp_path / "2d_text.spe"
        _write_text_2d(path)
        ds = scp.nmr.read_simpson(str(path))
        assert ds.ndim == 2
        assert ds.shape == (4, 8)

    def test_read_simpson_xreim_1d(self, tmp_path):
        path = tmp_path / "1d_xreim.spe"
        _write_xreim_1d(path)
        ds = scp.nmr.read_simpson(str(path))
        assert ds.ndim == 1
        assert ds.shape == (8,)

    def test_read_simpson_directory_with_in(self, tmp_path):
        _write_in_file(tmp_path / "experiment.in")
        data_path = tmp_path / "experiment.spe"
        _write_text_1d(data_path)
        ds = scp.nmr.read_simpson(str(tmp_path))
        assert ds is not None
        assert ds.ndim == 1
        assert ds.origin == "simpson"

    def test_read_simpson_in_with_sibling_data(self, tmp_path):
        in_path = tmp_path / "experiment.in"
        _write_in_file(in_path)
        _write_text_1d(tmp_path / "experiment.spe")
        ds = scp.nmr.read_simpson(str(in_path))
        assert ds is not None
        assert ds.ndim == 1

    def test_read_simpson_metadata_sw(self, tmp_path):
        path = tmp_path / "1d_text.spe"
        _write_text_1d(path)
        ds = scp.nmr.read_simpson(str(path))
        assert ds.meta.sw[0] == 10000.0

    def test_read_simpson_metadata_nucleus(self, tmp_path):
        in_path = tmp_path / "experiment.in"
        _write_in_file(in_path)
        _write_text_1d(tmp_path / "experiment.spe")
        ds = scp.nmr.read_simpson(str(in_path))
        assert ds.meta.nucleus[0] is not None
        assert "13C" in ds.meta.nucleus[0]

    def test_read_simpson_auto_detect(self, tmp_path):
        path = tmp_path / "1d_text.spe"
        _write_text_1d(path)
        ds = scp.nmr.read(str(path))
        assert ds is not None
        assert ds.origin == "simpson"

    def test_read_simpson_missing_file(self):
        result = scp.nmr.read_simpson("/nonexistent/path.spe")
        assert result is None
