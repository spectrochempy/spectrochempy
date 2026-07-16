# ruff: noqa: S101
"""
Validate the 2D FFT pipeline on real NMR data.

Type A: read → FFT → verify peak (non-zero magnitude)
Type B: read → em → FFT → verify peak (apodization + FFT chain)

Covers:
  - TopSpin 2D SER (Bruker)
  - Agilent 2D STATES, 2D TPPI
  - JEOL COSY, HSQC

Known limitations (not tested here):
  - pk() on 2D quaternion data (meta.phased is None)
  - em() on JEOL data (coords lack time units)
"""

from __future__ import annotations

import numpy as np
import pytest

import spectrochempy as scp

quaternion = pytest.importorskip("quaternion", reason="requires numpy-quaternion")

EXTRA_DATADIR = scp.preferences.datadir.parent / "testdata-extra"
EXTRA_NMR = EXTRA_DATADIR / "testdata" / "nmrdata"

TOPSPIN_2D = (
    scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"
)


def _has_agilent_data():
    return (EXTRA_NMR / "agilent" / "agilent_1d" / "fid").exists()


def _has_jeol_data():
    return (EXTRA_NMR / "jeol" / "1H.jdf").exists()


def _has_topspin_2d():
    return (TOPSPIN_2D / "1" / "ser").exists()


def _mag_from_quat_or_complex(ds):
    """Extract magnitude from quaternion or complex dataset."""
    if ds.dtype.kind == "V":
        qarr = quaternion.as_float_array(ds.data)
        return np.sqrt(np.sum(qarr**2, axis=-1))
    return np.abs(ds.data)


def _peak_info(mag):
    """Return (peak_indices, max_value)."""
    idx = np.unravel_index(np.argmax(mag), mag.shape)
    return idx, mag[idx]


# ---------------------------------------------------------------------------
# Type A: read → FFT → peak verification
# ---------------------------------------------------------------------------


class TestFFT2DTopSpin:
    """Type A: TopSpin 2D SER read + FFT."""

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data not available")
    def test_topspin_2d_fft_peak(self):
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        fft = ds.fft()

        assert fft.ndim == 2
        assert fft.shape[0] > 0 and fft.shape[1] > 0

        mag = _mag_from_quat_or_complex(fft)
        idx, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after TopSpin 2D FFT"
        assert all(0 <= i < s for i, s in zip(idx, fft.shape, strict=False))

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data not available")
    def test_topspin_2d_fft_encoding(self):
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        assert "encoding" in ds.meta
        assert len(ds.meta.encoding) == 2
        fft = ds.fft()
        assert fft.shape == ds.shape


@pytest.mark.skipif(not _has_agilent_data(), reason="Agilent test data not available")
class TestFFT2DAgilent:
    """Type A: Agilent 2D read + FFT."""

    def test_agilent_2d_states_fft_peak(self):
        ds = scp.read_agilent(EXTRA_NMR / "agilent" / "agilent_2d")
        fft = ds.fft()

        assert fft.ndim == 2
        assert fft.shape[0] > 0 and fft.shape[1] > 0

        mag = _mag_from_quat_or_complex(fft)
        idx, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after Agilent 2D STATES FFT"

    def test_agilent_2d_tppi_fft_peak(self):
        ds = scp.read_agilent(EXTRA_NMR / "agilent" / "agilent_2d_tppi")
        fft = ds.fft()

        assert fft.ndim == 2
        assert fft.shape[0] > 0 and fft.shape[1] > 0

        mag = _mag_from_quat_or_complex(fft)
        idx, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after Agilent 2D TPPI FFT"

    def test_agilent_2d_fft_encoding_preserved(self):
        ds = scp.read_agilent(EXTRA_NMR / "agilent" / "agilent_2d")
        enc_before = list(ds.meta.encoding)
        fft = ds.fft()
        assert fft.meta.encoding == enc_before


@pytest.mark.skipif(not _has_jeol_data(), reason="JEOL test data not available")
class TestFFT2DJEOL:
    """Type A: JEOL 2D read + FFT."""

    def test_jeol_cosy_fft_peak(self):
        ds = scp.read_jeol(EXTRA_NMR / "jeol" / "COSY.jdf")
        fft = ds.fft()

        assert fft.ndim == 2
        assert fft.shape[0] > 0 and fft.shape[1] > 0

        mag = _mag_from_quat_or_complex(fft)
        idx, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after JEOL COSY FFT"

    def test_jeol_hsqc_fft_peak(self):
        ds = scp.read_jeol(EXTRA_NMR / "jeol" / "HSQC.jdf")
        fft = ds.fft()

        assert fft.ndim == 2
        assert fft.shape[0] > 0 and fft.shape[1] > 0

        mag = _mag_from_quat_or_complex(fft)
        idx, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after JEOL HSQC FFT"


# ---------------------------------------------------------------------------
# Type B: read → em → FFT → peak verification
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data not available")
class TestEmFFT2DTopSpin:
    """Type B: TopSpin 2D em + FFT chain."""

    def test_topspin_2d_em_fft_peak(self):
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        ds.em(lb=2.0, inplace=True)
        fft = ds.fft()

        assert fft.ndim == 2
        mag = _mag_from_quat_or_complex(fft)
        idx, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after em + FFT"

    def test_topspin_2d_em_zero_lb(self):
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        ds.em(lb=0.0, inplace=True)
        fft = ds.fft()

        assert fft.ndim == 2
        mag = _mag_from_quat_or_complex(fft)
        _, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after em(lb=0) + FFT"


@pytest.mark.skipif(not _has_agilent_data(), reason="Agilent test data not available")
class TestEmFFT2DAgilent:
    """Type B: Agilent 2D em + FFT chain."""

    def test_agilent_2d_states_em_fft_peak(self):
        ds = scp.read_agilent(EXTRA_NMR / "agilent" / "agilent_2d")
        ds.em(lb=2.0, inplace=True)
        fft = ds.fft()

        assert fft.ndim == 2
        mag = _mag_from_quat_or_complex(fft)
        _, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after Agilent 2D STATES em + FFT"

    def test_agilent_2d_tppi_em_fft_peak(self):
        ds = scp.read_agilent(EXTRA_NMR / "agilent" / "agilent_2d_tppi")
        ds.em(lb=2.0, inplace=True)
        fft = ds.fft()

        assert fft.ndim == 2
        mag = _mag_from_quat_or_complex(fft)
        _, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after Agilent 2D TPPI em + FFT"
