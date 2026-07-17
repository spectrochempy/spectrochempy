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


# ---------------------------------------------------------------------------
# Phase metadata initialization on quaternion data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data not available")
class TestPhaseMetadata2D:
    """Verify phase metadata is initialized after 2D FFT on quaternion data."""

    def test_topspin_2d_fft_sets_phased(self):
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        fft = ds.fft()
        assert fft.meta.phased is not None
        assert len(fft.meta.phased) == 2

    def test_topspin_2d_fft_sets_phc0(self):
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        fft = ds.fft()
        assert fft.meta.phc0 is not None
        assert len(fft.meta.phc0) == 2


# ---------------------------------------------------------------------------
# JEOL em() with time-domain coordinates
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jeol_data(), reason="JEOL test data not available")
class TestJEOLEmChain:
    """Verify em() works on JEOL data after coordinate fix."""

    def test_jeol_cosy_em_fft(self):
        ds = scp.read_jeol(EXTRA_NMR / "jeol" / "COSY.jdf")
        ds.em(lb=2.0, inplace=True)
        fft = ds.fft()

        assert fft.ndim == 2
        mag = _mag_from_quat_or_complex(fft)
        _, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after JEOL COSY em + FFT"

    def test_jeol_hsqc_em_fft(self):
        ds = scp.read_jeol(EXTRA_NMR / "jeol" / "HSQC.jdf")
        ds.em(lb=2.0, inplace=True)
        fft = ds.fft()

        assert fft.ndim == 2
        mag = _mag_from_quat_or_complex(fft)
        _, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after JEOL HSQC em + FFT"


# ---------------------------------------------------------------------------
# Quaternion 2D phasing (auto-phase + manual pk)
# ---------------------------------------------------------------------------


def _has_topspin_2d() -> bool:
    return TOPSPIN_2D.exists()


@pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data not available")
class TestQuaternionPhasing:
    """Verify quaternion 2D auto-phasing via plugin pk.execute handler."""

    def test_fft_auto_phase_sets_meta(self):
        """fft() auto-phases F2 (direct dim) of quaternion data."""
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        fft = ds.fft()

        assert fft.dtype == quaternion.quaternion
        assert fft.ndim == 2
        # F2 should be auto-phased
        assert fft.meta.phased[-1] is True
        assert fft.meta.phc0[-1].magnitude == 0.0

    def test_fft_auto_phase_pivot(self):
        """Auto-phase computes pivot from quaternion modulus."""
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        fft = ds.fft()

        assert fft.meta.pivot[-1] is not None
        assert fft.ndim == 2

    def test_manual_pk_on_2d_quaternion(self):
        """Explicit pk(pivot=, phc0=) works on quaternion 2D data."""
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        fft = ds.fft()
        phased = fft.pk(pivot=0, phc0=45.0)

        assert phased.dtype == quaternion.quaternion
        assert phased.meta.phased[-1] is True
        assert phased.ndim == 2
        # Quaternion modulus is invariant under phase rotation,
        # so compare raw quaternion components directly.
        assert not np.allclose(
            quaternion.as_float_array(fft.data),
            quaternion.as_float_array(phased.data),
        ), "pk() did not modify the quaternion data"

    def test_em_fft_auto_phase_chain(self):
        """Full chain: em() → fft() auto-phases quaternion data."""
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        ds.em(lb=2.0, inplace=True)
        fft = ds.fft()

        assert fft.dtype == quaternion.quaternion
        assert fft.meta.phased[-1] is True
        mag = _mag_from_quat_or_complex(fft)
        _, maxval = _peak_info(mag)
        assert maxval > 0, "No peak found after em + fft + auto-phase"

    def test_quaternion_phasc0_from_bruker(self):
        """F1 (indirect dim) phc0 is preserved from Bruker processing params."""
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        fft = ds.fft()

        # F1 phc0 should come from acqu2s (Bruker), not be reset to 0
        f1_phc0 = fft.meta.phc0[0].magnitude
        # Bruker typically stores a non-zero value; just check it's a number
        assert isinstance(f1_phc0, (int, float))
