# ruff: noqa: S101
"""
2D FFT Chain Characterization.

This module traces the complete 2D FFT chain step by step on a synthetic
SER (Serial Experiment Result) with a single known peak. Each step is
documented, tested, and compared to analytical expectations.

Pipeline:
    quaternion raw data
        → representation adaptation (quaternion → complex)
        → FFT F2
        → FFT F1
        → final reconstruction

The goal is to verify that each step produces the expected result and to
document the exact state of the data at each stage.

Quaternion convention (spectrochempy-hypercomplex / numpy-quaternion):
    as_float_array(q) → [..., 4] with order [w, x, y, z]
    w = RR (real-real), x = RI (real-imag), y = IR (imag-real), z = II (imag-imag)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from spectrochempy_nmr.processing.hypercomplex import _extract_quaternion_components
from spectrochempy_nmr.processing.hypercomplex import _prepare_states
from spectrochempy_nmr.processing.hypercomplex import _rebuild_quaternion

quaternion = pytest.importorskip("quaternion", reason="requires numpy-quaternion")
as_float_array = quaternion.as_float_array


def _make_quat(array_4d):
    """Stack [..., 4] float array into quaternion array."""
    return quaternion.as_quat_array(np.asarray(array_4d, dtype=np.float64))


# ---------------------------------------------------------------------------
# Synthetic SER construction
# ---------------------------------------------------------------------------


def _make_states_ser(nf1, nf2, f1_freq, f2_freq):
    """
    Create a STATES-encoded SER for a single 2D peak.

    The SER is the raw data as stored by the Bruker spectrometer.
    For STATES encoding:
        RR = cos(w1*t1) * cos(w2*t2)
        RI = cos(w1*t1) * sin(w2*t2)
        IR = sin(w1*t1) * cos(w2*t2)
        II = sin(w1*t1) * sin(w2*t2)

    Parameters
    ----------
    nf1, nf2 : int
        Number of points in F1 and F2.
    f1_freq, f2_freq : float
        Frequency in cycles per dimension (e.g. 2.0 = 2 complete cycles).

    Returns
    -------
    ser : quaternion array
        Raw SER data with shape (nf1, nf2).
    """
    t1 = np.arange(nf1)
    t2 = np.arange(nf2)
    w1 = 2 * np.pi * f1_freq / nf1
    w2 = 2 * np.pi * f2_freq / nf2

    cos1 = np.cos(w1 * t1)[:, None]
    sin1 = np.sin(w1 * t1)[:, None]
    cos2 = np.cos(w2 * t2)[None, :]
    sin2 = np.sin(w2 * t2)[None, :]

    RR = cos1 * cos2
    RI = cos1 * sin2
    IR = sin1 * cos2
    II = sin1 * sin2

    return _make_quat(np.stack([RR, RI, IR, II], axis=-1))


# ---------------------------------------------------------------------------
# Step 1: Quaternion raw data
# ---------------------------------------------------------------------------


class TestStep1QuaternionRaw:
    """Step 1: Verify the raw quaternion SER data."""

    def test_ser_shape_and_dtype(self):
        """SER should have shape (nf1, nf2) and quaternion dtype."""
        nf1, nf2 = 16, 32
        ser = _make_states_ser(nf1, nf2, 2.0, 5.0)

        assert ser.shape == (nf1, nf2)
        assert ser.dtype == np.quaternion

    def test_ser_components_match_analytical(self):
        """SER components should match analytical formulas."""
        nf1, nf2 = 8, 16
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        RR, RI, IR, II = _extract_quaternion_components(ser)

        t1 = np.arange(nf1)
        t2 = np.arange(nf2)
        w1 = 2 * np.pi * f1_freq / nf1
        w2 = 2 * np.pi * f2_freq / nf2

        expected_RR = np.cos(w1 * t1)[:, None] * np.cos(w2 * t2)[None, :]
        expected_RI = np.cos(w1 * t1)[:, None] * np.sin(w2 * t2)[None, :]
        expected_IR = np.sin(w1 * t1)[:, None] * np.cos(w2 * t2)[None, :]
        expected_II = np.sin(w1 * t1)[:, None] * np.sin(w2 * t2)[None, :]

        assert_allclose(RR, expected_RR, atol=1e-10)
        assert_allclose(RI, expected_RI, atol=1e-10)
        assert_allclose(IR, expected_IR, atol=1e-10)
        assert_allclose(II, expected_II, atol=1e-10)


# ---------------------------------------------------------------------------
# Step 2: Representation adaptation
# ---------------------------------------------------------------------------


class TestStep2RepresentationAdaptation:
    """Step 2: Verify quaternion → complex subspectra adaptation."""

    def test_states_adaptation_produces_two_subspectra(self):
        """STATES adaptation should produce two complex subspectra."""
        nf1, nf2 = 8, 16
        ser = _make_states_ser(nf1, nf2, 2.0, 5.0)

        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)

        assert sr.shape == (nf1, nf2)
        assert si.shape == (nf1, nf2)
        assert np.iscomplexobj(sr)
        assert np.iscomplexobj(si)

    def test_states_adaptation_formulas(self):
        """STATES adaptation should compute (RR - 1j*RI)/2 and (IR - 1j*II)/2."""
        nf1, nf2 = 8, 16
        ser = _make_states_ser(nf1, nf2, 2.0, 5.0)

        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)

        expected_sr = (RR - 1j * RI) / 2.0
        expected_si = (IR - 1j * II) / 2.0

        assert_allclose(sr, expected_sr, atol=1e-10)
        assert_allclose(si, expected_si, atol=1e-10)

    def test_subspectra_are_complex_exponentials(self):
        """After adaptation, subspectra should be complex exponentials in F2."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)

        # sr = (RR - 1j*RI)/2 = cos(w1*t1) * (cos(w2*t2) - 1j*sin(w2*t2)) / 2
        #   = cos(w1*t1) * exp(-1j*w2*t2) / 2
        # So |sr| should be |cos(w1*t1)| / 2 for each t1

        t1 = np.arange(nf1)
        w1 = 2 * np.pi * f1_freq / nf1
        expected_amplitude = np.abs(np.cos(w1 * t1)) / 2.0

        # Check amplitude along F2 for each F1 row
        for i in range(nf1):
            actual_amplitude = np.abs(sr[i, :])
            # All points in this row should have the same amplitude
            assert_allclose(actual_amplitude, expected_amplitude[i], atol=1e-10)


# ---------------------------------------------------------------------------
# Step 3: FFT F2
# ---------------------------------------------------------------------------


class TestStep3FFTF2:
    """Step 3: Verify FFT along F2 dimension."""

    def test_fft_f2_produces_correct_peak_position(self):
        """FFT F2 should place the peak at the correct F2 frequency."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)

        # FFT F2
        fr = np.fft.fftshift(np.fft.fft(sr, axis=-1), axes=-1)
        fi = np.fft.fftshift(np.fft.fft(si, axis=-1), axes=-1)

        # Rebuild quaternion
        f2_result = _rebuild_quaternion(fr, fi)

        # Check shape and dtype
        assert f2_result.shape == (nf1, nf2)
        assert f2_result.dtype == np.quaternion

        # The peak should be visible in the RR component
        w, x, y, z = _extract_quaternion_components(f2_result)

        # After fftshift, the positive frequency peak is at n//2 + f2_freq
        # or the negative frequency peak at n//2 - f2_freq
        peak_f2 = int(np.argmax(np.abs(w[0, :])))
        expected_f2_positive = (nf2 // 2 + int(f2_freq)) % nf2
        expected_f2_negative = (nf2 // 2 - int(f2_freq)) % nf2

        assert (
            abs(peak_f2 - expected_f2_positive) <= 1
            or abs(peak_f2 - expected_f2_negative) <= 1
        ), f"F2 peak at {peak_f2}, expected near {expected_f2_positive} or {expected_f2_negative}"

    def test_fft_f2_amplitude_matches_analytical(self):
        """FFT F2 amplitude should match analytical expectation."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)

        # FFT F2
        fr = np.fft.fftshift(np.fft.fft(sr, axis=-1), axes=-1)
        _ = np.fft.fftshift(np.fft.fft(si, axis=-1), axes=-1)

        # sr = cos(w1*t1) * exp(-1j*w2*t2) / 2
        # FFT of exp(-1j*w2*t2) gives a delta at frequency -f2
        # After fftshift, this is at index n//2 - f2_freq
        # Amplitude: N/2 (for the delta) * |cos(w1*t1)| / 2

        t1 = np.arange(nf1)
        w1 = 2 * np.pi * f1_freq / nf1
        expected_amplitude = nf2 * np.abs(np.cos(w1 * t1)) / 2.0

        # Check amplitude at the peak position
        peak_idx = (nf2 // 2 - int(f2_freq)) % nf2
        for i in range(nf1):
            actual_amplitude = np.abs(fr[i, peak_idx])
            assert_allclose(actual_amplitude, expected_amplitude[i], rtol=0.1)


# ---------------------------------------------------------------------------
# Step 4: FFT F1
# ---------------------------------------------------------------------------


class TestStep4FFTF1:
    """Step 4: Verify FFT along F1 dimension."""

    def test_fft_f1_produces_correct_2d_peak(self):
        """FFT F1 should place the peak at the correct 2D position."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        # Step 2: Representation adaptation
        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)

        # Step 3: FFT F2
        fr = np.fft.fftshift(np.fft.fft(sr, axis=-1), axes=-1)
        fi = np.fft.fftshift(np.fft.fft(si, axis=-1), axes=-1)

        # Step 4: FFT F1
        f1_fr = np.fft.fftshift(np.fft.fft(fr, axis=0), axes=0)
        f1_fi = np.fft.fftshift(np.fft.fft(fi, axis=0), axes=0)

        # Check shape
        assert f1_fr.shape == (nf1, nf2)
        assert f1_fi.shape == (nf1, nf2)

        # The peak should be visible in the magnitude
        magnitude = np.abs(f1_fr) + np.abs(f1_fi)
        peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)

        # After fftshift, the positive frequency peak is at n//2 + f
        # or the negative frequency peak at n//2 - f
        expected_f1_positive = (nf1 // 2 + int(f1_freq)) % nf1
        expected_f1_negative = (nf1 // 2 - int(f1_freq)) % nf1
        expected_f2_positive = (nf2 // 2 + int(f2_freq)) % nf2
        expected_f2_negative = (nf2 // 2 - int(f2_freq)) % nf2

        f1_ok = (
            abs(peak[0] - expected_f1_positive) <= 1
            or abs(peak[0] - expected_f1_negative) <= 1
        )
        f2_ok = (
            abs(peak[1] - expected_f2_positive) <= 1
            or abs(peak[1] - expected_f2_negative) <= 1
        )

        assert f1_ok, f"F1 peak at {peak[0]}, expected near {expected_f1_positive} or {expected_f1_negative}"
        assert f2_ok, f"F2 peak at {peak[1]}, expected near {expected_f2_positive} or {expected_f2_negative}"

    def test_fft_f1_peak_symmetry(self):
        """After 2D FFT, the peak should be symmetric around the center."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        # Full pipeline
        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)
        fr = np.fft.fftshift(np.fft.fft(sr, axis=-1), axes=-1)
        fi = np.fft.fftshift(np.fft.fft(si, axis=-1), axes=-1)
        f1_fr = np.fft.fftshift(np.fft.fft(fr, axis=0), axes=0)
        f1_fi = np.fft.fftshift(np.fft.fft(fi, axis=0), axes=0)

        magnitude = np.abs(f1_fr) + np.abs(f1_fi)

        # Find all peaks above 50% of maximum
        threshold = 0.5 * np.max(magnitude)
        peaks = np.argwhere(magnitude > threshold)

        # All peaks should be clustered around the expected position
        center_f1 = nf1 // 2
        center_f2 = nf2 // 2

        for p in peaks:
            assert (
                abs(p[0] - center_f1) <= nf1 // 4
            ), f"Peak at F1={p[0]} too far from center"
            assert (
                abs(p[1] - center_f2) <= nf2 // 4
            ), f"Peak at F2={p[1]} too far from center"


# ---------------------------------------------------------------------------
# Step 5: Full pipeline characterization
# ---------------------------------------------------------------------------


class TestStep5FullPipeline:
    """Step 5: Verify the complete 2D FFT pipeline."""

    def test_full_pipeline_single_peak(self):
        """Full pipeline should produce a single 2D peak at the correct position."""
        nf1, nf2 = 32, 64
        f1_freq, f2_freq = 4.0, 10.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        # Step 2: Representation adaptation
        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)

        # Step 3: FFT F2
        fr = np.fft.fftshift(np.fft.fft(sr, axis=-1), axes=-1)
        fi = np.fft.fftshift(np.fft.fft(si, axis=-1), axes=-1)

        # Step 4: FFT F1
        f1_fr = np.fft.fftshift(np.fft.fft(fr, axis=0), axes=0)
        f1_fi = np.fft.fftshift(np.fft.fft(fi, axis=0), axes=0)

        # Final magnitude
        magnitude = np.abs(f1_fr) + np.abs(f1_fi)

        # Find the peak
        peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)

        # Verify peak position
        expected_f1 = (nf1 // 2 - int(f1_freq)) % nf1
        expected_f2 = (nf2 // 2 - int(f2_freq)) % nf2

        assert (
            abs(peak[0] - expected_f1) <= 1
        ), f"F1 peak at {peak[0]}, expected {expected_f1}"
        assert (
            abs(peak[1] - expected_f2) <= 1
        ), f"F2 peak at {peak[1]}, expected {expected_f2}"

    def test_full_pipeline_preserves_energy(self):
        """Full pipeline should preserve total energy (Parseval's theorem)."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        # Energy in time domain
        RR, RI, IR, II = _extract_quaternion_components(ser)
        energy_td = np.sum(RR**2 + RI**2 + IR**2 + II**2)

        # Full pipeline
        sr, si = _prepare_states(RR, RI, IR, II)
        fr = np.fft.fftshift(np.fft.fft(sr, axis=-1), axes=-1)
        fi = np.fft.fftshift(np.fft.fft(si, axis=-1), axes=-1)
        f1_fr = np.fft.fftshift(np.fft.fft(fr, axis=0), axes=0)
        f1_fi = np.fft.fftshift(np.fft.fft(fi, axis=0), axes=0)

        # Energy in frequency domain
        energy_fd = np.sum(np.abs(f1_fr) ** 2 + np.abs(f1_fi) ** 2)

        # Parseval's theorem: energy should be preserved (within numerical precision)
        # Note: the /2 factors in the STATES decomposition affect the scaling
        # so we check relative energy conservation
        assert_allclose(energy_fd, energy_td * nf1 * nf2 / 4, rtol=0.1)


# ---------------------------------------------------------------------------
# Step 6: Comparison with manual workflow
# ---------------------------------------------------------------------------


class TestStep6ComparisonWithManual:
    """Step 6: Verify that the pipeline matches the manual workflow."""

    def test_pipeline_matches_manual_workflow(self):
        """
        Full pipeline should produce same peak as manual 2D workflow.

        Both decompositions are valid: the pipeline uses the standard STATES
        convention (sr = (RR - 1j*RI)/2, si = (IR - 1j*II)/2) while the
        manual workflow swaps the pairing (sr = (RR - 1j*IR)/2, si =
        (RI - 1j*II)/2). Both produce peaks at the correct positions.
        We verify both produce the same peak location and comparable magnitude.
        """
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        # Pipeline (using representation layer)
        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)
        fr = np.fft.fftshift(np.fft.fft(sr, axis=-1), axes=-1)
        fi = np.fft.fftshift(np.fft.fft(si, axis=-1), axes=-1)
        f1_fr = np.fft.fftshift(np.fft.fft(fr, axis=0), axes=0)
        f1_fi = np.fft.fftshift(np.fft.fft(fi, axis=0), axes=0)

        # Manual workflow (direct quaternion manipulation)
        fa = as_float_array(ser)
        w, x, y, z = fa[..., 0], fa[..., 1], fa[..., 2], fa[..., 3]

        # Manual STATES decomposition (alternative pairing)
        manual_sr = (w - 1j * y) / 2.0
        manual_si = (x - 1j * z) / 2.0

        # Manual FFT F2
        manual_fr = np.fft.fftshift(np.fft.fft(manual_sr, axis=-1), axes=-1)
        manual_fi = np.fft.fftshift(np.fft.fft(manual_si, axis=-1), axes=-1)

        # Manual FFT F1
        manual_f1_fr = np.fft.fftshift(np.fft.fft(manual_fr, axis=0), axes=0)
        manual_f1_fi = np.fft.fftshift(np.fft.fft(manual_fi, axis=0), axes=0)

        # Both should produce peaks at the same 2D position
        pipe_mag = np.abs(f1_fr) + np.abs(f1_fi)
        manual_mag = np.abs(manual_f1_fr) + np.abs(manual_f1_fi)

        pipe_peak = np.unravel_index(np.argmax(pipe_mag), pipe_mag.shape)
        manual_peak = np.unravel_index(np.argmax(manual_mag), manual_mag.shape)

        assert (
            abs(pipe_peak[0] - manual_peak[0]) <= 1
            and abs(pipe_peak[1] - manual_peak[1]) <= 1
        ), f"Peak positions differ: pipeline={pipe_peak}, manual={manual_peak}"

        # Both should have comparable peak magnitudes
        assert_allclose(pipe_mag[pipe_peak], manual_mag[manual_peak], rtol=0.1)


# ---------------------------------------------------------------------------
# Step 7: End-to-end fft() on NDDataset with quaternion data
# ---------------------------------------------------------------------------


class TestStep7EndToEndFFT:
    """Step 7: Verify that the actual fft() function works on 2D quaternion NDDataset."""

    def test_fft_f2_on_2d_quaternion_nddataset(self):
        """fft(dim=-1) on a 2D quaternion NDDataset should return quaternion with correct peak."""
        from spectrochempy import Coord
        from spectrochempy import NDDataset

        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        c_y = Coord.arange(nf1, unit="s")
        c_x = Coord.arange(nf2, unit="s")
        ds = NDDataset(ser, coordset=[c_y, c_x])
        ds.meta.encoding = ["STATES", "DQD"]
        ds.meta.iscomplex = [True, True]
        ds.meta.td = [nf1, nf2]
        ds.meta.si = [nf1, nf2]

        ds_f2 = ds.fft(dim=-1)
        assert ds_f2.shape == (nf1, nf2)
        assert "quaternion" in str(ds_f2.data.dtype)

        fa = as_float_array(ds_f2.data)
        magnitude = np.sqrt(
            fa[..., 0] ** 2 + fa[..., 1] ** 2 + fa[..., 2] ** 2 + fa[..., 3] ** 2
        )
        peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)

        expected_f2_pos = (nf2 // 2 - int(f2_freq)) % nf2
        expected_f2_neg = (nf2 // 2 + int(f2_freq)) % nf2
        f2_ok = (
            abs(peak[1] - expected_f2_pos) <= 1 or abs(peak[1] - expected_f2_neg) <= 1
        )
        assert (
            f2_ok
        ), f"F2 peak at {peak[1]}, expected near {expected_f2_pos} or {expected_f2_neg}"

    def test_fft_full_2d_chain_on_nddataset(self):
        """fft(dim=-1) then fft(dim=0) on a 2D quaternion NDDataset should find the 2D peak."""
        from spectrochempy import Coord
        from spectrochempy import NDDataset

        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        c_y = Coord.arange(nf1, unit="s")
        c_x = Coord.arange(nf2, unit="s")
        ds = NDDataset(ser, coordset=[c_y, c_x])
        ds.meta.encoding = ["STATES", "DQD"]
        ds.meta.iscomplex = [True, True]
        ds.meta.td = [nf1, nf2]
        ds.meta.si = [nf1, nf2]

        ds_f2 = ds.fft(dim=-1)
        ds_2d = ds_f2.fft(dim=0)

        assert ds_2d.shape == (nf1, nf2)
        assert "quaternion" in str(ds_2d.data.dtype)

        fa = as_float_array(ds_2d.data)
        magnitude = np.sqrt(
            fa[..., 0] ** 2 + fa[..., 1] ** 2 + fa[..., 2] ** 2 + fa[..., 3] ** 2
        )
        peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)

        expected_f1 = (nf1 // 2 - int(f1_freq)) % nf1
        expected_f2 = (nf2 // 2 - int(f2_freq)) % nf2

        assert (
            abs(peak[0] - expected_f1) <= 1
        ), f"F1 peak at {peak[0]}, expected {expected_f1}"
        assert (
            abs(peak[1] - expected_f2) <= 1
        ), f"F2 peak at {peak[1]}, expected {expected_f2}"

    def test_fft_2d_peak_matches_manual_workflow(self):
        """fft() 2D chain peak position should match the manual numpy workflow."""
        from spectrochempy import Coord
        from spectrochempy import NDDataset

        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        ser = _make_states_ser(nf1, nf2, f1_freq, f2_freq)

        # fft() chain
        c_y = Coord.arange(nf1, unit="s")
        c_x = Coord.arange(nf2, unit="s")
        ds = NDDataset(ser, coordset=[c_y, c_x])
        ds.meta.encoding = ["STATES", "DQD"]
        ds.meta.iscomplex = [True, True]
        ds.meta.td = [nf1, nf2]
        ds.meta.si = [nf1, nf2]

        ds_2d = ds.fft(dim=-1).fft(dim=0)
        fa = as_float_array(ds_2d.data)
        pipe_mag = np.sqrt(
            fa[..., 0] ** 2 + fa[..., 1] ** 2 + fa[..., 2] ** 2 + fa[..., 3] ** 2
        )
        pipe_peak = np.unravel_index(np.argmax(pipe_mag), pipe_mag.shape)

        # Manual numpy workflow
        RR, RI, IR, II = _extract_quaternion_components(ser)
        sr, si = _prepare_states(RR, RI, IR, II)
        fr = np.fft.fftshift(np.fft.fft(sr, axis=-1), axes=-1)
        fi = np.fft.fftshift(np.fft.fft(si, axis=-1), axes=-1)
        f1_fr = np.fft.fftshift(np.fft.fft(fr, axis=0), axes=0)
        f1_fi = np.fft.fftshift(np.fft.fft(fi, axis=0), axes=0)
        manual_mag = np.abs(f1_fr) + np.abs(f1_fi)
        manual_peak = np.unravel_index(np.argmax(manual_mag), manual_mag.shape)

        assert (
            abs(pipe_peak[0] - manual_peak[0]) <= 1
            and abs(pipe_peak[1] - manual_peak[1]) <= 1
        ), f"Peak positions differ: fft()={pipe_peak}, manual={manual_peak}"
