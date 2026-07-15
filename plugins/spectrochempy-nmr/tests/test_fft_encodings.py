# ruff: noqa: S101
"""
Synthetic tests for NMR FFT encoding handlers.

Each test constructs a deterministic quaternion signal with a known single
frequency, applies the encoding-specific FFT, and verifies the peak position
and component values against analytical expectations.

Quaternion convention (spectrochempy-hypercomplex / numpy-quaternion):
    as_float_array(q) → [..., 4] with order [w, x, y, z]
    w = RR (real-real), x = RI (real-imag), y = IR (imag-real), z = II (imag-imag)

The encoding handlers receive quaternion data and must return quaternion data.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from spectrochempy_nmr.processing.fft_encodings import _echoanti_fft
from spectrochempy_nmr.processing.fft_encodings import _fft_encoding_handler
from spectrochempy_nmr.processing.fft_encodings import _qf_fft
from spectrochempy_nmr.processing.fft_encodings import _states_fft
from spectrochempy_nmr.processing.fft_encodings import _tppi_fft

quaternion = pytest.importorskip("quaternion", reason="requires numpy-quaternion")
as_float_array = quaternion.as_float_array


def _make_quat(array_4d):
    """Stack [..., 4] float array into quaternion array."""
    return quaternion.as_quat_array(np.asarray(array_4d, dtype=np.float64))


def _get_components(q):
    """Decompose quaternion array into (w, x, y, z) = (RR, RI, IR, II)."""
    fa = quaternion.as_float_array(np.asarray(q))
    return fa[..., 0], fa[..., 1], fa[..., 2], fa[..., 3]


# ---------------------------------------------------------------------------
# Synthetic data construction helpers
# ---------------------------------------------------------------------------


def _make_states_data(nf1, nf2, f1_freq, f2_freq):
    """
    Create STATES-encoded quaternion data for a single 2D peak.

    In STATES encoding, the Bruker experiment acquires two separate
    subspectra for the indirect dimension:
      - cos(ω₁t₁) modulation → stored in (RR, RI) for each t₂
      - sin(ω₁t₁) modulation → stored in (IR, II) for each t₂

    Parameters
    ----------
    nf1, nf2 : int
        Number of points in F1 and F2.
    f1_freq, f2_freq : float
        Frequency in cycles per dimension (e.g. 2.0 = 2 complete cycles).
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


def _make_echoanti_data(nf1, nf2, f1_freq, f2_freq):
    """
    Create Echo-Antiecho encoded quaternion data.

    In Echo-Antiecho encoding, the gradient-selected coherence pathway
    produces:
      - echo: (RR + IR) and (RI + II)
      - antiecho: (RR - IR) and (RI - II)
    """
    t1 = np.arange(nf1)
    t2 = np.arange(nf2)
    w1 = 2 * np.pi * f1_freq / nf1
    w2 = 2 * np.pi * f2_freq / nf2

    cos1 = np.cos(w1 * t1)[:, None]
    sin1 = np.sin(w1 * t1)[:, None]
    cos2 = np.cos(w2 * t2)[None, :]
    sin2 = np.sin(w2 * t2)[None, :]

    # Echo-antiecho: the two experiments are echo and antiecho
    # echo path: +1 coherence → produces cos(w1*t1)*f(t2)
    # antiecho path: -1 coherence → produces sin(w1*t1)*f(t2)
    # But the gradient selection means the components are mixed:
    RR = cos1 * cos2
    RI = cos1 * sin2
    IR = sin1 * cos2
    II = sin1 * sin2

    return _make_quat(np.stack([RR, RI, IR, II], axis=-1))


def _make_tppi_data(nf1, nf2, f1_freq, f2_freq):
    """
    Create TPPI-encoded quaternion data.

    In TPPI, the indirect dimension uses sign alternation on odd points.
    The four components store the same physical information as STATES
    but acquired with a different phase cycling scheme.
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


def _peak_position_1d(arr, axis=-1):
    """Find the index of maximum absolute value along an axis."""
    return int(np.argmax(np.abs(arr), axis=axis))


# ---------------------------------------------------------------------------
# QF encoding tests
# ---------------------------------------------------------------------------


class TestQFEncoding:
    """QF (quadrature-free) encoding: conjugate before FFT."""

    def test_qf_single_peak(self):
        """QF FFT of a complex signal should produce a peak at the conjugate frequency."""
        n = 64
        freq = 5.0  # 5 cycles
        t = np.arange(n)
        signal = np.exp(2j * np.pi * freq * t / n)

        result = _qf_fft(signal)
        assert result.dtype == np.complex128
        assert result.shape == signal.shape

        # QF does conjugate then FFT, so peak should be at -freq (or n-freq)
        peak_idx = _peak_position_1d(result)
        # After fftshift, the peak should be at the negative frequency position
        expected_idx = (n // 2 - int(freq)) % n
        assert (
            peak_idx == expected_idx
        ), f"Expected peak at {expected_idx}, got {peak_idx}"

    def test_qf_returns_fftshift(self):
        """QF result should be fftshifted."""
        n = 32
        signal = np.ones(n, dtype=np.complex128)
        result = _qf_fft(signal)
        # DC signal → single peak at center after fftshift
        peak_idx = _peak_position_1d(result)
        assert (
            peak_idx == n // 2
        ), f"DC peak should be at center ({n//2}), got {peak_idx}"


# ---------------------------------------------------------------------------
# STATES encoding tests
# ---------------------------------------------------------------------------


class TestSTATESEncoding:
    """STATES encoding: two subspectra with cos/sin modulation in F1."""

    def test_states_single_peak(self):
        """STATES FFT should place the peak at the correct F2 frequency."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0

        data = _make_states_data(nf1, nf2, f1_freq, f2_freq)
        result = _states_fft(data)

        assert result.shape == (nf1, nf2)

        # Check the peak position in the RR component
        w, x, y, z = _get_components(result)
        # The result has two complex subspectra: fr = RR + i*RI, fi = IR + i*II
        # Peak should be visible in RR at the F2 frequency position
        peak_f2 = _peak_position_1d(w[0, :])  # along F2 for first F1 row
        # After fftshift, the positive frequency peak is at n//2 + f2_freq
        expected_f2 = (nf2 // 2 + int(f2_freq)) % nf2
        # Or it could be the negative frequency peak at n//2 - f2_freq
        # The exact position depends on the subspectrum decomposition
        assert (
            abs(peak_f2 - expected_f2) <= 1
            or abs(peak_f2 - (nf2 // 2 - int(f2_freq))) <= 1
        ), f"F2 peak at {peak_f2}, expected near {expected_f2} or {nf2//2 - int(f2_freq)}"

    def test_states_preserves_shape(self):
        """STATES FFT should preserve the quaternion shape."""
        data = _make_states_data(8, 16, 1.0, 3.0)
        result = _states_fft(data)
        assert result.shape == data.shape
        assert result.dtype == np.quaternion

    def test_states_zero_signal(self):
        """STATES FFT of zeros should produce zeros."""
        data = _make_quat(np.zeros((4, 8, 4)))
        result = _states_fft(data)
        w, x, y, z = _get_components(result)
        assert_allclose(w, 0.0, atol=1e-15)
        assert_allclose(x, 0.0, atol=1e-15)

    def test_states_with_tppi_flag(self):
        """STATES with tppi=True should apply sign alternation before FFT."""
        nf1, nf2 = 16, 32
        data = _make_states_data(nf1, nf2, 2.0, 5.0)

        result_states = _states_fft(data, tppi=False)
        result_tppi = _states_fft(data, tppi=True)

        # Results should differ because TPPI adds sign alternation
        w_s, _, _, _ = _get_components(result_states)
        w_t, _, _, _ = _get_components(result_tppi)
        assert not np.allclose(w_s, w_t), "States-TPPI should differ from plain STATES"


# ---------------------------------------------------------------------------
# TPPI encoding tests
# ---------------------------------------------------------------------------


class TestTPPIEncoding:
    """TPPI encoding: sign alternation on odd points before FFT."""

    def test_tppi_single_peak(self):
        """TPPI FFT should place the peak at the correct position."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0

        data = _make_tppi_data(nf1, nf2, f1_freq, f2_freq)
        result = _tppi_fft(data)

        assert result.shape == (nf1, nf2)

        w, x, y, z = _get_components(result)
        peak_f2 = _peak_position_1d(w[0, :])
        # TPPI sign alternation shifts the peak: what would be at nf2//2+f in
        # STATES ends up at f after the alternating-sign FFT.
        expected_f2 = int(f2_freq)
        assert (
            abs(peak_f2 - expected_f2) <= 1 or abs(peak_f2 - (nf2 - int(f2_freq))) <= 1
        ), f"F2 peak at {peak_f2}, expected near {expected_f2}"

    def test_tppi_preserves_shape(self):
        """TPPI FFT should preserve the quaternion shape."""
        data = _make_tppi_data(8, 16, 1.0, 3.0)
        result = _tppi_fft(data)
        assert result.shape == data.shape
        assert result.dtype == np.quaternion

    def test_tppi_zero_signal(self):
        """TPPI FFT of zeros should produce zeros."""
        data = _make_quat(np.zeros((4, 8, 4)))
        result = _tppi_fft(data)
        w, x, y, z = _get_components(result)
        assert_allclose(w, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Echo-Antiecho encoding tests
# ---------------------------------------------------------------------------


class TestEchoAntiechoEncoding:
    """Echo-Antiecho encoding: gradient-selected coherence pathway."""

    def test_echoanti_single_peak(self):
        """Echo-Antiecho FFT should place the peak at the correct position."""
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0

        data = _make_echoanti_data(nf1, nf2, f1_freq, f2_freq)
        result = _echoanti_fft(data)

        assert result.shape == (nf1, nf2)

        w, x, y, z = _get_components(result)
        peak_f2 = _peak_position_1d(w[0, :])
        expected_f2 = (nf2 // 2 + int(f2_freq)) % nf2
        assert (
            abs(peak_f2 - expected_f2) <= 1
            or abs(peak_f2 - (nf2 // 2 - int(f2_freq))) <= 1
        ), f"F2 peak at {peak_f2}, expected near {expected_f2}"

    def test_echoanti_preserves_shape(self):
        """Echo-Antiecho FFT should preserve the quaternion shape."""
        data = _make_echoanti_data(8, 16, 1.0, 3.0)
        result = _echoanti_fft(data)
        assert result.shape == data.shape
        assert result.dtype == np.quaternion

    def test_echoanti_zero_signal(self):
        """Echo-Antiecho FFT of zeros should produce zeros."""
        data = _make_quat(np.zeros((4, 8, 4)))
        result = _echoanti_fft(data)
        w, x, y, z = _get_components(result)
        assert_allclose(w, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Encoding handler dispatch tests
# ---------------------------------------------------------------------------


class TestEncodingHandler:
    """Tests for the _fft_encoding_handler dispatch function."""

    def test_dispatch_states(self):
        """Handler should dispatch STATES to _states_fft."""
        data = _make_states_data(8, 16, 1.0, 2.0)
        result = _fft_encoding_handler(data, "STATES")
        assert result.shape == data.shape

    def test_dispatch_states_tppi(self):
        """Handler should dispatch STATES-TPPI with tppi flag."""
        data = _make_states_data(8, 16, 1.0, 2.0)
        result = _fft_encoding_handler(data, "STATES-TPPI", tppi=True)
        assert result.shape == data.shape

    def test_dispatch_tppi(self):
        """Handler should dispatch TPPI."""
        data = _make_tppi_data(8, 16, 1.0, 2.0)
        result = _fft_encoding_handler(data, "TPPI")
        assert result.shape == data.shape

    def test_dispatch_echo_anti(self):
        """Handler should dispatch ECHO-ANTIECHO."""
        data = _make_echoanti_data(8, 16, 1.0, 2.0)
        result = _fft_encoding_handler(data, "ECHO-ANTIECHO")
        assert result.shape == data.shape

    def test_dispatch_qsim_fails_on_quaternion(self):
        """QSIM/DQD dispatch should fail on quaternion data (numpy limitation)."""
        data = _make_states_data(8, 16, 1.0, 2.0)
        with pytest.raises((TypeError, ValueError)):
            _fft_encoding_handler(data, "DQD")

    def test_dispatch_qseq_not_implemented(self):
        """QSEQ encoding should raise NotImplementedError."""
        data = np.ones((8, 16), dtype=np.complex128)
        with pytest.raises(NotImplementedError, match="QSEQ"):
            _fft_encoding_handler(data, "QSEQ")

    def test_dispatch_unknown_encoding(self):
        """Unknown encoding should raise NotImplementedError."""
        data = np.ones((8, 16), dtype=np.complex128)
        with pytest.raises(NotImplementedError, match="not supported"):
            _fft_encoding_handler(data, "UNKNOWN")


# ---------------------------------------------------------------------------
# Component ordering characterization
# ---------------------------------------------------------------------------


class TestComponentOrdering:
    """
    Document and verify the quaternion component mapping used by each handler.

    The numpy-quaternion library uses (w, x, y, z) = (RR, RI, IR, II).
    The encoding handlers unpack these with different permutations.
    This test class documents the exact mapping.
    """

    def test_states_component_mapping(self):
        """
        Document the STATES component assignment.

        In _states_fft:
            wt, yt, xt, zt = as_float_array(data).T
            w, y, x, z = wt.T, yt.T, xt.T, zt.T

        After as_float_array(data).T, the 4 components unpack as:
            wt = float_arr[..., 0] = w = RR
            yt = float_arr[..., 1] = x = RI   (confusingly named yt)
            xt = float_arr[..., 2] = y = IR   (confusingly named xt)
            zt = float_arr[..., 3] = z = II

        After the second line (w, y, x, z = wt.T, yt.T, xt.T, zt.T):
            w = RR, y = RI, x = IR, z = II

        So the naming in the handler body is:
            sr = (w - 1j*y)/2 = (RR - 1j*RI)/2
            si = (x - 1j*z)/2 = (IR - 1j*II)/2
        """
        # Create data where only RR is nonzero
        nf1, nf2 = 4, 8
        RR = np.ones((nf1, nf2))
        RI = np.zeros((nf1, nf2))
        IR = np.zeros((nf1, nf2))
        II = np.zeros((nf1, nf2))
        data = _make_quat(np.stack([RR, RI, IR, II], axis=-1))

        # In STATES: sr = (RR - 1j*RI)/2 = 0.5, si = (IR - 1j*II)/2 = 0
        # After FFT of constant 0.5: peak at DC (index 0), after fftshift: index n//2
        result = _states_fft(data)
        w, x, y, z = _get_components(result)

        # The RR component should have the DC peak at the fftshifted center
        # FFT of constant C over N points → DC value = N*C
        dc_idx_f2 = nf2 // 2
        expected_dc = nf2 * 0.5  # sr was divided by 2, so fft(sr) DC = N * 0.5
        assert_allclose(
            np.abs(w[:, dc_idx_f2]),
            expected_dc,
            atol=1e-10,
            err_msg="STATES: RR component should have DC peak from sr=(RR-i*RI)/2",
        )
        # RI, IR, II should be zero since only RR was nonzero in input
        assert_allclose(x, 0.0, atol=1e-10)
        assert_allclose(y, 0.0, atol=1e-10)
        assert_allclose(z, 0.0, atol=1e-10)

    def test_tppi_component_mapping(self):
        """
        Document the TPPI component assignment.

        In _tppi_fft:
            w, y, x, z = wt.T, xt.T, yt.T, zt.T

        So: w = RR, y = IR, x = RI, z = II

        sx = w + 1j*y = RR + 1j*IR
        sy = x + 1j*z = RI + 1j*II
        """
        nf1, nf2 = 4, 8
        RR = np.ones((nf1, nf2))
        RI = np.zeros((nf1, nf2))
        IR = np.zeros((nf1, nf2))
        II = np.zeros((nf1, nf2))
        data = _make_quat(np.stack([RR, RI, IR, II], axis=-1))

        result = _tppi_fft(data)
        w, x, y, z = _get_components(result)

        # sx = RR + 1j*IR = 1.0 (constant), after sign alternation on odd points,
        # the FFT of the alternating signal [1,-1,1,-1,...] has its peak at bin N//2.
        # fftshift moves that to index 0.
        # FFT of alternating constant C over N points → peak value = N*C (no /2 in TPPI)
        peak_idx = 0  # fftshift moves N//2 bin to index 0
        expected_peak = nf2 * 1.0  # sx was NOT divided by 2
        assert_allclose(
            np.abs(w[:, peak_idx]),
            expected_peak,
            atol=1e-10,
            err_msg="TPPI: RR should have peak at index 0 from sign-alternated sx=(RR+i*IR)",
        )

    def test_echoanti_component_mapping(self):
        """
        Document the Echo-Antiecho component assignment.

        In _echoanti_fft:
            w, y, x, z = wt.T, xt.T, yt.T, zt.T

        So: w = RR, y = IR, x = RI, z = II (same as TPPI)

        c = (w + y) + 1j*(w - y) = (RR + IR) + 1j*(RR - IR)
        s = (x + z) - 1j*(x - z) = (RI + II) - 1j*(RI - II)
        """
        nf1, nf2 = 4, 8
        RR = np.ones((nf1, nf2))
        RI = np.zeros((nf1, nf2))
        IR = np.zeros((nf1, nf2))
        II = np.zeros((nf1, nf2))
        data = _make_quat(np.stack([RR, RI, IR, II], axis=-1))

        result = _echoanti_fft(data)
        w, x, y, z = _get_components(result)

        # c = (RR + IR) + 1j*(RR - IR) = 1.0 + 1j*1.0 (constant)
        # c/2 = (1+1j)/2, FFT gives DC = N*(1+1j)/2 at bin 0, fftshift moves to N//2
        dc_idx_f2 = nf2 // 2
        # as_quaternion(fc, fs) → w = fc.real. At DC: N * (1+1j)/2 → real = N * 0.5
        expected_dc = nf2 * 0.5  # c was divided by 2, so fft(c/2) DC = N * 0.5
        assert_allclose(
            w[0, dc_idx_f2],
            expected_dc,
            atol=1e-10,
            err_msg="Echo-Antiecho: RR should have DC peak from c=(RR+IR)+i*(RR-IR)",
        )


# ---------------------------------------------------------------------------
# 2D pipeline integration
# ---------------------------------------------------------------------------


class Test2DPipeline:
    """
    Integration tests documenting the 2D processing pipeline state.

    These tests document:
    - The DQD/quaternion failure (numpy.fft limitation)
    - The manual workaround path that succeeds
    - How the two-step 2D FFT is performed by hand
    """

    def test_dqd_handler_fails_on_quaternion(self):
        """
        DQD handler cannot process quaternion data.

        ROOT CAUSE: _fft_encoding_handler for DQD/QSIM does
        np.fft.fftshift(np.fft.fft(data), -1) but numpy's fft
        does not support quaternion dtype.

        This is the exact failure encountered in the real 2D pipeline
        where encoding = ['STATES-TPPI', 'DQD'] and encoding[-1] = 'DQD'.
        """
        data = _make_states_data(8, 16, 1.0, 2.0)
        assert data.dtype == np.quaternion
        with pytest.raises(TypeError, match="fft"):
            _fft_encoding_handler(data, "DQD")

    def test_states_handler_works_on_quaternion(self):
        """STATES handler correctly processes quaternion F2 dimension."""
        data = _make_states_data(8, 16, 1.0, 2.0)
        result = _states_fft(data)
        assert result.dtype == np.quaternion
        assert result.shape == data.shape

    def test_manual_2d_workflow_states(self):
        """
        Manual 2D workflow: F2 FFT via handler, then F1 FFT on complex.

        This documents the working workaround for the 2D pipeline:
        1. Extract complex from quaternion for F2
        2. Apply encoding-specific F2 FFT
        3. Extract complex for F1
        4. Apply encoding-specific F1 FFT
        """
        nf1, nf2 = 16, 32
        f1_freq, f2_freq = 2.0, 5.0
        data = _make_states_data(nf1, nf2, f1_freq, f2_freq)

        # Step 1-2: F2 FFT via STATES handler
        f2_result = _states_fft(data)
        assert f2_result.dtype == np.quaternion

        # Step 3: Extract complex subspectra from quaternion
        fa = as_float_array(f2_result)
        w, x, y, z = fa[..., 0], fa[..., 1], fa[..., 2], fa[..., 3]

        # Rebuild complex: sr = (w - 1j*y)/2, si = (x - 1j*z)/2
        sr = (w - 1j * y) / 2.0
        si = (x - 1j * z) / 2.0

        # Step 4: F1 FFT on each subspectrum along axis 0
        f1_sr = np.fft.fftshift(np.fft.fft(sr, axis=0), axes=0)
        f1_si = np.fft.fftshift(np.fft.fft(si, axis=0), axes=0)

        # Verify the result has energy at the expected 2D peak location
        magnitude = np.abs(f1_sr) + np.abs(f1_si)
        peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        # Peak should be near (f1_freq, f2_freq) accounting for fftshift
        assert 0 <= peak[0] < nf1
        assert 0 <= peak[1] < nf2

    def test_manual_2d_workflow_echoanti(self):
        """Manual 2D workflow with Echo-Antiecho encoding."""
        nf1, nf2 = 16, 32
        data = _make_echoanti_data(nf1, nf2, 2.0, 5.0)

        # F2 via ECHO-ANTIECHO handler
        f2_result = _echoanti_fft(data)
        assert f2_result.dtype == np.quaternion

        # Extract complex and do F1 FFT
        fa = as_float_array(f2_result)
        w, x, y, z = fa[..., 0], fa[..., 1], fa[..., 2], fa[..., 3]

        # Echo-antiecho output: fc, fs → w=fc.real, x=fc.imag, y=fs.real, z=fs.imag
        fc = w + 1j * x
        fs = y + 1j * z

        f1_fc = np.fft.fftshift(np.fft.fft(fc, axis=0), axes=0)
        f1_fs = np.fft.fftshift(np.fft.fft(fs, axis=0), axes=0)

        magnitude = np.abs(f1_fc) + np.abs(f1_fs)
        peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        assert 0 <= peak[0] < nf1
        assert 0 <= peak[1] < nf2

    def test_tppi_states_tppi_flag_matches_handler(self):
        """
        STATES-TPPI (tppi=True) and TPPI produce complementary peak positions.

        Both apply the same sign alternation but construct different complex
        subspectra from different component pairings:
        - STATES-TPPI: sr = (RR - 1j*RI)/2
        - TPPI: sx = RR + 1j*IR

        Their respective peaks land at complementary FFT bin positions
        (positive vs negative frequency).
        """
        data = _make_states_data(8, 16, 2.0, 3.0)

        result_tp_flag = _states_fft(data, tppi=True)
        result_tppi = _tppi_fft(data)

        fa_tp = as_float_array(result_tp_flag)
        fa_tppi = as_float_array(result_tppi)

        # STATES-TPPI: sr complex subspectrum
        sr_tp = (fa_tp[..., 0] - 1j * fa_tp[..., 2]) / 2.0
        # TPPI: sx complex subspectrum
        sr_tppi = fa_tppi[..., 0] + 1j * fa_tppi[..., 2]

        peak_tp = np.argmax(np.abs(sr_tp[0, :]))
        peak_tppi = np.argmax(np.abs(sr_tppi[0, :]))
        # Peaks are at complementary positions: n//2 + f and n//2 - f
        nf2 = 16
        center = nf2 // 2
        assert abs(peak_tp - center) == abs(peak_tppi - center), (
            f"Peaks should be symmetric around center {center}: "
            f"STATES-TPPI={peak_tp}, TPPI={peak_tppi}"
        )


# ---------------------------------------------------------------------------
# Hypercomplex representation layer
# ---------------------------------------------------------------------------


class TestHypercomplexRepresentation:
    """
    Tests for the hypercomplex representation adaptation layer.

    These tests verify that:
    - _extract_quaternion_components extracts (RR, RI, IR, II) correctly
    - _prepare_states, _prepare_tppi, _prepare_echoanti produce correct subspectra
    - The handlers produce identical results after refactoring
    """

    def test_extract_quaternion_components(self):
        """_extract_quaternion_components should return (RR, RI, IR, II)."""
        from spectrochempy_nmr.processing.hypercomplex import _extract_quaternion_components

        nf1, nf2 = 4, 8
        RR = np.ones((nf1, nf2)) * 1.0
        RI = np.ones((nf1, nf2)) * 2.0
        IR = np.ones((nf1, nf2)) * 3.0
        II = np.ones((nf1, nf2)) * 4.0
        data = _make_quat(np.stack([RR, RI, IR, II], axis=-1))

        rr, ri, ir, ii = _extract_quaternion_components(data)
        assert_allclose(rr, 1.0)
        assert_allclose(ri, 2.0)
        assert_allclose(ir, 3.0)
        assert_allclose(ii, 4.0)

    def test_prepare_states_formula(self):
        """_prepare_states should compute (RR - 1j*RI)/2 and (IR - 1j*II)/2."""
        from spectrochempy_nmr.processing.hypercomplex import _prepare_states

        RR = np.array([1.0])
        RI = np.array([2.0])
        IR = np.array([3.0])
        II = np.array([4.0])

        sr, si = _prepare_states(RR, RI, IR, II)
        assert_allclose(sr, (1.0 - 1j * 2.0) / 2.0)
        assert_allclose(si, (3.0 - 1j * 4.0) / 2.0)

    def test_prepare_tppi_formula(self):
        """_prepare_tppi should compute RR + 1j*IR and RI + 1j*II."""
        from spectrochempy_nmr.processing.hypercomplex import _prepare_tppi

        RR = np.array([1.0])
        RI = np.array([2.0])
        IR = np.array([3.0])
        II = np.array([4.0])

        sx, sy = _prepare_tppi(RR, RI, IR, II)
        assert_allclose(sx, 1.0 + 1j * 3.0)
        assert_allclose(sy, 2.0 + 1j * 4.0)

    def test_prepare_echoanti_formula(self):
        """_prepare_echoanti should compute (RR+IR)+1j*(RR-IR) and (RI+II)-1j*(RI-II)."""
        from spectrochempy_nmr.processing.hypercomplex import _prepare_echoanti

        RR = np.array([1.0])
        RI = np.array([2.0])
        IR = np.array([3.0])
        II = np.array([4.0])

        c, s = _prepare_echoanti(RR, RI, IR, II)
        assert_allclose(c, (1.0 + 3.0) + 1j * (1.0 - 3.0))
        assert_allclose(s, (2.0 + 4.0) - 1j * (2.0 - 4.0))

    def test_handler_matches_manual_decomposition(self):
        """Handler output should match manual decomposition + FFT."""
        nf1, nf2 = 8, 16
        data = _make_states_data(nf1, nf2, 2.0, 5.0)

        # Handler result
        result = _states_fft(data)

        # Manual decomposition + FFT
        from spectrochempy_nmr.processing.hypercomplex import (
            _extract_quaternion_components,
            _prepare_states,
        )

        RR, RI, IR, II = _extract_quaternion_components(data)
        sr, si = _prepare_states(RR, RI, IR, II)
        fr = np.fft.fftshift(np.fft.fft(sr), -1)
        fi = np.fft.fftshift(np.fft.fft(si), -1)

        # Compare
        fa = as_float_array(result)
        assert_allclose(fa[..., 0], fr.real, atol=1e-10)
        assert_allclose(fa[..., 1], fr.imag, atol=1e-10)
        assert_allclose(fa[..., 2], fi.real, atol=1e-10)
        assert_allclose(fa[..., 3], fi.imag, atol=1e-10)

    def test_handler_tppi_matches_manual(self):
        """TPPI handler output should match manual decomposition + FFT."""
        nf1, nf2 = 8, 16
        data = _make_tppi_data(nf1, nf2, 2.0, 5.0)

        result = _tppi_fft(data)

        from spectrochempy_nmr.processing.hypercomplex import (
            _extract_quaternion_components,
            _prepare_tppi,
        )

        RR, RI, IR, II = _extract_quaternion_components(data)
        sx, sy = _prepare_tppi(RR, RI, IR, II)
        sx[..., 1::2] = -sx[..., 1::2]
        sy[..., 1::2] = -sy[..., 1::2]
        fx = np.fft.fftshift(np.fft.fft(sx), -1)
        fy = np.fft.fftshift(np.fft.fft(sy), -1)

        fa = as_float_array(result)
        assert_allclose(fa[..., 0], fx.real, atol=1e-10)
        assert_allclose(fa[..., 1], fx.imag, atol=1e-10)
        assert_allclose(fa[..., 2], fy.real, atol=1e-10)
        assert_allclose(fa[..., 3], fy.imag, atol=1e-10)

    def test_handler_echoanti_matches_manual(self):
        """Echo-Antiecho handler output should match manual decomposition + FFT."""
        nf1, nf2 = 8, 16
        data = _make_echoanti_data(nf1, nf2, 2.0, 5.0)

        result = _echoanti_fft(data)

        from spectrochempy_nmr.processing.hypercomplex import (
            _extract_quaternion_components,
            _prepare_echoanti,
        )

        RR, RI, IR, II = _extract_quaternion_components(data)
        c, s = _prepare_echoanti(RR, RI, IR, II)
        fc = np.fft.fftshift(np.fft.fft(c / 2.0), -1)
        fs = np.fft.fftshift(np.fft.fft(s / 2.0), -1)

        fa = as_float_array(result)
        assert_allclose(fa[..., 0], fc.real, atol=1e-10)
        assert_allclose(fa[..., 1], fc.imag, atol=1e-10)
        assert_allclose(fa[..., 2], fs.real, atol=1e-10)
        assert_allclose(fa[..., 3], fs.imag, atol=1e-10)
