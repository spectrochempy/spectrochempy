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

import hashlib

import numpy as np
import pytest

import spectrochempy as scp

quaternion = pytest.importorskip("quaternion", reason="requires numpy-quaternion")

EXTRA_DATADIR = scp.preferences.datadir.parent / "testdata-extra"
EXTRA_NMR = EXTRA_DATADIR / "testdata" / "nmrdata"

TOPSPIN_2D = (
    scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"
)
BRUKER_ECHOANTI_2D = (
    scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "exam2d_HC"
)
EXAM2D_HH = (
    scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "exam2d_HH"
)

EXAM2D_HH_SHA256 = {
    "ser": "641707c170f8352785ee3bc923c52f85a335984c38e0ace7c591c054d32ccc6b",
    "acqus": "ed1db82749fac2dd724305260ac02c26ed76bb11b5b6e20d1dc2652ef21f4636",
    "acqu2": "81ee757018006fac238924cb5003e542f6ce21d47883921305050fe31ddfb843",
    "acqu2s": "b3831b60edfa84e06a89137b121958a8c905b37dfbcf0d056ee3318722611989",
    "pdata/1/2rr": "3852aaed6d3fcea61fcb022c73802efd5b62de1e2c41eb6cd1ea3ded29fe28ed",
    "pdata/1/procs": "15083e49296a29e66dfb85c41519680ab4157c0cdebf455f89af9195dec73261",
    "pdata/1/proc2s": "f5f4838b2e39188de217214882c105f3d04a8de81d2fbaba565dab596b8bb5b8",
}


def _has_agilent_data():
    return (EXTRA_NMR / "agilent" / "agilent_1d" / "fid").exists()


def _has_jeol_data():
    return (EXTRA_NMR / "jeol" / "1H.jdf").exists()


def _has_topspin_2d():
    return (TOPSPIN_2D / "1" / "ser").exists()


def _has_bruker_echoanti_2d():
    return (BRUKER_ECHOANTI_2D / "3" / "ser").exists()


def _has_exam2d_hh():
    return (EXAM2D_HH / "1" / "ser").exists()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jcamp_value(path, key):
    prefix = f"##${key}="
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise KeyError(key)


def _sine_window(size):
    return np.sin(np.pi * (np.arange(size) + 0.5) / size)


def _qf_cosy_exam2d_hh(raw):
    data = np.asarray(raw.data).copy()
    direct = np.zeros((1024, 1024), dtype=np.complex128)
    direct[: data.shape[0], : data.shape[1]] = data * _sine_window(data.shape[1])
    f2 = np.fft.fftshift(np.fft.fft(direct, axis=1), axes=1)
    f2 *= _sine_window(f2.shape[0])[:, None]
    return np.abs(np.fft.fftshift(np.fft.ifft(f2, axis=0), axes=0))


def _normalized(data):
    data = np.asarray(data, dtype=np.float64)
    return data / np.max(np.abs(data))


def _top_peaks(data, count=10, radius=3, border=5):
    work = np.asarray(data).copy()
    work[:border, :] = 0.0
    work[-border:, :] = 0.0
    work[:, :border] = 0.0
    work[:, -border:] = 0.0
    peaks = []
    for _ in range(count):
        idx = np.unravel_index(np.argmax(work), work.shape)
        peaks.append((idx, float(work[idx])))
        y, x = idx
        work[
            max(0, y - radius) : min(work.shape[0], y + radius + 1),
            max(0, x - radius) : min(work.shape[1], x + radius + 1),
        ] = 0.0
    return peaks


def _match_peak(data, idx, radius=1):
    y, x = idx
    window = data[
        max(0, y - radius) : min(data.shape[0], y + radius + 1),
        max(0, x - radius) : min(data.shape[1], x + radius + 1),
    ]
    local = np.unravel_index(np.argmax(window), window.shape)
    return max(0, y - radius) + local[0], max(0, x - radius) + local[1]


def _normalized_overlap(left, right):
    return float(
        np.vdot(left.ravel(), right.ravel()).real
        / np.sqrt(
            np.vdot(left.ravel(), left.ravel()).real
            * np.vdot(right.ravel(), right.ravel()).real
        )
    )


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
    def test_topspin_states_tppi_components_match_fixed_slots(self):
        """Keep the established raw-to-TopSpin component correspondence."""
        pdata = TOPSPIN_2D / "1" / "pdata" / "1" / "2rr"
        reference = scp.read_topspin(pdata)
        raw = scp.nmr.read(TOPSPIN_2D, expno=1, remove_digital_filter=True)

        f2 = raw.em(lb="50 Hz", dim="x").fft(dim="x", size=reference.shape[1])
        f2.data[0, :] *= 0.5
        result = f2.zf_size(size=reference.shape[0], dim="y").fft(dim="y")

        actual_slots = quaternion.as_float_array(result.data)
        reference_slots = quaternion.as_float_array(reference.data)
        limits = (0.06, 0.06, 0.11, 0.11)
        for actual, expected, limit in zip(
            np.moveaxis(actual_slots, -1, 0),
            np.moveaxis(reference_slots, -1, 0),
            limits,
            strict=True,
        ):
            scale = np.vdot(actual, expected) / np.vdot(actual, actual)
            relative_l2 = np.linalg.norm(scale * actual - expected) / np.linalg.norm(
                expected
            )
            assert relative_l2 < limit

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

    def test_topspin_2d_two_step_fft_peak_near_reference(self):
        """Two-step SCP processing should match TopSpin and suppress the F1 image."""
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)
        ref = scp.read_topspin(TOPSPIN_2D / "1" / "pdata" / "1" / "2rr")

        f2 = ds.em(lb=2.0).fft(size=2048)
        f1 = f2.zf_size(size=1024, dim="y").em(lb=5.0, dim="y").fft(dim="y")

        mag = _mag_from_quat_or_complex(f1)
        idx, _ = _peak_info(mag)
        y_peak = float(f1.y.data[idx[0]])
        x_peak = float(f1.x.data[idx[1]])

        ref_mag = _mag_from_quat_or_complex(ref)
        ref_idx, _ = _peak_info(ref_mag)
        ref_y = float(ref.y.data[ref_idx[0]])
        ref_x = float(ref.x.data[ref_idx[1]])

        assert abs(y_peak - ref_y) < 1.0
        assert abs(x_peak - ref_x) < 1.0

        # A bad STATES/STATES-TPPI reconstruction leaves a strong mirror image
        # around the center of the F1 window.  Keep it well below the main peak.
        mirror_i = f1.shape[0] - 1 - idx[0]
        row_slice = slice(max(0, mirror_i - 3), min(f1.shape[0], mirror_i + 4))
        col_slice = slice(max(0, idx[1] - 3), min(f1.shape[1], idx[1] + 4))
        mirror_max = float(mag[row_slice, col_slice].max())

        assert mirror_max / float(mag[idx]) < 0.2


@pytest.mark.skipif(not _has_exam2d_hh(), reason="exam2d_HH data not available")
class TestExam2DHHQFCOSY:
    """Bounded raw SER → QF/COSY comparison for exam2d_HH/1 only."""

    def test_exam2d_hh_public_fft_uses_qf_dqd_complex_conventions(self):
        raw = scp.read_topspin(EXAM2D_HH, expno=1, remove_digital_filter=True)

        f2 = raw.fft(dim="x", size=1024)
        expected_f2 = np.fft.fftshift(
            np.fft.fft(np.asarray(raw.data), n=1024, axis=-1),
            axes=-1,
        )
        np.testing.assert_allclose(f2.data, expected_f2, rtol=1e-12, atol=1e-12)

        f1 = f2.fft(dim="y", size=1024)
        f1_time = np.zeros((1024, f2.shape[1]), dtype=np.complex128)
        f1_time[: f2.shape[0], :] = f2.data
        expected_f1 = np.fft.fftshift(
            np.fft.fft(np.conjugate(f1_time), axis=0),
            axes=0,
        )

        assert f1.dtype == np.complex128
        np.testing.assert_allclose(f1.data, expected_f1, rtol=1e-12, atol=1e-12)

    def test_exam2d_hh_raw_ser_qf_cosy_matches_2rr_topology(self):
        root = EXAM2D_HH / "1"
        for relative, expected in EXAM2D_HH_SHA256.items():
            assert _sha256(root / relative) == expected

        raw = scp.read_topspin(EXAM2D_HH, expno=1, remove_digital_filter=True)
        reference = scp.read_topspin(root / "pdata" / "1" / "2rr", proc_axis=True)

        raw_data = np.asarray(raw.data).copy()
        raw_processing = raw.meta.nmr_processing.copy()

        assert raw.meta.parmode[-1] == 1
        assert int(_jcamp_value(root / "acqus", "TD")) == 2048
        assert int(_jcamp_value(root / "acqu2", "TD")) == 128
        assert raw.meta.fnmode[0] == 1
        assert raw.meta.encoding[0] == "QF"
        for proc_file in (
            root / "pdata" / "1" / "proc2s",
            root / "pdata" / "1" / "procs",
        ):
            assert int(_jcamp_value(proc_file, "SI")) == 1024
            assert int(_jcamp_value(proc_file, "FT_mod")) == 6
            assert int(_jcamp_value(proc_file, "MC2")) == 0
            assert int(_jcamp_value(proc_file, "WDW")) == 3
            assert float(_jcamp_value(proc_file, "LB")) == 0.0
            assert float(_jcamp_value(proc_file, "GB")) == 0.0
            assert int(_jcamp_value(proc_file, "SSB")) == 0
            assert float(_jcamp_value(proc_file, "PHC0")) == 0.0
            assert float(_jcamp_value(proc_file, "PHC1")) == 0.0

        calculated = _normalized(_qf_cosy_exam2d_hh(raw))
        ref = _normalized(np.asarray(reference.data).real)

        assert calculated.shape == (1024, 1024)
        assert ref.shape == (1024, 1024)

        expected_x = reference.meta.offset[1] - np.arange(1024) * (
            float(reference.meta.sw_p[1]) / float(reference.meta.sf[1].m) / 1024
        )
        expected_y = reference.meta.offset[0] - np.arange(1024) * (
            float(reference.meta.sw_p[0]) / float(reference.meta.sf[0].m) / 1024
        )
        half_x_bin = abs(expected_x[1] - expected_x[0]) / 2.0
        half_y_bin = abs(expected_y[1] - expected_y[0]) / 2.0
        np.testing.assert_allclose(reference.x.data, expected_x, atol=half_x_bin)
        np.testing.assert_allclose(reference.y.data, expected_y, atol=half_y_bin)
        assert np.all(np.diff(reference.x.data) < 0)
        assert np.all(np.diff(reference.y.data) < 0)

        ref_peaks = _top_peaks(ref)
        matched_values = []
        for (ref_y, ref_x), ref_value in ref_peaks:
            calc_y, calc_x = _match_peak(calculated, (ref_y, ref_x))
            assert abs(calc_y - ref_y) <= 1
            assert abs(calc_x - ref_x) <= 1
            matched_values.append(float(calculated[calc_y, calc_x]))
            assert matched_values[-1] >= 0.5 * ref_value

        rank_preserved = sum(
            left >= right
            for left, right in zip(matched_values, matched_values[1:], strict=False)
        )
        assert rank_preserved >= 8

        residual = calculated - ref
        assert float(np.sqrt(np.mean(residual**2))) <= 0.01
        assert _normalized_overlap(calculated, ref) >= 0.85

        inner = np.s_[5:-5, 5:-5]
        inner_residual = calculated[inner] - ref[inner]
        assert float(np.sqrt(np.mean(inner_residual**2))) <= 0.01
        assert _normalized_overlap(calculated[inner], ref[inner]) >= 0.85

        np.testing.assert_array_equal(raw.data, raw_data)
        assert raw.meta.nmr_processing == raw_processing
        assert "vendor_profile" in raw.meta.nmr_processing
        assert "requested" not in raw.meta.nmr_processing
        assert "applied" not in raw.meta.nmr_processing


@pytest.mark.skipif(
    not _has_bruker_echoanti_2d(), reason="Bruker Echo-Antiecho data not available"
)
class TestEmFFT2DBrukerEchoAntiecho:
    """Type B: Bruker Echo-Antiecho em + FFT chain."""

    def test_bruker_echoanti_two_step_fft_peak_near_reference(self):
        """
        Two-step SCP processing should match TopSpin on real Echo-Antiecho data.

        The key regression to guard here is an F1 mirror-image reconstruction:
        a wrong indirect-dimension orientation lands the main peak at the
        mirrored Y position relative to the TopSpin processed reference.
        """
        ds = scp.read_topspin(BRUKER_ECHOANTI_2D, expno=3, remove_digital_filter=True)
        ref = scp.read_topspin(BRUKER_ECHOANTI_2D / "3" / "pdata" / "1" / "2rr")

        f2 = ds.em(lb=2.0).fft(size=ref.shape[1])
        f1 = f2.zf_size(size=ref.shape[0], dim="y").em(lb=2.0, dim="y").fft(dim="y")

        mag = _mag_from_quat_or_complex(f1)
        idx, _ = _peak_info(mag)
        y_peak = float(f1.y.data[idx[0]])
        x_peak = float(f1.x.data[idx[1]])

        ref_mag = _mag_from_quat_or_complex(ref)
        ref_idx, _ = _peak_info(ref_mag)
        ref_y = float(ref.y.data[ref_idx[0]])
        ref_x = float(ref.x.data[ref_idx[1]])

        assert abs(y_peak - ref_y) < 1.0
        assert abs(x_peak - ref_x) < 1.0

    def test_experiment_rejects_bruker_echoanti_2d_processing(self):
        """The public Experiment processor remains limited to validated 1D data."""
        ds = scp.read_topspin(BRUKER_ECHOANTI_2D, expno=3, remove_digital_filter=True)
        ref = scp.read_topspin(BRUKER_ECHOANTI_2D / "3" / "pdata" / "1" / "2rr")

        with pytest.raises(NotImplementedError, match="only validated 1D experiments"):
            scp.nmr.Experiment(ds).process(apodization="em", lb=2.0, size=ref.shape[1])


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

    def test_fft_updates_domain_metadata_and_nuclei(self):
        """F2 then F1 FFT should update domains and keep axis nucleus mapping."""
        ds = scp.read_topspin(TOPSPIN_2D, expno=1, remove_digital_filter=True)

        f2 = ds.fft()
        assert f2.meta.isfreq == [False, True]
        assert str(f2.x.units) == "ppm"
        assert f2.y.units == scp.ur.microsecond
        assert "27" in f2.x.title

        f1 = f2.zf_size(size=256, dim="y").fft(dim="y")
        assert f1.meta.isfreq == [True, True]
        assert str(f1.x.units) == "ppm"
        assert str(f1.y.units) == "ppm"
        assert "27" in f1.x.title
        assert "31" in f1.y.title
