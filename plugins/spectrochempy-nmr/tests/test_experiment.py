# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

"""Tests for scp.nmr.Experiment — NMR-specific scientific model."""

import warnings

import numpy as np
import pytest
from spectrochempy_nmr.experiment import Experiment
from spectrochempy_nmr.experiment import ExperimentValidation

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur

DATADIR = scp.preferences.datadir
NMRDATA = DATADIR / "nmrdata"
nmrdir = NMRDATA / "bruker" / "tests" / "nmr"
EXTRA_DATADIR = scp.preferences.datadir.parent / "testdata-extra"
EXTRA_NMR = EXTRA_DATADIR / "testdata" / "nmrdata"


def _require_path(path):
    if not path.exists():
        pytest.skip(f"NMR test data not available: {path}")
    return path


def _read_or_skip(*args, **kwargs):
    try:
        result = scp.nmr.read(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"NMR test data incomplete: {exc}")
    if result is None:
        pytest.skip("NMR test data could not be read in this environment")
    return result


def _topspin_1d_oracle_metrics(spectrum, ref):
    """Return normalized comparison metrics against the bundled TopSpin oracle."""
    ref_axis = np.asarray(ref.x.data, dtype=float)
    ref_data = np.asarray(ref.data).squeeze()
    calc_axis = np.asarray(spectrum.x.data, dtype=float)
    calc_data = np.asarray(spectrum.data).squeeze()

    calc_axis_descending = bool(calc_axis[0] > calc_axis[-1])

    if ref_axis[0] > ref_axis[-1]:
        ref_axis = ref_axis[::-1]
        ref_data = ref_data[::-1]
    if calc_axis[0] > calc_axis[-1]:
        calc_axis = calc_axis[::-1]
        calc_data = calc_data[::-1]

    interp = np.interp(ref_axis, calc_axis, calc_data.real) + 1j * np.interp(
        ref_axis, calc_axis, calc_data.imag
    )

    amplitude_scale = np.vdot(interp, ref_data) / np.vdot(interp, interp)
    maxabs_ratio = np.max(np.abs(ref_data)) / np.max(np.abs(interp))

    interp_norm = interp / np.max(np.abs(interp))
    ref_norm = ref_data / np.max(np.abs(ref_data))
    residual = interp_norm - ref_norm

    complex_overlap = np.abs(np.vdot(interp_norm, ref_norm)) / (
        np.linalg.norm(interp_norm) * np.linalg.norm(ref_norm)
    )
    real_corr = np.corrcoef(interp_norm.real, ref_norm.real)[0, 1]

    calc_peak_ppm = float(ref_axis[int(np.argmax(np.abs(interp_norm)))])
    ref_peak_ppm = float(ref_axis[int(np.argmax(np.abs(ref_norm)))])

    return {
        "calc_axis_descending": calc_axis_descending,
        "amplitude_scale": amplitude_scale,
        "amplitude_scale_modulus": float(abs(amplitude_scale)),
        "phase_deg": float(np.angle(amplitude_scale, deg=True)),
        "maxabs_ratio": float(maxabs_ratio),
        "complex_overlap": float(complex_overlap),
        "real_corr": float(real_corr),
        "calc_peak_ppm": calc_peak_ppm,
        "ref_peak_ppm": ref_peak_ppm,
        "residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "residual_max": float(np.max(np.abs(residual))),
    }


def _manual_apodize_then_fft(dataset, apodization, *, size=None, **kwargs):
    apodizers = {
        "em": scp.em,
        "gm": scp.gm,
        "sp": scp.sp,
    }
    work = apodizers[apodization](dataset.copy(), inplace=False, **kwargs)
    if size is not None:
        work = work.zf_size(size=size)
    return work.fft()


def _manual_public_process(experiment, dataset, apodization, *, size=None, **kwargs):
    work = _manual_apodize_then_fft(dataset, apodization, size=size, **kwargs)
    return experiment._calibrate_1d_spectral_axis(work)


def _make_synthetic_vendor_fid(
    *,
    npts=64,
    sw_hz=6400.0,
    obs_mhz=400.0,
    freq_hz=500.0,
    offset_ppm=None,
    origin="tecmag",
    nucleus="1H",
):
    dt = 1.0 / sw_hz
    t = np.arange(npts, dtype=float) * dt
    fid = np.exp(2j * np.pi * freq_hz * t)
    coord = Coord(t, units="s", title="F1 acquisition time")
    coord.meta["acquisition_frequency"] = obs_mhz * ur.MHz

    ds = scp.NDDataset(fid, coordset=[coord])
    ds.origin = origin
    ds.meta.readonly = False
    ds.meta.origin = origin
    ds.meta.td = [npts]
    ds.meta.isfreq = [False]
    ds.meta.encoding = ["QSIM"]
    ds.meta.nucleus = [nucleus]
    ds.meta.datatype = "FID"
    ds.meta.iscomplex = [True]
    ds.meta.sw_h = [sw_hz]
    ds.meta.sfrq = [obs_mhz]
    ds.meta.offset = [offset_ppm]
    ds.meta.readonly = True
    return ds


def _has_topspin_1d():
    return (nmrdir / "topspin_1d/1/fid").exists()


def _has_topspin_1d_pdata():
    return (nmrdir / "topspin_1d/1/pdata/1/1r").exists()


def _has_topspin_2d():
    return (nmrdir / "topspin_2d/1/ser").exists()


def _has_topspin_2d_pdata():
    return (nmrdir / "topspin_2d/1/pdata/1/2rr").exists()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Test Experiment construction and input validation."""

    def test_construct_from_single_dataset(self):
        ds = scp.NDDataset(np.zeros(100))
        exp = Experiment(ds)
        assert exp.dataset is ds
        assert not exp.is_multi_dataset

    def test_construct_from_list_of_datasets(self):
        ds1 = scp.NDDataset(np.zeros(100))
        ds2 = scp.NDDataset(np.zeros(200))
        exp = Experiment([ds1, ds2])
        assert exp.dataset is ds1
        assert exp.is_multi_dataset
        assert len(exp.datasets) == 2
        assert exp.datasets[0] is ds1
        assert exp.datasets[1] is ds2

    def test_construct_from_tuple_of_datasets(self):
        ds1 = scp.NDDataset(np.zeros(100))
        ds2 = scp.NDDataset(np.zeros(200))
        exp = Experiment((ds1, ds2))
        assert exp.is_multi_dataset

    def test_construct_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Experiment([])

    def test_construct_non_dataset_raises(self):
        with pytest.raises(TypeError, match="NDDataset"):
            Experiment("not a dataset")

    def test_construct_list_with_non_dataset_raises(self):
        ds = scp.NDDataset(np.zeros(100))
        with pytest.raises(TypeError, match="NDDataset"):
            Experiment([ds, "bad"])

    def test_construct_from_unrelated_dataset(self):
        """Non-NMR NDDataset is accepted (validation warns)."""
        ds = scp.NDDataset(np.arange(100, dtype=float))
        exp = Experiment(ds)
        assert exp.ndim == 1

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_construct_from_real_fid(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        assert exp.dataset is fid
        assert not exp.is_multi_dataset


# ---------------------------------------------------------------------------
# Source identity preservation
# ---------------------------------------------------------------------------


class TestSourceIdentity:
    """Verify that Experiment does not copy or mutate the source dataset."""

    def test_dataset_identity_preserved(self):
        ds = scp.NDDataset(np.zeros(100))
        exp = Experiment(ds)
        assert exp.dataset is ds

    def test_no_source_mutation_on_construction(self):
        ds = scp.NDDataset(np.arange(50, dtype=float))
        original_data = ds.data.copy()
        Experiment(ds)
        np.testing.assert_array_equal(ds.data, original_data)

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_no_source_mutation_on_process(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        original_data = fid.data.copy()
        exp = Experiment(fid)
        exp.process()
        np.testing.assert_array_equal(fid.data, original_data)


# ---------------------------------------------------------------------------
# State classification — 1D
# ---------------------------------------------------------------------------


class TestStateClassification1D:
    """Test domain and source-kind classification for 1D data."""

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_fid_classification(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        assert exp.ndim == 1
        assert exp.domains == ("time",)
        assert exp.domain == "time"
        assert exp.source_kind == "fid"
        assert exp.is_time_domain
        assert not exp.is_frequency_domain
        assert not exp.is_mixed_domain
        assert exp.is_raw
        assert not exp.is_processed
        assert exp.is_processable

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    def test_processed_1d_classification(self):
        spec = _read_or_skip(nmrdir / "topspin_1d/1/pdata/1/1r")
        exp = Experiment(spec)
        assert exp.ndim == 1
        assert exp.domains == ("frequency",)
        assert exp.domain == "frequency"
        assert exp.source_kind == "processed_1d"
        assert not exp.is_time_domain
        assert exp.is_frequency_domain
        assert not exp.is_mixed_domain
        assert not exp.is_raw
        assert exp.is_processed
        assert exp.is_processable


# ---------------------------------------------------------------------------
# State classification — 2D
# ---------------------------------------------------------------------------


class TestStateClassification2D:
    """Test domain and source-kind classification for 2D data."""

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data missing")
    def test_ser_classification(self):
        ser = _read_or_skip(nmrdir / "topspin_2d/1/ser")
        exp = Experiment(ser)
        assert exp.ndim == 2
        assert exp.domains == ("time", "time")
        assert exp.domain == "time"
        assert exp.source_kind == "ser"
        assert exp.is_time_domain
        assert exp.is_raw
        assert exp.is_processable

    @pytest.mark.skipif(not _has_topspin_2d_pdata(), reason="TopSpin 2D pdata missing")
    def test_processed_2d_classification(self):
        spec2d = _read_or_skip(nmrdir / "topspin_2d/1/pdata/1/2rr")
        exp = Experiment(spec2d)
        assert exp.ndim == 2
        assert exp.domains == ("frequency", "frequency")
        assert exp.domain == "frequency"
        assert exp.source_kind == "processed_2d"
        assert exp.is_frequency_domain
        assert exp.is_processed
        assert exp.is_processable


# ---------------------------------------------------------------------------
# Metadata interpretation
# ---------------------------------------------------------------------------


class TestMetadataInterpretation:
    """Test metadata extraction and interpretation."""

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_fid_encoding(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        assert exp.encoding is not None
        assert len(exp.encoding) == 1
        assert exp.encoding[0] in ("QF", "QSIM", "QSEQ", "DQD")

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_fid_nuclei(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        assert exp.nuclei is not None
        assert len(exp.nuclei) == 1
        assert "H" in exp.nuclei[0]

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data missing")
    def test_ser_encoding_2d(self):
        ser = _read_or_skip(nmrdir / "topspin_2d/1/ser")
        exp = Experiment(ser)
        assert exp.encoding is not None
        assert len(exp.encoding) == 2

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data missing")
    def test_ser_nuclei_2d(self):
        ser = _read_or_skip(nmrdir / "topspin_2d/1/ser")
        exp = Experiment(ser)
        assert exp.nuclei is not None
        assert len(exp.nuclei) == 2

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    def test_processed_encoding_is_string(self):
        """Encoding integers are resolved to strings."""
        spec = _read_or_skip(nmrdir / "topspin_1d/1/pdata/1/1r")
        exp = Experiment(spec)
        assert exp.encoding is not None
        for e in exp.encoding:
            assert isinstance(e, str)

    def test_unrelated_dataset_metadata(self):
        """Non-NMR dataset has no NMR metadata."""
        ds = scp.NDDataset(np.arange(100, dtype=float))
        exp = Experiment(ds)
        assert exp.encoding is None
        assert exp.nuclei is None
        assert exp.source_kind == "unknown"
        assert exp.domain == "unknown"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Test validation API."""

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_fid_is_valid(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        report = exp.validate()
        assert isinstance(report, ExperimentValidation)
        assert report.is_valid
        assert len(report.errors) == 0

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    def test_processed_1d_is_valid(self):
        spec = _read_or_skip(nmrdir / "topspin_1d/1/pdata/1/1r")
        exp = Experiment(spec)
        report = exp.validate()
        assert report.is_valid

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data missing")
    def test_ser_has_info(self):
        ser = _read_or_skip(nmrdir / "topspin_2d/1/ser")
        exp = Experiment(ser)
        report = exp.validate()
        assert report.is_valid
        info_text = "\n".join(report.info)
        assert "2D" in info_text

    def test_no_metadata_reports_error(self):
        ds = scp.NDDataset(np.arange(100, dtype=float))
        exp = Experiment(ds)
        report = exp.validate()
        assert not report.is_valid
        assert any("metadata" in e.lower() for e in report.errors)

    def test_validation_repr(self):
        v = ExperimentValidation()
        v.add_info("test info")
        v.add_warning("test warning")
        v.add_error("test error")
        assert not v.is_valid
        text = repr(v)
        assert "test info" in text
        assert "test warning" in text
        assert "test error" in text


# ---------------------------------------------------------------------------
# Processing — time-domain 1D
# ---------------------------------------------------------------------------


class TestProcessTimeDomain:
    """Test state-aware processing of 1D time-domain data."""

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_fid_fft_with_apodization(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        spectrum = exp.process(apodization="em", lb=10.0)
        assert isinstance(spectrum, NDDataset)
        assert spectrum.ndim == 1
        # After FFT, coordinate should be in ppm (frequency domain)
        coord = spectrum.coord(0)
        assert str(coord.units) == "ppm"

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_fid_fft_without_apodization(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        spectrum = exp.process()
        assert isinstance(spectrum, NDDataset)
        assert spectrum.ndim == 1
        assert str(spectrum.coord(0).units) == "ppm"

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_fid_with_zerofilling(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        spectrum = exp.process(size=32768)
        assert spectrum.shape == (32768,)

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_fid_with_manual_phase(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        spectrum = exp.process(phase="manual", phc0=45.0)
        assert isinstance(spectrum, NDDataset)
        assert str(spectrum.coord(0).units) == "ppm"

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_source_unchanged_after_process(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        original_data = fid.data.copy()
        exp = Experiment(fid)
        _ = exp.process(apodization="em", lb=10.0)
        np.testing.assert_array_equal(fid.data, original_data)

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_experiment_unchanged_after_process(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        _ = exp.process(apodization="em", lb=10.0)
        assert exp.is_time_domain  # Experiment itself is unchanged

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_explicit_lb_changes_public_result_and_keeps_inputs_independent(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        source_data = np.asarray(fid.data).copy()

        fid_low = fid.copy()
        fid_high = fid.copy()
        assert not np.shares_memory(np.asarray(fid_low.data), np.asarray(fid.data))
        assert not np.shares_memory(np.asarray(fid_high.data), np.asarray(fid.data))
        assert not np.shares_memory(np.asarray(fid_low.data), np.asarray(fid_high.data))

        low = Experiment(fid_low).process(
            apodization="em",
            lb=2.0,
            size=16384,
            phase=None,
        )
        high = Experiment(fid_high).process(
            apodization="em",
            lb=20000.0,
            size=16384,
            phase=None,
        )

        low_data = np.asarray(low.data)
        high_data = np.asarray(high.data)

        assert not np.array_equal(low_data, high_data)
        assert not np.allclose(low_data, high_data)
        assert np.max(np.abs(low_data - high_data)) > 1.0e5
        assert np.linalg.norm(low_data - high_data) / np.linalg.norm(low_data) > 0.5
        assert np.max(np.abs(low_data)) > 1.0e5
        assert np.max(np.abs(high_data)) < 2.0e4

        np.testing.assert_array_equal(fid.data, source_data)
        np.testing.assert_array_equal(fid_low.data, source_data)
        np.testing.assert_array_equal(fid_high.data, source_data)

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_public_em_processing_matches_manual_apodize_then_fft(self):
        from spectrochempy.processing.fft.zero_filling import zf_size

        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        lb = 20000.0
        size = 16384

        exp = Experiment(fid.copy())
        public = exp.process(apodization="em", lb=lb, size=size, phase=None)

        direct_apodized = scp.em(fid.copy(), lb=lb, inplace=False)
        internal_apodized = exp._apply_apodization(fid.copy(), "em", lb=lb)

        np.testing.assert_allclose(
            np.asarray(internal_apodized.data),
            np.asarray(direct_apodized.data),
            atol=1.0e-12,
        )

        manual = zf_size(internal_apodized, size=size).fft()

        np.testing.assert_allclose(
            np.asarray(public.data),
            np.asarray(manual.data),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(public.x.data),
            np.asarray(manual.x.data),
            atol=1.0e-12,
        )
        assert str(public.x.units) == str(manual.x.units)
        np.testing.assert_array_equal(np.asarray(public.mask), np.asarray(manual.mask))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lb": 2.0}, "without apodization"),
            ({"apodization": "em", "gb": 0.5}, "does not accept parameter"),
            ({"apodization": "gm", "ssb": 2.0}, "does not accept parameter"),
            ({"apodization": "sp", "lb": 2.0}, "does not accept parameter"),
            ({"apodization": "sp", "pow": 3}, "must be 1 or 2"),
            ({"apodization": "sp", "ssb": 0.0}, "must be strictly positive"),
            (
                {"apodization": "gm", "gb": 1.0 * ur.s},
                "compatible with Hz",
            ),
        ],
    )
    def test_invalid_apodization_argument_combinations_raise(self, kwargs, match):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        with pytest.raises((TypeError, ValueError), match=match):
            exp.process(**kwargs)

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"apodization": "em", "lb": 2.0},
            {"apodization": "gm", "lb": 2.0, "gb": 1.0},
            {"apodization": "sp", "ssb": 2.0, "pow": 2},
        ],
    )
    def test_frequency_domain_rejects_explicit_apodization_requests(self, kwargs):
        spectrum = _read_or_skip(nmrdir / "topspin_1d", expno=1, procno=1)
        exp = Experiment(spectrum)

        with pytest.raises(
            RuntimeError,
            match="Frequency-domain datasets cannot accept apodization requests",
        ):
            exp.process(**kwargs)

    def test_gm_process_matches_manual_apodize_then_fft_on_synthetic_fid(self):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)
        public = Experiment(ds.copy()).process(
            apodization="gm",
            lb=-2.0,
            gb=4.0,
            size=128,
            phase=None,
        )
        manual = _manual_public_process(
            Experiment(ds.copy()),
            ds,
            "gm",
            lb=-2.0,
            gb=4.0,
            size=128,
        )

        np.testing.assert_allclose(public.data, manual.data, atol=1.0e-12)
        np.testing.assert_allclose(public.x.data, manual.x.data, atol=1.0e-12)
        assert str(public.x.units) == str(manual.x.units)
        np.testing.assert_array_equal(np.asarray(public.mask), np.asarray(manual.mask))

    def test_sp_process_matches_manual_apodize_then_fft_on_synthetic_fid(self):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)
        public = Experiment(ds.copy()).process(
            apodization="sp",
            ssb=2.0,
            pow=2,
            size=128,
            phase=None,
        )
        manual = _manual_public_process(
            Experiment(ds.copy()),
            ds,
            "sp",
            ssb=2.0,
            pow=2,
            size=128,
        )

        np.testing.assert_allclose(public.data, manual.data, atol=1.0e-12)
        np.testing.assert_allclose(public.x.data, manual.x.data, atol=1.0e-12)
        assert str(public.x.units) == str(manual.x.units)
        np.testing.assert_array_equal(np.asarray(public.mask), np.asarray(manual.mask))

    def test_gm_parameter_defaults_match_core_defaults_on_synthetic_fid(self):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)
        public = Experiment(ds.copy()).process(
            apodization="gm",
            size=128,
            phase=None,
        )
        manual = _manual_public_process(Experiment(ds.copy()), ds, "gm", size=128)

        np.testing.assert_allclose(public.data, manual.data, atol=1.0e-12)
        np.testing.assert_allclose(public.x.data, manual.x.data, atol=1.0e-12)

    def test_sp_parameter_defaults_match_core_defaults_on_synthetic_fid(self):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)
        public = Experiment(ds.copy()).process(
            apodization="sp",
            size=128,
            phase=None,
        )
        manual = _manual_public_process(Experiment(ds.copy()), ds, "sp", size=128)

        np.testing.assert_allclose(public.data, manual.data, atol=1.0e-12)
        np.testing.assert_allclose(public.x.data, manual.x.data, atol=1.0e-12)

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_real_fid_gm_gb_changes_public_result_without_mutating_source(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        source_data = np.asarray(fid.data).copy()

        low = Experiment(fid.copy()).process(
            apodization="gm",
            lb=-2.0,
            gb=1.0,
            size=16384,
            phase=None,
        )
        high = Experiment(fid.copy()).process(
            apodization="gm",
            lb=-2.0,
            gb=8.0,
            size=16384,
            phase=None,
        )

        assert not np.allclose(np.asarray(low.data), np.asarray(high.data))
        np.testing.assert_array_equal(fid.data, source_data)

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_real_fid_sp_ssb_and_pow_change_public_result_without_mutating_source(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        source_data = np.asarray(fid.data).copy()

        ssb_low = Experiment(fid.copy()).process(
            apodization="sp",
            ssb=1.0,
            pow=1,
            size=16384,
            phase=None,
        )
        ssb_high = Experiment(fid.copy()).process(
            apodization="sp",
            ssb=4.0,
            pow=1,
            size=16384,
            phase=None,
        )
        pow_high = Experiment(fid.copy()).process(
            apodization="sp",
            ssb=4.0,
            pow=2,
            size=16384,
            phase=None,
        )

        assert not np.allclose(np.asarray(ssb_low.data), np.asarray(ssb_high.data))
        assert not np.allclose(np.asarray(ssb_high.data), np.asarray(pow_high.data))
        np.testing.assert_array_equal(fid.data, source_data)

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_unknown_apodization_on_real_fid_raises(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        with pytest.raises(ValueError, match="Unknown apodization"):
            exp.process(apodization="bad_func")

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_unknown_phase_mode_on_real_fid_raises(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        with pytest.raises(ValueError, match="Unknown phase mode"):
            exp.process(phase="bad_mode")

    @pytest.mark.skipif(
        not (EXTRA_NMR / "agilent" / "agilent_1d" / "fid").exists(),
        reason="Agilent 1D data missing",
    )
    def test_agilent_1d_pipeline_calibrates_to_ppm(self):
        fid = _read_or_skip(EXTRA_NMR / "agilent" / "agilent_1d" / "fid")
        exp = Experiment(fid)
        spectrum = exp.process()
        assert exp.encoding == ("QSIM",)
        assert str(spectrum.coord(0).units) == "ppm"

    @pytest.mark.skipif(
        not (EXTRA_NMR / "jeol" / "1H.jdf").exists(),
        reason="JEOL 1D data missing",
    )
    def test_jeol_1d_pipeline_uses_direct_complex_encoding(self):
        fid = _read_or_skip(EXTRA_NMR / "jeol" / "1H.jdf")
        exp = Experiment(fid)
        assert exp.encoding == ("QSIM",)
        spectrum = exp.process()
        assert str(spectrum.coord(0).units) == "ppm"

    @pytest.mark.skipif(
        not (EXTRA_NMR / "tecmag" / "LiCl_ref1.tnt").exists(),
        reason="TecMag 1D data missing",
    )
    def test_tecmag_1d_pipeline_uses_direct_complex_encoding(self):
        fid = _read_or_skip(EXTRA_NMR / "tecmag" / "LiCl_ref1.tnt")
        exp = Experiment(fid)
        assert exp.encoding == ("QSIM",)
        spectrum = exp.process()
        assert str(spectrum.coord(0).units) == "ppm"


class TestPublic1DMathConventions:
    """Numerically characterize the public 1D FFT and axis conventions."""

    def test_positive_frequency_peak_appears_on_positive_ppm_side(self):
        ds = _make_synthetic_vendor_fid(freq_hz=500.0)
        spectrum = Experiment(ds).process()
        axis = np.asarray(spectrum.x.data)
        peak_idx = int(np.argmax(np.abs(np.asarray(spectrum.data))))

        assert str(spectrum.x.units) == "ppm"
        assert axis[0] > axis[-1]
        assert axis[peak_idx] > 0.0

    def test_negative_frequency_peak_appears_on_negative_ppm_side(self):
        ds = _make_synthetic_vendor_fid(freq_hz=-500.0)
        spectrum = Experiment(ds).process()
        axis = np.asarray(spectrum.x.data)
        peak_idx = int(np.argmax(np.abs(np.asarray(spectrum.data))))

        assert axis[peak_idx] < 0.0

    def test_zero_frequency_peak_stays_near_axis_center(self):
        ds = _make_synthetic_vendor_fid(freq_hz=0.0)
        spectrum = Experiment(ds).process()
        axis = np.asarray(spectrum.x.data)
        peak_idx = int(np.argmax(np.abs(np.asarray(spectrum.data))))

        assert peak_idx in (spectrum.shape[0] // 2 - 1, spectrum.shape[0] // 2)
        assert abs(axis[peak_idx]) <= abs(axis[0] - axis[-1]) / spectrum.shape[0]

    def test_zero_filling_preserves_peak_position(self):
        ds = _make_synthetic_vendor_fid(freq_hz=500.0)
        exp = Experiment(ds)
        base = exp.process()
        zfilled = exp.process(size=256)

        base_peak = float(
            np.asarray(base.x.data)[int(np.argmax(np.abs(np.asarray(base.data))))]
        )
        zf_peak = float(
            np.asarray(zfilled.x.data)[int(np.argmax(np.abs(np.asarray(zfilled.data))))]
        )

        assert abs(zf_peak - base_peak) < 0.1

    def test_vendor_offset_centers_axis_when_available(self):
        ds = _make_synthetic_vendor_fid(freq_hz=0.0, origin="jeol", offset_ppm=7.0)
        spectrum = Experiment(ds).process()
        axis = np.asarray(spectrum.x.data)
        center_ppm = (float(axis[0]) + float(axis[-1])) / 2.0

        assert center_ppm == pytest.approx(7.0, abs=0.05)

    def test_axis_is_convertible_back_to_hz_with_same_orientation(self):
        ds = _make_synthetic_vendor_fid(freq_hz=500.0)
        spectrum = Experiment(ds).process()
        hz = spectrum.x.to("Hz")

        assert str(hz.units) == "Hz"
        assert float(hz.data[0]) > float(hz.data[-1])


class TestPublic1DRealAxisValidation:
    """Validate final 1D spectral-axis calibration on real vendor data."""

    @pytest.mark.skipif(
        not (EXTRA_NMR / "agilent" / "agilent_1d" / "fid").exists(),
        reason="Agilent 1D data missing",
    )
    def test_agilent_1d_axis_is_centered_when_no_vendor_offset_exists(self):
        spectrum = Experiment(
            _read_or_skip(EXTRA_NMR / "agilent" / "agilent_1d" / "fid")
        ).process()
        axis = np.asarray(spectrum.x.data)
        peak_idx = int(np.argmax(np.abs(np.asarray(spectrum.data))))
        center_ppm = (float(axis[0]) + float(axis[-1])) / 2.0

        assert str(spectrum.x.units) == "ppm"
        assert center_ppm == pytest.approx(0.0, abs=0.05)
        assert 0 < peak_idx < spectrum.shape[0] - 1

    @pytest.mark.skipif(
        not (EXTRA_NMR / "jeol" / "1H.jdf").exists(),
        reason="JEOL 1H data missing",
    )
    def test_jeol_1h_axis_respects_vendor_offset(self):
        fid = _read_or_skip(EXTRA_NMR / "jeol" / "1H.jdf")
        spectrum = Experiment(fid).process()
        axis = np.asarray(spectrum.x.data)
        center_ppm = (float(axis[0]) + float(axis[-1])) / 2.0

        assert center_ppm == pytest.approx(float(fid.meta.offset[0]), abs=0.05)

    @pytest.mark.skipif(
        not (EXTRA_NMR / "jeol" / "13C.jdf").exists(),
        reason="JEOL 13C data missing",
    )
    def test_jeol_13c_axis_respects_vendor_offset(self):
        fid = _read_or_skip(EXTRA_NMR / "jeol" / "13C.jdf")
        spectrum = Experiment(fid).process()
        axis = np.asarray(spectrum.x.data)
        center_ppm = (float(axis[0]) + float(axis[-1])) / 2.0

        assert center_ppm == pytest.approx(float(fid.meta.offset[0]), abs=0.05)

    @pytest.mark.skipif(
        not (EXTRA_NMR / "tecmag" / "LiCl_ref1.tnt").exists(),
        reason="TecMag 1D data missing",
    )
    def test_tecmag_reference_peak_remains_near_zero_ppm(self):
        spectrum = Experiment(
            _read_or_skip(EXTRA_NMR / "tecmag" / "LiCl_ref1.tnt")
        ).process()
        axis = np.asarray(spectrum.x.data)
        peak_ppm = float(axis[int(np.argmax(np.abs(np.asarray(spectrum.data))))])
        center_ppm = (float(axis[0]) + float(axis[-1])) / 2.0

        assert center_ppm == pytest.approx(0.0, abs=0.05)
        assert peak_ppm == pytest.approx(0.0, abs=0.5)

    @pytest.mark.skipif(
        not (_has_topspin_1d() and _has_topspin_1d_pdata()),
        reason="TopSpin 1D raw/pdata pair missing",
    )
    def test_topspin_raw_process_matches_topspin_reference_oracle(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        ref = _read_or_skip(nmrdir / "topspin_1d", expno=1, procno=1)

        spectrum = Experiment(fid).process(size=int(ref.meta.si[0]), phase=None)
        metrics = _topspin_1d_oracle_metrics(spectrum, ref)

        assert str(spectrum.x.units) == "ppm"
        assert spectrum.x.linear
        assert metrics["calc_axis_descending"]
        assert metrics["calc_peak_ppm"] == pytest.approx(
            metrics["ref_peak_ppm"], abs=0.05
        )
        assert metrics["amplitude_scale_modulus"] == pytest.approx(1.0, abs=0.01)
        assert metrics["phase_deg"] == pytest.approx(0.0, abs=0.1)
        assert metrics["maxabs_ratio"] == pytest.approx(1.0, abs=0.01)
        assert metrics["complex_overlap"] > 0.999
        assert metrics["real_corr"] > 0.999
        assert metrics["residual_rms"] < 0.002
        assert metrics["residual_max"] < 0.005

    @pytest.mark.skipif(
        not (_has_topspin_1d() and _has_topspin_1d_pdata()),
        reason="TopSpin 1D raw/pdata pair missing",
    )
    def test_topspin_reference_oracle_metadata_phase_is_neutral_for_this_oracle(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        ref = _read_or_skip(nmrdir / "topspin_1d", expno=1, procno=1)
        size = int(ref.meta.si[0])

        unphased = Experiment(fid).process(size=size, phase=None)
        metadata_phased = Experiment(fid).process(size=size, phase="metadata")

        metrics_none = _topspin_1d_oracle_metrics(unphased, ref)
        metrics_metadata = _topspin_1d_oracle_metrics(metadata_phased, ref)

        assert metrics_none["complex_overlap"] == pytest.approx(
            metrics_metadata["complex_overlap"], abs=1.0e-12
        )
        assert metrics_none["real_corr"] == pytest.approx(
            metrics_metadata["real_corr"], abs=1.0e-12
        )
        assert metrics_none["residual_rms"] == pytest.approx(
            metrics_metadata["residual_rms"], abs=1.0e-12
        )
        assert metrics_none["residual_max"] == pytest.approx(
            metrics_metadata["residual_max"], abs=1.0e-12
        )
        assert metrics_none["phase_deg"] == pytest.approx(
            metrics_metadata["phase_deg"], abs=1.0e-12
        )

    @pytest.mark.skipif(
        not (_has_topspin_1d() and _has_topspin_1d_pdata()),
        reason="TopSpin 1D raw/pdata pair missing",
    )
    def test_topspin_reference_oracle_is_sensitive_to_historical_conventions(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        ref = _read_or_skip(nmrdir / "topspin_1d", expno=1, procno=1)
        size = int(ref.meta.si[0])

        baseline = _topspin_1d_oracle_metrics(
            Experiment(fid).process(size=size, phase=None), ref
        )

        rotated = fid.copy()
        rotated._data = np.asarray(fid.data) * np.exp(-1j * np.pi / 2.0)
        rotated_metrics = _topspin_1d_oracle_metrics(
            Experiment(rotated).process(size=size, phase=None), ref
        )

        conjugated = fid.copy()
        conjugated._data = np.conj(np.asarray(fid.data))
        conjugated_metrics = _topspin_1d_oracle_metrics(
            Experiment(conjugated).process(size=size, phase=None), ref
        )

        assert baseline["residual_rms"] < rotated_metrics["residual_rms"]
        assert baseline["residual_rms"] < conjugated_metrics["residual_rms"]
        assert baseline["residual_max"] < rotated_metrics["residual_max"]
        assert baseline["residual_max"] < conjugated_metrics["residual_max"]
        assert baseline["real_corr"] > rotated_metrics["real_corr"]
        assert baseline["real_corr"] > conjugated_metrics["real_corr"]
        assert abs(baseline["phase_deg"]) < abs(rotated_metrics["phase_deg"])
        assert abs(baseline["phase_deg"]) < abs(conjugated_metrics["phase_deg"])

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_topspin_1d_process_emits_no_runtime_warning(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            spectrum = Experiment(fid).process(
                apodization="em",
                lb=2.0,
                size=16384,
                phase="metadata",
            )

        runtime_messages = [
            str(w.message) for w in recorded if issubclass(w.category, RuntimeWarning)
        ]

        assert spectrum.x.linear
        assert runtime_messages == []

    def test_process_trace_records_only_explicit_requests_and_actual_time_domain_steps(
        self,
    ):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)

        result = Experiment(ds).process(apodization="em", lb=5.0, size=128)

        assert getattr(ds.meta, "nmr_processing", None) is None
        trace = result.meta.nmr_processing["scp_processing"]
        assert trace["requested"] == {
            "apodization": "em",
            "lb": 5.0 * ur.Hz,
            "size": 128,
        }
        assert trace["applied"] == {
            "apodization": "em",
            "lb": 5.0 * ur.Hz,
            "zero_filling": {"size": 128},
            "fft": True,
            "axis_calibration": "ppm",
        }
        assert result.meta.nmr_processing["observed_state"] == {
            "processing_history": "spectrochempy_process_recorded"
        }

    def test_process_trace_distinguishes_explicit_none_from_omitted_phase(self):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)

        implicit = Experiment(ds.copy()).process()
        explicit = Experiment(ds.copy()).process(phase=None)

        np.testing.assert_allclose(np.asarray(explicit.data), np.asarray(implicit.data))
        np.testing.assert_allclose(
            np.asarray(explicit.x.data),
            np.asarray(implicit.x.data),
        )
        assert implicit.meta.nmr_processing["scp_processing"]["requested"] == {}
        assert explicit.meta.nmr_processing["scp_processing"]["requested"] == {
            "phase": None
        }
        assert "phase" not in implicit.meta.nmr_processing["scp_processing"]["applied"]
        assert "phase" not in explicit.meta.nmr_processing["scp_processing"]["applied"]

    def test_process_trace_does_not_report_zero_filling_when_size_is_unchanged(self):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)

        result = Experiment(ds).process(size=64)

        trace = result.meta.nmr_processing["scp_processing"]
        assert trace["requested"] == {"size": 64}
        assert "zero_filling" not in trace["applied"]
        assert trace["applied"]["fft"] is True

    def test_process_trace_replaces_previous_trace_instead_of_building_history(self):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)

        first = Experiment(ds).process(apodization="em", lb=3.0, size=128)
        second = Experiment(first).process(phase=None)

        assert first.meta.nmr_processing["scp_processing"]["requested"] == {
            "apodization": "em",
            "lb": 3.0 * ur.Hz,
            "size": 128,
        }
        assert second.meta.nmr_processing["scp_processing"]["requested"] == {
            "phase": None,
        }
        assert second.meta.nmr_processing["scp_processing"]["applied"] == {}

    def test_process_trace_persists_through_copy_and_dump_roundtrip(self, tmp_path):
        ds = _make_synthetic_vendor_fid(npts=64, sw_hz=6400.0, freq_hz=250.0)
        result = Experiment(ds).process(apodization="gm", lb=2.0, gb=1.0, size=128)
        copied = result.copy()
        target = tmp_path / "nmr_trace.scp"

        assert copied.meta.nmr_processing == result.meta.nmr_processing
        assert copied.meta.nmr_processing is not result.meta.nmr_processing
        assert (
            copied.meta.nmr_processing["scp_processing"]
            is not result.meta.nmr_processing["scp_processing"]
        )

        result.dump(target)
        restored = scp.NDDataset.load(target)

        assert restored.meta.nmr_processing == result.meta.nmr_processing
        assert "scp_processing" in str(restored.meta)
        assert "scp_processing" in restored.meta._repr_html_()


# ---------------------------------------------------------------------------
# Processing — frequency-domain 1D
# ---------------------------------------------------------------------------


class TestProcessFrequencyDomain:
    """Test state-aware processing of 1D frequency-domain data."""

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    def test_no_fft_on_processed_data(self):
        """Verify that FFT is NOT called on frequency-domain input."""
        spec = _read_or_skip(nmrdir / "topspin_1d/1/pdata/1/1r")
        exp = Experiment(spec)
        result = exp.process()
        # If FFT were called, the data would be completely different
        # (FFT of a spectrum is nonsense).  Check data is preserved.
        np.testing.assert_allclose(spec.data, result.data, atol=1e-10)

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    def test_manual_phase_applied(self):
        spec = _read_or_skip(nmrdir / "topspin_1d/1/pdata/1/1r")
        exp = Experiment(spec)
        result = exp.process(phase="manual", phc0=10.0)
        # Phasing should change the data
        assert not np.allclose(spec.data, result.data)
        # But coordinate should remain ppm
        assert str(result.coord(0).units) == "ppm"

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    def test_no_phase_returns_copy(self):
        spec = _read_or_skip(nmrdir / "topspin_1d/1/pdata/1/1r")
        exp = Experiment(spec)
        result = exp.process()
        # Should be a copy, not the same object
        assert result is not spec
        np.testing.assert_allclose(spec.data, result.data, atol=1e-10)

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    def test_source_unchanged_after_process(self):
        spec = _read_or_skip(nmrdir / "topspin_1d/1/pdata/1/1r")
        original_data = spec.data.copy()
        exp = Experiment(spec)
        _ = exp.process(phase="manual", phc0=10.0)
        np.testing.assert_array_equal(spec.data, original_data)

    @pytest.mark.skipif(not _has_topspin_1d_pdata(), reason="TopSpin 1D pdata missing")
    def test_frequency_domain_trace_records_phase_without_fft(self):
        spec = _read_or_skip(nmrdir / "topspin_1d/1/pdata/1/1r")

        result = Experiment(spec).process(phase="manual", phc0=10.0)

        trace = result.meta.nmr_processing["scp_processing"]
        assert trace["requested"] == {
            "phase": "manual",
            "phc0": 10.0 * ur.deg,
        }
        assert trace["applied"] == {
            "phase": {
                "mode": "manual",
                "phc0": 10.0 * ur.deg,
                "phc1": 0.0 * ur.deg,
            }
        }
        assert "scp_processing" not in spec.meta.nmr_processing


# ---------------------------------------------------------------------------
# Processing — 2D limitations
# ---------------------------------------------------------------------------


class TestProcess2D:
    """Test that 2D datasets are outside the public processing scope."""

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data missing")
    def test_2d_ser_is_rejected(self):
        ser = _read_or_skip(nmrdir / "topspin_2d/1/ser")
        exp = Experiment(ser)
        with pytest.raises(NotImplementedError, match="only validated 1D experiments"):
            exp.process()

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data missing")
    def test_2d_ser_with_apodization_is_rejected(self):
        ser = _read_or_skip(nmrdir / "topspin_2d/1/ser")
        exp = Experiment(ser)
        with pytest.raises(NotImplementedError, match="only validated 1D experiments"):
            exp.process(apodization="em", lb=2.0)

    @pytest.mark.skipif(not _has_topspin_2d_pdata(), reason="TopSpin 2D pdata missing")
    def test_2d_processed_is_rejected(self):
        """Processed 2D data is still outside the public processing scope."""
        spec2d = _read_or_skip(nmrdir / "topspin_2d/1/pdata/1/2rr")
        exp = Experiment(spec2d)
        with pytest.raises(NotImplementedError, match="only validated 1D experiments"):
            exp.process()


# ---------------------------------------------------------------------------
# Summary and representation
# ---------------------------------------------------------------------------


class TestSummaryAndRepr:
    """Test summary() and __repr__."""

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_summary_fid(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        s = exp.summary()
        assert "NMR Experiment" in s
        assert "fid" in s
        assert "time" in s
        assert "1H" in s or "H" in s

    @pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data missing")
    def test_summary_2d_ser(self):
        ser = _read_or_skip(nmrdir / "topspin_2d/1/ser")
        exp = Experiment(ser)
        s = exp.summary()
        assert "time × time" in s
        assert "public processing: 1D only" in s

    @pytest.mark.skipif(not _has_topspin_1d(), reason="TopSpin 1D data missing")
    def test_repr_fid(self):
        fid = _read_or_skip(nmrdir / "topspin_1d/1/fid")
        exp = Experiment(fid)
        r = repr(exp)
        assert "Experiment(" in r
        assert "fid" in r
        assert "ndim=1" in r


# ---------------------------------------------------------------------------
# Synthetic non-TopSpin / canonical metadata
# ---------------------------------------------------------------------------


class TestCanonicalMetadata:
    """Verify that NMRMetadata and Experiment work without Bruker keys."""

    def test_nmr_metadata_no_bruker_keys(self):
        """NMRMetadata can be constructed with vendor-neutral values only."""
        from spectrochempy_nmr.nmr_metadata import NMRMetadata
        from spectrochempy_nmr.nmr_metadata import infer_source_kind
        from spectrochempy_nmr.nmr_metadata import summarise_domain

        meta = NMRMetadata(
            ndim=2,
            domains=("time", "time"),
            encoding=("States", "DQD"),
            nuclei=("13C", "1H"),
            pulse_program="hsqc",
            source_kind="ser",
            datatype="2D",
            iscomplex=(True, True),
            spectral_width_hz=(15000.0, 6000.0),
            spectrometer_freq_mhz=(125.0, 500.0),
        )

        # All fields accessible — no Bruker key needed.
        assert meta.ndim == 2
        assert meta.domains == ("time", "time")
        assert meta.encoding == ("States", "DQD")
        assert meta.nuclei == ("13C", "1H")
        assert meta.pulse_program == "hsqc"
        assert meta.source_kind == "ser"
        assert meta.spectral_width_hz == (15000.0, 6000.0)
        assert meta.spectrometer_freq_mhz == (125.0, 500.0)

        # Shared logic works on pure canonical data.
        assert infer_source_kind(2, ("time", "time")) == "ser"
        assert summarise_domain(("time", "time")) == "time"
        assert summarise_domain(("frequency", "frequency")) == "frequency"
        assert summarise_domain(("time", "frequency")) == "mixed"

    def test_nmr_metadata_various_source_kinds(self):
        """infer_source_kind covers all canonical cases."""
        from spectrochempy_nmr.nmr_metadata import infer_source_kind

        assert infer_source_kind(1, ("time",)) == "fid"
        assert infer_source_kind(1, ("frequency",)) == "processed_1d"
        assert infer_source_kind(2, ("time", "time")) == "ser"
        assert infer_source_kind(2, ("frequency", "frequency")) == "processed_2d"
        assert infer_source_kind(2, ("time", "frequency")) == "partially_processed"
        assert infer_source_kind(3, ("time", "time", "time")) == "unknown"

    def test_nmr_metadata_frozen(self):
        """NMRMetadata is immutable."""
        from spectrochempy_nmr.nmr_metadata import NMRMetadata

        meta = NMRMetadata(ndim=1, domains=("time",))
        with pytest.raises(AttributeError):
            meta.ndim = 2  # type: ignore[misc]

    def test_synthetic_jeol_dataset(self):
        """Experiment classifies data from a non-TopSpin reader."""
        # The canonical extraction layer can consume metadata that contains
        # no Bruker-specific keys — the mock has only the fields that
        # extract_topspin_metadata reads via getattr.
        import numpy as np
        from spectrochempy_nmr.nmr_metadata import NMRMetadata

        # --- Direct canonical extraction (simulating a future vendor adapter) ---
        nmr_meta = NMRMetadata(
            ndim=1,
            domains=("time",),
            encoding=("QSIM",),
            nuclei=("13C",),
            pulse_program="ja3",
            source_kind="fid",
            datatype="FID",
            iscomplex=(True,),
            spectral_width_hz=(15000.0,),
            spectrometer_freq_mhz=(125.0,),
        )
        assert nmr_meta.nuclei == ("13C",)
        assert nmr_meta.encoding == ("QSIM",)
        assert nmr_meta.source_kind == "fid"

        # --- Experiment instantiation via the standard Bruker path ---
        # Set attributes directly on ds.meta, mimicking what any reader
        # could do.  The important point is that *Experiment itself* never
        # references these names — only the extraction layer does.
        ds = scp.NDDataset(np.arange(1024, dtype=np.complex128))
        ds.meta.ndim = 1
        ds.meta.isfreq = [False]
        ds.meta.encoding = ["QSIM"]
        ds.meta.nuc1 = ["13C"]
        ds.meta.pulprog = "ja3"
        ds.meta.datatype = "FID"
        ds.meta.iscomplex = [True]
        ds.meta.sw_h = [15000.0]
        ds.meta.sfo1 = [125.0]
        ds.meta.readonly = True

        exp = Experiment(ds)
        assert exp.ndim == 1
        assert exp.domains == ("time",)
        assert exp.nuclei == ("13C",)
        assert exp.encoding == ("QSIM",)
        assert exp.source_kind == "fid"
        assert exp.is_time_domain

    def test_experiment_no_bruker_keys_on_dataset(self):
        """Experiment instantiates from empty metadata — no Bruker keys."""
        ds = scp.NDDataset(np.arange(100, dtype=float))
        exp = Experiment(ds)
        assert exp.domain == "unknown"
        assert exp.source_kind == "unknown"


# ---------------------------------------------------------------------------
# 2D processing via Experiment.process()
# ---------------------------------------------------------------------------


def _has_topspin_2d():
    return (nmrdir / "topspin_2d/1/ser").exists()


@pytest.mark.skipif(not _has_topspin_2d(), reason="TopSpin 2D data not available")
class TestExperiment2DProcessing:
    """Verify multi-dimensional processing is kept out of the public API."""

    def test_process_2d_em_fft_rejected(self):
        """The public processing workflow is intentionally 1D-only."""
        ds = scp.nmr.read(nmrdir / "topspin_2d", expno=1, remove_digital_filter=True)
        exp = Experiment(ds)
        assert exp.is_time_domain
        assert exp.ndim == 2

        with pytest.raises(NotImplementedError, match="only validated 1D experiments"):
            exp.process(apodization="em", lb=2.0)

    def test_validate_2d_reports_public_scope_warning(self):
        """Validation should make the current public scope explicit."""
        ds = scp.nmr.read(nmrdir / "topspin_2d", expno=1, remove_digital_filter=True)
        exp = Experiment(ds)
        report = exp.validate()
        assert any("public supported workflow" in msg for msg in report.warnings)
