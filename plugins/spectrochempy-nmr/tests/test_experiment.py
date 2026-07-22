# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

"""Tests for scp.nmr.Experiment — NMR-specific scientific model."""

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

        base_peak = float(np.asarray(base.x.data)[int(np.argmax(np.abs(np.asarray(base.data))))])
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
        spectrum = Experiment(_read_or_skip(EXTRA_NMR / "agilent" / "agilent_1d" / "fid")).process()
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
        spectrum = Experiment(_read_or_skip(EXTRA_NMR / "tecmag" / "LiCl_ref1.tnt")).process()
        axis = np.asarray(spectrum.x.data)
        peak_ppm = float(axis[int(np.argmax(np.abs(np.asarray(spectrum.data))))])
        center_ppm = (float(axis[0]) + float(axis[-1])) / 2.0

        assert center_ppm == pytest.approx(0.0, abs=0.05)
        assert peak_ppm == pytest.approx(0.0, abs=0.5)


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
