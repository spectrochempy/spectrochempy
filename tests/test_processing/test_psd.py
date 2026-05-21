# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for PSD (Phase-Sensitive Detection) signal-processing transform."""

import numpy as np
import pytest
import traitlets as tr

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.processing.psd import PSD
from spectrochempy.processing.psd import PSDResult


# Fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture
def raw_2d():
    """Create raw 2D data: (N_cycles * n_spectra_per_cycle, n_wavenumbers)."""
    n_cycles = 4
    n_spectra = 30
    n_wavenumbers = 100
    data = np.random.rand(n_cycles * n_spectra, n_wavenumbers)
    return NDDataset(data), n_cycles, n_spectra, n_wavenumbers


@pytest.fixture
def grouped_3d():
    """Create grouped 3D data: (N_cycles, n_spectra_per_cycle, n_wavenumbers)."""
    n_cycles = 4
    n_spectra = 30
    n_wavenumbers = 100
    data = np.random.rand(n_cycles, n_spectra, n_wavenumbers)
    ds = NDDataset(data)
    ds.set_coordset(
        z=Coord(np.arange(n_cycles), title="cycle"),
        y=Coord(np.linspace(0, 1, n_spectra), title="time", units="s"),
        x=Coord(np.arange(n_wavenumbers), title="wavenumber", units="cm^-1"),
    )
    return ds, n_cycles, n_spectra, n_wavenumbers


@pytest.fixture
def averaged_2d():
    """Create averaged 2D data: (n_spectra_per_cycle, n_wavenumbers)."""
    n_spectra = 30
    n_wavenumbers = 100
    data = np.random.rand(n_spectra, n_wavenumbers)
    return NDDataset(data), n_spectra, n_wavenumbers


@pytest.fixture
def synthetic_sine():
    """Create synthetic sine-modulated data with known phase and amplitude."""
    n_cycles = 2
    n_spectra = 60
    n_wavenumbers = 50
    time = np.linspace(0, 1, n_spectra, endpoint=False)

    amplitude_true = 2.0
    phase_deg_true = 30.0
    phase_rad_true = np.radians(phase_deg_true)

    base_spectrum = np.random.rand(n_wavenumbers) + 1.0

    data = np.zeros((n_cycles, n_spectra, n_wavenumbers))
    for i in range(n_cycles):
        modulation = 1.0 + amplitude_true * np.sin(
            2.0 * np.pi * 1.0 * time + phase_rad_true
        )
        data[i] = base_spectrum[np.newaxis, :] * modulation[:, np.newaxis]

    ds = NDDataset(data)
    ds.set_coordset(
        z=Coord(np.arange(n_cycles), title="cycle"),
        y=Coord(time, title="time", units="s"),
        x=Coord(np.arange(n_wavenumbers), title="wavenumber", units="cm^-1"),
    )
    return ds, amplitude_true, phase_deg_true


# Tests
# --------------------------------------------------------------------------------------
class TestPSD:
    """Lean test suite for PSD signal-processing transform."""

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def test_psd_api(self):
        """Test public API surface: import, result class, and callable."""
        assert hasattr(scp, "PSD"), "PSD not found in spectrochempy namespace"
        from spectrochempy.processing.psd import PSD as _PSD

        assert scp.PSD is _PSD

        from spectrochempy.processing.psd import PSDResult as _PSDResult

        assert _PSDResult is PSDResult

        # __call__ delegates to transform
        ds = NDDataset(np.random.rand(10, 20))
        psd = PSD()
        result_call = psd(ds)
        result_transform = psd.transform(ds)
        np.testing.assert_allclose(result_call.prs.data, result_transform.prs.data)

    # --------------------------------------------------------------------------
    # Input normalization (all three supported formats)
    # --------------------------------------------------------------------------
    def test_psd_input_raw_2d(self, raw_2d):
        """Raw 2D concatenated input with n_spectra_per_cycle."""
        ds, _, n_spectra, n_wavenumbers = raw_2d
        psd = PSD(n_spectra_per_cycle=n_spectra)
        result = psd.transform(ds)

        assert isinstance(result, PSDResult)
        assert result.prs.shape == (len(psd._get_phi()), n_wavenumbers)
        assert result.T is not None
        assert result.T.shape == (len(psd._get_phi()), n_spectra)

    def test_psd_input_grouped_3d(self, grouped_3d):
        """Grouped 3D input."""
        ds, _, n_spectra, n_wavenumbers = grouped_3d
        psd = PSD()
        result = psd.transform(ds)

        assert isinstance(result, PSDResult)
        assert result.prs.shape == (len(psd._get_phi()), n_wavenumbers)
        assert result.T is not None
        assert result.T.shape == (len(psd._get_phi()), n_spectra)

    def test_psd_input_averaged_2d(self, averaged_2d):
        """Already-averaged 2D input without n_spectra_per_cycle."""
        ds, n_spectra, n_wavenumbers = averaged_2d
        psd = PSD()
        result = psd.transform(ds)

        assert isinstance(result, PSDResult)
        assert result.prs.shape == (len(psd._get_phi()), n_wavenumbers)
        assert result.T is not None
        assert result.T.shape == (len(psd._get_phi()), n_spectra)

    # --------------------------------------------------------------------------
    # Invalid input
    # --------------------------------------------------------------------------
    def test_psd_invalid_dimensions(self):
        """1D and 4D input raise ValueError."""
        psd = PSD()

        with pytest.raises(ValueError, match="1D input is not supported"):
            psd.transform(NDDataset(np.random.rand(100)))

        with pytest.raises(ValueError, match="4D input is not supported"):
            psd.transform(NDDataset(np.random.rand(2, 3, 4, 5)))

    def test_psd_invalid_n_spectra_per_cycle(self, raw_2d, averaged_2d):
        """Non-matching n_spectra_per_cycle raises ValueError."""
        ds_raw, _, n_spectra, _ = raw_2d
        psd = PSD(n_spectra_per_cycle=n_spectra + 1)
        with pytest.raises(ValueError, match="not divisible"):
            psd.transform(ds_raw)

        ds_ave, n_spectra_ave, _ = averaged_2d
        psd = PSD(n_spectra_per_cycle=n_spectra_ave + 1)
        with pytest.raises(ValueError, match="not divisible"):
            psd.transform(ds_ave)

    # --------------------------------------------------------------------------
    # Private helper (_normalize_to_cycle_average)
    # --------------------------------------------------------------------------
    def test_psd_normalize_to_cycle_average(self, raw_2d, grouped_3d, averaged_2d):
        """Normalization always returns a correct 2D cycle-averaged array."""
        # Raw 2D
        ds_raw, n_cycles, n_spectra, n_wavenumbers = raw_2d
        psd = PSD(n_spectra_per_cycle=n_spectra)
        A, *_ = psd._normalize_to_cycle_average(ds_raw)
        assert A.ndim == 2
        assert A.shape == (n_spectra, n_wavenumbers)
        expected = ds_raw.data.reshape(n_cycles, n_spectra, n_wavenumbers).mean(axis=0)
        np.testing.assert_allclose(A, expected)

        # Grouped 3D
        ds_grouped, _, n_spectra_g, n_wavenumbers_g = grouped_3d
        psd = PSD()
        A, *_ = psd._normalize_to_cycle_average(ds_grouped)
        assert A.ndim == 2
        assert A.shape == (n_spectra_g, n_wavenumbers_g)
        expected = ds_grouped.data.mean(axis=0)
        np.testing.assert_allclose(A, expected)

        # Averaged 2D
        ds_ave, n_spectra_a, n_wavenumbers_a = averaged_2d
        psd = PSD()
        A, *_ = psd._normalize_to_cycle_average(ds_ave)
        assert A.ndim == 2
        assert A.shape == (n_spectra_a, n_wavenumbers_a)

    # --------------------------------------------------------------------------
    # Coordinate handling
    # --------------------------------------------------------------------------
    def test_psd_coordinate_preservation(self, grouped_3d, averaged_2d):
        """Time and spectral coordinates are preserved correctly."""
        # Grouped 3D: time coord should be preserved in T
        ds, _, n_spectra, n_wavenumbers = grouped_3d
        psd = PSD()
        result = psd.transform(ds)
        assert result.T.shape[1] == n_spectra
        np.testing.assert_allclose(result.T.coordset.x.data, ds.coordset.y.data)

        # Averaged 2D: time coord should be preserved in T
        ds_ave, n_spectra_a, _ = averaged_2d
        result = psd.transform(ds_ave)
        assert result.T.shape[1] == n_spectra_a

        # Raw 2D with explicit time: averaged relative time should match cycle time
        n_cycles = 4
        n_spectra_r = 30
        n_wavenumbers_r = 100
        cycle_time = np.linspace(0.0, 2.9, n_spectra_r)
        full_time = np.concatenate([cycle_time + i * 3.0 for i in range(n_cycles)])
        data = np.random.rand(n_cycles * n_spectra_r, n_wavenumbers_r)
        ds_raw = NDDataset(data)
        ds_raw.set_coordset(
            y=Coord(full_time, title="time", units="s"),
            x=Coord(np.arange(n_wavenumbers_r), title="wavenumber", units="cm^-1"),
        )
        psd = PSD(n_spectra_per_cycle=n_spectra_r)
        result = psd.transform(ds_raw)
        np.testing.assert_allclose(result.T.coordset.x.data, cycle_time, rtol=1e-5)

        # Spectral coordinate preservation on grouped 3D with non-trivial values
        spectral_values = np.linspace(7000, 8200, n_wavenumbers)
        ds.set_coordset(
            z=ds.coordset.z,
            y=ds.coordset.y,
            x=Coord(spectral_values, title="wavenumber", units="cm^-1"),
        )
        result = psd.transform(ds)
        np.testing.assert_allclose(result.prs.coordset.x.data, spectral_values)
        np.testing.assert_allclose(result.in_phase.coordset.x.data, spectral_values)
        np.testing.assert_allclose(result.quadrature.coordset.x.data, spectral_values)

    def test_psd_zero_period_time_coord_raises(self):
        """Constant time coordinate raises ValueError."""
        n_spectra = 5
        n_wavenumbers = 10
        ds = NDDataset(np.random.rand(n_spectra, n_wavenumbers))
        ds.set_coordset(
            y=Coord(np.full(n_spectra, 3.0), title="time", units="s"),
            x=Coord(np.arange(n_wavenumbers), title="wavenumber"),
        )
        with pytest.raises(ValueError, match="zero span"):
            PSD().transform(ds)

    def test_psd_nonzero_time_origin_agreement(self):
        """Matrix and integration agree when time origin is not zero."""
        n_spectra = 7
        n_wavenumbers = 20
        time = np.linspace(5.0, 7.0, n_spectra)
        ds = NDDataset(np.random.rand(n_spectra, n_wavenumbers))
        ds.set_coordset(
            y=Coord(time, title="time", units="s"),
            x=Coord(np.arange(n_wavenumbers), title="wavenumber"),
        )
        phi = np.arange(0.0, 360.0, 30.0)
        result_matrix = PSD(demodulation="matrix", phi=phi).transform(ds)
        result_int = PSD(demodulation="integration", phi=phi).transform(ds)
        np.testing.assert_allclose(
            result_matrix.prs.data,
            result_int.prs.data,
            atol=1e-12,
            err_msg="Matrix and integration disagree for non-zero time origin",
        )

    # --------------------------------------------------------------------------
    # Stateless semantics
    # --------------------------------------------------------------------------
    def test_psd_transform_stateless(self, grouped_3d, raw_2d):
        """Repeated transforms are independent and do not mutate prior results."""
        ds1, _, _, _ = grouped_3d
        ds2, _, n_spectra, _ = raw_2d

        psd = PSD()
        result1 = psd.transform(ds1)
        original_prs = result1.prs.data.copy()

        # Repeated call on same data: independent objects, equal values
        result2 = psd.transform(ds1)
        np.testing.assert_allclose(result1.prs.data, result2.prs.data)
        assert result1.prs is not result2.prs
        assert result1.in_phase is not result2.in_phase
        assert result1.T is not result2.T

        # Call on different data: prior result unchanged
        psd.transform(ds2)
        np.testing.assert_array_equal(result1.prs.data, original_prs)

    # --------------------------------------------------------------------------
    # T-matrix and integration-rule weights
    # --------------------------------------------------------------------------
    def test_psd_T_shape(self, grouped_3d):
        """T matrix has correct shape and coords."""
        ds, _, n_spectra, _ = grouped_3d
        phi = np.arange(0.0, 360.0, 30.0)
        result = PSD(phi=phi).transform(ds)
        T = result.T
        assert T.shape == (len(phi), n_spectra)
        assert "y" in T.dims
        assert T.coordset["y"] is not None

    def test_psd_simpson_weights(self):
        """Simpson T-matrix coefficients match theoretical weights."""
        n_spectra = 5
        data = np.random.rand(n_spectra, 10)
        ds = NDDataset(data)
        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            integration_rule="simpson",
            demodulation="matrix",
        )
        result = psd.transform(ds)

        T = result.T
        expected_weights = np.array([1.0, 4.0, 2.0, 4.0, 1.0])
        prefactor = 2.0 / (3.0 * (n_spectra - 1))  # = 1/6
        phi = psd._get_phi()
        phi_0_idx = np.where(np.isclose(phi, 0.0))[0][0]
        t_norm = np.linspace(0.0, 1.0, n_spectra)
        expected_T_row = prefactor * expected_weights * np.sin(2.0 * np.pi * t_norm)
        np.testing.assert_allclose(T.data[phi_0_idx], expected_T_row, atol=1e-10)

    def test_psd_simpson_validation(self):
        """Simpson requires odd n >= 3."""
        data = np.random.rand(4, 10)
        ds = NDDataset(data)
        psd = PSD(
            n_spectra_per_cycle=4, integration_rule="simpson", demodulation="matrix"
        )
        with pytest.raises(ValueError, match="odd number of points"):
            psd.transform(ds)

        data = np.random.rand(1, 10)
        ds = NDDataset(data)
        psd = PSD(
            n_spectra_per_cycle=1, integration_rule="simpson", demodulation="matrix"
        )
        with pytest.raises(ValueError, match="odd number of points"):
            psd.transform(ds)

    # --------------------------------------------------------------------------
    # Matrix / integration agreement per rule
    # --------------------------------------------------------------------------
    def test_psd_matrix_integration_simpson_agreement(self):
        """Matrix and integration Simpson agree on sinusoidal data."""
        n_spectra = 5
        t = np.linspace(0, 1, n_spectra)
        data = np.tile(np.sin(2.0 * np.pi * t), (10, 1)).T
        ds = NDDataset(data)
        phi = np.array([0.0, 90.0])

        psd_matrix = PSD(
            n_spectra_per_cycle=n_spectra,
            demodulation="matrix",
            integration_rule="simpson",
            phi=phi,
        )
        psd_int = PSD(
            n_spectra_per_cycle=n_spectra,
            demodulation="integration",
            integration_rule="simpson",
            phi=phi,
        )

        result_matrix = psd_matrix.transform(ds)
        result_int = psd_int.transform(ds)
        np.testing.assert_allclose(
            result_matrix.prs.data, result_int.prs.data, atol=1e-12
        )

    def test_psd_matrix_integration_riemann_agreement(self):
        """Matrix and integration Riemann agree on random data."""
        n_spectra = 7
        data = np.random.rand(n_spectra, 20)
        ds = NDDataset(data)
        phi = np.arange(0.0, 360.0, 30.0)

        psd_matrix = PSD(
            n_spectra_per_cycle=n_spectra,
            demodulation="matrix",
            integration_rule="riemann",
            phi=phi,
        )
        psd_int = PSD(
            n_spectra_per_cycle=n_spectra,
            demodulation="integration",
            integration_rule="riemann",
            phi=phi,
        )

        result_matrix = psd_matrix.transform(ds)
        result_int = psd_int.transform(ds)
        np.testing.assert_allclose(
            result_matrix.prs.data, result_int.prs.data, atol=1e-12
        )

    def test_psd_matrix_vs_integration(self, grouped_3d):
        """Matrix and integration (default trapezoid) approximately agree."""
        ds, _, _, _ = grouped_3d
        phi = np.arange(0.0, 360.0, 30.0)

        result_matrix = PSD(demodulation="matrix", phi=phi).transform(ds)
        result_int = PSD(demodulation="integration", phi=phi).transform(ds)

        np.testing.assert_allclose(
            result_matrix.prs.data,
            result_int.prs.data,
            atol=1e-3,
            err_msg="Matrix and integration methods give different results",
        )

    # --------------------------------------------------------------------------
    # Demodulation method outputs
    # --------------------------------------------------------------------------
    def test_psd_demodulation_methods(self, grouped_3d):
        """Both demodulation strategies return correct shapes."""
        ds, _, _, n_wavenumbers = grouped_3d

        result_matrix = PSD(demodulation="matrix").transform(ds)
        assert result_matrix.prs.shape[0] == len(PSD()._get_phi())
        assert result_matrix.prs.shape[1] == n_wavenumbers
        assert result_matrix.T is not None

        result_int = PSD(demodulation="integration").transform(ds)
        assert result_int.prs.shape[0] == len(PSD()._get_phi())
        assert result_int.prs.shape[1] == n_wavenumbers
        assert result_int.T is None

    # --------------------------------------------------------------------------
    # Output components and physical properties
    # --------------------------------------------------------------------------
    def test_psd_synthetic_sine(self, synthetic_sine):
        """Synthetic sine modulation produces expected outputs."""
        ds, amplitude_true, phase_deg_true = synthetic_sine

        psd = PSD(demodulation="matrix", phi=np.arange(0.0, 360.0, 15.0))
        result = psd.transform(ds)

        assert result.in_phase is not None
        assert result.quadrature is not None
        assert result.amplitude is not None
        assert result.phase is not None
        assert np.all(np.isfinite(result.phase.data))

        mean_amp = np.mean(result.amplitude.data)
        assert 0.5 < mean_amp < 5.0, f"Amplitude {mean_amp} seems off"

    def test_psd_dc_offset_insensitive(self):
        """PSD result is unchanged by a constant DC offset."""
        n_spectra = 7
        n_wavenumbers = 10
        t = np.linspace(0, 1, n_spectra)
        data_offset = np.tile(np.sin(2.0 * np.pi * t) + 10.0, (n_wavenumbers, 1)).T
        data_clean = np.tile(np.sin(2.0 * np.pi * t), (n_wavenumbers, 1)).T

        psd = PSD(n_spectra_per_cycle=n_spectra, demodulation="matrix")
        result_offset = psd.transform(NDDataset(data_offset))
        result_clean = psd.transform(NDDataset(data_clean))

        np.testing.assert_allclose(
            result_offset.prs.data,
            result_clean.prs.data,
            atol=1e-12,
            err_msg="PSD result changed by a constant DC offset",
        )

    # --------------------------------------------------------------------------
    # Phase output
    # --------------------------------------------------------------------------
    def test_psd_phase_unit_degrees(self, grouped_3d):
        """phase_unit='degrees' produces values in [-180, 360]."""
        ds, _, _, _ = grouped_3d
        psd = PSD(
            demodulation="matrix", phi=np.arange(0.0, 360.0, 15.0), phase_unit="degrees"
        )
        result = psd.transform(ds)
        assert np.all(result.phase.data >= -180.0 - 1e-6)
        assert np.all(result.phase.data <= 360.0 + 1e-6)

    def test_psd_phase_unit_radians(self, grouped_3d):
        """phase_unit='radians' produces values in [-π, 2π] with radian units."""
        ds, _, _, _ = grouped_3d
        psd = PSD(
            demodulation="matrix", phi=np.arange(0.0, 360.0, 15.0), phase_unit="radians"
        )
        result = psd.transform(ds)
        assert np.all(result.phase.data >= -np.pi - 1e-6)
        assert np.all(result.phase.data <= 2.0 * np.pi + 1e-6)
        assert result.phase.units == "radian"

    # --------------------------------------------------------------------------
    # Trait validation
    # --------------------------------------------------------------------------
    def test_psd_trait_validation(self):
        """Invalid trait values raise on construction or transform."""
        with pytest.raises(ValueError, match="harmonic must be positive"):
            PSD(harmonic=0)

        with pytest.raises(ValueError, match="n_spectra_per_cycle must be positive"):
            PSD(n_spectra_per_cycle=0)

        with pytest.raises(tr.TraitError):
            PSD(phase_unit="gradians")

        # phi must be 1D
        with pytest.raises(ValueError, match="phi must be 1-dimensional"):
            PSD(phi=[[0, 90]])

        # Missing 0° or 90° in phi
        psd = PSD(phi=[10, 20])
        ds = NDDataset(np.random.rand(5, 10))
        with pytest.raises(ValueError, match="phi must contain 0"):
            psd.transform(ds)

        psd = PSD(phi=[0, 45])
        with pytest.raises(ValueError, match="phi must contain 90"):
            psd.transform(ds)
