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


# Tests
# --------------------------------------------------------------------------------------
class TestPSD:
    """Lean test suite for PSD signal-processing transform."""

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def test_psd_api(self):
        """Public API: import, result class, and __call__ delegation."""
        assert hasattr(scp, "PSD")
        from spectrochempy.processing.psd import PSD as _PSD

        assert scp.PSD is _PSD
        from spectrochempy.processing.psd import PSDResult as _PSDResult

        assert _PSDResult is PSDResult

        ds = NDDataset(np.random.rand(10, 20))
        psd = PSD()
        np.testing.assert_allclose(psd(ds).prs.data, psd.transform(ds).prs.data)

    # --------------------------------------------------------------------------
    # Input formats (parametrized)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "fixture_name",
        ["raw_2d", "grouped_3d", "averaged_2d"],
    )
    def test_psd_input_formats(self, fixture_name, request):
        """All three supported input formats produce correct output shapes."""
        fixture = request.getfixturevalue(fixture_name)
        if fixture_name == "raw_2d":
            ds, _, n_spectra, n_wavenumbers = fixture
            psd = PSD(n_spectra_per_cycle=n_spectra)
        elif fixture_name == "grouped_3d":
            ds, _, n_spectra, n_wavenumbers = fixture
            psd = PSD()
        else:  # averaged_2d
            ds, n_spectra, n_wavenumbers = fixture
            psd = PSD()

        result = psd.transform(ds)
        assert isinstance(result, PSDResult)
        assert result.prs.shape == (len(psd._get_phi()), n_wavenumbers)
        assert result.T is not None
        assert result.T.shape == (len(psd._get_phi()), n_spectra)

    # --------------------------------------------------------------------------
    # Invalid input
    # --------------------------------------------------------------------------
    def test_psd_invalid_input(self, raw_2d, averaged_2d):
        """Invalid dimensions and non-matching n_spectra_per_cycle raise."""
        psd = PSD()
        with pytest.raises(ValueError, match="1D input is not supported"):
            psd.transform(NDDataset(np.random.rand(100)))
        with pytest.raises(ValueError, match="4D input is not supported"):
            psd.transform(NDDataset(np.random.rand(2, 3, 4, 5)))

        ds_raw, _, n_spectra, _ = raw_2d
        psd = PSD(n_spectra_per_cycle=n_spectra + 1)
        with pytest.raises(ValueError, match="not divisible"):
            psd.transform(ds_raw)

        ds_ave, n_spectra_ave, _ = averaged_2d
        psd = PSD(n_spectra_per_cycle=n_spectra_ave + 1)
        with pytest.raises(ValueError, match="not divisible"):
            psd.transform(ds_ave)

    # --------------------------------------------------------------------------
    # Coordinate handling
    # --------------------------------------------------------------------------
    def test_psd_coordinate_preservation(self, grouped_3d, averaged_2d):
        """Time and spectral coordinates are preserved correctly."""
        ds, _, n_spectra, n_wavenumbers = grouped_3d
        psd = PSD()
        result = psd.transform(ds)
        np.testing.assert_allclose(result.T.coordset.x.data, ds.coordset.y.data)

        ds_ave, n_spectra_a, _ = averaged_2d
        result = psd.transform(ds_ave)
        assert result.T.shape[1] == n_spectra_a

        # Raw 2D with explicit time: averaged relative time matches cycle time
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

        # Spectral coordinate preservation
        spectral_values = np.linspace(7000, 8200, n_wavenumbers)
        ds.set_coordset(
            z=ds.coordset.z,
            y=ds.coordset.y,
            x=Coord(spectral_values, title="wavenumber", units="cm^-1"),
        )
        result = psd.transform(ds)
        np.testing.assert_allclose(result.prs.coordset.x.data, spectral_values)
        np.testing.assert_allclose(result.in_phase.coordset.x.data, spectral_values)

    def test_psd_zero_period_time_coord_raises(self):
        """Constant time coordinate raises ValueError."""
        ds = NDDataset(np.random.rand(5, 10))
        ds.set_coordset(
            y=Coord(np.full(5, 3.0), title="time", units="s"),
            x=Coord(np.arange(10), title="wavenumber"),
        )
        with pytest.raises(ValueError, match="zero span"):
            PSD().transform(ds)

    def test_psd_nonzero_time_origin_agreement(self):
        """Matrix and integration agree when time origin is not zero."""
        n_spectra = 7
        time = np.linspace(5.0, 7.0, n_spectra)
        ds = NDDataset(np.random.rand(n_spectra, 20))
        ds.set_coordset(
            y=Coord(time, title="time", units="s"),
            x=Coord(np.arange(20), title="wavenumber"),
        )
        phi = np.arange(0.0, 360.0, 30.0)
        result_matrix = PSD(demodulation="matrix", phi=phi).transform(ds)
        result_int = PSD(demodulation="integration", phi=phi).transform(ds)
        np.testing.assert_allclose(
            result_matrix.prs.data,
            result_int.prs.data,
            atol=1e-12,
        )

    # --------------------------------------------------------------------------
    # Stateless semantics
    # --------------------------------------------------------------------------
    def test_psd_transform_stateless(self, grouped_3d, raw_2d):
        """Repeated transforms are independent and do not mutate prior results."""
        ds1, _, _, _ = grouped_3d
        ds2, _, _, _ = raw_2d

        psd = PSD()
        result1 = psd.transform(ds1)
        original_prs = result1.prs.data.copy()

        result2 = psd.transform(ds1)
        np.testing.assert_allclose(result1.prs.data, result2.prs.data)
        assert result1.prs is not result2.prs
        assert result1.in_phase is not result2.in_phase
        assert result1.T is not result2.T

        psd.transform(ds2)
        np.testing.assert_array_equal(result1.prs.data, original_prs)

    # --------------------------------------------------------------------------
    # Demodulation methods and T-matrix
    # --------------------------------------------------------------------------
    def test_psd_demodulation_methods(self, grouped_3d):
        """Both strategies return correct shapes; matrix has T, integration does not."""
        ds, _, n_spectra, n_wavenumbers = grouped_3d
        phi = np.arange(0.0, 360.0, 30.0)

        result_matrix = PSD(demodulation="matrix", phi=phi).transform(ds)
        assert result_matrix.prs.shape == (len(phi), n_wavenumbers)
        assert result_matrix.T is not None
        assert result_matrix.T.shape == (len(phi), n_spectra)
        assert "y" in result_matrix.T.dims

        result_int = PSD(demodulation="integration", phi=phi).transform(ds)
        assert result_int.prs.shape == (len(phi), n_wavenumbers)
        assert result_int.T is None

    # --------------------------------------------------------------------------
    # Simpson rule
    # --------------------------------------------------------------------------
    def test_psd_simpson_weights(self):
        """Simpson T-matrix coefficients match theoretical weights."""
        n_spectra = 5
        ds = NDDataset(np.random.rand(n_spectra, 10))
        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            integration_rule="simpson",
            demodulation="matrix",
        )
        result = psd.transform(ds)
        T = result.T
        expected_weights = np.array([1.0, 4.0, 2.0, 4.0, 1.0])
        prefactor = 2.0 / (3.0 * (n_spectra - 1))
        phi = psd._get_phi()
        phi_0_idx = np.where(np.isclose(phi, 0.0))[0][0]
        t_norm = np.linspace(0.0, 1.0, n_spectra)
        expected_T_row = prefactor * expected_weights * np.sin(2.0 * np.pi * t_norm)
        np.testing.assert_allclose(T.data[phi_0_idx], expected_T_row, atol=1e-10)

    def test_psd_simpson_validation(self):
        """Simpson requires odd n >= 3."""
        for n in [4, 1]:
            ds = NDDataset(np.random.rand(n, 10))
            psd = PSD(
                n_spectra_per_cycle=n,
                integration_rule="simpson",
                demodulation="matrix",
            )
            with pytest.raises(ValueError, match="odd number of points"):
                psd.transform(ds)

    # --------------------------------------------------------------------------
    # Matrix / integration agreement (parametrized by rule)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("rule", ["riemann", "trapezoid", "simpson"])
    def test_psd_matrix_integration_agreement(self, rule):
        """Matrix and integration agree for each integration rule."""
        if rule == "simpson":
            n_spectra = 5
            t = np.linspace(0, 1, n_spectra)
            data = np.tile(np.sin(2.0 * np.pi * t), (10, 1)).T
        else:
            n_spectra = 7
            data = np.random.rand(n_spectra, 20)

        ds = NDDataset(data)
        phi = (
            np.array([0.0, 90.0]) if rule == "simpson" else np.arange(0.0, 360.0, 30.0)
        )

        psd_matrix = PSD(
            n_spectra_per_cycle=n_spectra,
            demodulation="matrix",
            integration_rule=rule,
            phi=phi,
        )
        psd_int = PSD(
            n_spectra_per_cycle=n_spectra,
            demodulation="integration",
            integration_rule=rule,
            phi=phi,
        )

        np.testing.assert_allclose(
            psd_matrix.transform(ds).prs.data,
            psd_int.transform(ds).prs.data,
            atol=1e-12,
        )

    # --------------------------------------------------------------------------
    # Regression: non-uniform time coordinate (matrix = integration)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("rule", ["trapezoid", "simpson"])
    def test_psd_nonuniform_time_agreement(self, rule):
        """
        Matrix and integration agree for NON-UNIFORM time coordinates.

        This is a regression test for the fix where the T-matrix
        construction previously assumed uniform weights, causing
        disagreement with explicit integration on non-uniform grids.
        """
        n_spectra = 5 if rule == "simpson" else 7
        n_wavenumbers = 10

        # Slightly jittered time grid (endpoints fixed at 0 and 1)
        rng = np.random.RandomState(42)
        t_base = np.linspace(0.0, 1.0, n_spectra)
        jitter = rng.uniform(-0.03, 0.03, n_spectra)
        jitter[0] = 0.0
        jitter[-1] = 0.0
        t_nonuniform = t_base + jitter

        data = np.sin(2.0 * np.pi * t_nonuniform)[:, np.newaxis] * np.ones(
            n_wavenumbers
        )
        ds = NDDataset(data)
        ds.set_coordset(
            y=Coord(t_nonuniform, title="time", units="s"),
            x=Coord(np.arange(n_wavenumbers), title="wavenumber"),
        )

        phi = np.array([0.0, 90.0])
        result_matrix = PSD(
            demodulation="matrix",
            integration_rule=rule,
            phi=phi,
        ).transform(ds)
        result_int = PSD(
            demodulation="integration",
            integration_rule=rule,
            phi=phi,
        ).transform(ds)

        np.testing.assert_allclose(
            result_matrix.prs.data,
            result_int.prs.data,
            atol=1e-12,
        )

    # --------------------------------------------------------------------------
    # Riemann exact sinusoid (protects right-endpoint restoration)
    # --------------------------------------------------------------------------
    def test_psd_riemann_exact_on_sinusoid(self):
        """Riemann rule gives exact amplitude/phase for pure sinusoids."""
        n_spectra = 61
        t = (np.arange(n_spectra) + 1) / n_spectra  # right-endpoint sampling

        amplitude_true = 2.0
        phase_deg_true = 30.0
        phase_rad_true = np.radians(phase_deg_true)

        data = amplitude_true * np.sin(2.0 * np.pi * t + phase_rad_true)
        ds = NDDataset(data[:, np.newaxis])

        phi = np.arange(0.0, 360.0, 15.0)
        psd = PSD(demodulation="matrix", integration_rule="riemann", phi=phi)
        result = psd.transform(ds)

        np.testing.assert_allclose(result.amplitude.data[0], amplitude_true, atol=1e-10)
        np.testing.assert_allclose(result.phase_lag.data[0], phase_deg_true, atol=1e-10)

    # --------------------------------------------------------------------------
    # Output components and physical properties
    # --------------------------------------------------------------------------
    def test_psd_output_components(self, grouped_3d):
        """Output components exist, are finite, and DC offset does not affect results."""
        ds, _, _, _ = grouped_3d
        psd = PSD(demodulation="matrix", phi=np.arange(0.0, 360.0, 15.0))
        result = psd.transform(ds)

        assert result.in_phase is not None
        assert result.quadrature is not None
        assert result.amplitude is not None
        assert result.phase_lag is not None
        assert np.all(np.isfinite(result.phase_lag.data))

        # DC offset insensitivity
        n_spectra = 7
        t = np.linspace(0, 1, n_spectra)
        data_offset = np.tile(np.sin(2.0 * np.pi * t) + 10.0, (10, 1)).T
        data_clean = np.tile(np.sin(2.0 * np.pi * t), (10, 1)).T
        psd2 = PSD(n_spectra_per_cycle=n_spectra, demodulation="matrix")
        np.testing.assert_allclose(
            psd2.transform(NDDataset(data_offset)).prs.data,
            psd2.transform(NDDataset(data_clean)).prs.data,
            atol=1e-12,
        )

    # --------------------------------------------------------------------------
    # Phase units (parametrized)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "unit, lo, hi, expected_unit",
        [
            ("degrees", -180.0, 360.0, None),
            ("radians", -np.pi, 2.0 * np.pi, "radian"),
        ],
    )
    def test_psd_phase_units(self, grouped_3d, unit, lo, hi, expected_unit):
        """phase_unit produces values in expected range with correct units."""
        ds, _, _, _ = grouped_3d
        psd = PSD(
            demodulation="matrix",
            phi=np.arange(0.0, 360.0, 15.0),
            phase_unit=unit,
        )
        result = psd.transform(ds)
        assert np.all(result.phase_lag.data >= lo - 1e-6)
        assert np.all(result.phase_lag.data <= hi + 1e-6)
        if expected_unit is not None:
            assert result.phase_lag.units == expected_unit

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
        with pytest.raises(ValueError, match="phi must be 1-dimensional"):
            PSD(phi=np.array([[0], [90]]))

        # Missing 0° or 90° in phi
        psd = PSD(phi=[10, 20])
        ds = NDDataset(np.random.rand(5, 10))
        with pytest.raises(ValueError, match="phi must contain 0"):
            psd.transform(ds)

        psd = PSD(phi=[0, 45])
        with pytest.raises(ValueError, match="phi must contain 90"):
            psd.transform(ds)

    # --------------------------------------------------------------------------
    # Phi default and instance isolation
    # --------------------------------------------------------------------------
    def test_psd_phi_handling(self, grouped_3d):
        """Default phi, custom phi, and instance isolation all work correctly."""
        ds, _, _, _ = grouped_3d
        expected_default = np.arange(0.0, 360.0, 15.0)

        # Default phi
        psd = PSD()
        np.testing.assert_allclose(psd._get_phi(), expected_default)
        np.testing.assert_allclose(psd.phi, expected_default)

        # Custom phi as list
        phi_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        psd_list = PSD(phi=phi_list)
        result = psd_list.transform(ds)
        assert result.prs.shape[0] == len(phi_list)
        np.testing.assert_allclose(psd_list._get_phi(), phi_list)

        # Custom phi as numpy array
        phi_array = np.arange(0.0, 360.0, 45.0)
        psd_arr = PSD(phi=phi_array)
        result = psd_arr.transform(ds)
        assert result.prs.shape[0] == len(phi_array)
        np.testing.assert_allclose(psd_arr._get_phi(), phi_array)

        # Instance isolation
        psd1 = PSD(phi=[0.0, 90.0, 180.0, 270.0])
        psd2 = PSD()
        np.testing.assert_allclose(psd2._get_phi(), expected_default)
        psd1.phi = [0.0, 180.0]
        np.testing.assert_allclose(psd2._get_phi(), expected_default)

        # Default not shared between instances
        psd_a = PSD()
        psd_b = PSD()
        psd_a.phi = [0.0, 90.0]
        np.testing.assert_allclose(psd_b._get_phi(), expected_default)
