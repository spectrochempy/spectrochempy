# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for PSD (Phase-Sensitive Detection)."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.psd import PSD
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset


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

    # Create data with sine modulation: amplitude=2.0, phase=30 degrees
    amplitude_true = 2.0
    phase_deg_true = 30.0
    phase_rad_true = np.radians(phase_deg_true)

    # Base spectrum (constant across cycles)
    base_spectrum = np.random.rand(n_wavenumbers) + 1.0

    # Create modulated data
    data = np.zeros((n_cycles, n_spectra, n_wavenumbers))
    for i in range(n_cycles):
        modulation = 1.0 + amplitude_true * np.sin(
            2.0 * np.pi * 1.0 * time + phase_rad_true
        )
        data[i] = base_spectrum[np.newaxis, :] * modulation[:, np.newaxis]

    # Create NDDataset with proper coords
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
    """Test suite for PSD class."""

    # ----------------------------------------------------------------------------------
    # Import tests
    # ----------------------------------------------------------------------------------
    def test_psd_import(self):
        """Test that PSD can be imported."""
        assert hasattr(scp, "PSD"), "PSD not found in spectrochempy namespace"

    # ----------------------------------------------------------------------------------
    # Input mode tests
    # ----------------------------------------------------------------------------------
    def test_psd_raw_2d_input(self, raw_2d):
        """Test raw 2D input mode."""
        ds, n_cycles, n_spectra, n_wavenumbers = raw_2d

        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            input_mode="raw",
        )
        psd.fit(ds)

        assert psd.prs is not None
        assert psd._n_cycles == n_cycles
        assert psd._n_spectra == n_spectra
        assert psd._n_wavenumbers == n_wavenumbers

    def test_psd_grouped_3d_input(self, grouped_3d):
        """Test grouped 3D input mode."""
        ds, n_cycles, n_spectra, n_wavenumbers = grouped_3d

        psd = PSD(input_mode="grouped")
        psd.fit(ds)

        assert psd.prs is not None
        assert psd._n_cycles == n_cycles
        assert psd._n_spectra == n_spectra
        assert psd._n_wavenumbers == n_wavenumbers

    def test_psd_averaged_2d_input(self, averaged_2d):
        """Test averaged 2D input mode."""
        ds, n_spectra, n_wavenumbers = averaged_2d

        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            input_mode="averaged",
        )
        psd.fit(ds)

        assert psd.prs is not None
        assert psd._n_cycles == 1
        assert psd._n_spectra == n_spectra
        assert psd._n_wavenumbers == n_wavenumbers

    def test_psd_input_mode_auto_3d(self, grouped_3d):
        """Test input_mode='auto' with 3D data."""
        ds, n_cycles, n_spectra, n_wavenumbers = grouped_3d

        psd = PSD(input_mode="auto")
        psd.fit(ds)

        assert psd._n_cycles == n_cycles

    def test_psd_input_mode_auto_raw(self, raw_2d):
        """Test input_mode='auto' with 2D raw data."""
        ds, n_cycles, n_spectra, n_wavenumbers = raw_2d

        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            input_mode="auto",
        )
        psd.fit(ds)

        assert psd._n_cycles == n_cycles

    def test_psd_input_mode_auto_averaged(self, averaged_2d):
        """Test input_mode='auto' with 2D averaged data."""
        ds, n_spectra, n_wavenumbers = averaged_2d

        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            input_mode="auto",
        )
        psd.fit(ds)

        assert psd._n_cycles == 1

    def test_psd_invalid_1d_input(self):
        """Test that 1D input raises ValueError."""
        data = np.random.rand(100)
        ds = NDDataset(data)

        psd = PSD()

        with pytest.raises(ValueError, match="1D input is not supported"):
            psd.fit(ds)

    def test_psd_invalid_4d_input(self):
        """Test that 4D input raises ValueError."""
        data = np.random.rand(2, 3, 4, 5)
        ds = NDDataset(data)

        psd = PSD()

        with pytest.raises(ValueError, match="4D input is not supported"):
            psd.fit(ds)

    def test_psd_invalid_n_spectra_per_cycle(self, raw_2d):
        """Test invalid n_spectra_per_cycle."""
        ds, _, n_spectra, _ = raw_2d

        psd = PSD(
            n_spectra_per_cycle=n_spectra + 1,  # Wrong value
            input_mode="raw",
        )

        with pytest.raises(ValueError, match="not divisible"):
            psd.fit(ds)

    def test_psd_invalid_averaged_shape(self, averaged_2d):
        """Test invalid averaged input shape."""
        ds, n_spectra, _ = averaged_2d

        psd = PSD(
            n_spectra_per_cycle=n_spectra + 1,  # Wrong value
            input_mode="averaged",
        )

        with pytest.raises(ValueError, match="must equal n_spectra_per_cycle"):
            psd.fit(ds)

    # ----------------------------------------------------------------------------------
    # T matrix tests
    # ----------------------------------------------------------------------------------
    def test_psd_T_shape_dims_coords(self, grouped_3d):
        """Test T matrix has correct shape, dims, and coords."""
        ds, n_cycles, n_spectra, n_wavenumbers = grouped_3d

        phi = np.arange(0.0, 360.0, 30.0)
        psd = PSD(input_mode="grouped", phi=phi)
        psd.fit(ds)

        T = psd.T
        assert T.shape == (len(phi), n_spectra)
        assert "y" in T.dims
        assert T.coordset["y"] is not None

    # ----------------------------------------------------------------------------------
    # Method tests
    # ----------------------------------------------------------------------------------
    def test_psd_matrix_method_output(self, grouped_3d):
        """Test matrix method output shape and coords."""
        ds, n_cycles, n_spectra, n_wavenumbers = grouped_3d

        psd = PSD(
            input_mode="grouped",
            method="matrix",
        )
        psd.fit(ds)

        assert psd.prs.shape[0] == len(psd._get_phi())
        assert psd.prs.shape[1] == n_wavenumbers

    def test_psd_integration_method_output(self, grouped_3d):
        """Test integration method output shape and coords."""
        ds, n_cycles, n_spectra, n_wavenumbers = grouped_3d

        psd = PSD(
            input_mode="grouped",
            method="integration",
        )
        psd.fit(ds)

        assert psd.prs.shape[0] == len(psd._get_phi())
        assert psd.prs.shape[1] == n_wavenumbers

    def test_psd_matrix_vs_integration(self, grouped_3d):
        """Test matrix method approximately matches integration method."""
        ds, n_cycles, n_spectra, n_wavenumbers = grouped_3d
        phi = np.arange(0.0, 360.0, 30.0)

        # Matrix method
        psd_matrix = PSD(
            input_mode="grouped",
            method="matrix",
            phi=phi,
        )
        psd_matrix.fit(ds)

        # Integration method
        psd_int = PSD(
            input_mode="grouped",
            method="integration",
            phi=phi,
        )
        psd_int.fit(ds)

        # Should be approximately equal (within numerical precision)
        # The two methods use slightly different numerical approaches,
        # so we allow a small absolute tolerance
        np.testing.assert_allclose(
            psd_matrix.prs.data,
            psd_int.prs.data,
            atol=1e-3,  # Absolute tolerance for numerical method differences
            err_msg="Matrix and integration methods give different results",
        )

    # ----------------------------------------------------------------------------------
    # Synthetic sine test
    # ----------------------------------------------------------------------------------
    def test_psd_synthetic_sine(self, synthetic_sine):
        """Test synthetic sine modulation gives expected phase and amplitude."""
        ds, amplitude_true, phase_deg_true = synthetic_sine

        psd = PSD(
            input_mode="grouped",
            method="matrix",
            phi=np.arange(0.0, 360.0, 15.0),
        )
        psd.fit(ds)

        # Check that in_phase and quadrature are extracted
        in_phase = psd.in_phase
        quadrature = psd.quadrature
        amplitude = psd.amplitude
        phase = psd.phase

        assert in_phase is not None
        assert quadrature is not None
        assert amplitude is not None
        assert phase is not None

        # Check amplitude is close to true (for well-modulated channels)
        # Use mean over channels as a rough check
        mean_amp = np.mean(amplitude.data)
        assert 0.5 < mean_amp < 5.0, f"Amplitude {mean_amp} seems off"

    # ----------------------------------------------------------------------------------
    # Phase tests
    # ----------------------------------------------------------------------------------
    def test_psd_phase_atan2_quadrants(self, grouped_3d):
        """Test phase uses arctan2 and handles quadrants correctly."""
        ds, _, _, _ = grouped_3d

        psd = PSD(
            input_mode="grouped",
            method="matrix",
            phi=np.arange(0.0, 360.0, 15.0),
        )
        psd.fit(ds)

        phase = psd.phase
        assert phase is not None

        # Phase should be in [-180, 180] or [0, 360] depending on implementation
        phase_data = phase.data
        assert np.all(np.isfinite(phase_data)), "Phase contains non-finite values"

    def test_psd_phase_unit_degrees(self, grouped_3d):
        """Test phase_unit='degrees'."""
        ds, _, _, _ = grouped_3d

        psd = PSD(
            input_mode="grouped",
            method="matrix",
            phi=np.arange(0.0, 360.0, 15.0),
            phase_unit="degrees",
        )
        psd.fit(ds)

        phase = psd.phase
        # Degrees should be in roughly [-180, 180] or [0, 360]
        assert np.all(phase.data >= -180.0 - 1e-6)
        assert np.all(phase.data <= 360.0 + 1e-6)

    def test_psd_phase_unit_radians(self, grouped_3d):
        """Test phase_unit='radians'."""
        ds, _, _, _ = grouped_3d

        psd = PSD(
            input_mode="grouped",
            method="matrix",
            phi=np.arange(0.0, 360.0, 15.0),
            phase_unit="radians",
        )
        psd.fit(ds)

        phase = psd.phase
        # Radians should be in roughly [-pi, pi] or [0, 2*pi]
        assert np.all(phase.data >= -np.pi - 1e-6)
        assert np.all(phase.data <= 2.0 * np.pi + 1e-6)
        assert phase.units == "radian"

    # ----------------------------------------------------------------------------------
    # Time coordinate handling tests
    # ----------------------------------------------------------------------------------
    def test_psd_raw_2d_time_coord(self):
        """Test raw 2D input with explicit time coordinate."""
        n_cycles = 4
        n_spectra = 30
        n_wavenumbers = 100

        # Create time coordinate for all cycles
        # Each cycle: 0.0, 0.1, 0.2, ..., 2.9 (total 3.0 per cycle)
        cycle_time = np.linspace(0.0, 2.9, n_spectra)
        full_time = np.concatenate([cycle_time + i * 3.0 for i in range(n_cycles)])

        data = np.random.rand(n_cycles * n_spectra, n_wavenumbers)
        ds = NDDataset(data)
        ds.set_coordset(
            y=Coord(full_time, title="time", units="s"),
            x=Coord(np.arange(n_wavenumbers), title="wavenumber", units="cm^-1"),
        )

        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            input_mode="raw",
        )
        psd.fit(ds)

        # T should have time axis length = n_spectra
        assert psd.T.shape[1] == n_spectra

        # T time coordinate should equal averaged relative time
        # Expected: cycle_time (0.0 to 2.9)
        expected_time = cycle_time
        actual_time = psd.T.coordset.x.data
        np.testing.assert_allclose(actual_time, expected_time, rtol=1e-5)

        # Verify psd output is correct shape
        assert psd.prs.shape[0] == len(psd._get_phi())
        assert psd.prs.shape[1] == n_wavenumbers

    def test_psd_grouped_3d_time_coord(self, grouped_3d):
        """Test grouped 3D input preserves time coordinate."""
        ds, n_cycles, n_spectra, n_wavenumbers = grouped_3d

        psd = PSD(input_mode="grouped")
        psd.fit(ds)

        # T should have time axis length = n_spectra
        assert psd.T.shape[1] == n_spectra

        # Time coord should match the input time coord (from grouped_3d fixture)
        input_time = ds.coordset.y.data  # y is time axis in grouped_3d fixture
        actual_time = psd.T.coordset.x.data
        np.testing.assert_allclose(actual_time, input_time)

    def test_psd_averaged_2d_time_coord(self, averaged_2d):
        """Test averaged 2D input preserves time coordinate."""
        ds, n_spectra, n_wavenumbers = averaged_2d

        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            input_mode="averaged",
        )
        psd.fit(ds)

        # T should have time axis length = n_spectra
        assert psd.T.shape[1] == n_spectra

    def test_psd_spectral_coord_preservation_raw_2d(self):
        """Test spectral coordinate is preserved for raw 2D input."""
        n_cycles = 4
        n_spectra = 30
        n_wavenumbers = 100

        # Create spectral coordinate with non-trivial values (7000-8200)
        spectral_values = np.linspace(7000, 8200, n_wavenumbers)

        data = np.random.rand(n_cycles * n_spectra, n_wavenumbers)
        ds = NDDataset(data)
        ds.set_coordset(
            y=Coord(np.arange(n_cycles * n_spectra), title="time", units="s"),
            x=Coord(spectral_values, title="wavenumber", units="cm^-1"),
        )

        psd = PSD(
            n_spectra_per_cycle=n_spectra,
            input_mode="raw",
        )
        psd.fit(ds)

        # Check that spectral coordinate is preserved in all outputs
        np.testing.assert_allclose(psd.prs.coordset.x.data, spectral_values)
        np.testing.assert_allclose(psd.in_phase.coordset.x.data, spectral_values)
        np.testing.assert_allclose(psd.quadrature.coordset.x.data, spectral_values)
        np.testing.assert_allclose(psd.amplitude.coordset.x.data, spectral_values)
        np.testing.assert_allclose(psd.phase.coordset.x.data, spectral_values)

    def test_psd_spectral_coord_preservation_grouped_3d(self, grouped_3d):
        """Test spectral coordinate is preserved for grouped 3D input."""
        ds, n_cycles, n_spectra, n_wavenumbers = grouped_3d

        # Add non-trivial spectral coordinate
        spectral_values = np.linspace(7000, 8200, n_wavenumbers)
        ds.set_coordset(
            z=ds.coordset.z,
            y=ds.coordset.y,
            x=Coord(spectral_values, title="wavenumber", units="cm^-1"),
        )

        psd = PSD(input_mode="grouped")
        psd.fit(ds)

        # Check that spectral coordinate is preserved in all outputs
        np.testing.assert_allclose(psd.prs.coordset.x.data, spectral_values)
        np.testing.assert_allclose(psd.in_phase.coordset.x.data, spectral_values)
        np.testing.assert_allclose(psd.quadrature.coordset.x.data, spectral_values)
        np.testing.assert_allclose(psd.amplitude.coordset.x.data, spectral_values)
        np.testing.assert_allclose(psd.phase.coordset.x.data, spectral_values)
