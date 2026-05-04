# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the CP module."""

import numpy as np
import pytest

from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis.decomposition.cp import CP
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

# Skip all tests in this module if tensorly is not available
tensorly = pytest.importorskip("tensorly")


def _make_synthetic_3d():
    """Create synthetic 3D NDDataset for testing."""
    np.random.seed(42)
    # Create a 3D tensor with known CP structure (6 samples, 8 features, 10 time points)
    A = np.random.rand(6, 2)
    B = np.random.rand(8, 2)
    C = np.random.rand(10, 2)
    X = np.zeros((6, 8, 10))
    for r in range(2):
        X += np.outer(np.outer(A[:, r], B[:, r]), C[:, r]).reshape(6, 8, 10)
    ds = NDDataset(X)
    ds.dims = ["y", "x", "z"]  # Use valid SpectroChemPy dim names: y=6, x=8, z=10
    ds.set_coordset(
        y=Coord(np.arange(6)),
        x=Coord(np.arange(8)),
        z=Coord(np.arange(10)),
    )
    ds.name = "synthetic_3d"
    ds.title = "intensity"
    return ds


class TestCP:
    """Test suite for CP decomposition class."""

    def test_cp_import_without_tensorly(self):
        """
        Test that SpectroChemPy imports without tensorly.

        Note: This test is mainly for manual verification - in CI without tensorly,
        the import should still work.
        """
        pass  # Import is tested by CI without tensorly installed.

    def test_cp_error_without_tensorly(self):
        """Test that clear error is raised when using CP without tensorly."""
        # This test is skipped if tensorly is available
        pytest.skip("Test only relevant when tensorly is not installed")

    def test_cp_basic_fit(self):
        """Test basic fit on synthetic 3D dataset."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        result = cp.fit(ds)

        assert result is cp, "fit should return self"
        assert cp._fitted
        assert cp.n_components == 2

    def test_cp_factor_shapes(self):
        """Test output factor shapes."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        assert cp.A.shape == (6, 2), f"A shape should be (6, 2), got {cp.A.shape}"
        assert cp.B.shape == (8, 2), f"B shape should be (8, 2), got {cp.B.shape}"
        assert cp.C.shape == (10, 2), f"C shape should be (10, 2), got {cp.C.shape}"

    def test_cp_factor_dims_and_coords(self):
        """Test factor dims and coordinates."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        # Check dims - 'a' is used as the components dimension (valid SpectroChemPy dim name)
        assert cp.A.dims == ["y", "a"], f"A dims: {cp.A.dims}"
        assert cp.B.dims == ["x", "a"], f"B dims: {cp.B.dims}"
        assert cp.C.dims == ["z", "a"], f"C dims: {cp.C.dims}"

        # Check that components coordinate exists by accessing it
        a_coord = cp.A.coordset["a"]
        assert a_coord is not None, "Component coord 'a' should exist in A"
        assert (
            a_coord.title == "components"
        ), "Component coord should have title 'components'"

        b_coord = cp.B.coordset["a"]
        assert b_coord is not None, "Component coord 'a' should exist in B"

        c_coord = cp.C.coordset["a"]
        assert c_coord is not None, "Component coord 'a' should exist in C"

    def test_cp_loadings(self):
        """Test loadings property returns tuple of factors."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        loadings = cp.loadings
        assert isinstance(loadings, tuple)
        assert len(loadings) == 3
        assert loadings[0] is cp.A
        assert loadings[1] is cp.B
        assert loadings[2] is cp.C

    def test_cp_fit_transform(self):
        """Test fit_transform returns factors."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        result = cp.fit_transform(ds)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(f, NDDataset) for f in result)

    def test_cp_inverse_transform(self):
        """Test inverse_transform returns reconstructed NDDataset."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        X_hat = cp.inverse_transform()

        assert isinstance(X_hat, NDDataset)
        assert X_hat.shape == ds.shape, f"Shape mismatch: {X_hat.shape} vs {ds.shape}"
        assert X_hat.dims == ds.dims
        assert X_hat.units == ds.units
        assert X_hat.title == ds.title

    def test_cp_weights(self):
        """Test weights property."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        weights = cp.weights
        assert weights is not None
        assert len(weights) == 2

    def test_cp_errors(self):
        """Test errors property when return_errors is True."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2, return_errors=True)
        cp.fit(ds)

        errors = cp.errors
        assert errors is not None
        assert isinstance(errors, list)

    def test_cp_SSE(self):
        """Test SSE calculation."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        sse = cp.SSE
        assert isinstance(sse, float)
        assert sse >= 0, f"SSE should be non-negative, got {sse}"

    def test_cp_explained_variance(self):
        """Test explained variance calculation."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        ev = cp.explained_variance
        assert isinstance(ev, float)
        assert 0 <= ev <= 100, f"Explained variance should be in [0, 100], got {ev}"

    def test_cp_core_consistency(self):
        """Test core consistency diagnostic (finite only)."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        cc = cp.core_consistency
        # Core consistency can be negative if overfactoring, so just check it's finite
        assert np.isfinite(cc), f"Core consistency should be finite, got {cc}"

    def test_cp_invalid_2d_input(self):
        """Test error on 2D input."""
        ds = NDDataset(np.random.rand(10, 20))
        ds.dims = ["x", "y"]
        ds.set_coordset(x=Coord(np.arange(10)), y=Coord(np.arange(20)))

        cp = CP(n_components=3)
        with pytest.raises(ValueError, match="3D input"):
            cp.fit(ds)

    def test_cp_invalid_rank_too_large(self):
        """Test error on invalid rank (too large)."""
        ds = _make_synthetic_3d()  # shape (6, 8, 10)
        cp = CP(n_components=50)  # Too many

        with pytest.raises(ValueError, match="cannot exceed"):
            cp.fit(ds)

    def test_cp_invalid_rank_zero(self):
        """Test error on zero components."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=0)

        with pytest.raises(ValueError, match="positive"):
            cp.fit(ds)

    def test_cp_reproducibility_with_random_state(self):
        """Test reproducibility with random_state."""
        ds = _make_synthetic_3d()

        cp1 = CP(n_components=2, random_state=42)
        cp1.fit(ds)

        cp2 = CP(n_components=2, random_state=42)
        cp2.fit(ds)

        np.testing.assert_array_almost_equal(cp1.A.data, cp2.A.data)
        np.testing.assert_array_almost_equal(cp1.B.data, cp2.B.data)
        np.testing.assert_array_almost_equal(cp1.C.data, cp2.C.data)

    def test_cp_non_negative_constraint(self):
        """Test non-negative constraint."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2, non_negative=True)
        cp.fit(ds)

        assert np.all(cp.A.data >= 0), "A should be non-negative"
        assert np.all(cp.B.data >= 0), "B should be non-negative"
        assert np.all(cp.C.data >= 0), "C should be non-negative"

    def test_cp_components_property(self):
        """Test components property (should return B factor)."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        # components should return B (following PCA convention)
        # Need to compare .data since components returns NDDataset
        np.testing.assert_array_equal(cp.components.data, cp.B.data)

    def test_cp_not_fitted_error(self):
        """Test that properties raise NotFittedError when not fitted."""
        cp = CP(n_components=2)

        with pytest.raises(NotFittedError):
            _ = cp.A
        with pytest.raises(NotFittedError):
            _ = cp.B
        with pytest.raises(NotFittedError):
            _ = cp.C
        with pytest.raises(NotFittedError):
            _ = cp.SSE
        with pytest.raises(NotFittedError):
            _ = cp.explained_variance
        with pytest.raises(NotFittedError):
            _ = cp.core_consistency

    def test_cp_svd_options(self):
        """Test different svd options."""
        ds = _make_synthetic_3d()

        for svd_option in ["numpy_svd", "truncated_svd", "randomized_svd"]:
            cp = CP(n_components=2, svd=svd_option)
            cp.fit(ds)
            assert cp._fitted

    def test_cp_init_options(self):
        """Test different init options."""
        ds = _make_synthetic_3d()

        for init_option in ["random", "svd"]:
            cp = CP(n_components=2, init=init_option)
            cp.fit(ds)
            assert cp._fitted

    def test_cp_transform_not_implemented(self):
        """Test that transform raises NotImplementedError."""
        ds = _make_synthetic_3d()
        cp = CP(n_components=2)
        cp.fit(ds)

        with pytest.raises(NotImplementedError, match="does not support transform"):
            cp.transform(ds)

    def test_cp_inverse_transform_not_fitted(self):
        """Test inverse_transform before fit raises error."""
        cp = CP(n_components=2)

        with pytest.raises(NotFittedError):
            cp.inverse_transform()

    def test_cp_with_units(self):
        """Test CP with units on input data."""
        ds = _make_synthetic_3d()
        ds.units = "absorbance"

        cp = CP(n_components=2)
        cp.fit(ds)

        X_hat = cp.inverse_transform()
        assert X_hat.units == ds.units

    def test_cp_fixed_modes_validation(self):
        """Test fixed_modes validation."""
        ds = _make_synthetic_3d()

        # Last mode cannot be fixed
        cp = CP(n_components=2, fixed_modes=[2])  # mode 2 is last
        with pytest.raises(ValueError, match="Last mode cannot be fixed"):
            cp.fit(ds)

        # Invalid mode should raise error
        cp2 = CP(n_components=2, fixed_modes=[5])  # Invalid mode
        with pytest.raises(ValueError, match="invalid mode"):
            cp2.fit(ds)

    def test_cp_cvg_criterion(self):
        """Test different convergence criteria."""
        ds = _make_synthetic_3d()

        for criterion in ["abs_rec_error", "rec_error"]:
            cp = CP(n_components=2, cvg_criterion=criterion)
            cp.fit(ds)
            assert cp._fitted
