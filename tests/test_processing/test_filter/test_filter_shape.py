# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.processing.filter.denoise import denoise


@pytest.fixture
def dataset_1d():
    return NDDataset(np.random.rand(100), title="intensity")


@pytest.fixture
def dataset_2d():
    return NDDataset(np.random.rand(5, 100), title="intensity")


@pytest.fixture
def dataset_2d_single_row():
    return NDDataset(np.random.rand(1, 100), title="intensity")


class TestFilterShapePreservation:
    def test_filter_1d_input_returns_1d(self, dataset_1d):
        result = scp.Filter(method="savgol", size=5, order=2).transform(dataset_1d)
        assert result.ndim == 1, f"Expected 1D output for 1D input, got {result.ndim}D"
        assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"

    def test_filter_2d_input_returns_2d(self, dataset_2d):
        result = scp.Filter(method="savgol", size=5, order=2).transform(dataset_2d)
        assert result.ndim == 2, f"Expected 2D output for 2D input, got {result.ndim}D"
        assert result.shape == (5, 100), f"Expected shape (5, 100), got {result.shape}"

    def test_filter_2d_single_row_preserves_shape(self, dataset_2d_single_row):
        result = scp.Filter(method="savgol", size=5, order=2).transform(
            dataset_2d_single_row
        )
        assert (
            result.ndim == 2
        ), f"Expected 2D output for 2D (1, N) input, got {result.ndim}D"
        assert result.shape == (
            1,
            100,
        ), f"Expected shape (1, 100) to be preserved, got {result.shape}"

    def test_filter_2d_single_row_preserves_coords(self, dataset_2d_single_row):
        dataset_2d_single_row.set_coordset(
            x=scp.Coord(np.linspace(4000, 800, 100), title="wavenumber"),
            y=scp.Coord([0], title="spectrum"),
        )
        result = scp.Filter(method="savgol", size=5, order=2).transform(
            dataset_2d_single_row
        )
        assert result.coordset is not None, "Coordset should be preserved"
        assert result.dims == ["y", "x"], f"Expected dims ['y', 'x'], got {result.dims}"

    def test_smooth_2d_single_row_preserves_shape(self, dataset_2d_single_row):
        result = scp.smooth(dataset_2d_single_row, size=5, window="avg")
        assert result.shape == (
            1,
            100,
        ), f"Expected shape (1, 100) to be preserved, got {result.shape}"

    def test_savgol_2d_single_row_preserves_shape(self, dataset_2d_single_row):
        result = scp.savgol(dataset_2d_single_row, size=5, order=2)
        assert result.shape == (
            1,
            100,
        ), f"Expected shape (1, 100) to be preserved, got {result.shape}"

    def test_whittaker_2d_single_row_preserves_shape(self, dataset_2d_single_row):
        result = scp.whittaker(dataset_2d_single_row, lamb=1.0, order=2)
        assert result.shape == (
            1,
            100,
        ), f"Expected shape (1, 100) to be preserved, got {result.shape}"


class TestFilterEdgeCases:
    def test_filter_preserves_units(self, dataset_2d_single_row):
        dataset_2d_single_row.units = "absorbance"
        result = scp.Filter(method="savgol", size=5, order=2).transform(
            dataset_2d_single_row
        )
        assert result.units == "absorbance", "Units should be preserved"

    def test_filter_preserves_title(self, dataset_2d_single_row):
        dataset_2d_single_row.title = "IR spectrum"
        result = scp.Filter(method="savgol", size=5, order=2).transform(
            dataset_2d_single_row
        )
        assert result.title == "IR spectrum", "Title should be preserved"


class TestSavgolDerivativeMetadata:
    """
    Metadata contract for ``savgol(..., deriv=n)`` (issue #1095).

    The Savitzky-Golay path is index-based and does not validate even spacing,
    so the derivative output keeps the original units (no physically
    transformed units are claimed), preserves the coordinate metadata, and its
    title is annotated to make the derivative explicit.
    """

    @pytest.fixture
    def dataset_2d_with_meta(self):
        ds = NDDataset(np.random.rand(3, 100), title="intensity", units="absorbance")
        ds.set_coordset(
            x=scp.Coord(np.linspace(4000, 800, 100), title="wavenumber"),
            y=scp.Coord([0, 1, 2], title="spectrum"),
        )
        return ds

    @pytest.mark.parametrize("deriv, label", [(1, "1st"), (2, "2nd"), (3, "3rd")])
    def test_deriv_title_is_explicit(self, dataset_2d_with_meta, deriv, label):
        result = scp.savgol(dataset_2d_with_meta, size=7, order=3, deriv=deriv)
        assert result.title == f"intensity ({label} derivative)", result.title

    def test_deriv_higher_order_title(self, dataset_2d_with_meta):
        result = scp.savgol(dataset_2d_with_meta, size=11, order=5, deriv=4)
        assert result.title == "intensity (4th derivative)", result.title

    def test_deriv_zero_keeps_title_unannotated(self, dataset_2d_with_meta):
        result = scp.savgol(dataset_2d_with_meta, size=7, order=3, deriv=0)
        assert result.title == "intensity", result.title

    def test_deriv_preserves_coordinates(self, dataset_2d_with_meta):
        result = scp.savgol(dataset_2d_with_meta, size=7, order=3, deriv=1)
        assert result.coordset is not None
        assert result.dims == ["y", "x"]
        np.testing.assert_array_equal(
            result.coord("x").data, dataset_2d_with_meta.coord("x").data
        )

    def test_deriv_keeps_original_units(self, dataset_2d_with_meta):
        # Conservative: index-based derivative scaling is not tied to a
        # validated physical spacing, so units are unchanged.
        result = scp.savgol(dataset_2d_with_meta, size=7, order=3, deriv=1)
        assert result.units == "absorbance", result.units


class TestSavgolDimSelection:
    """
    Dim selection across savgol entry points (PR 1 for issue #1091).

    Verifies that ``dim`` is correctly transmitted from the public API to the
    underlying ``Filter.transform()`` call, for both string names and integer
    indices.
    """

    @pytest.fixture
    def ds_2d(self):
        """
        2D dataset with distinct numerical patterns per axis.

        Row 0 along x is sin(x), row 1 is cos(x), row 2 is linear.
        The y-axis has only 5 points so we must use a small window for dim='y'.
        """
        np.random.seed(0)
        x = np.linspace(4000, 800, 50)
        data = np.zeros((5, 50))
        data[0, :] = np.sin(x)
        data[1, :] = np.cos(x)
        data[2, :] = x * 0.001
        data[3, :] = np.sin(x) * 2
        data[4, :] = np.cos(x) * 3
        ds = NDDataset(data, title="test")
        ds.set_coordset(
            y=scp.Coord(np.arange(5, dtype=float), title="sample"),
            x=scp.Coord(x, title="wavenumber"),
        )
        ds.x.units = "1/centimeter"
        return ds

    def test_baseline_no_dim(self, ds_2d):
        r = scp.savgol(ds_2d, size=5, order=2, deriv=0)
        assert r.shape == (5, 50)
        assert r.dims == ["y", "x"]

    def test_dim_x_string(self, ds_2d):
        r = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        assert r.shape == (5, 50)
        assert r.dims == ["y", "x"]

    def test_dim_y_string(self, ds_2d):
        r = scp.savgol(ds_2d, size=3, order=1, deriv=0, dim="y")
        assert r.shape == (5, 50)
        assert r.dims == ["y", "x"]

    def test_dim_neg1(self, ds_2d):
        r = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim=-1)
        baseline = scp.savgol(ds_2d, size=5, order=2, deriv=0)
        np.testing.assert_array_equal(r.data, baseline.data)

    def test_dim_positive_int(self, ds_2d):
        r = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim=1)
        baseline = scp.savgol(ds_2d, size=5, order=2, deriv=0)
        np.testing.assert_array_equal(r.data, baseline.data)

    def test_dim_0_matches_y_string(self, ds_2d):
        r_int = scp.savgol(ds_2d, size=3, order=1, deriv=0, dim=0)
        r_str = scp.savgol(ds_2d, size=3, order=1, deriv=0, dim="y")
        np.testing.assert_array_equal(r_int.data, r_str.data)

    def test_axis_x_treated_not_axis_y(self, ds_2d):
        """Verify that dim='x' actually processes along x, not y."""
        r_x = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        r_y = scp.savgol(ds_2d, size=3, order=1, deriv=0, dim="y")
        assert not np.allclose(r_x.data, r_y.data)

    def test_api_equivalence_function_method_alias(self, ds_2d):
        r_func = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        r_method = ds_2d.savgol(size=5, order=2, deriv=0, dim="x")
        r_alias = scp.savgol_filter(ds_2d, size=5, order=2, deriv=0, dim="x")
        np.testing.assert_array_equal(r_func.data, r_method.data)
        np.testing.assert_array_equal(r_func.data, r_alias.data)
        assert r_func.dims == r_method.dims == r_alias.dims
        assert r_func.shape == r_method.shape == r_alias.shape

    def test_api_equivalence_transformer(self, ds_2d):
        r_func = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        r_trans = scp.Filter(method="savgol", size=5, order=2).transform(ds_2d, dim=-1)
        np.testing.assert_array_equal(r_func.data, r_trans.data)

    def test_api_equivalence_integer_dim(self, ds_2d):
        r_func = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim=1)
        r_trans = scp.Filter(method="savgol", size=5, order=2).transform(ds_2d, dim=1)
        np.testing.assert_array_equal(r_func.data, r_trans.data)

    def test_validation_unknown_dim(self, ds_2d):
        with pytest.raises(ValueError, match="not recognized"):
            scp.savgol(ds_2d, dim="nonexistent")

    def test_validation_out_of_bounds(self, ds_2d):
        with pytest.raises(IndexError):
            scp.savgol(ds_2d, dim=99)

    def test_validation_bool(self, ds_2d):
        with pytest.raises(TypeError, match="Boolean"):
            scp.savgol(ds_2d, dim=True)

    def test_validation_np_bool(self, ds_2d):
        with pytest.raises(TypeError, match="Boolean"):
            scp.savgol(ds_2d, dim=np.bool_(True))

    def test_validation_tuple(self, ds_2d):
        with pytest.raises(TypeError, match="Tuple/list"):
            scp.savgol(ds_2d, dim=(0, 1))

    def test_validation_list(self, ds_2d):
        with pytest.raises(TypeError, match="Tuple/list"):
            scp.savgol(ds_2d, dim=[0])

    def test_source_not_mutated(self, ds_2d):
        original = ds_2d.data.copy()
        scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        np.testing.assert_array_equal(ds_2d.data, original)

    def test_coords_preserved(self, ds_2d):
        r = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        assert r.coordset is not None
        assert r.dims == ["y", "x"]
        np.testing.assert_array_equal(r.coord("x").data, ds_2d.coord("x").data)
        np.testing.assert_array_equal(r.coord("y").data, ds_2d.coord("y").data)

    def test_shape_preserved(self, ds_2d):
        r = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        assert r.shape == ds_2d.shape

    def test_title_preserved(self, ds_2d):
        r = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        assert r.title == ds_2d.title

    def test_dims_preserved(self, ds_2d):
        r = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        assert r.dims == ds_2d.dims

    def test_default_dim_unchanged(self, ds_2d):
        """Calling without dim produces the same result as dim=-1."""
        r_default = scp.savgol(ds_2d, size=5, order=2, deriv=0)
        r_explicit = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim=-1)
        np.testing.assert_array_equal(r_default.data, r_explicit.data)

    def test_numeric_unchanged_without_dim(self, ds_2d):
        """No numeric change for calls without dim (regression guard)."""
        r_before = scp.savgol(ds_2d, size=5, order=2, deriv=0)
        r_after = scp.savgol(ds_2d, size=5, order=2, deriv=0, dim="x")
        np.testing.assert_array_equal(r_before.data, r_after.data)


class TestFilterFamilyDimSelection:
    """Dim selection for smooth() and whittaker() (same fix as savgol)."""

    @pytest.fixture
    def ds_2d(self):
        np.random.seed(1)
        x = np.linspace(4000, 800, 50)
        data = np.random.rand(5, 50)
        ds = NDDataset(data, title="test")
        ds.set_coordset(
            y=scp.Coord(np.arange(5, dtype=float), title="sample"),
            x=scp.Coord(x, title="wavenumber"),
        )
        return ds

    def test_smooth_dim_x(self, ds_2d):
        r = scp.smooth(ds_2d, size=3, dim="x")
        assert r.shape == (5, 50)
        assert r.dims == ["y", "x"]

    def test_smooth_dim_y(self, ds_2d):
        r = scp.smooth(ds_2d, size=3, dim="y")
        assert r.shape == (5, 50)

    def test_smooth_dim_int(self, ds_2d):
        r_str = scp.smooth(ds_2d, size=3, dim="x")
        r_int = scp.smooth(ds_2d, size=3, dim=-1)
        np.testing.assert_array_equal(r_str.data, r_int.data)

    def test_smooth_validation_bool(self, ds_2d):
        with pytest.raises(TypeError, match="Boolean"):
            scp.smooth(ds_2d, dim=True)

    def test_whittaker_dim_x(self, ds_2d):
        r = scp.whittaker(ds_2d, lamb=1.0, dim="x")
        assert r.shape == (5, 50)
        assert r.dims == ["y", "x"]

    def test_whittaker_dim_y(self, ds_2d):
        r = scp.whittaker(ds_2d, lamb=1.0, order=1, dim="y")
        assert r.shape == (5, 50)

    def test_whittaker_dim_int(self, ds_2d):
        r_str = scp.whittaker(ds_2d, lamb=1.0, dim="x")
        r_int = scp.whittaker(ds_2d, lamb=1.0, dim=-1)
        np.testing.assert_array_equal(r_str.data, r_int.data)

    def test_whittaker_validation_tuple(self, ds_2d):
        with pytest.raises(TypeError, match="Tuple/list"):
            scp.whittaker(ds_2d, dim=(0, 1))

    def test_filter_transform_string_dim(self, ds_2d):
        r = scp.Filter(method="savgol", size=5, order=2).transform(ds_2d, dim="x")
        assert r.shape == (5, 50)
        assert r.dims == ["y", "x"]

    def test_filter_transform_validation(self, ds_2d):
        with pytest.raises(TypeError, match="Boolean"):
            scp.Filter(method="savgol", size=5, order=2).transform(ds_2d, dim=True)


class TestDenoiseGuard:
    """Regression: denoise dimensionality guard (fix #xxx)."""

    def test_denoise_1d_does_not_crash(self):
        """1D input returns early without error."""
        ds = NDDataset(np.random.rand(100))
        result = denoise(ds)
        assert result is ds  # returned unchanged

    def test_denoise_2d_works(self, dataset_2d):
        """2D input runs PCA denoising."""
        result = denoise(dataset_2d, ratio=99.0)
        assert result is not dataset_2d
        assert result.shape == dataset_2d.shape
