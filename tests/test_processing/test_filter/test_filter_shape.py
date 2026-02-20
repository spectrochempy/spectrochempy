# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset


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
        assert result.ndim == 2, (
            f"Expected 2D output for 2D (1, N) input, got {result.ndim}D"
        )
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
