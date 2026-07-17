# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest
from scipy.io import savemat

from spectrochempy import Coord, NDDataset, read_matlab
from spectrochempy import preferences as prefs

MATLABDATA = prefs.datadir / "matlabdata"


@pytest.fixture
def matlabdata():
    if not MATLABDATA.exists():
        pytest.skip("test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")
    return MATLABDATA


@pytest.mark.data
def test_read_matlab(matlabdata):
    A = read_matlab(matlabdata / "als2004dataset.MAT")
    assert len(A) == 6
    assert A[3].shape == (4, 96)

    A = read_matlab(matlabdata / "dso.mat")
    assert A.name == "Group sust_base line withoutEQU.SPG"
    assert A.shape == (20, 426)


def test_read_matlab_multiple_variables(tmp_path):
    path = tmp_path / "multi.mat"
    savemat(
        path,
        {
            "alpha": np.linspace(0.0, 1.0, 5).reshape(1, 5),
            "beta": np.linspace(1.0, 2.0, 7).reshape(1, 7),
            "label": "a non-numeric variable",
        },
    )

    datasets = read_matlab(path)

    # one NDDataset per numeric variable, named after the MATLAB variable
    assert isinstance(datasets, list)
    assert all(isinstance(ds, NDDataset) for ds in datasets)
    names = {ds.name for ds in datasets}
    assert names == {"alpha", "beta"}

    shapes = {ds.name: ds.shape for ds in datasets}
    assert shapes["alpha"] == (1, 5)
    assert shapes["beta"] == (1, 7)

    assert "label" not in names
    assert not any(ds.name.startswith("__") for ds in datasets)


def test_read_matlab_single_1d_array_currently_uses_row_shape_and_no_coordset(tmp_path):
    path = tmp_path / "single_1d.mat"
    savemat(path, {"trace": np.arange(5)})

    dataset = read_matlab(path)

    assert isinstance(dataset, NDDataset)
    assert dataset.name == "trace"
    assert dataset.shape == (1, 5)
    assert np.array_equal(dataset.data, np.arange(5).reshape(1, 5))
    assert dataset.origin == "matlab"
    assert dataset.coordset is None
    assert dataset.x is None
    assert dataset.y is None
    assert any("Imported from .mat file" in str(entry) for entry in dataset.history)


def test_read_matlab_single_2d_array_preserves_values_without_materialized_coords(
    tmp_path,
):
    path = tmp_path / "single_2d.mat"
    values = np.arange(6).reshape(2, 3)
    savemat(path, {"image": values})

    dataset = read_matlab(path)

    assert isinstance(dataset, NDDataset)
    assert dataset.name == "image"
    assert dataset.shape == (2, 3)
    assert np.array_equal(dataset.data, values)
    assert dataset.origin == "matlab"
    assert dataset.coordset is None
    assert dataset.x is None
    assert dataset.y is None
    assert any("Imported from .mat file" in str(entry) for entry in dataset.history)


def test_read_matlab_same_shape_arrays_are_stacked(tmp_path):
    path = tmp_path / "stack.mat"
    savemat(
        path,
        {
            "spectrum1": np.linspace(0.0, 1.0, 5).reshape(1, 5),
            "spectrum2": np.linspace(1.0, 2.0, 5).reshape(1, 5),
        },
    )

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.shape == (2, 5)


def test_read_matlab_three_dimensional_array_preserves_shape_and_values(tmp_path):
    path = tmp_path / "cube.mat"
    values = np.arange(24).reshape((2, 3, 4), order="F")
    savemat(path, {"cube": values})

    dataset = read_matlab(path)

    assert isinstance(dataset, NDDataset)
    assert dataset.name == "cube"
    assert dataset.shape == (2, 3, 4)
    assert np.array_equal(dataset.data, values)
    assert dataset.origin == "matlab"
    assert dataset.coordset is None


def test_read_matlab_reconstructs_write_matlab_exchange_payload(tmp_path):
    x = Coord(np.linspace(4000.0, 1000.0, 6), units="cm^-1", title="wavenumber")
    y = Coord(np.array([0.0, 10.0]), units="s", title="time")
    original = NDDataset(np.random.rand(2, 6), coordset=[y, x])
    original.name = "myspectrum"
    original.title = "absorbance"
    original.units = "absorbance"
    original.description = "round trip check"

    path = tmp_path / "exchange.mat"
    original.write_matlab(path)

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.name == "myspectrum"
    assert result.title == "absorbance"
    assert str(result.units) == str(original.units)
    assert result.description == "round trip check"
    assert result.dims == original.dims
    assert result.shape == original.shape
    assert np.allclose(result.data, original.data)

    assert result.coordset is not None
    assert np.allclose(result.coord("x").data, x.data)
    assert str(result.coord("x").units) == str(x.units)
    assert result.coord("x").title == "wavenumber"
    assert np.allclose(result.coord("y").data, y.data)
    assert str(result.coord("y").units) == str(y.units)
    assert result.coord("y").title == "time"


def test_read_matlab_exchange_payload_without_coordinates(tmp_path):
    original = NDDataset(np.arange(10.0).reshape(2, 5))
    original.name = "plain"
    original.title = "raw signal"

    path = tmp_path / "exchange_no_coords.mat"
    original.write_matlab(path)

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.name == "plain"
    assert result.title == "raw signal"
    assert result.shape == (2, 5)
    assert np.allclose(result.data, original.data)


def test_read_matlab_does_not_crash_on_plain_cell_array_variable(tmp_path):
    path = tmp_path / "generic_with_cell.mat"
    savemat(
        path,
        {
            "signal": np.linspace(0.0, 1.0, 5).reshape(1, 5),
            "channel_names": np.array(["ch1", "ch2"], dtype=object),
        },
    )

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.name == "signal"
    assert result.shape == (1, 5)


def test_read_matlab_partial_exchange_keys_use_generic_path(tmp_path):
    path = tmp_path / "partial_signature.mat"
    savemat(
        path,
        {
            "data": np.linspace(0.0, 1.0, 5).reshape(1, 5),
            "dims": np.array(["x"], dtype=object),
            "coords": {"x": np.arange(5.0)},
        },
    )

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.name == "data"
    assert result.shape == (1, 5)


def test_read_matlab_exchange_payload_restores_non_default_dimension_names(tmp_path):
    q = Coord(np.linspace(4000.0, 1000.0, 6), units="cm^-1", title="wavenumber")
    v = Coord(np.array([0.0, 10.0]), units="s", title="time")
    original = NDDataset(np.random.rand(2, 6), dims=["v", "q"], coordset=[v, q])
    original.name = "myspectrum"

    path = tmp_path / "nondefault_dims.mat"
    original.write_matlab(path)

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.dims == ["v", "q"]
    assert result.shape == original.shape
    assert np.allclose(result.data, original.data)
    assert np.allclose(result.coord("q").data, q.data)
    assert result.coord("q").title == "wavenumber"
    assert np.allclose(result.coord("v").data, v.data)
    assert result.coord("v").title == "time"


def test_read_matlab_exchange_detection_rejects_wrong_field_structure(tmp_path):
    path = tmp_path / "wrong_structure.mat"
    savemat(
        path,
        {
            "data": np.linspace(0.0, 1.0, 5).reshape(1, 5),
            "dims": np.array(["x"], dtype=object),
            "coords": np.array([1, 2, 3]),
            "coord_units": np.array([1, 2, 3]),
            "coord_titles": np.array([1, 2, 3]),
            "name": "somefile",
            "title": "sometitle",
        },
    )

    result = read_matlab(path)

    names = {ds.name for ds in result} if isinstance(result, list) else {result.name}
    assert "somefile" not in names


def test_read_matlab_exchange_detection_rejects_dims_length_mismatch(tmp_path):
    path = tmp_path / "dims_mismatch.mat"
    savemat(
        path,
        {
            "data": np.linspace(0.0, 1.0, 10).reshape(2, 5),
            "dims": np.array(["x"], dtype=object),
            "coords": {"x": np.arange(5.0)},
            "coord_units": {"x": "s"},
            "coord_titles": {"x": "time"},
            "name": "mismatched",
            "title": "t",
        },
    )

    result = read_matlab(path)

    assert getattr(result, "name", None) != "mismatched"


def test_read_matlab_exchange_payload_without_coordinates_round_trips_via_writer(
    tmp_path,
):
    original = NDDataset(np.arange(6.0).reshape(1, 6))
    original.name = "no_coords"

    path = tmp_path / "no_coords_exchange.mat"
    original.write_matlab(path)

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.name == "no_coords"
    assert np.allclose(result.data, original.data)
