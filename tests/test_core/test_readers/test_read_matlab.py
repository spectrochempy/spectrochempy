# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest
from scipy.io import savemat

from spectrochempy import NDDataset, read_matlab
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
    # read_matlab() imports each numeric variable of a .mat file as its own
    # NDDataset, named after the MATLAB variable, and skips non-numeric and
    # MATLAB-internal (``__header__``/``__version__``/``__globals__``)
    # variables (#1142). Distinct shapes are used so the importer keeps the
    # arrays as separate datasets rather than stacking same-shape arrays into
    # a single one.
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

    # the non-numeric (string) variable and the MATLAB internals are not
    # returned as datasets
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


def test_read_matlab_single_2d_array_preserves_values_without_materialized_coords(tmp_path):
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
    # same-shape numeric arrays are stacked by the importer into a single
    # NDDataset (the documented merge behaviour), so two (1, n) arrays come
    # back as one (2, n) dataset (#1142).
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
