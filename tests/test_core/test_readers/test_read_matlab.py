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


# --------------------------------------------------------------------------------------
# #1270 - reconstruction of write_matlab() minimal exchange payloads
# --------------------------------------------------------------------------------------
def test_read_matlab_reconstructs_write_matlab_exchange_payload(tmp_path):
    # Full round trip: a dataset written with write_matlab() must come back
    # from read_matlab() as a single NDDataset with its data, name, title,
    # units, description, and both coordinates (values, units, titles)
    # intact (#1270).
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
    # A dataset written without explicit coordinates should still round-trip
    # its values, name, title, units, and description; the reconstructed
    # dataset simply has no coordset (#1270).
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
    # Regression test (#1270): before the fix, any .mat file containing a
    # plain MATLAB cell array (a numpy object array with dtype.names is
    # None) crashed read_matlab() entirely -- first with an unguarded
    # TypeError in the DSO-signature check (surfaced only as a UserWarning,
    # with the function silently returning None), and, once that guard is
    # added, with an AttributeError in merge_datasets() because the
    # unrecognized variable was appended as a raw [name, data] list instead
    # of being skipped. Neither should happen: the numeric variable must
    # still come back as a valid NDDataset, and the cell array must simply
    # be skipped with a warning.
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
    # Safety net for the signature-based detection (#1270): a file that only
    # partially overlaps write_matlab()'s key set (here, missing name,
    # title, coord_units, and coord_titles) must not be mistaken for an
    # exchange payload. It falls back to the generic per-variable import
    # path instead of being incorrectly reconstructed.
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
    # Regression test (#1270): dimension names read from the `dims` field
    # must actually be applied to the reconstructed NDDataset, not just
    # used internally to look up coordinates. A dataset using SpectroChemPy's
    # default dimension names (`y`, `x`) would pass even if the stored names
    # were silently ignored, since a freshly constructed NDDataset defaults
    # to those names anyway -- so this uses non-default names (`v`, `q`) to
    # actually exercise the restoration.
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
    # Regression test (#1270): having all the right variable names is not
    # enough to be treated as a write_matlab() exchange payload -- the
    # fields must also have the expected structure. Here `coords`,
    # `coord_units`, and `coord_titles` are plain numeric arrays instead of
    # MATLAB structs, so this must fall back to the generic per-variable
    # import path (and must not crash) rather than being misreconstructed.
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
    # Regression test (#1270): the number of names in `dims` must match the
    # number of dimensions in `data`. A mismatch (here, one dimension name
    # for 2D data) indicates the file does not actually match the
    # write_matlab() contract, so it must fall back to the generic
    # per-variable import path instead of being misreconstructed.
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
    # Edge case uncovered while tightening the detection in
    # _is_scp_matlab_exchange_payload (#1270): when a dataset has no
    # coordinates at all, write_matlab() writes empty {} dicts for
    # coords/coord_units/coord_titles. scipy round-trips an empty dict as a
    # bare `array([[None]], dtype=object)` rather than a structured array,
    # which must still be recognized as a valid (empty) exchange payload
    # rather than rejected as malformed.
    original = NDDataset(np.arange(6.0).reshape(1, 6))
    original.name = "no_coords"

    path = tmp_path / "no_coords_exchange.mat"
    original.write_matlab(path)

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.name == "no_coords"
    assert np.allclose(result.data, original.data)


def test_read_matlab_exchange_payload_true_1d_round_trip(tmp_path):
    # Regression test (#1270): scipy.io.loadmat() always reads MATLAB
    # arrays back as at least 2D, so a genuinely 1D NDDataset (ndim == 1,
    # a single dimension name) is written by write_matlab() with one dims
    # entry, but its `data` variable comes back from loadmat() as a
    # (1, n) row vector -- dims.size == 1 while data.ndim == 2. Without
    # accounting for this, the payload is wrongly rejected as malformed
    # and falls back to the generic per-variable import path, which loses
    # the dataset's name, title, units, and description, and returns a
    # 2D (y:1, x:n) dataset instead of a true 1D one.
    x = Coord(np.linspace(0.0, 9.0, 10), units="s", title="time")
    original = NDDataset(np.arange(10.0), coordset=[x])
    original.name = "trueoned"
    original.title = "signal"
    original.units = "V"
    original.description = "a genuinely 1D dataset"

    path = tmp_path / "true_1d_exchange.mat"
    original.write_matlab(path)

    result = read_matlab(path)

    assert isinstance(result, NDDataset)
    assert result.ndim == 1
    assert result.shape == (10,)
    assert result.dims == ["x"]
    assert result.name == "trueoned"
    assert result.title == "signal"
    assert str(result.units) == str(original.units)
    assert result.description == "a genuinely 1D dataset"
    assert np.allclose(result.data, original.data)
    assert np.allclose(result.coord("x").data, x.data)
    assert result.coord("x").title == "time"


def test_read_matlab_exchange_detection_still_rejects_genuine_dims_mismatch_for_2d(
    tmp_path,
):
    # Safety net alongside the true-1D fix above: the (1, n) row-vector
    # allowance must only apply when data actually collapses to a single
    # row. A genuinely 2D array (more than one row) with too few dims
    # names is still a real mismatch and must still fall back to the
    # generic path, not be swept in by the same exception.
    path = tmp_path / "genuine_2d_mismatch.mat"
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
