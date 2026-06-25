# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest
from scipy.io import loadmat

import spectrochempy as scp


def test_write_matlab_1d_with_coordinate_and_metadata(tmp_path):
    coord = scp.Coord(
        np.linspace(4000.0, 1000.0, 5),
        title="wavenumber",
        units="cm^-1",
    )
    dataset = scp.NDDataset(
        np.linspace(0.1, 0.5, 5),
        coordset=[coord],
        name="spectrum",
        title="absorbance",
        units="mV",
        description="Simple exchange test",
    )

    path = tmp_path / "spectrum.mat"
    returned = dataset.write_matlab(path, confirm=False)

    assert returned == path
    exported = loadmat(path, simplify_cells=True)

    assert np.allclose(exported["data"], np.linspace(0.1, 0.5, 5))
    assert list(np.ravel(exported["dims"])) == ["x"]
    assert np.allclose(exported["coords"]["x"], np.linspace(4000.0, 1000.0, 5))
    assert exported["coord_units"]["x"] == "cm^-1"
    assert exported["coord_titles"]["x"] == "wavenumber"
    assert exported["name"] == "spectrum"
    assert exported["title"] == "absorbance"
    assert exported["units"] == "mV"
    assert exported["description"] == "Simple exchange test"


def test_write_matlab_2d_with_two_coordinates(tmp_path):
    x = scp.Coord(np.linspace(4000.0, 1000.0, 4), title="wavenumber", units="cm^-1")
    y = scp.Coord(np.array([0.0, 10.0, 20.0]), title="time", units="s")
    values = np.arange(12, dtype=float).reshape(3, 4)
    dataset = scp.NDDataset(values, coordset=[y, x], name="image", title="intensity")

    path = tmp_path / "image.mat"
    dataset.write_matlab(path, confirm=False)

    exported = loadmat(path, simplify_cells=True)

    assert exported["data"].shape == (3, 4)
    assert np.array_equal(exported["data"], values)
    assert list(np.ravel(exported["dims"])) == ["y", "x"]
    assert np.array_equal(exported["coords"]["x"], x.data)
    assert np.array_equal(exported["coords"]["y"], y.data)
    assert exported["coord_units"]["x"] == "cm^-1"
    assert exported["coord_units"]["y"] == "s"
    assert exported["coord_titles"]["x"] == "wavenumber"
    assert exported["coord_titles"]["y"] == "time"


def test_write_matlab_alias_uses_mat_suffix(tmp_path):
    dataset = scp.NDDataset(np.arange(5.0), name="trace")

    path = dataset.write_mat(tmp_path / "trace", confirm=False)

    assert path.name == "trace.mat"
    assert path.exists()


def test_write_matlab_generic_dispatch_uses_mat_protocol(tmp_path):
    dataset = scp.NDDataset(np.arange(6.0).reshape(2, 3), name="grid")

    path = dataset.write(tmp_path / "grid.mat", confirm=False)

    assert path.name == "grid.mat"
    exported = loadmat(path, simplify_cells=True)
    assert np.array_equal(exported["data"], np.arange(6.0).reshape(2, 3))


def test_write_matlab_rejects_unsupported_object():
    with pytest.raises(
        TypeError, match="the API write method needs a NDDataset object"
    ):
        scp.write_matlab(object(), "invalid.mat")


def test_write_matlab_rejects_complex_data(tmp_path):
    dataset = scp.NDDataset(np.array([1 + 1j, 2 + 0j]), name="complex")

    with pytest.raises(TypeError, match="does not support complex"):
        dataset.write_matlab(tmp_path / "complex.mat", confirm=False)


def test_write_matlab_rejects_higher_dimensional_data(tmp_path):
    dataset = scp.NDDataset(np.arange(24.0).reshape(2, 3, 4), name="cube")

    with pytest.raises(NotImplementedError, match="only implemented for 1D and 2D"):
        dataset.write_matlab(tmp_path / "cube.mat", confirm=False)


def test_write_matlab_does_not_modify_source_dataset(tmp_path):
    dataset = scp.NDDataset(np.arange(5.0), name="trace")
    original_filename = dataset.filename

    dataset.write_matlab(tmp_path / "trace.mat", confirm=False)

    assert dataset.filename == original_filename
