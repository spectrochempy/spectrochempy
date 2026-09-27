# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import csv

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs


# ======================================================================================
# Round-trip tests for CSV read/write (issue #1077)
# These tests use synthetic data and do not require external test data
# ======================================================================================


def test_read_csv_roundtrip_no_coords(tmp_path):
    """Test that a 1D dataset without coords can be written and read back."""
    ds = scp.NDDataset([1.0, 2.0, 3.0, 4.0, 5.0])
    filepath = tmp_path / "test_no_coords.csv"
    ds.write_csv(filepath)

    # Read it back
    loaded = scp.read_csv(filepath)
    # read_csv always creates 2D datasets, so squeeze to compare
    assert loaded.squeeze().shape == ds.shape
    assert np.allclose(loaded.data.squeeze(), ds.data)


def test_read_csv_roundtrip_with_coords(tmp_path):
    """Test that a 1D dataset with coords can be written and read back."""
    coord = scp.Coord(np.linspace(4000, 1000, 5), title="wavenumber", units="cm^-1")
    ds = scp.NDDataset(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), coordset=[coord])
    filepath = tmp_path / "test_with_coords.csv"
    ds.write_csv(filepath)

    # Read it back
    loaded = scp.read_csv(filepath)
    # read_csv always creates 2D datasets, so squeeze to compare
    assert loaded.squeeze().shape == ds.shape
    assert np.allclose(loaded.data.squeeze(), ds.data)
    assert np.allclose(loaded.x.data, ds.x.data)


def test_read_csv_roundtrip_semicolon(tmp_path):
    """Test roundtrip with semicolon delimiter."""
    ds = scp.NDDataset([1.0, 2.0, 3.0])
    filepath = tmp_path / "test_semicolon.csv"
    ds.write_csv(filepath, delimiter=";")

    loaded = scp.read_csv(filepath, csv_delimiter=";")
    assert loaded.squeeze().shape == ds.shape
    assert np.allclose(loaded.data.squeeze(), ds.data)


def test_read_csv_roundtrip_with_units(tmp_path):
    """Test that units are preserved in roundtrip."""
    coord = scp.Coord(np.linspace(100, 500, 5), title="wavelength", units="nm")
    ds = scp.NDDataset(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]), coordset=[coord], units="absorbance"
    )
    filepath = tmp_path / "test_with_units.csv"
    ds.write_csv(filepath)

    loaded = scp.read_csv(filepath)
    assert loaded.squeeze().shape == ds.shape
    assert np.allclose(loaded.data.squeeze(), ds.data)
    assert np.allclose(loaded.x.data, ds.x.data)


def test_read_csv_roundtrip_preserves_simple_scp_metadata_header(tmp_path):
    coord = scp.Coord(np.linspace(100, 500, 5), title="wavelength", units="nm")
    ds = scp.NDDataset(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        coordset=[coord],
        title="absorbance",
        units="absorbance",
    )
    filepath = tmp_path / "test_metadata_roundtrip.csv"
    ds.write_csv(filepath)

    loaded = scp.read_csv(filepath)

    assert loaded.title == "absorbance"
    assert loaded.units == ds.units
    assert loaded.x.title == "wavelength"
    assert loaded.x.units == ds.x.units
    assert np.allclose(loaded.x.data, ds.x.data)
    assert np.allclose(loaded.data.squeeze(), ds.data)


def test_read_csv_external_header_without_scp_metadata_keeps_current_semantics(
    tmp_path,
):
    filepath = tmp_path / "external.csv"
    filepath.write_text("x,y\n1,10\n2,20\n3,30\n")

    loaded = scp.read_csv(filepath)

    assert loaded.name == "external"
    assert loaded.title == "<untitled>"
    assert loaded.units is None
    assert loaded.x.title == "<untitled>"
    assert loaded.x.units is None
    assert np.allclose(loaded.x.data, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(loaded.data.squeeze(), np.array([10.0, 20.0, 30.0]))


def test_read_csv_numeric_first_row_is_data():
    content = "0,1.25\n0.5,2.5\n1,3.75\n"

    loaded = scp.read_csv({"no_header.csv": content.encode("utf-8")})

    np.testing.assert_allclose(loaded.x.data, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(loaded.data.ravel(), [1.25, 2.5, 3.75])


@pytest.mark.parametrize(
    ("coord_units", "dataset_units", "expected_header"),
    [
        (None, "V", ["elapsed time", "signal / V"]),
        ("s", None, ["elapsed time / s", "signal"]),
        ("s", "V", ["elapsed time / s", "signal / V"]),
    ],
)
def test_read_csv_roundtrip_preserves_partial_metadata_header(
    tmp_path,
    coord_units,
    dataset_units,
    expected_header,
):
    coord = scp.Coord([0.0, 0.5, 1.0], title="elapsed time", units=coord_units)
    dataset = scp.NDDataset(
        [1.25, 2.5, 3.75],
        coordset=[coord],
        title="signal",
        units=dataset_units,
    )
    filepath = tmp_path / "partial_metadata.csv"

    dataset.write_csv(filepath, confirm=False)

    with filepath.open(newline="") as fid:
        assert next(csv.reader(fid)) == expected_header

    loaded = scp.read_csv(filepath)

    np.testing.assert_allclose(loaded.x.data, dataset.x.data)
    np.testing.assert_allclose(loaded.data.ravel(), dataset.data)
    assert loaded.x.title == dataset.x.title
    assert loaded.x.units == dataset.x.units
    assert loaded.title == dataset.title
    assert loaded.units == dataset.units


def test_read_csv_roundtrip_title_only_header_remains_ambiguous(tmp_path):
    coord = scp.Coord([0.0, 0.5, 1.0], title="elapsed time", units=None)
    dataset = scp.NDDataset(
        [1.25, 2.5, 3.75],
        coordset=[coord],
        title="signal",
        units=None,
    )
    filepath = tmp_path / "title_only.csv"

    dataset.write_csv(filepath, confirm=False)

    with filepath.open(newline="") as fid:
        assert next(csv.reader(fid)) == ["elapsed time", "signal"]

    loaded = scp.read_csv(filepath)

    np.testing.assert_allclose(loaded.x.data, dataset.x.data)
    np.testing.assert_allclose(loaded.data.ravel(), dataset.data)
    assert loaded.x.title == "<untitled>"
    assert loaded.x.units is None
    assert loaded.title == "<untitled>"
    assert loaded.units is None


def test_read_csv_roundtrip_preserves_explicit_dimensionless_units(tmp_path):
    coord = scp.Coord(
        [0.0, 0.5, 1.0],
        title="relative position",
        units="dimensionless",
    )
    dataset = scp.NDDataset(
        [1.25, 2.5, 3.75],
        coordset=[coord],
        title="normalized signal",
        units="dimensionless",
    )
    filepath = tmp_path / "dimensionless.csv"

    dataset.write_csv(filepath, confirm=False)

    with filepath.open(newline="") as fid:
        assert next(csv.reader(fid)) == ["relative position / ", "normalized signal / "]

    loaded = scp.read_csv(filepath)

    assert loaded.x.title == "relative position"
    assert loaded.x.units == scp.ur.dimensionless
    assert loaded.title == "normalized signal"
    assert loaded.units == scp.ur.dimensionless
    np.testing.assert_allclose(loaded.x.data, dataset.x.data)
    np.testing.assert_allclose(loaded.data.ravel(), dataset.data)


@pytest.mark.parametrize("dataset_units", [None, "V"])
def test_read_csv_roundtrip_single_column_metadata(tmp_path, dataset_units):
    dataset = scp.NDDataset(
        [1.25, 2.5, 3.75],
        title="signal",
        units=dataset_units,
    )
    filepath = tmp_path / "single_column.csv"

    dataset.write_csv(filepath, confirm=False)

    with filepath.open(newline="") as fid:
        expected = ["signal"] if dataset_units is None else ["signal / V"]
        assert next(csv.reader(fid)) == expected

    loaded = scp.read_csv(filepath)

    np.testing.assert_allclose(loaded.data.ravel(), dataset.data)
    np.testing.assert_allclose(loaded.x.data, np.arange(dataset.size))
    assert loaded.title == ("<untitled>" if dataset_units is None else "signal")
    assert loaded.units == dataset.units


def test_read_csv_partial_metadata_respects_explicit_dataset_options(tmp_path):
    coord = scp.Coord([0.0, 0.5, 1.0], title="elapsed time", units=None)
    dataset = scp.NDDataset(
        [1.25, 2.5, 3.75],
        coordset=[coord],
        title="signal",
        units="V",
    )
    filepath = tmp_path / "explicit_options.csv"
    dataset.write_csv(filepath, confirm=False)

    loaded = scp.read_csv(filepath, title="current", units="A")

    assert loaded.title == "current"
    assert loaded.units == scp.ur.A
    assert loaded.x.title == "elapsed time"
    assert loaded.x.units is None
    np.testing.assert_allclose(loaded.x.data, dataset.x.data)
    np.testing.assert_allclose(loaded.data.ravel(), dataset.data)


def test_read_csv_unknown_header_unit_is_not_guessed():
    content = "elapsed time / not_a_unit,signal / V\n0,1.25\n0.5,2.5\n1,3.75\n"

    with pytest.warns(UserWarning, match="'not_a_unit' is not defined"):
        loaded = scp.read_csv({"unknown_unit.csv": content.encode("utf-8")})

    assert loaded is None


def test_read_csv_malformed_header_unit_preserves_other_valid_metadata():
    content = " / s,signal / V\n0,1.25\n0.5,2.5\n1,3.75\n"

    loaded = scp.read_csv({"malformed_unit.csv": content.encode("utf-8")})

    assert loaded.x.title == "<untitled>"
    assert loaded.x.units is None
    assert loaded.title == "signal"
    assert loaded.units == scp.ur.V
    np.testing.assert_allclose(loaded.x.data, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(loaded.data.ravel(), [1.25, 2.5, 3.75])
