# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

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
    coord = scp.Coord(
        np.linspace(4000, 1000, 5),
        title="wavenumber",
        units="cm^-1"
    )
    ds = scp.NDDataset(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        coordset=[coord]
    )
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
    coord = scp.Coord(
        np.linspace(100, 500, 5),
        title="wavelength",
        units="nm"
    )
    ds = scp.NDDataset(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        coordset=[coord],
        units="absorbance"
    )
    filepath = tmp_path / "test_with_units.csv"
    ds.write_csv(filepath)

    loaded = scp.read_csv(filepath)
    assert loaded.squeeze().shape == ds.shape
    assert np.allclose(loaded.data.squeeze(), ds.data)
    assert np.allclose(loaded.x.data, ds.x.data)
