# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp


@pytest.fixture
def ds_1d_with_coords():
    coord = scp.Coord(np.linspace(4000, 1000, 5), title="wavenumber", units="cm^-1")
    return scp.NDDataset(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), coordset=[coord])


@pytest.fixture
def ds_2d_with_coords(ds_1d_with_coords):
    coord = ds_1d_with_coords.coordset[-1]
    y_coord = scp.Coord([1.0, 2.0, 3.0], title="time", units="s")
    return scp.NDDataset(np.ones((3, 5)), coordset=[y_coord, coord])


def test_write_csv(mock_cwd):
    # 1D dataset without coords
    ds = scp.NDDataset([1, 2, 3])
    f = ds.write_csv("myfile.csv")
    assert f == mock_cwd / "myfile.csv"
    assert f.name == "myfile.csv"
    f.unlink()

    f = ds.write_csv("myfile")
    assert f == mock_cwd / "myfile.csv"
    assert f.name == "myfile.csv"
    f.unlink()


def test_write_csv_with_coords(mock_cwd, ds_1d_with_coords, ds_2d_with_coords):
    # 1D dataset with coords
    ds = ds_1d_with_coords
    f = ds.write_csv("myfile.csv")
    assert f.name == "myfile.csv"
    f.unlink()

    # 2D dataset with coords - should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        ds_2d_with_coords.write_csv("myfile.csv")

    # Relative path handling (issue 706)
    ds.write_csv("myfile.csv")
    assert mock_cwd.joinpath("myfile.csv").exists()
    f = ds.write_csv("../myfile.csv")
    assert f.name == "myfile.csv"
    assert scp.pathclean(mock_cwd.joinpath("../myfile.csv")).exists()
    f.unlink()


def test_write_csv_masked_values(tmp_path):
    # masked samples must be exported as missing values (NaN), not as their
    # (stale) underlying data (#1135), mirroring the JCAMP-DX writer (#1132)
    coord = scp.Coord(np.linspace(4000, 1000, 10), title="wavenumber", units="cm^-1")
    data = np.linspace(0.1, 0.9, 10)
    data[3:6] = 999.0  # sentinel hidden under the mask
    ds = scp.NDDataset(data, coordset=[coord], units="absorbance", name="masked")
    mask = np.zeros(10, dtype=bool)
    mask[3:6] = True
    ds.mask = mask

    f = ds.write_csv(tmp_path / "masked.csv", confirm=False)
    text = f.read_text()

    # the masked underlying value never leaks into the file
    assert "999" not in text
    # each masked sample is written as the missing-value marker (NaN)
    assert text.lower().count("nan") == 3

    # round-trip: the masked region reads back as NaN, the rest stays finite
    back = scp.read_csv(f)
    arr = np.asarray(back.data, dtype=float).ravel()
    assert np.isnan(arr[3:6]).all()
    assert np.isfinite(np.concatenate([arr[:3], arr[6:]])).all()
