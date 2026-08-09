# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import warnings

import numpy as np
import pytest

import spectrochempy as scp


def _dataset_without_y():
    x = scp.Coord(np.linspace(4000.0, 1000.0, 6), units="cm^-1", title="wavenumber")
    ds = scp.NDDataset(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), coordset=[x])
    ds.name = "no_y"
    return ds


def _dataset_with_y():
    x = scp.Coord(np.linspace(4000.0, 1000.0, 6), units="cm^-1", title="wavenumber")
    y = scp.Coord([0.0])
    return scp.NDDataset(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(1, 6),
        coordset=[y, x],
        units="absorbance",
        name="valid_1d",
    )


def _complex_dataset_with_y():
    x = scp.Coord([0.0, 1.0])
    y = scp.Coord([0.0, 10.0])
    return scp.NDDataset(
        np.array([[1 + 2j, 2 + 1j], [3 + 0j, 4 + 1j]]),
        coordset=[y, x],
        name="complex",
    )


def test_write_jcamp_rejects_dataset_without_y_coordinate(tmp_path):
    dataset = _dataset_without_y()
    target = tmp_path / "no_y.jdx"
    original_filename = dataset.filename

    with pytest.raises(ValueError, match="`y` coordinate"):
        dataset.write_jcamp(target, confirm=False)

    assert not target.exists()
    assert dataset.filename == original_filename


def test_write_jcamp_rejects_dataset_without_y_keeps_existing_file(tmp_path):
    dataset = _dataset_without_y()
    target = tmp_path / "no_y.jdx"
    target.write_text("SENTINEL")
    original_filename = dataset.filename

    with pytest.raises(ValueError, match="`y` coordinate"):
        dataset.write_jcamp(target, confirm=False, overwrite=True)

    assert target.read_text() == "SENTINEL"
    assert dataset.filename == original_filename


def test_write_jcamp_rejects_complex_data(tmp_path):
    dataset = _complex_dataset_with_y()
    target = tmp_path / "complex.jdx"
    original_filename = dataset.filename

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=np.exceptions.ComplexWarning)
        with pytest.raises(TypeError, match="does not support complex"):
            dataset.write_jcamp(target, confirm=False)

    assert not target.exists()
    assert dataset.filename == original_filename


def test_write_jcamp_rejects_complex_data_keeps_existing_file(tmp_path):
    dataset = _complex_dataset_with_y()
    target = tmp_path / "complex.jdx"
    target.write_text("SENTINEL")
    original_filename = dataset.filename

    with pytest.raises(TypeError, match="does not support complex"):
        dataset.write_jcamp(target, confirm=False, overwrite=True)

    assert target.read_text() == "SENTINEL"
    assert dataset.filename == original_filename


def test_write_jcamp_generic_dispatch_applies_validation(tmp_path):
    no_y = _dataset_without_y()
    target = tmp_path / "dispatch.jdx"

    with pytest.raises(ValueError, match="`y` coordinate"):
        scp.write(no_y, target, confirm=False)

    assert not target.exists()

    complex_ds = _complex_dataset_with_y()
    target = tmp_path / "dispatch_complex.jdx"

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=np.exceptions.ComplexWarning)
        with pytest.raises(TypeError, match="does not support complex"):
            scp.write(complex_ds, target, confirm=False)

    assert not target.exists()


def test_write_jcamp_valid_dataset_round_trip(tmp_path):
    dataset = _dataset_with_y()

    path = dataset.write_jcamp(tmp_path / "valid.jdx", confirm=False)

    assert path.exists()
    assert path.stat().st_size > 0
    text = path.read_text()
    assert "##JCAMP-DX=5.01" in text
    assert "##DATA TYPE=INFRARED SPECTRUM" in text
    assert "##XYDATA=(X++(Y..Y))" in text

    back = scp.read_jcamp(path)
    assert back.shape == (1, 6)
    assert np.allclose(back.data, dataset.data)
