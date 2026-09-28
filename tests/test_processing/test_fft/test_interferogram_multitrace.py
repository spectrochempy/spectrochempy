# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Regression tests for multidimensional interferogram FFTs."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.units import ur


def _trace(size, zpd, frequency):
    index = np.arange(size)
    envelope = np.exp(-((index - zpd) / 5.0) ** 2)
    return envelope * np.cos(frequency * (index - zpd))


def _interferogram_dataset(zpds, *, shape=None):
    size = 64
    flat_zpds = np.asarray(zpds).reshape(-1)
    traces = np.asarray(
        [_trace(size, zpd, 0.31 + 0.02 * i) for i, zpd in enumerate(flat_zpds)]
    )

    if shape is None:
        shape = (len(flat_zpds), size)
    data = traces.reshape(*shape[:-1], size)
    x = scp.Coord(np.arange(size) * 0.5, title="time", units="ps")

    if len(shape) == 2:
        y = scp.Coord(
            np.arange(shape[0]) * 2.0,
            labels=[f"trace-{i}" for i in range(shape[0])],
            title="delay",
            units="s",
        )
        coordset = [y, x]
    else:
        z = scp.Coord(np.arange(shape[0]), title="series")
        y = scp.Coord(np.arange(shape[1]) * 2.0, title="delay", units="s")
        coordset = [z, y, x]

    dataset = scp.NDDataset(
        data,
        coordset=coordset,
        meta={"interferogram": True, "td": list(shape)},
    )
    dataset.name = "multitrace-interferogram"
    dataset.history = "synthetic source"
    return dataset


def _transform_each_trace(dataset, dim="x"):
    axis, dim = dataset.get_axis(dim, negative_axis=False)
    moved = np.moveaxis(dataset.data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    coordinate = dataset.coordset[dim]
    rows = []

    for trace in flat:
        single = scp.NDDataset(
            trace[np.newaxis, :],
            coordset=[scp.Coord([0.0]), coordinate.copy()],
            meta={"interferogram": True, "td": [1, trace.size]},
        )
        transformed = single.fft()
        assert np.isfinite(transformed.data).all()
        rows.append(transformed.data[0])

    transformed = np.asarray(rows).reshape(*moved.shape[:-1], rows[0].size)
    return np.moveaxis(transformed, -1, axis)


def _assert_unchanged(dataset, before):
    np.testing.assert_array_equal(dataset.data, before.data)
    np.testing.assert_array_equal(dataset.mask, before.mask)
    assert dataset.shape == before.shape
    assert dataset.dims == before.dims
    assert dataset.coordset.references == before.coordset.references
    for dim in dataset.dims:
        coordinate = dataset.coordset[dim]
        reference = before.coordset[dim]
        np.testing.assert_array_equal(coordinate.data, reference.data)
        if coordinate.labels is None or reference.labels is None:
            assert coordinate.labels is reference.labels
        else:
            np.testing.assert_array_equal(coordinate.labels, reference.labels)
        assert coordinate.name == reference.name
        assert coordinate.title == reference.title
        assert coordinate.units == reference.units
        assert coordinate.linear == reference.linear
        assert coordinate.meta == reference.meta
        assert coordinate._zpd == reference._zpd
        assert coordinate._use_time_axis == reference._use_time_axis
    assert dataset.meta == before.meta
    assert dataset.history_entries == before.history_entries


def test_single_interferogram_numerical_convention_is_unchanged():
    data = np.asarray([[0.1, 0.5, 1.5, 5.0, 2.0, 0.7, 0.2, 0.1]])
    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord([0.0]), scp.Coord.arange(8) * (0.5 * ur.ps)],
        meta={"interferogram": True, "td": [1, 8]},
    )

    transformed = dataset.fft()

    expected = np.asarray(
        [[0.6185169325199483, 0.9282050893933123, 1.7157848354955576, 2.65]]
    )
    np.testing.assert_allclose(transformed.data, expected, rtol=1.0e-14)
    assert np.isfinite(transformed.data).all()


def test_multitrace_fft_matches_individual_transforms_and_preserves_source():
    dataset = _interferogram_dataset([8, 13, 19])
    before = dataset.copy()
    expected = _transform_each_trace(dataset)

    assert np.argmax(np.abs(dataset.data), axis=-1).tolist() == [8, 13, 19]

    transformed = dataset.fft()

    assert np.isfinite(transformed.data).all()
    np.testing.assert_allclose(transformed.data, expected, rtol=1.0e-13, atol=1.0e-15)
    assert transformed.shape == (3, 32)
    assert transformed.dims == dataset.dims
    assert transformed.y == dataset.y
    assert transformed.x.units == ur("cm^-1")
    assert transformed.x.size == transformed.shape[-1]
    _assert_unchanged(dataset, before)

    work = dataset.copy()
    returned = work.fft(inplace=True)
    assert returned is work
    assert np.isfinite(work.data).all()
    np.testing.assert_allclose(work.data, expected, rtol=1.0e-13, atol=1.0e-15)
    assert work.y == dataset.y


def test_multitrace_fft_on_nonfinal_dimension_matches_individual_transforms():
    final_dimension = _interferogram_dataset([8, 13, 19])
    expected = final_dimension.fft()
    dataset = final_dimension.T
    before = dataset.copy()

    transformed = dataset.fft(dim="x")

    assert np.isfinite(transformed.data).all()
    np.testing.assert_allclose(
        transformed.data,
        expected.data.T,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    assert transformed.shape == (32, 3)
    assert transformed.dims == dataset.dims
    assert transformed.y == dataset.y
    assert transformed.x == expected.x
    _assert_unchanged(dataset, before)

    work = dataset.copy()
    returned = work.fft(dim="x", inplace=True)
    assert returned is work
    assert np.isfinite(work.data).all()
    np.testing.assert_allclose(work.data, expected.data.T, rtol=1.0e-13, atol=1.0e-15)
    assert work.dims == dataset.dims
    assert work.y == dataset.y


def test_three_dimensional_interferogram_fft_transforms_every_trace():
    dataset = _interferogram_dataset(
        [[8, 11, 14], [17, 20, 23]],
        shape=(2, 3, 64),
    )
    before = dataset.copy()
    expected = _transform_each_trace(dataset)

    transformed = dataset.fft()

    assert np.isfinite(transformed.data).all()
    np.testing.assert_allclose(transformed.data, expected, rtol=1.0e-13, atol=1.0e-15)
    assert transformed.shape == (2, 3, 32)
    assert transformed.dims == dataset.dims
    assert transformed.z == dataset.z
    assert transformed.y == dataset.y
    assert transformed.x.units == ur("cm^-1")
    _assert_unchanged(dataset, before)


@pytest.mark.data
def test_calibrated_ig_multi_matches_individual_transforms():
    path = prefs.datadir / "galacticdata" / "IG_MULTI.SPC"
    if not path.exists():
        pytest.skip("IG_MULTI.SPC is not available")

    dataset = scp.read_spc(path)
    dataset.x.set_laser_frequency(15798.26 * ur("cm^-1"))
    before = dataset.copy()
    expected = _transform_each_trace(dataset)

    transformed = dataset.fft()

    assert np.isfinite(transformed.data).all()
    np.testing.assert_allclose(transformed.data, expected, rtol=1.0e-13, atol=1.0e-15)
    assert transformed.shape == (10, 2048)
    assert transformed.dims == dataset.dims
    assert transformed.y == dataset.y
    assert transformed.x.units == ur("cm^-1")
    assert transformed.x.size == transformed.shape[-1]
    _assert_unchanged(dataset, before)
