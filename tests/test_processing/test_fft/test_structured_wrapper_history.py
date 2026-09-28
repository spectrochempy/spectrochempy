"""Structured history for shared shift and zero-filling wrappers."""

import copy

import numpy as np
import pytest

import spectrochempy as scp


def _shift_dataset(*, complex_data=False):
    data = np.arange(12.0).reshape(3, 4)
    if complex_data:
        data = data + 1.0j * np.arange(12.0).reshape(3, 4)
    dataset = scp.NDDataset(
        data,
        dims=["y", "x"],
        coordset=[
            scp.Coord.arange(3, units="s", title="time y"),
            scp.Coord.arange(4, units="s", title="time x"),
        ],
        units="V",
        title="processing history",
    )
    dataset.meta.sample = "synthetic"
    dataset.annotate("prepared")
    return dataset


def _zero_fill_dataset():
    dataset = _shift_dataset()
    dataset.meta.td = [3, 4]
    return dataset


def _snapshot(dataset):
    return {
        "data": dataset.data.copy(),
        "mask": dataset.mask.copy(),
        "dims": list(dataset.dims),
        "coords": {dim: dataset.coord(dim).copy() for dim in dataset.dims},
        "units": dataset.units,
        "meta": copy.deepcopy(dataset.meta),
        "history": dataset.history_entries,
    }


def _assert_snapshot(dataset, snapshot):
    np.testing.assert_array_equal(dataset.data, snapshot["data"])
    np.testing.assert_array_equal(dataset.mask, snapshot["mask"])
    assert dataset.dims == snapshot["dims"]
    for dim, coord in snapshot["coords"].items():
        assert dataset.coord(dim) == coord
    assert dataset.units == snapshot["units"]
    assert dataset.meta == snapshot["meta"]
    assert dataset.history_entries == snapshot["history"]


@pytest.mark.parametrize("inplace", [False, True])
def test_roll_records_one_structured_nonfinal_shift(inplace):
    source = _shift_dataset()
    source.mask = np.array(
        [
            [False, True, False, False],
            [True, False, False, False],
            [False, False, True, False],
        ]
    )
    snapshot = _snapshot(source)

    result = source.roll(pts=1, neg=True, axis=0, inplace=inplace)

    assert (result is source) is inplace
    expected_data = np.roll(snapshot["data"], 1, axis=0)
    expected_data[0] = -expected_data[0]
    np.testing.assert_array_equal(result.data, expected_data)
    np.testing.assert_array_equal(result.mask, np.roll(snapshot["mask"], 1, axis=0))
    assert result.dims == snapshot["dims"]
    for dim, coord in snapshot["coords"].items():
        assert result.coord(dim) == coord
    assert result.units == snapshot["units"]
    assert result.meta == snapshot["meta"]
    assert result.history_entries[:-1] == snapshot["history"]
    assert len(result.history_entries) == len(snapshot["history"]) + 1
    assert result.history_entries[-1]["operation"] == "roll"
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": 0,
        "resolved_dim": "y",
        "resolved_axis": 0,
        "scientific_parameters": {"pts": 1, "neg": True},
        "inplace": inplace,
    }
    assert "`roll` shift performed on dimension `y`" in result.history[-1]
    if not inplace:
        _assert_snapshot(source, snapshot)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("rs", [0.0, 1.0, 99.0, 3.0]),
        ("ls", [99.0, 3.0, 4.0, 0.0]),
    ],
)
def test_zero_filled_shifts_structure_default_axis_and_move_mask(name, expected):
    source = scp.NDDataset(np.array([1.0, 99.0, 3.0, 4.0]), units="V")
    source.mask = np.array([False, True, False, False])
    source.annotate("prepared")
    source_history = source.history_entries

    result = getattr(source, name)(pts=1)

    np.testing.assert_array_equal(result.data, expected)
    assert result.history_entries[:-1] == source_history
    assert result.history_entries[-1]["operation"] == name
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": None,
        "resolved_dim": "x",
        "resolved_axis": 0,
        "scientific_parameters": {"pts": 1},
        "inplace": False,
    }
    assert source.history_entries == source_history


def test_cs_delegates_to_roll_without_duplicate_entry_and_preserves_noop_policy():
    source = _shift_dataset()
    source_history = source.history_entries

    result = source.cs(pts=0, neg=True, dim="x")

    np.testing.assert_array_equal(result.data, source.data)
    assert result.history_entries[:-1] == source_history
    assert len(result.history_entries) == len(source_history) + 1
    assert result.history_entries[-1]["operation"] == "roll"
    assert result.history_entries[-1]["parameters"]["scientific_parameters"] == {
        "pts": 0,
        "neg": True,
    }
    assert source.history_entries == source_history


@pytest.mark.parametrize("name", ["fsh", "fsh2"])
def test_fourier_shifts_record_the_executed_kernel(name):
    source = _shift_dataset(complex_data=True)
    source_history = source.history_entries

    result = getattr(source, name)(pts=0.5, dim="x")

    size = source.shape[-1]
    phase_sign = -1.0 if name == "fsh" else 1.0
    if name == "fsh":
        transformed = np.fft.ifft(np.fft.ifftshift(source.data, -1))
        shifted = (
            np.exp(phase_sign * 2.0j * np.pi * 0.5 * np.arange(size) / size)
            * transformed
        )
        expected = np.fft.fftshift(np.fft.fft(shifted), -1)
    else:
        transformed = np.fft.fft(np.fft.ifftshift(source.data, -1)) * size
        shifted = (
            np.exp(phase_sign * 2.0j * np.pi * 0.5 * np.arange(size) / size)
            * transformed
        )
        expected = np.fft.fftshift(
            np.fft.ifft(shifted).astype(shifted.dtype)
        ) * size

    assert result.shape == source.shape
    assert result.dims == source.dims
    np.testing.assert_allclose(result.data, expected)
    assert np.isfinite(result.data).all()
    assert result.history_entries[:-1] == source_history
    assert result.history_entries[-1]["operation"] == name
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": "x",
        "resolved_dim": "x",
        "resolved_axis": 1,
        "scientific_parameters": {"pts": 0.5},
        "inplace": False,
    }
    assert source.history_entries == source_history


def test_shift_failure_does_not_record_success():
    source = _shift_dataset()
    snapshot = _snapshot(source)

    with pytest.raises(ValueError, match="not recognized"):
        source.roll(pts=1, dim="missing", inplace=True)

    _assert_snapshot(source, snapshot)


def test_shift_history_parameters_are_detached():
    source = _shift_dataset()
    result = source.roll(pts=1)

    entries = result.history_entries
    entries[-1]["parameters"]["scientific_parameters"]["pts"] = 99

    assert result.history_entries[-1]["parameters"]["scientific_parameters"] == {
        "pts": 1,
        "neg": False,
    }


def test_zf_size_records_effective_nonfinal_geometry_and_preserves_source():
    source = _zero_fill_dataset()
    snapshot = _snapshot(source)

    result = source.zf_size(size=5, dim="y")

    expected = np.concatenate([snapshot["data"], np.zeros((2, 4))], axis=0)
    np.testing.assert_array_equal(result.data, expected)
    assert result.shape == (5, 4)
    assert result.dims == snapshot["dims"]
    assert result.y.units == snapshot["coords"]["y"].units
    np.testing.assert_array_equal(result.y.data, np.arange(5.0))
    assert result.x == snapshot["coords"]["x"]
    assert result.units == snapshot["units"]
    assert result.meta.sample == snapshot["meta"].sample
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == "zf_size"
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": "y",
        "resolved_dim": "y",
        "resolved_axis": 0,
        "scientific_parameters": {"size": 5, "mid": False},
        "inplace": False,
        "source_size": 3,
        "result_size": 5,
    }
    assert "Applied zf_size zero filling on dimension y" in result.history[-1]
    _assert_snapshot(source, snapshot)


def test_zf_double_records_inplace_final_dimension():
    source = _zero_fill_dataset()
    history = source.history_entries
    original = source.data.copy()

    result = source.zf_double(n=1, mid=True, dim="x", inplace=True)

    assert result is source
    assert result.shape == (3, 8)
    np.testing.assert_array_equal(result.data[..., :2], original[..., :2])
    np.testing.assert_array_equal(result.data[..., 6:], original[..., 2:])
    assert result.history_entries[:-1] == history
    assert result.history_entries[-1]["operation"] == "zf_double"
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": "x",
        "resolved_dim": "x",
        "resolved_axis": 1,
        "scientific_parameters": {"n": 1, "mid": True},
        "inplace": True,
        "source_size": 4,
        "result_size": 8,
    }


def test_zf_size_noop_records_requested_and_effective_size():
    source = _zero_fill_dataset()
    source_history = source.history_entries

    result = source.zf_size()

    np.testing.assert_array_equal(result.data, source.data)
    assert len(result.history_entries) == len(source_history) + 1
    assert result.history_entries[-1]["operation"] == "zf_size"
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": None,
        "resolved_dim": "x",
        "resolved_axis": 1,
        "scientific_parameters": {"size": 4, "mid": False},
        "inplace": False,
        "source_size": 4,
        "result_size": 4,
        "requested_parameters": {"size": None, "mid": False},
    }
    assert source.history_entries == source_history


@pytest.mark.parametrize("name", ["zf", "zf_auto"])
def test_zero_fill_aliases_keep_executed_zf_size_convention(name):
    source = _zero_fill_dataset()
    kwargs = {"size": 6} if name == "zf" else {}

    result = getattr(source, name)(**kwargs)

    assert len(result.history_entries) == len(source.history_entries) + 1
    assert result.history_entries[-1]["operation"] == "zf_size"
    assert result.history_entries[-1]["parameters"]["result_size"] == result.shape[-1]


@pytest.mark.parametrize(
    "coord",
    [
        scp.Coord([0.0, 1.0, 3.0], units="s"),
        scp.Coord.arange(3, units="mm"),
    ],
)
def test_refused_nonfinal_zero_fill_returns_source_without_mutation(coord):
    source = _zero_fill_dataset()
    source.y = coord
    snapshot = _snapshot(source)

    result = source.zf_size(size=5, dim="y", inplace=True)

    assert result is source
    _assert_snapshot(source, snapshot)


def test_zero_fill_failure_does_not_record_success():
    source = _zero_fill_dataset()
    snapshot = _snapshot(source)

    with pytest.raises(ValueError, match="not recognized"):
        source.zf_size(size=5, dim="missing", inplace=True)

    _assert_snapshot(source, snapshot)
