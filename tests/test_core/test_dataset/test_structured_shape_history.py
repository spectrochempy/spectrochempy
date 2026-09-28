"""Structured history coverage for public NDDataset shape operations."""

import copy

import numpy as np
import pytest

import spectrochempy as scp


def _dataset(shape=(2, 3, 4)):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    dataset = scp.NDDataset(
        data,
        dims=["z", "y", "x"],
        coordset=[
            scp.Coord.arange(shape[0], units="s", title="depth"),
            scp.Coord.arange(shape[1], units="mm", title="position"),
            scp.Coord.arange(shape[2], units="cm^-1", title="wavenumber"),
        ],
        units="K",
        title="shape history",
    )
    mask = np.zeros(shape, dtype=bool)
    mask.flat[-2] = True
    dataset.mask = mask
    dataset.meta.sample = "synthetic"
    dataset.annotate("prepared")
    return dataset


def _snapshot(dataset):
    return {
        "data": dataset.data.copy(),
        "mask": dataset.mask.copy(),
        "dims": list(dataset.dims),
        "coords": {
            dim: dataset.coord(dim).copy() for dim in dataset.dims
        },
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
def test_squeeze_records_requested_and_resolved_dimensions(inplace):
    source = _dataset((1, 3, 4))
    snapshot = _snapshot(source)

    result = source.squeeze("z", inplace=inplace)

    assert (result is source) is inplace
    np.testing.assert_array_equal(result.data, snapshot["data"].squeeze(axis=0))
    np.testing.assert_array_equal(result.mask, snapshot["mask"].squeeze(axis=0))
    assert result.dims == ["y", "x"]
    assert result.coord("y") == snapshot["coords"]["y"]
    assert result.coord("x") == snapshot["coords"]["x"]
    assert result.units == snapshot["units"]
    assert result.meta == snapshot["meta"]
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1] == {
        "date": result.history_entries[-1]["date"],
        "operation": "squeeze",
        "parameters": {
            "requested_dims": ["z"],
            "resolved_dims": ["z"],
            "resolved_axes": [0],
            "result_dims": ["y", "x"],
            "inplace": inplace,
        },
        "message": "Data squeezed",
    }
    assert "Data squeezed" in result.history[-1]
    if not inplace:
        _assert_snapshot(source, snapshot)


def test_squeeze_noop_retains_existing_recording_convention():
    source = _dataset()

    result = source.squeeze()

    np.testing.assert_array_equal(result.data, source.data)
    assert result.dims == source.dims
    assert result.history_entries[-1]["operation"] == "squeeze"
    assert result.history_entries[-1]["parameters"]["resolved_dims"] == []
    assert result.history_entries[-1]["parameters"]["resolved_axes"] == []


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    ("selectors", "requested"),
    [((0, -1), [0, -1]), (("z", "x"), ["z", "x"])],
)
def test_swapdims_records_requested_and_resolved_axes(
    inplace, selectors, requested
):
    source = _dataset()
    snapshot = _snapshot(source)

    result = source.swapdims(*selectors, inplace=inplace)

    assert (result is source) is inplace
    np.testing.assert_array_equal(result.data, np.swapaxes(snapshot["data"], 0, 2))
    np.testing.assert_array_equal(result.mask, np.swapaxes(snapshot["mask"], 0, 2))
    assert result.dims == ["x", "y", "z"]
    for dim in snapshot["dims"]:
        assert result.coord(dim) == snapshot["coords"][dim]
    assert result.units == snapshot["units"]
    assert result.meta.sample == snapshot["meta"].sample
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == "swapdims"
    assert result.history_entries[-1]["parameters"] == {
        "requested_dims": requested,
        "resolved_dims": ["z", "x"],
        "resolved_axes": [0, 2],
        "result_dims": ["x", "y", "z"],
        "inplace": inplace,
    }
    assert "Data swapped between dims" in result.history[-1]
    if not inplace:
        _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("inplace", [False, True])
def test_reshape_records_requested_and_resolved_geometry(inplace):
    source = _dataset()
    snapshot = _snapshot(source)

    result = source.reshape(
        (2, -1, 2),
        dims=("z", "y", "x"),
        coord_policy="drop",
        inplace=inplace,
    )

    assert (result is source) is inplace
    np.testing.assert_array_equal(result.data, snapshot["data"].reshape(2, 6, 2))
    np.testing.assert_array_equal(result.mask, snapshot["mask"].reshape(2, 6, 2))
    assert result.dims == ["z", "y", "x"]
    assert result.coordset is None
    assert result.units == snapshot["units"]
    assert result.meta == snapshot["meta"]
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == "reshape"
    assert result.history_entries[-1]["parameters"] == {
        "requested_shape": [2, -1, 2],
        "resolved_shape": [2, 6, 2],
        "source_dims": ["z", "y", "x"],
        "requested_dims": ["z", "y", "x"],
        "result_dims": ["z", "y", "x"],
        "coord_policy": "drop",
        "coordinate_overrides": [],
        "inplace": inplace,
    }
    assert "Data reshaped from (2, 3, 4) to (2, 6, 2)" in result.history[-1]
    if not inplace:
        _assert_snapshot(source, snapshot)


def test_shape_failures_do_not_append_success_entries():
    calls = [
        lambda dataset: dataset.squeeze("y", inplace=True),
        lambda dataset: dataset.swapdims("missing", "x", inplace=True),
        lambda dataset: dataset.reshape((5, 5), inplace=True),
        lambda dataset: dataset.reshape(
            (2, 3, 4), dims=("a", "b"), inplace=True
        ),
        lambda dataset: dataset.reshape(
            (2, 3, 4),
            dims=("z", "y", "x"),
            coords={"missing": scp.Coord.arange(2)},
            inplace=True,
        ),
    ]

    for call in calls:
        source = _dataset()
        snapshot = _snapshot(source)
        with pytest.raises(ValueError):
            call(source)
        _assert_snapshot(source, snapshot)


def test_transpose_structured_contract_is_unchanged():
    source = _dataset()

    result = source.transpose("x", "y", "z")

    assert result.history_entries[-1]["operation"] == "transpose"
    assert result.history_entries[-1]["parameters"] == {
        "requested_dims": ["x", "y", "z"],
        "result_dims": ["x", "y", "z"],
        "inplace": False,
    }
    assert len(result.history_entries) == len(source.history_entries) + 1


def test_shape_history_entries_are_detached():
    source = _dataset()
    result = source.swapdims("z", "x")

    entries = result.history_entries
    entries[-1]["parameters"]["result_dims"][0] = "changed"

    assert result.history_entries[-1]["parameters"]["result_dims"] == ["x", "y", "z"]
    assert source.history_entries == _snapshot(source)["history"]


def test_internal_nonfinal_swaps_do_not_add_shape_entries():
    source = scp.NDDataset(
        np.arange(12.0).reshape(3, 4),
        dims=["y", "x"],
        coordset=[
            scp.Coord.arange(3, units="s"),
            scp.Coord.arange(4, units="s"),
        ],
    )
    source.meta.td = [3, 4]
    source.annotate("prepared")
    source_history = source.history_entries

    zero_filled = source.zf_size(size=5, dim="y")

    assert zero_filled.shape == (5, 4)
    assert zero_filled.dims == ["y", "x"]
    assert zero_filled.history_entries[:-1] == source_history
    assert len(zero_filled.history_entries) == len(source_history) + 1
    assert zero_filled.history_entries[-1]["operation"] is None
    assert zero_filled.history_entries[-1]["message"].startswith(
        "Applied zf_size zero filling on dimension y"
    )
    assert source.history_entries == source_history


def test_internal_phasing_swap_does_not_add_shape_entries():
    source = scp.NDDataset(
        np.arange(12.0).reshape(3, 4) + 1.0j,
        dims=["y", "x"],
        coordset=[
            scp.Coord.arange(3, units="Hz"),
            scp.Coord.arange(4, units="Hz"),
        ],
    )
    source.meta.phc0 = [0, 0]
    source.meta.phc1 = [0, 0]
    source.meta.pivot = [0, 0]
    source.meta.exptc = [0, 0]
    source.meta.phased = [False, False]
    source.annotate("prepared")
    source_history = source.history_entries

    phased = source.pk(phc0=10, dim="y")

    assert phased.shape == source.shape
    assert phased.dims == source.dims
    assert phased.history_entries[:-1] == source_history
    assert len(phased.history_entries) == len(source_history) + 1
    assert phased.history_entries[-1]["operation"] is None
    assert phased.history_entries[-1]["message"].startswith(
        "Applied pk phasing on dimension y"
    )
    assert source.history_entries == source_history
