"""Structured history for the remaining unit-agnostic FFT wrappers."""

import copy
import json

import numpy as np
import pytest
from scipy.signal import hilbert

import spectrochempy as scp


def _dataset(*, shape=(3, 4), complex_data=True):
    real = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    data = real + 1.0j * (real + 0.5) if complex_data else real
    dims = ["z", "y", "x"][-len(shape) :]
    coords = [
        scp.Coord.arange(size, units="s", title=f"time {dim}")
        for size, dim in zip(shape, dims, strict=True)
    ]
    dataset = scp.NDDataset(
        data,
        dims=dims,
        coordset=coords,
        units="V",
        title="unit-agnostic history",
    )
    mask = np.zeros(shape, dtype=bool)
    mask.reshape(-1)[min(2, mask.size - 1)] = True
    dataset.mask = mask
    dataset.meta.sample = "synthetic"
    dataset.annotate("prepared")
    return dataset


def _snapshot(dataset):
    return {
        "data": dataset.data.copy(),
        "mask": np.asarray(dataset.mask).copy(),
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


def _assert_preserved_geometry(dataset, snapshot):
    np.testing.assert_array_equal(dataset.mask, snapshot["mask"])
    assert dataset.dims == snapshot["dims"]
    for dim, coord in snapshot["coords"].items():
        assert dataset.coord(dim) == coord
    assert dataset.units == snapshot["units"]
    assert dataset.meta == snapshot["meta"]


@pytest.mark.parametrize("name", ["mc", "ps"])
@pytest.mark.parametrize("complex_data", [False, True])
def test_modulus_and_power_spectrum_record_the_executed_operation(name, complex_data):
    source = _dataset(complex_data=complex_data)
    snapshot = _snapshot(source)

    result = getattr(scp, name)(source)

    expected = source.data.real**2 + source.data.imag**2
    if name == "mc":
        expected = np.sqrt(expected)
    np.testing.assert_allclose(result.data, expected)
    assert result.dtype == expected.dtype
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == name
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": None,
        "resolved_dim": "x",
        "resolved_axis": 1,
        "scientific_parameters": {},
        "inplace": False,
    }
    description = "modulus calculated" if name == "mc" else "power spectrum calculated"
    assert description in result.history_entries[-1]["message"]
    assert "shift performed" not in result.history_entries[-1]["message"]
    _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("name", ["mc", "ps"])
def test_modulus_and_power_spectrum_preserve_inplace_contract(name):
    source = _dataset()
    snapshot = _snapshot(source)
    expected = source.data.real**2 + source.data.imag**2
    if name == "mc":
        expected = np.sqrt(expected)

    result = getattr(source, name)(inplace=True)

    assert result is source
    np.testing.assert_allclose(result.data, expected)
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["parameters"]["inplace"] is True


@pytest.mark.parametrize("inplace", [False, True])
def test_hilbert_transform_records_explicit_n_and_matches_scipy(inplace):
    source = _dataset()
    snapshot = _snapshot(source)
    size = source.shape[-1]
    expected = hilbert(snapshot["data"].real, size)
    expected.real = snapshot["data"].real

    result = source.ht(N=size, inplace=inplace)

    assert (result is source) is inplace
    np.testing.assert_allclose(result.data, expected)
    assert result.dtype == expected.dtype
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == "ht"
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": None,
        "resolved_dim": "x",
        "resolved_axis": 1,
        "scientific_parameters": {"N": size},
        "inplace": inplace,
    }
    assert "Hilbert transform performed" in result.history_entries[-1]["message"]
    assert "shift performed" not in result.history_entries[-1]["message"]
    if not inplace:
        _assert_snapshot(source, snapshot)


def test_one_dimensional_hilbert_transform_retains_larger_explicit_n():
    source = _dataset(shape=(4,), complex_data=False)
    snapshot = _snapshot(source)
    requested_n = 8
    expected = hilbert(snapshot["data"], requested_n)[: source.size]
    expected *= requested_n / source.size
    expected.real = snapshot["data"]

    result = source.ht(N=requested_n)

    np.testing.assert_allclose(result.data, expected)
    assert result.shape == source.shape
    assert result.history_entries[-1]["parameters"]["scientific_parameters"] == {
        "N": requested_n
    }
    _assert_snapshot(source, snapshot)


@pytest.mark.parametrize(
    ("name", "kwargs", "error"),
    [
        ("mc", {"dim": "y"}, TypeError),
        ("ps", {"axis": 0}, TypeError),
        ("ht", {"N": 4, "dim": "y"}, TypeError),
        ("ht", {}, TypeError),
        ("ht", {"N": 2}, ValueError),
    ],
)
def test_preexisting_unsupported_calls_do_not_record_success(name, kwargs, error):
    source = _dataset()
    snapshot = _snapshot(source)

    with pytest.raises(error):
        getattr(source, name)(**kwargs)

    _assert_snapshot(source, snapshot)


def test_three_dimensional_hilbert_with_different_n_remains_unsupported():
    source = _dataset(shape=(2, 3, 4))
    snapshot = _snapshot(source)

    with pytest.raises(ValueError, match="could not broadcast"):
        source.ht(N=6)

    _assert_snapshot(source, snapshot)


@pytest.mark.parametrize(
    ("kwargs", "tail_points"),
    [({}, 1), ({"len": 0.6}, 2)],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_dc_records_the_actual_tail_and_preserves_results(kwargs, tail_points, inplace):
    source = _dataset()
    snapshot = _snapshot(source)
    expected = snapshot["data"] - np.mean(snapshot["data"][..., -tail_points:])

    result = source.dc(inplace=inplace, **kwargs)

    assert (result is source) is inplace
    np.testing.assert_allclose(result.data, expected)
    assert result.dtype == expected.dtype
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == "dc"
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": None,
        "resolved_dim": "x",
        "resolved_axis": 1,
        "scientific_parameters": {"tail_points": tail_points},
        "inplace": inplace,
        "requested_parameters": {"len": kwargs.get("len", 0.25)},
    }
    assert "DC baseline correction performed" in result.history_entries[-1]["message"]
    assert "shift performed" not in result.history_entries[-1]["message"]
    if not inplace:
        _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("inplace", [False, True])
def test_dc_nonfinal_dimension_records_resolved_geometry(inplace):
    source = _dataset(complex_data=False)
    snapshot = _snapshot(source)
    expected = snapshot["data"] - np.mean(snapshot["data"][-1:, :])

    result = source.dc(len=0.5, dim="y", inplace=inplace)

    assert (result is source) is inplace
    np.testing.assert_allclose(result.data, expected)
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": "y",
        "resolved_dim": "y",
        "resolved_axis": 0,
        "scientific_parameters": {"tail_points": 1},
        "inplace": inplace,
        "requested_parameters": {"len": 0.5},
    }
    if not inplace:
        _assert_snapshot(source, snapshot)


def test_dc_zero_rounded_length_records_the_full_executed_selection():
    source = _dataset(complex_data=False)

    result = source.dc(len=0.3, dim="y")

    np.testing.assert_allclose(result.data, source.data - np.mean(source.data))
    assert result.history_entries[-1]["parameters"]["scientific_parameters"] == {
        "tail_points": source.shape[0]
    }


def test_dc_failure_does_not_record_success_or_mutate_source():
    source = _dataset()
    snapshot = _snapshot(source)

    with pytest.raises(ValueError):
        source.dc(len="invalid", inplace=True)

    _assert_snapshot(source, snapshot)


def test_parameters_are_detached_and_json_serializable():
    result = _dataset().dc(len=0.6)

    parameters = result.history_entries[-1]["parameters"]
    assert json.loads(json.dumps(parameters)) == parameters
    parameters["scientific_parameters"]["tail_points"] = 999
    parameters["requested_parameters"]["len"] = 999

    stored = result.history_entries[-1]["parameters"]
    assert stored["scientific_parameters"]["tail_points"] == 2
    assert stored["requested_parameters"]["len"] == pytest.approx(0.6)


def test_scp_roundtrip_preserves_structured_dc_history(tmp_path):
    result = _dataset().dc(len=0.6)
    filename = tmp_path / "unit-agnostic-history.scp"

    result.save_as(filename, confirm=False)
    rebuilt = scp.load(filename)

    assert rebuilt.history_entries == result.history_entries
    assert rebuilt.history == result.history
