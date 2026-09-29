"""Structured history for the remaining unit-agnostic FFT wrappers."""

import copy
import json

import numpy as np
import pytest
from scipy.signal import hilbert

import spectrochempy as scp


def _dataset(*, shape=(3, 4), complex_data=True, dtype=float):
    real = np.arange(np.prod(shape), dtype=dtype).reshape(shape) + 1.0
    data = real + 1.0j * (real + 0.5) if complex_data else real
    dims = ["z", "y", "x"][-len(shape) :]
    coords = [
        scp.Coord(
            np.arange(size, dtype=float) * (index + 1) + 10 ** (index + 1),
            units="s",
            title=f"coordinate {dim}",
        )
        for index, (size, dim) in enumerate(zip(shape, dims, strict=True))
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
    dataset.meta.axis_marker = [f"metadata {dim}" for dim in dims]
    dataset.annotate("prepared")
    return dataset


def _snapshot(dataset):
    mask = np.asarray(dataset.mask).copy()
    assert mask.any()
    return {
        "data": dataset.data.copy(),
        "mask": mask,
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


def _hilbert_reference(data, n, axis):
    source_size = data.shape[axis]
    transformed = hilbert(data.real, n, axis=axis)
    selection = [slice(None)] * data.ndim
    selection[axis] = slice(source_size)
    result = transformed[tuple(selection)] * (n / source_size)
    result.real = data.real
    return result


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
    expected = _hilbert_reference(snapshot["data"], requested_n, axis=0)

    result = source.ht(N=requested_n)

    np.testing.assert_allclose(result.data, expected)
    assert result.shape == source.shape
    assert result.history_entries[-1]["parameters"]["scientific_parameters"] == {
        "N": requested_n
    }
    _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    ("kwargs", "requested_parameters"),
    [({}, None), ({"N": None}, {"N": None})],
)
def test_hilbert_transform_resolves_default_n(kwargs, requested_parameters, inplace):
    source = _dataset(shape=(3, 5))
    snapshot = _snapshot(source)
    expected = _hilbert_reference(snapshot["data"], source.shape[-1], axis=-1)

    result = source.ht(inplace=inplace, **kwargs)

    assert (result is source) is inplace
    np.testing.assert_allclose(result.data, expected)
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == "ht"
    parameters = result.history_entries[-1]["parameters"]
    assert parameters["scientific_parameters"] == {"N": source.shape[-1]}
    assert json.loads(json.dumps(parameters)) == parameters
    if requested_parameters is None:
        assert "requested_parameters" not in parameters
    else:
        assert parameters["requested_parameters"] == requested_parameters
    if not inplace:
        _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize(
    ("shape", "selector", "axis"),
    [((1,), {}, 0), ((2, 1, 3), {"dim": "y"}, 1)],
)
def test_one_point_hilbert_transform_does_not_require_a_second_value(
    shape, selector, axis, complex_data
):
    source = _dataset(shape=shape, complex_data=complex_data, dtype=np.float32)
    snapshot = _snapshot(source)
    expected = _hilbert_reference(snapshot["data"], 1, axis=axis)

    result = source.ht(**selector)

    np.testing.assert_allclose(result.data, expected)
    assert result.dtype == expected.dtype
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == "ht"
    assert not any(entry["operation"] == "swapdims" for entry in result.history_entries)
    _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("name", ["mc", "ps"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    ("selector", "requested"),
    [
        ({"dim": 0}, 0),
        ({"dims": 0}, 0),
        ({"axis": 0}, 0),
        ({"dim": "y"}, "y"),
        ({"dim": -2}, -2),
    ],
)
def test_modulus_and_power_spectrum_accept_dimension_selectors(
    name, inplace, selector, requested
):
    source = _dataset()
    snapshot = _snapshot(source)
    expected = snapshot["data"].real ** 2 + snapshot["data"].imag ** 2
    if name == "mc":
        expected = np.sqrt(expected)

    result = getattr(source, name)(inplace=inplace, **selector)

    assert (result is source) is inplace
    np.testing.assert_allclose(result.data, expected)
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[-1]["parameters"]["requested_dim"] == requested
    assert result.history_entries[-1]["parameters"]["resolved_dim"] == "y"
    assert result.history_entries[-1]["parameters"]["resolved_axis"] == 0
    assert not any(entry["operation"] == "swapdims" for entry in result.history_entries)
    if not inplace:
        _assert_snapshot(source, snapshot)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    ("selector", "requested"),
    [
        ({"dim": 0}, 0),
        ({"dims": 0}, 0),
        ({"axis": 0}, 0),
        ({"dim": "y"}, "y"),
        ({"axis": -2}, -2),
    ],
)
def test_hilbert_transform_accepts_dimension_selectors(selector, requested, inplace):
    source = _dataset()
    snapshot = _snapshot(source)
    size = source.shape[0]
    expected = hilbert(snapshot["data"].real, size, axis=0)
    expected.real = snapshot["data"].real

    result = source.ht(N=size, inplace=inplace, **selector)

    assert (result is source) is inplace
    np.testing.assert_allclose(result.data, expected)
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["parameters"] == {
        "requested_dim": requested,
        "resolved_dim": "y",
        "resolved_axis": 0,
        "scientific_parameters": {"N": size},
        "inplace": inplace,
    }
    assert not any(entry["operation"] == "swapdims" for entry in result.history_entries)
    if not inplace:
        _assert_snapshot(source, snapshot)


def test_dimension_selector_precedence_matches_get_axis():
    source = _dataset()
    snapshot = _snapshot(source)

    default_result = source.ht(N=source.shape[-1], dim=None, axis=0)
    primary_result = source.ht(N=source.shape[0], dims=0, dim="x", axis=1)

    expected_default = hilbert(snapshot["data"].real, source.shape[-1], axis=-1)
    expected_default.real = snapshot["data"].real
    expected_primary = hilbert(snapshot["data"].real, source.shape[0], axis=0)
    expected_primary.real = snapshot["data"].real
    np.testing.assert_allclose(default_result.data, expected_default)
    np.testing.assert_allclose(primary_result.data, expected_primary)
    assert default_result.history_entries[-1]["parameters"]["requested_dim"] is None
    assert default_result.history_entries[-1]["parameters"]["resolved_dim"] == "x"
    assert primary_result.history_entries[-1]["parameters"]["requested_dim"] == 0
    assert primary_result.history_entries[-1]["parameters"]["resolved_dim"] == "y"
    _assert_snapshot(source, snapshot)


def test_invalid_selector_is_rejected_before_inplace_permutation():
    source = _dataset()
    snapshot = _snapshot(source)

    with pytest.raises(ValueError):
        source.mc(dim="invalid", inplace=True)

    _assert_snapshot(source, snapshot)


def test_kernel_failure_restores_inplace_permutation():
    source = _dataset(shape=(3, 5))
    snapshot = _snapshot(source)

    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown_option'"):
        source.ht(
            N=source.shape[0],
            dim="y",
            inplace=True,
            unknown_option=True,
        )

    _assert_snapshot(source, snapshot)


@pytest.mark.parametrize(
    ("shape", "selector", "axis", "requested_n"),
    [
        ((3, 5), {"dim": "x"}, 1, 8),
        ((3, 5), {"dims": "y"}, 0, 6),
        ((2, 3, 4), {"axis": -1}, 2, 7),
        ((2, 3, 4), {"dim": "y"}, 1, 5),
        ((2, 3, 4), {"dims": "z"}, 0, 4),
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_hilbert_transform_supports_larger_n_on_any_dimension(
    shape, selector, axis, requested_n, inplace
):
    source = _dataset(shape=shape)
    snapshot = _snapshot(source)
    expected = _hilbert_reference(snapshot["data"], requested_n, axis=axis)

    result = source.ht(N=np.int64(requested_n), inplace=inplace, **selector)

    assert (result is source) is inplace
    np.testing.assert_allclose(result.data, expected)
    _assert_preserved_geometry(result, snapshot)
    assert result.history_entries[:-1] == snapshot["history"]
    assert result.history_entries[-1]["operation"] == "ht"
    parameters = result.history_entries[-1]["parameters"]
    assert parameters["scientific_parameters"] == {"N": requested_n}
    assert "requested_parameters" not in parameters
    assert not any(entry["operation"] == "swapdims" for entry in result.history_entries)
    if not inplace:
        _assert_snapshot(source, snapshot)


@pytest.mark.parametrize(
    ("invalid_n", "error"),
    [
        (2, ValueError),
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (np.bool_(False), TypeError),
        (3.0, TypeError),
        (np.float64(3.0), TypeError),
        ("3", TypeError),
    ],
)
def test_invalid_hilbert_sizes_are_rejected_without_mutation(invalid_n, error):
    source = _dataset(shape=(3, 5))
    snapshot = _snapshot(source)
    assert np.asarray(source.mask).any()

    with pytest.raises(error, match="N must be"):
        source.ht(N=invalid_n, dim="y", inplace=True)

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
