"""Structured history for the shared apodization wrapper."""

import copy
import json
import unicodedata

import numpy as np
import pytest

import spectrochempy as scp

DIRECT_KERNELS = [
    ("em", {}),
    ("gm", {}),
    ("sp", {}),
    ("general_hamming", {"alpha": 0.7}),
    ("triang", {}),
    ("bartlett", {}),
    ("blackmanharris", {}),
]

# Delegating wrappers must record only the kernel that actually ran.
DELEGATIONS = [
    ("hamming", "general_hamming", {"alpha": 0.54}),
    ("hann", "general_hamming", {"alpha": 0.5}),
    ("sine", "sp", {}),
    ("sinm", "sp", {"pow": 1}),
    ("qsin", "sp", {"pow": 2}),
]


def _dataset(shape=(1, 64), units="ps", sample_size=64):
    dataset = scp.NDDataset(
        np.arange(float(np.prod(shape))).reshape(shape) + 1.0,
        dims=["y", "x"][-len(shape) :] if len(shape) > 1 else ["x"],
        units="K",
        title="apodization history",
    )
    dataset.x = scp.Coord(np.linspace(0, 10, sample_size), units=units, title="time")
    dataset.meta.sample = "synthetic"
    dataset.annotate("prepared")
    return dataset


def _two_dimensional():
    dataset = scp.NDDataset(
        np.arange(12.0).reshape(3, 4) + 1.0,
        dims=["y", "x"],
        coordset=[
            scp.Coord.arange(3, units="s", title="time y"),
            scp.Coord.arange(4, units="ps", title="time x"),
        ],
        units="V",
    )
    dataset.meta.sample = "synthetic"
    dataset.annotate("prepared")
    return dataset


def _interferogram():
    ncols = 100
    data = np.zeros((3, ncols))
    data[:, 40] = 1.0
    return scp.NDDataset(
        data,
        coordset=[scp.Coord.arange(3), scp.Coord.arange(ncols, units="us")],
        meta={"interferogram": True},
    )


def _snapshot(dataset):
    return {
        "data": dataset.data.copy(),
        "dims": list(dataset.dims),
        "shape": dataset.shape,
        "coords": {dim: dataset.coord(dim).copy() for dim in dataset.dims},
        "units": dataset.units,
        "meta": copy.deepcopy(dataset.meta),
        "history": dataset.history_entries,
    }


def _assert_unchanged(dataset, snapshot):
    np.testing.assert_array_equal(dataset.data, snapshot["data"])
    assert dataset.dims == snapshot["dims"]
    assert dataset.shape == snapshot["shape"]
    for dim, coord in snapshot["coords"].items():
        assert dataset.coord(dim) == coord
    assert dataset.units == snapshot["units"]
    assert dataset.meta == snapshot["meta"]
    assert dataset.history_entries == snapshot["history"]


def _last(dataset):
    return dataset.history_entries[-1]


def _assert_requested_parameters(actual, expected):
    """
    Compare requested parameters, ignoring equivalent unit spellings.

    The micro prefix has two canonically equivalent code points, MICRO SIGN and
    GREEK SMALL LETTER MU. Which one a units backend returns depends on the
    installed version and the platform, so pinning one of them here would test
    the backend rather than the history contract. NFKC maps both to the same
    character, so comparing under it keeps the assertion about the retained
    magnitude and unit rather than about their encoding.
    """
    assert set(actual) == set(expected)
    for key, reference in expected.items():
        assert actual[key]["value"] == pytest.approx(reference["value"])
        assert unicodedata.normalize(
            "NFKC", actual[key]["units"]
        ) == unicodedata.normalize("NFKC", reference["units"])


def _general_hamming_reference(size, alpha):
    """
    Independent reference for the generalized Hamming window.

    Written from the documented functional form rather than delegated to the
    same SciPy helper the kernel uses, so the comparison is not circular.
    """
    n = np.arange(size)
    return alpha - (1 - alpha) * np.cos(2 * np.pi * n / (size - 1))


@pytest.mark.parametrize(("kernel", "kwargs"), DIRECT_KERNELS)
def test_direct_kernel_records_exactly_one_structured_entry(kernel, kwargs):
    dataset = _dataset()
    before = len(dataset.history_entries)

    result = getattr(dataset, kernel)(**kwargs)

    assert len(result.history_entries) == before + 1
    entry = _last(result)
    assert entry["operation"] == kernel
    assert entry["message"].startswith(f"Applied {kernel} apodization on dimension")
    parameters = entry["parameters"]
    assert parameters["requested_dim"] is None
    assert parameters["resolved_dim"] == "x"
    assert parameters["resolved_axis"] == 1
    assert parameters["inplace"] is False
    assert parameters["inv"] is False
    assert parameters["rev"] is False
    assert "requested_parameters" not in parameters


@pytest.mark.parametrize(
    ("wrapper", "executed", "expected"), DELEGATIONS, ids=[d[0] for d in DELEGATIONS]
)
def test_delegating_wrapper_records_only_the_executed_kernel(
    wrapper, executed, expected
):
    dataset = _dataset()
    before = len(dataset.history_entries)

    result = getattr(dataset, wrapper)()

    # A delegation must not add a duplicate entry of its own.
    assert len(result.history_entries) == before + 1
    entry = _last(result)
    assert entry["operation"] == executed
    scientific = entry["parameters"]["scientific_parameters"]
    for name, value in expected.items():
        assert scientific[name] == pytest.approx(value)


def test_omitted_parameters_record_the_effective_kernel_defaults():
    dataset = _dataset()

    sp = _last(dataset.sp())["parameters"]["scientific_parameters"]
    hamming = _last(dataset.hamming())["parameters"]["scientific_parameters"]
    hann = _last(dataset.hann())["parameters"]["scientific_parameters"]

    assert sp == {"ssb": 1, "pow": 1}
    assert hamming == {"alpha": 0.54}
    assert hann == {"alpha": 0.5}


def test_explicit_dimensionless_parameter_is_recorded_effectively():
    dataset = _dataset()

    result = dataset.general_hamming(alpha=0.7)

    scientific = _last(result)["parameters"]["scientific_parameters"]
    assert scientific == {"alpha": 0.7}
    # No conversion happened, so no separate request is retained.
    assert "requested_parameters" not in _last(result)["parameters"]


def test_unit_bearing_parameters_record_effective_and_requested_forms():
    dataset = _dataset()

    result = dataset.em(lb="250 Hz", shifted="1.5 us")

    parameters = _last(result)["parameters"]
    scientific = parameters["scientific_parameters"]
    # Effective values are expressed in the units of the dataset coordinate.
    assert scientific["lb"] == pytest.approx(2.5e-10)
    assert scientific["shifted"] == pytest.approx(1500000.0)
    # The request is kept separately, serializable, and physically meaningful.
    _assert_requested_parameters(
        parameters["requested_parameters"],
        {
            "lb": {"value": 250, "units": "Hz"},
            "shifted": {"value": 1.5, "units": "µs"},
        },
    )
    # No live Quantity object is retained.
    json.dumps(parameters)


def test_unit_normalization_to_zero_is_not_duplicated():
    # A zero band width is normalized to a plain 0.0. The effective value
    # equals the requested one, so no separate request is recorded.
    dataset = _dataset()

    result = dataset.em(lb="0 Hz")

    parameters = _last(result)["parameters"]
    assert parameters["scientific_parameters"]["lb"] == 0.0
    assert "requested_parameters" not in parameters


def test_requested_and_resolved_dimensions_are_both_recorded():
    named = _last(_dataset().hamming(dim="x"))["parameters"]
    positive = _last(_dataset().hamming(axis=1))["parameters"]
    negative = _last(_dataset().hamming(axis=-1))["parameters"]
    by_dims = _last(_dataset().hamming(dims="x"))["parameters"]

    assert named["requested_dim"] == "x"
    assert named["resolved_dim"] == "x"
    assert positive["requested_dim"] == 1
    assert positive["resolved_dim"] == "x"
    assert negative["requested_dim"] == -1
    assert negative["resolved_dim"] == "x"
    assert by_dims["requested_dim"] == "x"


def test_non_final_dimension_records_one_entry_and_keeps_dimensions():
    dataset = _two_dimensional()
    before = len(dataset.history_entries)

    result = dataset.hamming(axis=0)

    # The temporary dimension permutation must not appear as a public entry.
    assert len(result.history_entries) == before + 1
    assert [e["operation"] for e in result.history_entries] == [None, "general_hamming"]
    parameters = _last(result)["parameters"]
    assert parameters["requested_dim"] == 0
    assert parameters["resolved_dim"] == "y"
    assert parameters["resolved_axis"] == 0
    assert result.dims == ["y", "x"]
    assert result.shape == (3, 4)
    # The data really was apodized along the requested axis.
    assert not np.array_equal(result.data, dataset.data)


@pytest.mark.parametrize("inplace", [False, True])
def test_inplace_and_outof_place_modes(inplace):
    dataset = _dataset()
    snapshot = _snapshot(dataset)
    original = dataset.data.copy()
    expected = _general_hamming_reference(64, 0.54)

    result = dataset.hamming(inplace=inplace)

    assert _last(result)["parameters"]["inplace"] is inplace
    assert (result is dataset) is inplace
    # The applied window is the ratio of the result to the source data.
    np.testing.assert_allclose(result.data[0] / original[0], expected, atol=1e-12)
    if not inplace:
        # The source and its coordinates stay untouched out of place.
        _assert_unchanged(dataset, snapshot)


@pytest.mark.parametrize("wrapper,alpha", [("hamming", 0.54), ("hann", 0.5)])
def test_window_matches_an_independent_reference(wrapper, alpha):
    dataset = _dataset()
    size = 64
    original = dataset.data.copy()

    treated, window = getattr(dataset, wrapper)(retapod=True)

    reference = _general_hamming_reference(size, alpha)
    np.testing.assert_allclose(np.asarray(window.data), reference, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(treated.data)[0] / original[0], reference, atol=1e-12
    )


def test_inverse_option_is_recorded_and_is_the_reciprocal_window():
    dataset = _dataset()
    treated, window = dataset.hamming(retapod=True)
    inverted, inverse = dataset.hamming(retapod=True, inv=True)

    plain = np.asarray(window.data)
    # alpha=0.54 keeps the window strictly positive, so the inverse is finite.
    np.testing.assert_allclose(np.asarray(inverse.data), 1.0 / plain, atol=1e-12)
    assert _last(inverted)["parameters"]["inv"] is True
    assert _last(treated)["parameters"]["inv"] is False
    # The returned window carries no entry of its own, by the arbitrated contract.
    assert len(window.history_entries) == 0
    assert len(inverse.history_entries) == 0


def test_inverse_of_a_window_with_a_zero_endpoint_is_non_finite():
    # A Hann window starts at exactly zero, so its inverse legitimately blows
    # up. This records the real behavior instead of hiding it.
    dataset = _dataset()
    treated, inverse = dataset.hann(retapod=True, inv=True)

    assert not np.all(np.isfinite(np.asarray(inverse.data)))
    assert _last(treated)["parameters"]["inv"] is True


def test_reverse_option_is_recorded_and_reverses_the_window():
    dataset = _dataset()
    treated, window = dataset.hamming(retapod=True)
    reversed_treated, reversed_window = dataset.hamming(retapod=True, rev=True)

    np.testing.assert_array_equal(
        np.asarray(reversed_window.data), np.asarray(window.data)[::-1]
    )
    assert _last(reversed_treated)["parameters"]["rev"] is True
    assert _last(treated)["parameters"]["rev"] is False


def test_retapod_preserves_the_return_structure_and_leaves_the_window_history_empty():
    dataset = _dataset()

    returned = dataset.hann(retapod=True)

    # The arbitrated contract: the return structure is unchanged, the treated
    # dataset records the operation, and the returned window -- which never
    # underwent an apodization -- records nothing.
    assert isinstance(returned, tuple)
    assert len(returned) == 2
    treated, window = returned
    assert len(treated.history_entries) == 2
    assert len(window.history_entries) == 0
    assert window is not treated
    assert np.shape(window.data) == (64,)
    assert np.shape(treated.data) == (1, 64)


def test_interferogram_path_records_one_entry():
    dataset = _interferogram()
    before = len(dataset.history_entries)

    result = dataset.em(lb=50.0)

    assert len(result.history_entries) == before + 1
    assert _last(result)["operation"] == "em"
    assert result.shape == (3, 100)


def test_refused_call_appends_no_success_entry():
    refused = scp.NDDataset(np.arange(8.0) + 1.0, units="K")
    refused.x = scp.Coord(np.linspace(1000, 4000, 8), units="cm^-1")

    result = refused.hamming()

    assert len(result.history_entries) == 0
    np.testing.assert_array_equal(result.data, refused.data)


def test_refused_inplace_call_leaves_the_source_intact():
    refused = scp.NDDataset(np.arange(8.0) + 1.0, units="K")
    refused.x = scp.Coord(np.linspace(1000, 4000, 8), units="cm^-1")
    snapshot = _snapshot(refused)

    refused.hamming(inplace=True)

    _assert_unchanged(refused, snapshot)
    assert len(refused.history_entries) == 0


def test_dryrun_records_nothing_and_touches_nothing():
    dataset = _dataset()
    snapshot = _snapshot(dataset)
    before = len(dataset.history_entries)

    result = dataset.hamming(dryrun=True)

    # A cancelled operation must not claim a success entry.
    assert len(result.history_entries) == before
    _assert_unchanged(dataset, snapshot)


def test_previous_entries_are_preserved():
    dataset = _dataset()
    earlier = copy.deepcopy(dataset.history_entries)

    result = dataset.sp()

    assert result.history_entries[: len(earlier)] == earlier
    assert len(result.history_entries) == len(earlier) + 1


def test_history_entries_are_detached_from_the_dataset():
    dataset = _dataset()
    result = dataset.sp()

    entries = result.history_entries
    entries[-1]["parameters"]["scientific_parameters"]["ssb"] = 999
    entries[-1]["parameters"]["injected"] = True

    assert _last(result)["parameters"]["scientific_parameters"]["ssb"] == 1
    assert "injected" not in _last(result)["parameters"]


def test_parameters_contain_no_live_objects():
    dataset = _dataset()

    result = dataset.em(lb="250 Hz", shifted="1.5 us")

    # Round-tripping through JSON must not describe anything as unretained.
    assert (
        json.loads(json.dumps(_last(result)["parameters"]))
        == _last(result)["parameters"]
    )


def test_scp_roundtrip_preserves_the_unit_bearing_parameters(tmp_path):
    dataset = _dataset()
    result = dataset.em(lb="250 Hz", shifted="1.5 us")
    filename = tmp_path / "apodization-history.scp"

    result.save_as(filename, confirm=False)
    rebuilt = scp.load(filename)

    assert rebuilt.history_entries == result.history_entries
    assert rebuilt.history == result.history
    _assert_requested_parameters(
        rebuilt.history_entries[-1]["parameters"]["requested_parameters"],
        {
            "lb": {"value": 250, "units": "Hz"},
            "shifted": {"value": 1.5, "units": "µs"},
        },
    )
