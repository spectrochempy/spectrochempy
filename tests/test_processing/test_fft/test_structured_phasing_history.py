"""Structured history for the shared phasing wrapper."""

import copy
import json
import unicodedata

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.processing.fft import phasing as phasing_module

Q = scp.Quantity


def _dataset(
    shape=(64,), units="Hz", phc0=0.0, phc1=0.0, phased=True, pivot=0.0, exptc=0.0
):
    """
    A complex dataset carrying the phase metadata the wrapper reads back.

    ``pk`` reads the current phase, the pivot and the exponential time constant
    from the metadata, so a dataset without them is refused. Building them here
    keeps the tests focused on the history rather than on metadata setup.
    """
    rng = np.random.default_rng(0)
    dataset = scp.NDDataset(
        rng.normal(size=shape) + 1.0j * rng.normal(size=shape),
        dims=["y", "x"][-len(shape) :] if len(shape) > 1 else ["x"],
        units="V",
        title="phasing history",
    )
    dataset.x = scp.Coord(np.linspace(100.0, 0.0, shape[-1]), units=units, title="freq")
    ndim = len(shape)
    dataset.meta.phc0 = [Q(phc0, "degree")] * ndim
    dataset.meta.phc1 = [Q(phc1, "degree")] * ndim
    dataset.meta.phased = [phased] * ndim
    dataset.meta.pivot = [pivot] * ndim
    dataset.meta.exptc = [Q(exptc, "us")] * ndim
    dataset.annotate("prepared")
    return dataset


def _unphased(**kwargs):
    return _dataset(phased=False, **kwargs)


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


def _operations(dataset):
    """The recorded operations, leaving the free form ``annotate`` entries out."""
    return [
        entry["operation"]
        for entry in dataset.history_entries
        if entry["operation"] is not None
    ]


def _annotated_only(dataset):
    return [entry for entry in dataset.history_entries if entry["parameters"] == {}]


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
        if reference["units"] is None:
            assert actual[key]["units"] is None
        else:
            assert unicodedata.normalize(
                "NFKC", actual[key]["units"]
            ) == unicodedata.normalize("NFKC", reference["units"])


def _pk_window(size, phc0, phc1, exptc, pivot):
    """
    Independent reference for the ``pk`` window, in the units the kernel uses.

    Written from the documented functional form rather than delegated to the
    kernel, so that reconstructing a window from a history entry is not
    circular. ``phc0`` and ``phc1`` are degrees, ``pivot`` is in the units of
    the phased coordinate, and ``exptc`` is in the inverse of those units when
    the request carried units, as ``_check_units(..., inv=True)`` produces.
    """
    index = np.arange(size)
    phc0 = np.pi * phc0 / 180.0
    if exptc > 0.0:
        return np.exp(1.0j * (phc0 * np.exp(-exptc * (index - pivot) / size)))
    phc1 = np.pi * phc1 / 180.0
    return np.exp(1.0j * (phc0 + (phc1 * (index - pivot) / size)))


def test_one_call_records_exactly_one_structured_entry():
    result = _dataset().pk(phc0=30.0)
    assert _operations(result) == ["pk"]
    entry = _last(result)
    assert entry["operation"] == "pk"
    assert entry["parameters"]["requested_dim"] is None
    assert entry["parameters"]["resolved_dim"] == "x"
    assert entry["parameters"]["resolved_axis"] == 0
    assert entry["parameters"]["inplace"] is False
    assert "pk" in entry["message"]


def test_pk_exp_records_the_delegated_kernel_only():
    result = _dataset().pk_exp(phc0=30.0)
    assert _operations(result) == ["pk"]
    # pk_exp hands a zero first order phase to pk, so none is claimed.
    assert _last(result)["parameters"]["scientific_parameters"] == {
        "phc0": 30.0,
        "phc1": 0.0,
        "exptc": 0.0,
        "pivot": 0.0,
    }


def test_omitted_parameters_record_the_effective_kernel_defaults():
    result = _dataset().pk()
    parameters = _last(result)["parameters"]
    assert parameters["scientific_parameters"] == {
        "phc0": 0.0,
        "phc1": 0.0,
        "exptc": 0.0,
        "pivot": 0.0,
    }
    # Nothing was requested, so there is no request to retain.
    assert "requested_parameters" not in parameters


def test_recorded_parameters_are_the_applied_correction_not_the_target():
    # Already phased at 10 degree, asking for 30 degree corrects by 20.
    result = _dataset(phc0=10.0).pk(phc0=30.0)
    parameters = _last(result)["parameters"]
    assert parameters["scientific_parameters"]["phc0"] == pytest.approx(20.0)
    _assert_requested_parameters(
        parameters["requested_parameters"], {"phc0": {"value": 30, "units": None}}
    )


def test_first_call_on_an_unphased_dimension_records_the_negated_correction():
    result = _unphased().pk(phc0=30.0)
    parameters = _last(result)["parameters"]
    assert parameters["scientific_parameters"]["phc0"] == pytest.approx(-30.0)
    _assert_requested_parameters(
        parameters["requested_parameters"], {"phc0": {"value": 30, "units": None}}
    )


def test_relative_correction_accumulates_and_is_recorded():
    dataset = _dataset(phc0=10.0)
    result = dataset.pk(phc0=30.0, rel=True)
    parameters = _last(result)["parameters"]
    assert parameters["rel"] is True
    assert parameters["scientific_parameters"]["phc0"] == pytest.approx(30.0)
    # The target and the applied correction coincide, so nothing is retained.
    assert "requested_parameters" not in parameters
    assert result.meta.phc0[-1].magnitude == pytest.approx(40.0)
    assert result.meta.phc0[-1].units == scp.ur.degree


def test_absolute_correction_is_recorded_as_not_relative():
    assert _last(_dataset(phc0=10.0).pk(phc0=30.0))["parameters"]["rel"] is False


def test_successive_calls_each_record_the_increment_they_applied():
    dataset = _dataset()
    dataset.pk(phc0=10.0, inplace=True)
    dataset.pk(phc0=20.0, inplace=True)
    dataset.pk_exp(phc0=5.0, inplace=True)
    entries = dataset.history_entries[1:]
    assert _operations(dataset) == ["pk", "pk", "pk"]
    assert [
        entry["parameters"]["scientific_parameters"]["phc0"] for entry in entries
    ] == pytest.approx([10.0, 10.0, -5.0])
    # Only the calls whose target differs from the applied correction keep it.
    assert ["requested_parameters" in entry["parameters"] for entry in entries] == [
        False,
        True,
        True,
    ]


def test_first_order_phase_is_not_claimed_when_a_positive_exptc_ignores_it():
    result = _dataset().pk(phc0=30.0, phc1=10.0, exptc=5.0)
    parameters = _last(result)["parameters"]
    # The kernel drops phc1 entirely, so the entry must not report it applied.
    assert "phc1" not in parameters["scientific_parameters"]
    # A plain exptc is already expressed in inverse coordinate units, so it
    # matches its request; only the unused first order phase is retained.
    assert parameters["scientific_parameters"]["exptc"] == pytest.approx(5.0)
    _assert_requested_parameters(
        parameters["requested_parameters"],
        {"phc1": {"value": 10, "units": None}},
    )


def test_first_order_phase_is_recorded_when_no_exponential_correction_applies():
    result = _dataset().pk(phc0=30.0, phc1=10.0, exptc=0.0)
    parameters = _last(result)["parameters"]
    assert parameters["scientific_parameters"]["phc1"] == pytest.approx(10.0)
    # phc0 and exptc match their request, so only exptc's spelling is retained
    # by the conversion below; here nothing differs.
    assert "requested_parameters" not in parameters


def test_a_first_order_phase_really_has_no_effect_on_the_window():
    # The recorded omission of phc1 must match the data, not just the schema.
    with_first = _dataset().pk(phc0=30.0, phc1=10.0, exptc=5.0)
    with_other = _dataset().pk(phc0=30.0, phc1=99.0, exptc=5.0)
    np.testing.assert_allclose(with_first.data, with_other.data)

    without = _dataset().pk(phc0=30.0, phc1=10.0, exptc=0.0)
    other = _dataset().pk(phc0=30.0, phc1=99.0, exptc=0.0)
    assert not np.allclose(without.data, other.data)


def test_the_first_order_rule_is_shared_with_the_kernel():
    # A change of the kernel rule must not silently desynchronise the recorder.
    assert phasing_module._first_order_contributes(0.0) is True
    assert phasing_module._first_order_contributes(5.0) is False
    assert phasing_module._first_order_contributes(-1.0) is True


def test_unit_bearing_request_retains_its_magnitude_and_unit():
    result = _dataset().pk(phc0=Q(30, "degree"), pivot=Q(0.5, "Hz"), exptc=Q(5, "us"))
    parameters = _last(result)["parameters"]
    # A unit-bearing exptc is a time converted to the inverse of the coordinate
    # units, so the request is retained next to the applied magnitude.
    _assert_requested_parameters(
        parameters["requested_parameters"],
        {"exptc": {"value": 5, "units": "µs"}},
    )
    assert parameters["scientific_parameters"]["exptc"] == pytest.approx(5e-06)
    assert parameters["scientific_parameters"]["pivot"] == pytest.approx(0.5)


def test_pivot_and_exptc_use_opposite_unit_conventions():
    # pivot is converted to the coordinate units, exptc to their inverse, so the
    # same physical time is a different effective number on a different scale.
    on_hz = _dataset(units="Hz").pk(phc0=30.0, exptc=Q(5, "us"))
    on_khz = _dataset(units="kHz").pk(phc0=30.0, exptc=Q(5, "us"))
    assert _last(on_hz)["parameters"]["scientific_parameters"][
        "exptc"
    ] == pytest.approx(5e-06)
    assert _last(on_khz)["parameters"]["scientific_parameters"][
        "exptc"
    ] == pytest.approx(0.005)

    # pivot goes the other way: it lands in the coordinate units themselves.
    pivot_hz = _dataset(units="Hz").pk(phc0=30.0, pivot=Q(0.5, "Hz"))
    pivot_khz = _dataset(units="kHz").pk(phc0=30.0, pivot=Q(0.5, "Hz"))
    assert _last(pivot_hz)["parameters"]["scientific_parameters"][
        "pivot"
    ] == pytest.approx(0.5)
    assert _last(pivot_khz)["parameters"]["scientific_parameters"][
        "pivot"
    ] == pytest.approx(0.0005)


def test_a_dimensionless_exptc_is_read_as_a_plain_number():
    # Without units the wrapper multiplies by the coordinate units, so 5.0 is
    # 5.0 whatever the coordinate scale, unlike the unit-bearing form.
    for units, expected in (("Hz", 5.0), ("kHz", 5.0)):
        result = _dataset(units=units).pk(phc0=30.0, exptc=5.0)
        parameters = _last(result)["parameters"]
        assert parameters["scientific_parameters"]["exptc"] == pytest.approx(expected)
        # A plain number matches what the kernel is given, so it is not retained.
        assert "exptc" not in parameters.get("requested_parameters", {})


def test_request_surviving_conversion_unchanged_is_not_duplicated():
    # 0.5 Hz requested and 0.5 applied describe the same quantity.
    result = _dataset().pk(phc0=30.0, pivot=Q(0.5, "Hz"))
    parameters = _last(result)["parameters"]
    assert parameters["scientific_parameters"]["pivot"] == pytest.approx(0.5)
    assert "requested_parameters" not in parameters


def test_phase_requested_in_radians_is_recorded_in_the_applied_unit():
    result = _dataset().pk(phc0=Q(0.5, "radian"))
    parameters = _last(result)["parameters"]
    assert parameters["scientific_parameters"]["phc0"] == pytest.approx(
        0.5 * 180.0 / np.pi
    )
    _assert_requested_parameters(
        parameters["requested_parameters"], {"phc0": {"value": 0.5, "units": "rad"}}
    )


def test_pivot_inherited_from_the_metadata_is_recorded_as_the_effective_value():
    result = _dataset(pivot=0.25).pk(phc0=30.0)
    parameters = _last(result)["parameters"]
    assert parameters["scientific_parameters"]["pivot"] == pytest.approx(0.25)
    # Nothing was requested for the pivot, so no request is invented.
    assert "requested_parameters" not in parameters


def test_requested_and_resolved_dimensions_are_both_recorded():
    result = _dataset(shape=(8, 64)).pk(phc0=30.0, dim=0)
    parameters = _last(result)["parameters"]
    assert parameters["requested_dim"] == 0
    assert parameters["resolved_dim"] == "y"
    assert parameters["resolved_axis"] == 0


def test_negative_dimension_is_resolved_to_a_positive_axis():
    result = _dataset(shape=(8, 64)).pk(phc0=30.0, dim=-2)
    parameters = _last(result)["parameters"]
    assert parameters["requested_dim"] == -2
    assert parameters["resolved_dim"] == "y"
    assert parameters["resolved_axis"] == 0


def test_non_final_dimension_records_one_entry_and_keeps_dimensions():
    result = _dataset(shape=(8, 64)).pk(phc0=30.0, dim=0)
    assert _operations(result) == ["pk"]
    assert result.shape == (8, 64)
    assert result.dims == ["y", "x"]


def test_non_final_dimension_uses_the_metadata_of_the_phased_dimension():
    dataset = _dataset(shape=(8, 64))
    dataset.meta.phc0[0] = 10.0 * scp.ur.degree
    dataset.meta.phc0[1] = 40.0 * scp.ur.degree
    result = dataset.pk(phc0=50.0, dim=0)
    # The y metadata travels with the swap, so the correction that is recorded
    # and applied follows the dimension that was asked for, not the last one.
    assert _last(result)["parameters"]["scientific_parameters"][
        "phc0"
    ] == pytest.approx(40.0)


def test_non_final_dimension_updates_the_metadata_of_that_dimension():
    # The correction is read from and written back to the same dimension. The
    # wrapper writes `meta.phc0[-1]` while the phased dimension is still last,
    # and only then undoes the swap, so the write lands on the requested
    # dimension. The stored value is the applied correction, which is the
    # pre-existing semantics: a `dim=-1` call stores the same kind of value.
    dataset = _dataset(shape=(8, 64))
    dataset.meta.phc0[0] = 10.0 * scp.ur.degree
    dataset.meta.phc0[1] = 40.0 * scp.ur.degree
    result = dataset.pk(phc0=50.0, dim=0)
    assert result.meta.phc0[0].magnitude == pytest.approx(40.0)  # y: 50 - 10
    assert result.meta.phc0[1].magnitude == pytest.approx(40.0)  # x: untouched

    last = _dataset()
    last.meta.phc0[0] = 40.0 * scp.ur.degree
    assert last.pk(phc0=50.0).meta.phc0[0].magnitude == pytest.approx(10.0)


def test_relative_and_first_time_calls_target_the_requested_dimension():
    dataset = _dataset(shape=(8, 64))
    dataset.meta.phc0[0] = 10.0 * scp.ur.degree
    dataset.meta.phc0[1] = 40.0 * scp.ur.degree
    accumulated = dataset.pk(phc0=30.0, dim=0, rel=True)
    assert accumulated.meta.phc0[0].magnitude == pytest.approx(40.0)
    assert accumulated.meta.phc0[1].magnitude == pytest.approx(40.0)

    fresh = _dataset(shape=(8, 64))
    fresh.meta.phc0[0] = 10.0 * scp.ur.degree
    fresh.meta.phc0[1] = 40.0 * scp.ur.degree
    fresh.meta.phased[0] = False
    first = fresh.pk(phc0=30.0, dim=0)
    # A not yet phased dimension is reset rather than corrected.
    assert first.meta.phc0[0].magnitude == pytest.approx(0.0)
    assert first.meta.phc0[1].magnitude == pytest.approx(40.0)
    assert first.meta.phased[0] is True


@pytest.mark.parametrize("inplace", [True, False])
def test_inplace_and_outof_place_modes(inplace):
    source = _dataset()
    result = source.pk(phc0=30.0, inplace=inplace)
    assert _last(result)["parameters"]["inplace"] is inplace
    if inplace:
        assert result is source
    else:
        assert result is not source
        assert _annotated_only(source) == source.history_entries


def test_recorded_parameters_reproduce_the_executed_window():
    # Rebuilding the window from the entry alone must explain the data change.
    source = _dataset()
    result = source.pk(phc0=30.0, phc1=10.0, pivot=0.2, rel=True)
    effective = _last(result)["parameters"]["scientific_parameters"]
    window = _pk_window(
        source.shape[-1],
        effective["phc0"],
        effective["phc1"],
        effective["exptc"],
        effective["pivot"],
    )
    np.testing.assert_allclose(result.data, source.data * window)


def test_recorded_parameters_reproduce_an_exponential_window():
    source = _dataset()
    result = source.pk(phc0=30.0, phc1=10.0, exptc=5.0)
    effective = _last(result)["parameters"]["scientific_parameters"]
    window = _pk_window(
        source.shape[-1],
        effective["phc0"],
        # The entry correctly omits phc1, and the window must match anyway.
        effective.get("phc1", 0.0),
        effective["exptc"],
        effective["pivot"],
    )
    np.testing.assert_allclose(result.data, source.data * window)


def test_inverse_request_is_refused_and_records_nothing():
    source = _dataset()
    with pytest.raises(NotImplementedError):
        source.pk(phc0=30.0, inv=True)
    assert _annotated_only(source) == source.history_entries


def test_time_coordinate_is_refused_and_records_nothing():
    source = _dataset(units="ms")
    snapshot = _snapshot(source)
    result = source.pk(phc0=30.0)
    # A time coordinate is outside the documented domain: the call is cancelled
    # and must not claim a successful correction.
    assert result.history_entries == snapshot["history"]
    _assert_unchanged(result, snapshot)


def test_unsupported_string_units_are_refused_and_record_nothing():
    # Pre-existing behaviour: a unit string is not a Quantity and the wrapper
    # does not build one. The history must not describe a call that failed.
    source = _dataset()
    with pytest.raises(TypeError):
        source.pk(phc0="30 degree")
    assert _annotated_only(source) == source.history_entries


def test_previous_entries_are_preserved():
    dataset = _dataset()
    dataset.pk(phc0=10.0, inplace=True)
    dataset.annotate("manual")
    dataset.pk(phc0=20.0, inplace=True)
    assert _operations(dataset) == ["pk", "pk"]
    # The free form annotations survive both phasing calls, in order.
    assert [entry["message"] for entry in _annotated_only(dataset)] == [
        "prepared",
        "manual",
    ]


def test_history_entries_are_detached_from_the_dataset():
    result = _dataset().pk(phc0=30.0)
    entries = result.history_entries
    entries[-1]["parameters"]["scientific_parameters"]["phc0"] = 999.0
    assert _last(result)["parameters"]["scientific_parameters"][
        "phc0"
    ] == pytest.approx(30.0)


def test_parameters_contain_no_live_objects():
    result = _dataset().pk(phc0=Q(30, "degree"), exptc=Q(5, "us"))
    parameters = _last(result)["parameters"]
    for value in parameters["scientific_parameters"].values():
        assert not isinstance(value, (Q, np.ndarray))
    for described in parameters.get("requested_parameters", {}).values():
        assert not isinstance(described["value"], (Q, np.ndarray))
    # Every recorded value must be a plain serializable scalar. The entry
    # itself also carries a timestamp, so only the parameters are dumped.
    json.loads(json.dumps(parameters))
    assert isinstance(_last(result)["message"], str)


def test_scp_roundtrip_preserves_the_phasing_entry(tmp_path):
    result = _dataset().pk(phc0=Q(30, "degree"), pivot=Q(0.5, "Hz"), exptc=Q(5, "us"))
    path = tmp_path / "phasing.scp"
    scp.write(result, path, overwrite=True)
    reloaded = scp.read(path)
    entry = _last(reloaded)
    assert entry["operation"] == "pk"
    assert entry["parameters"]["scientific_parameters"] == pytest.approx(
        {"phc0": 30.0, "exptc": 5e-06, "pivot": 0.5}
    )
    _assert_requested_parameters(
        entry["parameters"]["requested_parameters"],
        {"exptc": {"value": 5, "units": "\u00b5s"}},
    )
