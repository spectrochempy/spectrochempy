import gc
import weakref

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.processing.transformation.concatenate import concatenate
from spectrochempy.processing.transformation.concatenate import stack
from spectrochempy.provenance import ProvenanceCapture
from spectrochempy.provenance import _instrument
from spectrochempy.utils.exceptions import DimensionsCompatibilityError
from spectrochempy.utils.exceptions import UnitsCompatibilityError


def _dataset(rows=2, columns=3, *, offset=0.0, units=None):
    data = np.arange(rows * columns, dtype=float).reshape(rows, columns) + offset
    y = scp.Coord(np.arange(rows, dtype=float), name="y")
    x = scp.Coord(np.arange(columns, dtype=float), name="x")
    return scp.NDDataset(data, coordset=[y, x], dims=["y", "x"], units=units)


def _values(record, kind):
    return record.to_dict()["parameters"][kind]["values"]


@pytest.mark.parametrize(
    ("call", "operation_id"),
    [
        (
            lambda left, right: concatenate(left, right, dims="y"),
            _instrument.CONCATENATE_OPERATION_ID,
        ),
        (
            lambda left, right: left.concatenate(right, dims="y"),
            _instrument.CONCATENATE_OPERATION_ID,
        ),
        (lambda left, right: stack(left, right), _instrument.STACK_OPERATION_ID),
        (lambda left, right: left.stack(right), _instrument.STACK_OPERATION_ID),
    ],
)
def test_function_and_method_entrypoints_emit_one_record(call, operation_id):
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)

    with ProvenanceCapture() as capture:
        result = call(left, right)

    assert result.size > 0
    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.operation_id == operation_id
    assert [link.role for link in record.inputs] == ["source[0]", "source[1]"]
    assert [link.role for link in record.outputs] == ["result"]


def test_order_reversal_and_repeated_inputs_keep_positions_and_identity(monkeypatch):
    left = _dataset(offset=10.0)
    right = _dataset(offset=20.0)

    with ProvenanceCapture() as capture:
        observe = capture._observe_source
        observed_ids = []

        def count_observation(value):
            observed_ids.append(id(value))
            return observe(value)

        monkeypatch.setattr(capture, "_observe_source", count_observation)
        forward = concatenate(left, right, left, dims="y")
        reverse = concatenate(right, left, dims="y")

    first, second = capture.ledger.operation_records
    assert [link.role for link in first.inputs] == [
        "source[0]",
        "source[1]",
        "source[2]",
    ]
    assert first.inputs[0].reference == first.inputs[2].reference
    assert first.inputs[0].reference == second.inputs[1].reference
    assert first.inputs[1].reference == second.inputs[0].reference
    assert observed_ids.count(id(left)) == 2
    assert observed_ids.count(id(right)) == 2
    np.testing.assert_array_equal(
        forward.data,
        np.concatenate([left.data, right.data, left.data], axis=0),
    )
    np.testing.assert_array_equal(
        reverse.data,
        np.concatenate([right.data, left.data], axis=0),
    )


@pytest.mark.parametrize(
    ("kwargs", "requested", "axis", "dim"),
    [
        ({}, {}, 1, "x"),
        ({"dims": "y"}, {"dims": "y"}, 0, "y"),
        ({"dim": "y"}, {"dim": "y"}, 0, "y"),
        ({"axis": 0}, {"axis": 0}, 0, "y"),
        (
            {"axis": 1, "dim": "y", "dims": "x"},
            {"axis": 1, "dim": "y", "dims": "x"},
            1,
            "x",
        ),
    ],
)
def test_concatenate_requested_and_resolved_dimension(kwargs, requested, axis, dim):
    left = _dataset()
    right = _dataset(offset=10.0)

    with ProvenanceCapture() as capture:
        concatenate(left, right, **kwargs)

    record = capture.ledger.operation_records[0]
    assert _values(record, "requested") == requested
    resolved = _values(record, "resolved")
    assert resolved["axis"] == axis
    assert resolved["dim"] == dim
    assert resolved["mode"] == "existing_dimension"
    assert resolved["input_count"] == 2


@pytest.mark.parametrize(
    ("kwargs", "requested", "axis"),
    [
        ({}, {}, 0),
        ({"axis": 0}, {"axis": 0}, 0),
        ({"axis": None}, {"axis": None}, 0),
        ({"axis": 1}, {"axis": 1}, 1),
        ({"axis": -1}, {"axis": -1}, 1),
    ],
)
def test_stack_requested_and_resolved_new_dimension(kwargs, requested, axis):
    left = _dataset(rows=1).squeeze()
    right = _dataset(rows=1, offset=10.0).squeeze()

    with ProvenanceCapture() as capture:
        result = stack(left, right, **kwargs)

    record = capture.ledger.operation_records[0]
    assert _values(record, "requested") == requested
    resolved = _values(record, "resolved")
    assert resolved["axis"] == axis
    assert resolved["dim"] == result.dims[axis]
    assert resolved["mode"] == "new_dimension"


def test_concatenate_axis_one_profile_promotion_is_reported_truthfully():
    left = _dataset(rows=1).squeeze()
    right = _dataset(rows=1, offset=10.0).squeeze()

    with ProvenanceCapture() as capture:
        result = concatenate(left, right, axis=1)

    resolved = _values(capture.ledger.operation_records[0], "resolved")
    assert result.shape == (3, 2)
    assert resolved["axis"] == 1
    assert resolved["mode"] == "new_dimension"


def test_p2_p5_p4_p3_chain_links_result_states():
    source = _dataset(rows=6, columns=4)
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as capture:
        selected = source[:3]
        transformer.fit(selected)
        centered = transformer.transform(selected)
        assembled = concatenate(selected, centered, dims="y")
        residual = assembled - assembled
        result = residual.transpose()

    records = capture.ledger.operation_records
    assembly = records[3]
    arithmetic = records[4]
    transpose = records[5]
    assert assembly.inputs[0].reference == records[0].outputs[0].reference
    assert assembly.inputs[1].reference == records[2].outputs[0].reference
    assert arithmetic.inputs[0].reference == assembly.outputs[0].reference
    assert arithmetic.inputs[1].reference == assembly.outputs[0].reference
    assert transpose.inputs[0].reference == arithmetic.outputs[0].reference
    assert result.shape == (4, 6)


def test_capture_preserves_values_coordinates_units_masks_metadata_title_and_history():
    left = _dataset(offset=1.0, units="m")
    right = _dataset(offset=2.0, units="cm")
    left[0, 1] = scp.MASKED
    right[1, 2] = scp.MASKED
    left.title = right.title = "shared"
    left.meta.project = right.meta.project = "p4"
    expected = concatenate(left, right, dims="y")

    with ProvenanceCapture() as capture:
        result = concatenate(left, right, dims="y")

    np.testing.assert_array_equal(result.data, expected.data)
    np.testing.assert_array_equal(result.mask, expected.mask)
    assert result.units == expected.units
    assert result.dims == expected.dims
    assert result.coordset == expected.coordset
    assert result.meta == expected.meta
    assert result.title == expected.title
    assert [entry.split("> ", 1)[-1] for entry in result.history] == [
        entry.split("> ", 1)[-1] for entry in expected.history
    ]
    assert len(capture.ledger) == 1


@pytest.mark.parametrize("failure", ["shape", "units"])
def test_failed_assembly_preserves_exception_and_records_no_output(failure):
    left = _dataset(rows=2, columns=3, units="m")
    if failure == "shape":
        right = _dataset(rows=2, columns=4, units="m")
        kwargs = {"dims": "y"}
        error = DimensionsCompatibilityError
    else:
        right = _dataset(rows=2, columns=3, units="s")
        kwargs = {}
        error = UnitsCompatibilityError

    with pytest.raises(error) as expected:
        concatenate(left, right, **kwargs)

    with ProvenanceCapture() as capture, pytest.raises(error) as caught:
        concatenate(left, right, **kwargs)

    assert str(caught.value) == str(expected.value)

    record = capture.ledger.operation_records[0]
    assert record.status == "failed"
    assert record.outputs == ()
    assert [link.role for link in record.inputs] == ["source[0]", "source[1]"]


def test_boundary_capture_failure_invalidates_successful_result(monkeypatch):
    left = _dataset()
    right = _dataset(offset=10.0)

    with ProvenanceCapture() as capture:
        observe = capture._observe_source

        def fail_observation(value):
            raise RuntimeError("identity registry unavailable")

        monkeypatch.setattr(capture, "_observe_source", fail_observation)
        result = concatenate(left, right, dims="y")
        monkeypatch.setattr(capture, "_observe_source", observe)
        result.transpose()

    assert [record.operation_id for record in capture.ledger] == [
        _instrument.TRANSPOSE_OPERATION_ID
    ]
    assert capture.ledger.operation_records[0].capture["status"] == "partial"


def test_record_write_failure_invalidates_successful_result(monkeypatch):
    left = _dataset()
    right = _dataset(offset=10.0)

    with ProvenanceCapture() as capture:
        append = capture.ledger.append
        calls = 0

        def fail_first(record):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("ledger unavailable")
            append(record)

        monkeypatch.setattr(capture.ledger, "append", fail_first)
        result = stack(left, right)
        result.transpose()

    assert [record.operation_id for record in capture.ledger] == [
        _instrument.TRANSPOSE_OPERATION_ID
    ]
    assert capture.ledger.operation_records[0].capture["status"] == "partial"


def test_warning_sink_failure_never_masks_successful_assembly(monkeypatch):
    left = _dataset()
    right = _dataset(offset=10.0)

    with ProvenanceCapture() as capture:
        append = capture.ledger.append
        note_warning = capture._note_capture_warning

        def fail_append(record):
            raise RuntimeError("ledger unavailable")

        def fail_warning(operation_id, exc):
            raise RuntimeError("warning sink unavailable")

        monkeypatch.setattr(capture.ledger, "append", fail_append)
        monkeypatch.setattr(capture, "_note_capture_warning", fail_warning)
        result = concatenate(left, right, dims="y")
        monkeypatch.setattr(capture.ledger, "append", append)
        monkeypatch.setattr(capture, "_note_capture_warning", note_warning)
        result.transpose()

    np.testing.assert_array_equal(
        result.data, np.concatenate([left.data, right.data], axis=0)
    )
    assert capture.ledger.operation_records[0].capture["status"] == "partial"


def test_unrecorded_input_mutation_marks_each_repeated_position_partial():
    source = _dataset()

    with ProvenanceCapture() as capture:
        concatenate(source, source, dims="y")
        source.data[0, 0] = -100.0
        concatenate(source, source, dims="y")

    second = capture.ledger.operation_records[1]
    assert second.capture["status"] == "partial"
    assert [item["field"] for item in second.capture["omissions"]] == [
        "inputs[0].reference",
        "inputs[1].reference",
    ]


def test_nested_contexts_and_numpy_paths_are_bounded():
    left = _dataset()
    right = _dataset(offset=10.0)

    with ProvenanceCapture() as outer:
        concatenate(left, right, dims="y")
        with ProvenanceCapture() as inner:
            stack(left, right)
        np_concatenated = np.concatenate([left, right], axis=0)
        np_stacked = np.stack([left, right], axis=0)

    assert [record.operation_id for record in outer.ledger] == [
        _instrument.CONCATENATE_OPERATION_ID
    ]
    assert [record.operation_id for record in inner.ledger] == [
        _instrument.STACK_OPERATION_ID
    ]
    assert isinstance(np_concatenated, np.ndarray)
    assert isinstance(np_stacked, np.ndarray)


def test_all_input_references_are_retained_without_truncation():
    source = _dataset(rows=1, columns=1)
    sources = [source] * 80

    with ProvenanceCapture() as capture:
        concatenate(sources, dims="y")

    record = capture.ledger.operation_records[0]
    assert len(record.inputs) == 80
    assert [link.role for link in record.inputs] == [
        f"source[{index}]" for index in range(80)
    ]
    assert len({link.reference for link in record.inputs}) == 1
    assert record.capture["status"] == "complete"


def test_pipeline_suppresses_internal_assembly_records(monkeypatch):
    def concatenate_self(self, dataset):
        return concatenate(dataset, dataset, dims="y")

    source = _dataset(rows=4)
    monkeypatch.setattr(scp.CenterTransformer, "_transform", concatenate_self)
    pipeline = scp.Pipeline([("concatenate-self", scp.CenterTransformer(dim="y"))])

    with ProvenanceCapture() as capture:
        pipeline.fit(source)
        result = pipeline.transform(source)
        direct = concatenate(source, source, dims="y")

    assert result.shape == direct.shape
    assert [record.operation_id for record in capture.ledger] == [
        _instrument.CONCATENATE_OPERATION_ID
    ]


def test_capture_does_not_retain_inputs_or_result():
    left = _dataset()
    right = _dataset(offset=10.0)
    left_ref = weakref.ref(left)
    right_ref = weakref.ref(right)

    with ProvenanceCapture() as capture:
        result = concatenate(left, right, dims="y")
        result_ref = weakref.ref(result)

    del left, right, result
    gc.collect()
    assert capture._identity_size == 0
    assert left_ref() is None
    assert right_ref() is None
    assert result_ref() is None


def test_single_list_tuple_and_invalid_iterable_behaviors_are_preserved():
    source = _dataset()

    with ProvenanceCapture() as capture:
        direct = concatenate(source, dims="y")
        listed = concatenate([source, source], dims="y")
        tupled = stack((source, source))

    assert direct.shape == source.shape
    assert listed.shape[0] == 2 * source.shape[0]
    assert tupled.shape[0] == 2
    assert len(capture.ledger) == 3

    with pytest.raises(IndexError):
        concatenate()
    with pytest.raises(AttributeError):
        concatenate(iter([source, source]))


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda _source: stack([]), DimensionsCompatibilityError),
        (lambda _source: stack([], axis=1), ValueError),
        (lambda source: concatenate([[source, source]], dims="y"), AttributeError),
    ],
)
def test_argument_validation_order_is_preserved_with_and_without_capture(call, error):
    source = _dataset()

    with pytest.raises(error) as without_capture:
        call(source)

    with ProvenanceCapture() as capture, pytest.raises(error) as with_capture:
        call(source)

    assert str(with_capture.value) == str(without_capture.value)
    assert len(capture.ledger) == 0
