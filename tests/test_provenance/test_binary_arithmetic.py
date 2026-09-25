import gc
import operator
import weakref

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.provenance import ProvenanceCapture
from spectrochempy.provenance import _instrument


def _dataset(rows=4, columns=6, *, offset=0.0, units=None):
    data = np.arange(rows * columns, dtype=float).reshape(rows, columns) + offset
    return scp.NDDataset(data, dims=["y", "x"], units=units)


def _binary_records(capture):
    return [
        record
        for record in capture.ledger
        if record.operation_id in _instrument.BINARY_OPERATION_IDS
    ]


@pytest.mark.parametrize(
    ("operation", "operation_id"),
    [
        (operator.add, _instrument.BINARY_ADD_OPERATION_ID),
        (operator.sub, _instrument.BINARY_SUBTRACT_OPERATION_ID),
        (operator.mul, _instrument.BINARY_MULTIPLY_OPERATION_ID),
        (operator.truediv, _instrument.BINARY_TRUE_DIVIDE_OPERATION_ID),
        (np.add, _instrument.BINARY_ADD_OPERATION_ID),
        (np.subtract, _instrument.BINARY_SUBTRACT_OPERATION_ID),
        (np.multiply, _instrument.BINARY_MULTIPLY_OPERATION_ID),
        (np.true_divide, _instrument.BINARY_TRUE_DIVIDE_OPERATION_ID),
    ],
)
def test_four_operator_and_ufunc_paths_emit_one_ordered_record(
    operation,
    operation_id,
):
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)

    with ProvenanceCapture() as capture:
        result = operation(left, right)

    assert result.shape == left.shape
    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.operation_id == operation_id
    assert record.category == "combine"
    assert [link.role for link in record.inputs] == ["left", "right"]
    assert [link.role for link in record.outputs] == ["result"]
    dispatch = record.to_dict()["parameters"]["resolved"]["values"]["dispatch"]
    assert dispatch == (
        "numpy_ufunc" if isinstance(operation, np.ufunc) else "operator"
    )


def test_noncommutative_operations_preserve_left_and_right_order():
    left = _dataset(offset=10.0)
    right = _dataset(offset=1.0)

    with ProvenanceCapture() as capture:
        difference = left - right
        quotient = right / left

    subtract, divide = capture.ledger.operation_records
    assert subtract.inputs[0].reference == divide.inputs[1].reference
    assert subtract.inputs[1].reference == divide.inputs[0].reference
    np.testing.assert_allclose(difference.data, left.data - right.data)
    np.testing.assert_allclose(quotient.data, right.data / left.data)


def test_same_dataset_keeps_two_roles_on_one_state():
    source = _dataset()

    with ProvenanceCapture() as capture:
        result = source - source

    record = capture.ledger.operation_records[0]
    assert [link.role for link in record.inputs] == ["left", "right"]
    assert record.inputs[0].reference == record.inputs[1].reference
    assert record.inputs[0].reference.object == record.inputs[1].reference.object
    np.testing.assert_array_equal(result.data, np.zeros_like(source.data))


def test_p2_center_p3_and_transpose_chain_share_state_references():
    source = _dataset(rows=6)
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as capture:
        calibration = source[:4]
        transformer.fit(calibration)
        validation = source[4:]
        centered = transformer.transform(validation)
        residual = validation - centered
        result = residual.transpose()

    (
        calibration_slice,
        fit,
        validation_slice,
        transform,
        subtract,
        transpose,
    ) = capture.ledger.operation_records
    assert fit.inputs[1].reference == calibration_slice.outputs[0].reference
    assert transform.inputs[1].reference == validation_slice.outputs[0].reference
    assert subtract.inputs[0].reference == validation_slice.outputs[0].reference
    assert subtract.inputs[1].reference == transform.outputs[0].reference
    assert transpose.inputs[0].reference == subtract.outputs[0].reference
    assert result.shape == (source.shape[1], 2)


def test_broadcasting_is_captured_in_both_orders():
    matrix = _dataset(rows=3, columns=4)

    with ProvenanceCapture() as capture:
        reference = matrix[0]
        forward = matrix - reference
        reverse = reference - matrix

    selection, forward_record, reverse_record = capture.ledger.operation_records
    assert forward_record.inputs[1].reference == selection.outputs[0].reference
    assert reverse_record.inputs[0].reference == selection.outputs[0].reference
    assert forward.shape == reverse.shape == matrix.shape
    np.testing.assert_allclose(forward.data, matrix.data - reference.data)
    np.testing.assert_allclose(reverse.data, reference.data - matrix.data)


def test_capture_preserves_units_masks_geometry_values_and_history():
    left = _dataset(rows=2, columns=3, offset=1.0, units="m")
    right = _dataset(rows=1, columns=3, offset=2.0, units="m")
    left[0, 1] = scp.MASKED
    right[0, 2] = scp.MASKED
    expected = left + right

    with ProvenanceCapture() as capture:
        result = left + right

    np.testing.assert_array_equal(result.data, expected.data)
    np.testing.assert_array_equal(result.mask, expected.mask)
    assert result.units == expected.units
    assert result.dims == expected.dims
    assert result.coordset == expected.coordset
    assert result.history == expected.history
    assert len(_binary_records(capture)) == 1


@pytest.mark.parametrize("mismatch", ["shape", "coordinates"])
def test_failed_arithmetic_keeps_scientific_exception_and_has_no_output(mismatch):
    left = _dataset(rows=2, columns=4)
    if mismatch == "shape":
        right = _dataset(rows=3, columns=5)
    else:
        right = _dataset(rows=2, columns=4)
        left.x = np.arange(4.0)
        right.x = np.arange(4.0) + 0.5

    with pytest.raises(Exception) as expected:
        left + right

    with (
        ProvenanceCapture() as capture,
        pytest.raises(type(expected.value)) as caught,
    ):
        left + right

    assert str(caught.value) == str(expected.value)
    record = capture.ledger.operation_records[0]
    assert record.status == "failed"
    assert record.outputs == ()
    assert [link.role for link in record.inputs] == ["left", "right"]


def test_excluded_arithmetic_paths_emit_no_p3_record():
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)

    with ProvenanceCapture() as capture:
        left + 2.0
        2.0 - left
        left * np.ones(left.shape)
        _negated = -left
        _powered = left**2
        _comparison = left < right
        mutated = left.copy()
        mutated += right
        np.add(left, right, out=(left.copy(),))
        with pytest.raises(TypeError):
            np.add.at(left, (0, 0), 1.0)

    assert _binary_records(capture) == []


def test_unrecorded_numeric_mutation_marks_next_binary_record_partial():
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)

    with ProvenanceCapture() as capture:
        first = left + right
        left.data[0, 0] = -100.0
        second = left + right

    first_record, second_record = capture.ledger.operation_records
    assert first_record.capture["status"] == "complete"
    assert second_record.capture["status"] == "partial"
    assert second_record.inputs[0].reference != first_record.inputs[0].reference
    assert any(
        omission["reason"] == "unrecorded_state_change"
        for omission in second_record.capture["omissions"]
    )
    assert first.data[0, 0] != second.data[0, 0]


def test_boundary_capture_failure_invalidates_successful_result(monkeypatch):
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)

    with ProvenanceCapture() as capture:
        observe = capture._observe_source

        def fail_observation(value):
            raise RuntimeError("identity registry unavailable")

        monkeypatch.setattr(capture, "_observe_source", fail_observation)
        result = left + right
        monkeypatch.setattr(capture, "_observe_source", observe)
        result.transpose()

    assert len(capture.capture_warnings) == 1
    assert [record.operation_id for record in capture.ledger] == [
        _instrument.TRANSPOSE_OPERATION_ID
    ]
    transpose = capture.ledger.operation_records[0]
    assert transpose.capture["status"] == "partial"


def test_record_write_failure_invalidates_successful_result(monkeypatch):
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)

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
        result = left + right
        result.transpose()

    assert len(capture.capture_warnings) == 1
    assert [record.operation_id for record in capture.ledger] == [
        _instrument.TRANSPOSE_OPERATION_ID
    ]
    assert capture.ledger.operation_records[0].capture["status"] == "partial"


def test_capture_warning_failure_never_masks_successful_arithmetic(monkeypatch):
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)

    with ProvenanceCapture() as capture:
        append = capture.ledger.append
        note_warning = capture._note_capture_warning

        def fail_append(record):
            raise RuntimeError("ledger unavailable")

        def fail_warning(operation_id, exc):
            raise RuntimeError("warning sink unavailable")

        monkeypatch.setattr(capture.ledger, "append", fail_append)
        monkeypatch.setattr(capture, "_note_capture_warning", fail_warning)
        result = left + right
        monkeypatch.setattr(capture.ledger, "append", append)
        monkeypatch.setattr(capture, "_note_capture_warning", note_warning)
        result.transpose()

    np.testing.assert_allclose(result.data, left.data + right.data)
    transpose = capture.ledger.operation_records[0]
    assert transpose.operation_id == _instrument.TRANSPOSE_OPERATION_ID
    assert transpose.capture["status"] == "partial"


def test_nested_capture_contexts_do_not_leak_records():
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)

    with ProvenanceCapture() as outer:
        left + right
        with ProvenanceCapture() as inner:
            left - right
        left * right

    assert [record.operation_id for record in outer.ledger] == [
        _instrument.BINARY_ADD_OPERATION_ID,
        _instrument.BINARY_MULTIPLY_OPERATION_ID,
    ]
    assert [record.operation_id for record in inner.ledger] == [
        _instrument.BINARY_SUBTRACT_OPERATION_ID
    ]


def test_capture_does_not_retain_binary_operands_or_result():
    left = _dataset(offset=1.0)
    right = _dataset(offset=2.0)
    left_ref = weakref.ref(left)
    right_ref = weakref.ref(right)

    with ProvenanceCapture() as capture:
        result = left + right
        result_ref = weakref.ref(result)

    del left, right, result
    gc.collect()
    assert capture._identity_size == 0
    assert left_ref() is None
    assert right_ref() is None
    assert result_ref() is None


def test_pipeline_suppresses_internal_binary_records(monkeypatch):
    def add_self(self, dataset):
        return dataset + dataset

    source = _dataset()
    monkeypatch.setattr(scp.CenterTransformer, "_transform", add_self)
    pipeline = scp.Pipeline([("add-self", scp.CenterTransformer(dim="y"))])

    with ProvenanceCapture() as capture:
        selected = source[:]
        pipeline.fit(selected)
        result = pipeline.transform(selected)
        direct = source + source

    assert result.shape == source.shape
    assert direct.shape == source.shape
    assert [record.operation_id for record in capture.ledger] == [
        _instrument.SLICE_OPERATION_ID,
        _instrument.BINARY_ADD_OPERATION_ID,
    ]
