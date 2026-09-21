import gc
import inspect
import weakref

import numpy as np

import spectrochempy as scp
from spectrochempy.provenance import ProvenanceCapture
from spectrochempy.utils.constants import INPLACE


def _dataset():
    return scp.NDDataset(np.arange(24.0).reshape(4, 6))


def test_slicing_outside_capture_is_unchanged():
    source = _dataset()
    result = source[:, 1:3]

    assert ProvenanceCapture.current() is None
    assert isinstance(result, scp.NDDataset)
    assert result.shape == (4, 2)
    assert np.array_equal(result.data, source.data[:, 1:3])
    assert result.history[-1].endswith(
        "Slice extracted: ((slice(None, None, None), slice(1, 3, None)))"
    )


def test_slice_inside_capture_records_source_and_result():
    source = _dataset()
    with ProvenanceCapture() as capture:
        result = source[:, 1:3]

    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.operation_id == "org.spectrochempy.dataset.slice"
    assert record.category == "transform"
    assert record.status == "succeeded"
    assert [link.role for link in record.inputs] == ["source"]
    assert [link.role for link in record.outputs] == ["result"]
    assert record.inputs[0].reference.id != record.outputs[0].reference.id
    assert np.array_equal(result.data, source.data[:, 1:3])

    payload = record.to_dict()
    selection = payload["parameters"]["requested"]["values"]["selection"]
    assert selection == [
        {"type": "slice", "start": None, "stop": None, "step": None},
        {"type": "slice", "start": 1, "stop": 3, "step": None},
    ]


def test_transpose_inside_capture_records_dims():
    source = _dataset()
    with ProvenanceCapture() as capture:
        result = source.transpose(1, 0)

    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.operation_id == "org.spectrochempy.dataset.transpose"
    assert record.to_dict()["parameters"]["requested"]["values"]["dims"] == [1, 0]
    assert np.array_equal(result.data, source.data.T)


def test_chained_operations_share_state_reference():
    source = _dataset()
    with ProvenanceCapture() as capture:
        result = source[:, 1:3].transpose()

    first, second = capture.ledger.operation_records
    assert second.inputs[0].reference.id == first.outputs[0].reference.id
    assert second.outputs[0].reference.id != first.outputs[0].reference.id
    assert np.array_equal(result.data, source.data[:, 1:3].T)


def test_verified_chain_reports_complete_capture():
    source = _dataset()
    with ProvenanceCapture() as capture:
        result = source[:, 1:3].transpose()

    first, second = capture.ledger.operation_records
    assert first.capture["status"] == "complete"
    assert second.capture["status"] == "complete"
    assert second.inputs[0].reference.id == first.outputs[0].reference.id
    assert np.array_equal(result.data, source.data[:, 1:3].T)


def test_unrecorded_mutation_breaks_false_chaining():
    source = _dataset()
    with ProvenanceCapture() as capture:
        selection = source[:, 1:3]
        selection.data[:] = 0.0
        result = selection.transpose()

    slice_record, transpose_record = capture.ledger.operation_records
    assert slice_record.capture["status"] == "complete"
    assert transpose_record.capture["status"] == "partial"
    assert (
        transpose_record.inputs[0].reference.id != slice_record.outputs[0].reference.id
    )
    assert any(
        "unrecorded_state_change" in str(item)
        for item in transpose_record.capture["omissions"]
    )
    assert result.shape == (2, 4)
    assert result.data.sum() == 0.0
    assert source.data[:, 1:3].sum() != 0.0


def test_reused_source_with_unrecorded_change_is_detected():
    source = _dataset()
    with ProvenanceCapture() as capture:
        source[:, 0:1]
        source.data[:, 0] = 0.0
        source[:, 2:3]

    first, second = capture.ledger.operation_records
    assert first.capture["status"] == "complete"
    assert second.capture["status"] == "partial"
    assert first.inputs[0].reference.id != second.inputs[0].reference.id


def test_repeated_source_reuses_state_reference():
    source = _dataset()
    with ProvenanceCapture() as capture:
        source[:, 0:1]
        source[:, 2:3]

    first, second = capture.ledger.operation_records
    assert first.inputs[0].reference.id == second.inputs[0].reference.id
    assert first.outputs[0].reference.id != second.outputs[0].reference.id


def test_inplace_transpose_is_not_recorded():
    source = _dataset()
    original_id = id(source)
    with ProvenanceCapture() as capture:
        returned = source.transpose(1, 0, inplace=True)

    assert id(returned) == original_id
    assert len(capture.ledger) == 0


def test_inplace_slicing_is_not_recorded():
    source = _dataset()
    original_id = id(source)
    with ProvenanceCapture() as capture:
        returned = source[:, 1:3, INPLACE]

    assert id(returned) == original_id
    assert returned.shape == (4, 2)
    assert np.array_equal(returned.data, np.arange(24.0).reshape(4, 6)[:, 1:3])
    assert len(capture.ledger) == 0


def test_inplace_slice_followed_by_transpose_is_single_record():
    source = _dataset()
    with ProvenanceCapture() as capture:
        source[:, 1:3, INPLACE]
        result = source.transpose()

    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.operation_id == "org.spectrochempy.dataset.transpose"
    assert record.status == "succeeded"
    assert record.capture["status"] == "complete"
    assert result.shape == (2, 4)
    assert np.array_equal(result.data, source.data.T)
    assert result.history[-1].endswith("Data transposed")


def test_location_slicing_records_requested_coordinate_selection():
    source = scp.NDDataset(np.arange(24.0).reshape(4, 6))
    source.set_coordset(x=np.linspace(0, 1, 6), y=np.arange(4.0))
    with ProvenanceCapture() as capture:
        result = source[:, 0.2:0.6]

    assert result.shape == (4, 3)
    assert np.array_equal(result.data, source.data[:, 1:4])
    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    selection = record.to_dict()["parameters"]["requested"]["values"]["selection"]
    assert selection == [
        {"type": "slice", "start": None, "stop": None, "step": None},
        {"type": "slice", "start": 0.2, "stop": 0.6, "step": None},
    ]
    assert record.capture["status"] == "complete"


def test_coordinate_name_access_is_not_recorded():
    source = scp.NDDataset(np.arange(24.0).reshape(4, 6))
    source.set_coordset(x=np.linspace(0, 1, 6), y=np.arange(4.0))
    with ProvenanceCapture() as capture:
        coord = source["x"]

    assert isinstance(coord, scp.Coord)
    assert len(capture.ledger) == 0


def test_fancy_indexing_array_is_recorded_as_omission_when_large():
    source = scp.NDDataset(np.arange(10000.0).reshape(100, 100))
    fancy = np.arange(70)
    with ProvenanceCapture() as capture:
        result = source[:, fancy]

    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    payload = record.to_dict()
    assert "array_size_limit" in str(payload["parameters"]["requested"]["values"])
    assert record.capture["status"] == "partial"
    assert np.array_equal(result.data, source.data[:, fancy])


def test_failed_transpose_records_failure_and_preserves_exception():
    source = _dataset()
    with ProvenanceCapture() as capture:
        try:
            source.transpose(5, 6)
        except Exception as exc:  # noqa: BLE001
            assert type(exc).__name__ == "AxisError"

    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.status == "failed"
    assert record.outputs == ()
    assert record.capture["status"] == "complete"
    payload = record.to_dict()
    assert payload["warnings"][0]["reason"] == "operation_failed"
    assert payload["warnings"][0]["exception_type"].endswith("AxisError")
    assert payload["parameters"]["requested"]["values"]["dims"] == [5, 6]


def test_capture_failure_does_not_alter_result(monkeypatch):
    source = _dataset()

    def _boom(record):
        raise RuntimeError("ledger unavailable")

    with ProvenanceCapture() as capture:
        monkeypatch.setattr(capture.ledger, "append", _boom)
        result = source[:, 1:3]

    assert len(capture.ledger) == 0
    assert np.array_equal(result.data, source.data[:, 1:3])
    assert capture.capture_warnings
    assert capture.capture_warnings[0]["status"] == "capture_failed"


def test_parameter_description_failure_does_not_break_slicing(monkeypatch):
    from spectrochempy.provenance import _instrument

    def _boom(items):
        raise RuntimeError("selection not serializable")

    monkeypatch.setattr(_instrument, "describe_selection", _boom)

    source = _dataset()
    with ProvenanceCapture() as capture:
        result = source[:, 1:3]

    assert len(capture.ledger) == 1
    assert np.array_equal(result.data, source.data[:, 1:3])
    assert capture.capture_warnings
    assert capture.capture_warnings[0]["status"] == "capture_failed"
    record = capture.ledger.operation_records[0]
    assert record.to_dict()["parameters"]["requested"]["values"]["status"] == "omitted"
    assert record.capture["status"] == "partial"


def test_transpose_history_is_preserved_inside_capture():
    source = _dataset()
    outside = _dataset().transpose(1, 0).history[-1]
    with ProvenanceCapture() as capture:
        result = source.transpose(1, 0)

    assert len(capture.ledger) == 1
    assert result.history[-1].endswith("Data transposed between dims: (1, 0)")
    assert result.history[-1].split("> ")[1:] == outside.split("> ")[1:]


def test_data_isolation_between_source_and_result():
    source = _dataset()
    with ProvenanceCapture() as capture:
        result = source[:, 1:3]
        result.data[:] = 0.0

    assert len(capture.ledger) == 1
    assert source.data[:, 1:3].sum() != 0.0


def _apply_viewer_region_then_transpose(data, x_bounds, y_bounds):
    selection = data[slice(*x_bounds), slice(*y_bounds)]
    return selection.transpose()


def test_interactive_region_selection_then_transpose_is_traced():
    source = _dataset()
    x_bounds, y_bounds = (1, 3), (0, 2)
    source_code = inspect.getsource(_apply_viewer_region_then_transpose)
    assert "(1, 3)" not in source_code
    assert "(0, 2)" not in source_code

    with ProvenanceCapture() as capture:
        result = _apply_viewer_region_then_transpose(source, x_bounds, y_bounds)

    slice_record, transpose_record = capture.ledger.operation_records
    selection = slice_record.to_dict()["parameters"]["requested"]["values"]["selection"]
    assert selection == [
        {"type": "slice", "start": 1, "stop": 3, "step": None},
        {"type": "slice", "start": 0, "stop": 2, "step": None},
    ]
    assert (
        transpose_record.inputs[0].reference.id == slice_record.outputs[0].reference.id
    )
    assert result.shape == (2, 2)
    assert np.array_equal(result.data, source.data[1:3, 0:2].T)


def test_capture_does_not_retain_live_objects():
    source = _dataset()
    source_ref = weakref.ref(source)

    with ProvenanceCapture() as capture:
        result = source[:, 1:3]
        assert len(capture.ledger) == 1
        assert capture._identity_size == 2

    assert capture._identity_size == 0
    del source, result
    gc.collect()
    assert source_ref() is None
