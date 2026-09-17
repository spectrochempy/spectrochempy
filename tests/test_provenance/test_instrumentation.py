import gc
import weakref

import numpy as np

import spectrochempy as scp
from spectrochempy.provenance import ProvenanceCapture


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
    payload = record.to_dict()
    assert payload["warnings"][0]["reason"] == "operation_failed"
    assert payload["warnings"][0]["exception_type"].endswith("AxisError")


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
