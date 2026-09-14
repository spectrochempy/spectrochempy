import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from datetime import datetime

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.provenance import ObjectRef
from spectrochempy.provenance import OperationRecord
from spectrochempy.provenance import OperationRef
from spectrochempy.provenance import ProvenanceCapture
from spectrochempy.provenance import ProvenanceLedger
from spectrochempy.provenance import ReferenceLink
from spectrochempy.provenance import ResultRef
from spectrochempy.provenance import StateRef


def _record(index, record_id=None, *, inputs=(), outputs=None, parent_operation=None):
    if outputs is None:
        outputs = (ReferenceLink("result", ResultRef(f"result-{index:06d}")),)
    return OperationRecord(
        id=record_id or f"op-{index:06d}",
        operation_id="org.spectrochempy.test.operation",
        category="transform",
        implementation="spectrochempy.testing.operation",
        provider_name="spectrochempy",
        provider_version="0.13.dev0",
        started_at=datetime(2026, 9, 14, 8, index, tzinfo=UTC),
        inputs=inputs,
        outputs=outputs,
        parent_operation=parent_operation,
    )


def test_capture_is_disabled_by_default():
    assert ProvenanceCapture.current() is None


def test_public_api_is_grouped_in_provenance_namespace():
    assert scp.provenance.ProvenanceCapture is ProvenanceCapture
    assert not hasattr(scp, "ProvenanceCapture")


def test_capture_owns_append_only_ordered_ledger():
    with ProvenanceCapture() as capture:
        capture.ledger.append(_record(1))
        capture.ledger.append(_record(2))
        assert ProvenanceCapture.current() is capture
        assert capture.active

    assert [record.id for record in capture.ledger.operation_records] == [
        "op-000001",
        "op-000002",
    ]
    assert ProvenanceCapture.current() is None
    assert not capture.active


def test_ledger_returns_immutable_snapshots():
    ledger = ProvenanceLedger()
    ledger.append(_record(1))
    snapshot = ledger.operation_records

    assert isinstance(snapshot, tuple)
    snapshot += (_record(2),)
    assert len(snapshot) == 2
    assert len(ledger) == 1


def test_ledger_rejects_duplicate_operation_id_atomically():
    ledger = ProvenanceLedger()
    first = _record(1)
    ledger.append(first)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.append(_record(2, record_id="op-000001"))
    assert ledger.operation_records == (first,)


def test_ledger_allows_consistent_forward_operation_reference():
    ledger = ProvenanceLedger()
    ledger.append(_record(1, parent_operation=OperationRef("op-parent")))
    ledger.append(_record(2, record_id="op-parent"))

    assert [record.id for record in ledger] == ["op-000001", "op-parent"]


def test_ledger_allows_valid_reference_reuse_and_monotonic_progression():
    ledger = ProvenanceLedger()
    object_ref = ObjectRef("object-000001")
    initial = StateRef("state-000001", object_ref, 0)
    changed = StateRef("state-000002", object_ref, 1)
    result = ResultRef("result-000001")

    ledger.append(
        _record(
            1,
            outputs=(
                ReferenceLink("dataset", initial),
                ReferenceLink("diagnostic", result),
            ),
        )
    )
    ledger.append(
        _record(
            2,
            inputs=(
                ReferenceLink("source", initial),
                ReferenceLink("previous_diagnostic", result),
            ),
            outputs=(ReferenceLink("result", changed),),
            parent_operation=OperationRef("op-000001"),
        )
    )
    ledger.append(_record(3, inputs=(ReferenceLink("original", initial),)))

    assert len(ledger) == 3


@pytest.mark.parametrize(
    "conflicting",
    [
        StateRef("state-shared", ObjectRef("object-a"), 1),
        StateRef("state-shared", ObjectRef("object-b"), 0),
    ],
)
def test_ledger_rejects_state_id_with_conflicting_semantics(conflicting):
    ledger = ProvenanceLedger()
    first = StateRef("state-shared", ObjectRef("object-a"), 0)
    ledger.append(_record(1, outputs=(ReferenceLink("result", first),)))

    with pytest.raises(ValueError, match="conflicting state reference"):
        ledger.append(_record(2, inputs=(ReferenceLink("source", conflicting),)))

    assert len(ledger) == 1


def test_ledger_rejects_duplicate_state_identity():
    ledger = ProvenanceLedger()
    object_ref = ObjectRef("object-000001")
    first = StateRef("state-a", object_ref, 0)
    duplicate_identity = StateRef("state-b", object_ref, 0)
    ledger.append(_record(1, outputs=(ReferenceLink("result", first),)))

    with pytest.raises(ValueError, match="already registered as state"):
        ledger.append(_record(2, inputs=(ReferenceLink("source", duplicate_identity),)))

    assert len(ledger) == 1


def test_ledger_rejects_non_monotonic_new_state_version():
    ledger = ProvenanceLedger()
    object_ref = ObjectRef("object-000001")
    later = StateRef("state-later", object_ref, 2)
    regression = StateRef("state-regression", object_ref, 1)
    ledger.append(_record(1, outputs=(ReferenceLink("result", later),)))

    with pytest.raises(ValueError, match="is not greater than 2"):
        ledger.append(_record(2, inputs=(ReferenceLink("source", regression),)))

    assert len(ledger) == 1


@pytest.mark.parametrize("existing_kind", ["object", "state", "result", "operation"])
def test_ledger_rejects_cross_kind_identifier_reuse(existing_kind):
    ledger = ProvenanceLedger()
    object_ref = ObjectRef("object-id")
    state_ref = StateRef("state-id", object_ref, 0)
    result_ref = ResultRef("result-id")
    ledger.append(
        _record(
            1,
            record_id="operation-id",
            outputs=(
                ReferenceLink("dataset", state_ref),
                ReferenceLink("diagnostic", result_ref),
            ),
        )
    )
    identifiers = {
        "object": object_ref.id,
        "state": state_ref.id,
        "result": result_ref.id,
        "operation": "operation-id",
    }
    conflicting_reference = ResultRef(identifiers[existing_kind])
    expected_kind = "result" if existing_kind == "result" else existing_kind
    if existing_kind == "result":
        conflicting_reference = StateRef(
            result_ref.id,
            ObjectRef("object-conflict"),
            0,
        )

    with pytest.raises(ValueError, match=f"already registered as {expected_kind}"):
        ledger.append(
            _record(2, inputs=(ReferenceLink("source", conflicting_reference),))
        )

    assert len(ledger) == 1


def test_failed_append_leaves_all_reference_registries_unchanged():
    ledger = ProvenanceLedger()
    object_ref = ObjectRef("object-000001")
    initial = StateRef("state-initial", object_ref, 0)
    ledger.append(_record(1, outputs=(ReferenceLink("result", initial),)))

    staged_then_rejected = StateRef("state-staged", object_ref, 1)
    conflicting_result = ResultRef(object_ref.id)
    with pytest.raises(ValueError, match="already registered as object"):
        ledger.append(
            _record(
                2,
                outputs=(
                    ReferenceLink("result", staged_then_rejected),
                    ReferenceLink("diagnostic", conflicting_result),
                ),
            )
        )

    replacement = StateRef("state-replacement", object_ref, 1)
    ledger.append(
        _record(
            3,
            record_id="op-000002",
            outputs=(ReferenceLink("result", replacement),),
        )
    )
    assert [record.id for record in ledger] == ["op-000001", "op-000002"]


def test_nested_capture_uses_innermost_and_restores_outer():
    with ProvenanceCapture() as outer:
        assert ProvenanceCapture.current() is outer
        with ProvenanceCapture() as inner:
            assert ProvenanceCapture.current() is inner
            inner.ledger.append(_record(1))
        assert ProvenanceCapture.current() is outer
        outer.ledger.append(_record(1))

    assert len(outer.ledger) == 1
    assert len(inner.ledger) == 1


def test_exception_cleanup_restores_previous_context():
    outer = ProvenanceCapture()
    inner = ProvenanceCapture()
    with outer:
        with pytest.raises(RuntimeError, match="scientific failure"), inner:
            raise RuntimeError("scientific failure")
        assert ProvenanceCapture.current() is outer
    assert ProvenanceCapture.current() is None


def test_same_capture_cannot_be_reentered():
    capture = ProvenanceCapture()
    with capture:
        with pytest.raises(RuntimeError, match="re-entered"), capture:
            pass
        assert ProvenanceCapture.current() is capture


def test_new_thread_has_independent_context():
    with ProvenanceCapture() as capture:
        with ThreadPoolExecutor(max_workers=1) as executor:
            active_in_thread = executor.submit(ProvenanceCapture.current).result()
        assert active_in_thread is None
        assert ProvenanceCapture.current() is capture


def test_concurrent_tasks_have_independent_contexts():
    async def worker(ready, release):
        with ProvenanceCapture() as capture:
            ready.set()
            await release.wait()
            return capture, ProvenanceCapture.current()

    async def run_workers():
        first_ready = asyncio.Event()
        second_ready = asyncio.Event()
        release = asyncio.Event()
        first = asyncio.create_task(worker(first_ready, release))
        second = asyncio.create_task(worker(second_ready, release))
        await first_ready.wait()
        await second_ready.wait()
        release.set()
        return await asyncio.gather(first, second)

    (first_capture, first_current), (second_capture, second_current) = asyncio.run(
        run_workers()
    )
    assert first_current is first_capture
    assert second_current is second_capture
    assert first_capture is not second_capture


def test_p1_does_not_instrument_operations_or_change_legacy_history():
    dataset = scp.NDDataset(np.arange(6.0).reshape(2, 3))
    dataset.history = "seed event"
    expected = scp.center(dataset, dim="x")

    with ProvenanceCapture() as capture:
        actual = scp.center(dataset.copy(), dim="x", inplace=True)

    assert np.allclose(actual.data, expected.data)
    assert [entry.split("> ", 1)[1] for entry in actual.history] == [
        "Seed event",
        "CenterTransformer applied on dimension x",
    ]
    assert capture.ledger.operation_records == ()
