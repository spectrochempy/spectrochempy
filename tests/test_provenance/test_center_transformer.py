import gc
import weakref

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.provenance import ProvenanceCapture
from spectrochempy.provenance import _instrument
from spectrochempy.utils.exceptions import NotFittedError
from spectrochempy.utils.exceptions import SpectroChemPyError


def _dataset(rows=6):
    data = np.arange(rows * 4.0).reshape(rows, 4)
    return scp.NDDataset(data, dims=["y", "x"])


def _records(capture, operation_id):
    return [
        record
        for record in capture.ledger.operation_records
        if record.operation_id == operation_id
    ]


def test_direct_fit_and_transform_are_opt_in_and_preserve_results_and_history():
    source = _dataset()
    plain = scp.CenterTransformer(dim="y")
    expected = plain.fit(source).transform(source)

    transformer = scp.CenterTransformer(dim="y")
    with ProvenanceCapture() as capture:
        returned = transformer.fit(source)
        result = transformer.transform(source)

    assert returned is transformer
    np.testing.assert_array_equal(result.data, expected.data)
    assert result.history == expected.history
    assert [record.operation_id for record in capture.ledger] == [
        _instrument.CENTER_FIT_OPERATION_ID,
        _instrument.CENTER_TRANSFORM_OPERATION_ID,
    ]


def test_fit_records_stable_transformer_identity_and_bounded_learned_state():
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as capture:
        transformer.fit(source)

    record = capture.ledger.operation_records[0]
    assert record.status == "succeeded"
    assert [link.role for link in record.inputs] == ["transformer", "calibration"]
    assert [link.role for link in record.outputs] == ["fitted_transformer"]
    before = record.inputs[0].reference
    after = record.outputs[0].reference
    assert before.object == after.object
    assert (before.version, after.version) == (0, 1)
    payload = record.to_dict()
    assert payload["parameters"]["requested"]["values"] == {}
    assert payload["parameters"]["resolved"]["values"] == {
        "axis": 0,
        "dim": "y",
        "dimension": "y",
    }
    mean = payload["fitted_state_summary"]["mean"]
    assert mean == {
        "dtype": "float64",
        "masked": 0,
        "reason": "learned_array_values",
        "shape": [1, 4],
        "size": 4,
        "status": "omitted",
    }
    assert record.capture["status"] == "partial"


def test_transform_reuses_fitted_state_and_chains_a_p2_selection():
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as capture:
        calibration = source[:4]
        transformer.fit(calibration)
        validation = source[4:]
        result = transformer.transform(validation)

    (
        slice_calibration,
        fit,
        slice_validation,
        transform,
    ) = capture.ledger.operation_records
    assert fit.inputs[1].reference == slice_calibration.outputs[0].reference
    assert transform.inputs[0].reference == fit.outputs[0].reference
    assert transform.inputs[1].reference == slice_validation.outputs[0].reference
    assert transform.outputs[0].reference.object != transform.inputs[1].reference.object
    np.testing.assert_array_equal(
        result.data,
        validation.data - np.mean(calibration.data, axis=0, keepdims=True),
    )


def test_successful_refit_advances_one_transformer_identity():
    first = _dataset()
    second = scp.NDDataset(first.data + 10.0, dims=first.dims)
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as capture:
        transformer.fit(first)
        transformer.fit(second)

    first_fit, second_fit = capture.ledger.operation_records
    assert second_fit.inputs[0].reference == first_fit.outputs[0].reference
    assert (
        second_fit.outputs[0].reference.object == first_fit.outputs[0].reference.object
    )
    assert second_fit.outputs[0].reference.version == 2


def test_failed_refit_records_known_invalidation_and_does_not_reuse_fitted_state():
    source = _dataset()
    incompatible = scp.NDDataset(np.arange(4.0), dims=["x"])
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as capture:
        transformer.fit(source)
        with pytest.raises(ValueError) as caught:
            transformer.fit(incompatible)
        with pytest.raises(NotFittedError):
            transformer.transform(source)

    assert "y" in str(caught.value)
    successful_fit, failed_fit, failed_transform = capture.ledger.operation_records
    assert failed_fit.status == "failed"
    assert failed_fit.outputs == ()
    assert failed_fit.inputs[0].reference == successful_fit.outputs[0].reference
    assert failed_fit.state_effect == {
        "effect": "changed",
        "reconstructable": False,
    }
    assert (
        failed_transform.inputs[0].reference.object
        == successful_fit.outputs[0].reference.object
    )
    assert failed_transform.inputs[0].reference.version == 2
    assert failed_transform.inputs[0].reference != successful_fit.outputs[0].reference


def test_transform_before_fit_records_failure_without_masking_exception():
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")

    with (
        ProvenanceCapture() as capture,
        pytest.raises(NotFittedError) as caught,
    ):
        transformer.transform(source)

    assert "transform" in str(caught.value)
    record = capture.ledger.operation_records[0]
    assert record.status == "failed"
    assert record.outputs == ()
    assert record.state_effect == {
        "effect": "unchanged",
        "reconstructable": False,
    }


def test_incompatible_transform_failure_preserves_fitted_state_and_exception():
    source = _dataset()
    incompatible = scp.NDDataset(np.arange(18.0).reshape(6, 3), dims=["y", "x"])
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as capture:
        transformer.fit(source)
        with pytest.raises(SpectroChemPyError, match="incompatible"):
            transformer.transform(incompatible)
        result = transformer.transform(source)

    fit, failed, succeeded = capture.ledger.operation_records
    assert result.shape == source.shape
    assert failed.status == "failed"
    assert failed.outputs == ()
    assert failed.inputs[0].reference == fit.outputs[0].reference
    assert failed.state_effect == {
        "effect": "unchanged",
        "reconstructable": False,
    }
    assert succeeded.inputs[0].reference == fit.outputs[0].reference


@pytest.mark.parametrize("mutation", ["configuration", "learned_state"])
def test_unrecorded_transformer_changes_break_verified_continuity(mutation):
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as capture:
        transformer.fit(source)
        if mutation == "configuration":
            transformer.dim = 0
        else:
            transformer.mean_ += 1.0
        transformer.transform(source)

    fit, transform = capture.ledger.operation_records
    assert transform.inputs[0].reference != fit.outputs[0].reference
    assert transform.inputs[0].reference.object == fit.outputs[0].reference.object
    assert transform.capture["status"] == "partial"
    assert any(
        omission["reason"] == "unrecorded_state_change"
        for omission in transform.capture["omissions"]
    )


def test_fit_capture_failure_invalidates_continuity_without_failing_fit(monkeypatch):
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")
    with ProvenanceCapture() as capture:
        original_append = capture.ledger.append
        calls = 0

        def fail_first(record):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("ledger unavailable")
            original_append(record)

        monkeypatch.setattr(capture.ledger, "append", fail_first)
        returned = transformer.fit(source)
        result = transformer.transform(source)

    assert returned is transformer
    assert result.shape == source.shape
    assert capture.capture_warnings[0]["status"] == "capture_failed"
    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.operation_id == _instrument.CENTER_TRANSFORM_OPERATION_ID
    assert record.capture["status"] == "partial"
    assert any(
        omission["reason"] == "unrecorded_state_change"
        for omission in record.capture["omissions"]
    )


def test_fit_boundary_capture_failure_starts_conservative_identity(monkeypatch):
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")
    with ProvenanceCapture() as capture:
        observe_state = capture._observe_state

        def fail_observation(value, fingerprint):
            raise RuntimeError("identity registry unavailable")

        monkeypatch.setattr(capture, "_observe_state", fail_observation)
        returned = transformer.fit(source)
        monkeypatch.setattr(capture, "_observe_state", observe_state)
        result = transformer.transform(source)

    assert returned is transformer
    assert result.shape == source.shape
    assert capture.capture_warnings[0]["status"] == "capture_failed"
    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.inputs[0].reference.version == 1
    assert record.capture["status"] == "partial"
    assert any(
        omission["reason"] == "unrecorded_state_change"
        for omission in record.capture["omissions"]
    )


def test_transform_capture_failure_breaks_later_p2_result_continuity(monkeypatch):
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")
    with ProvenanceCapture() as capture:
        original_append = capture.ledger.append
        calls = 0

        def fail_second(record):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("ledger unavailable")
            original_append(record)

        monkeypatch.setattr(capture.ledger, "append", fail_second)
        transformer.fit(source)
        result = transformer.transform(source)
        transposed = result.transpose()

    assert transposed.shape == source.T.shape
    assert [record.operation_id for record in capture.ledger] == [
        _instrument.CENTER_FIT_OPERATION_ID,
        _instrument.TRANSPOSE_OPERATION_ID,
    ]
    transpose = capture.ledger.operation_records[-1]
    assert transpose.capture["status"] == "partial"
    assert any(
        omission["reason"] == "unrecorded_state_change"
        for omission in transpose.capture["omissions"]
    )


def test_excluded_direct_paths_emit_no_center_records():
    source = _dataset()
    with ProvenanceCapture() as capture:
        fitted = scp.CenterTransformer(dim="y")
        fitted.fit_transform(source)
        fitted.inverse_transform(source)
        scp.center(source, dim="y")

    assert not _records(capture, _instrument.CENTER_FIT_OPERATION_ID)
    assert not _records(capture, _instrument.CENTER_TRANSFORM_OPERATION_ID)


def test_pipeline_suppresses_center_records_without_suppressing_p2():
    source = _dataset()
    pipeline = scp.Pipeline([("center", scp.CenterTransformer(dim="y"))])

    with ProvenanceCapture() as capture:
        selected = source[:4]
        pipeline.fit(selected)
        pipeline.transform(selected)

    assert [record.operation_id for record in capture.ledger] == [
        _instrument.SLICE_OPERATION_ID
    ]


def test_cross_validate_pipeline_emits_no_center_records():
    source = _dataset(rows=8)
    target = scp.NDDataset(
        (1.0 + source.data[:, 0])[:, None],
        dims=["y", "t"],
    )
    pipeline = scp.Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=1)),
        ]
    )

    with ProvenanceCapture() as capture:
        result = scp.cross_validate(pipeline, source, target, cv=2)

    assert result.n_splits == 2
    assert not _records(capture, _instrument.CENTER_FIT_OPERATION_ID)
    assert not _records(capture, _instrument.CENTER_TRANSFORM_OPERATION_ID)
    assert {record.operation_id for record in capture.ledger} == {
        _instrument.SLICE_OPERATION_ID
    }


def test_nested_suppression_and_capture_do_not_leak():
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")

    with ProvenanceCapture() as outer:
        with (
            _instrument.suppress_provenance(_instrument.CENTER_FIT_OPERATION_ID),
            ProvenanceCapture() as inner,
        ):
            transformer.fit(source)
        transformer.fit(source)

    assert len(inner.ledger) == 0
    assert len(outer.ledger) == 1
    assert outer.ledger.operation_records[0].operation_id == (
        _instrument.CENTER_FIT_OPERATION_ID
    )


def test_capture_releases_transformer_datasets_and_learned_arrays():
    source = _dataset()
    transformer = scp.CenterTransformer(dim="y")
    source_ref = weakref.ref(source)
    transformer_ref = weakref.ref(transformer)

    with ProvenanceCapture() as capture:
        transformer.fit(source)
        result = transformer.transform(source)
        result_ref = weakref.ref(result)

    del source, transformer, result
    gc.collect()
    assert capture._identity_size == 0
    assert source_ref() is None
    assert transformer_ref() is None
    assert result_ref() is None
