import gc
import weakref

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.provenance import ProvenanceCapture
from spectrochempy.provenance import _instrument
from spectrochempy.utils.exceptions import NotFittedError


def _dataset(rows=12, columns=5, *, offset=0.0):
    observations = np.linspace(-1.0, 1.0, rows)
    features = np.linspace(0.2, 1.2, columns)
    data = (
        np.outer(observations, features)
        + 0.4 * np.outer(np.sin(2.3 * observations), features**2)
        + 0.05 * np.arange(rows * columns).reshape(rows, columns)
        + offset
    )
    y = scp.Coord.arange(rows, title="sample")
    x = scp.Coord.arange(columns, title="feature")
    dataset = scp.NDDataset(
        data,
        coordset=[y, x],
        dims=["y", "x"],
        units="absorbance",
        title="PCA source",
    )
    dataset.meta.experiment = "p6"
    return dataset


def _records(capture, operation_id):
    return [
        record
        for record in capture.ledger.operation_records
        if record.operation_id == operation_id
    ]


def _parameter_values(record, kind):
    return record.to_dict()["parameters"][kind]["values"]


def test_direct_fit_transform_preserve_scores_metadata_and_history():
    source = _dataset()
    plain = scp.PCA(n_components=3, svd_solver="full")
    expected = plain.fit(source).transform(source, n_components=2)

    estimator = scp.PCA(n_components=3, svd_solver="full")
    with ProvenanceCapture() as capture:
        returned = estimator.fit(source)
        result = estimator.transform(source, n_components=2)

    assert returned is estimator
    np.testing.assert_allclose(np.abs(result.data), np.abs(expected.data), atol=1.0e-12)
    np.testing.assert_array_equal(result.mask, expected.mask)
    assert result.dims == expected.dims
    assert result.coordset == expected.coordset
    assert result.units == expected.units
    assert result.title == expected.title
    assert [entry.split("> ", 1)[-1] for entry in result.history] == [
        entry.split("> ", 1)[-1] for entry in expected.history
    ]
    assert [record.operation_id for record in capture.ledger] == [
        _instrument.PCA_FIT_OPERATION_ID,
        _instrument.PCA_TRANSFORM_OPERATION_ID,
    ]


def test_fit_and_transform_link_one_estimator_identity_and_describe_state():
    source = _dataset()
    estimator = scp.PCA(n_components=3, svd_solver="auto", whiten=True)

    with ProvenanceCapture() as capture:
        estimator.fit(source)
        result = estimator.transform(source, n_components=2)

    fit, transform = capture.ledger.operation_records
    assert [link.role for link in fit.inputs] == ["estimator", "calibration"]
    assert [link.role for link in fit.outputs] == ["fitted_estimator"]
    assert [link.role for link in transform.inputs] == ["estimator", "source"]
    assert [link.role for link in transform.outputs] == ["result"]
    assert fit.inputs[0].reference.object == fit.outputs[0].reference.object
    assert (fit.inputs[0].reference.version, fit.outputs[0].reference.version) == (
        0,
        1,
    )
    assert transform.inputs[0].reference == fit.outputs[0].reference
    assert transform.inputs[1].reference == fit.inputs[1].reference
    assert transform.outputs[0].reference.object != transform.inputs[1].reference.object

    requested_fit = _parameter_values(fit, "requested")
    assert requested_fit["call"] == {}
    assert requested_fit["configuration"]["n_components"] == 3
    assert requested_fit["configuration"]["svd_solver"] == "auto"
    resolved_fit = _parameter_values(fit, "resolved")
    assert resolved_fit["configuration"]["n_components"] == 3
    assert resolved_fit["configuration"]["svd_solver"] == "full"
    assert resolved_fit["input"] == {"dims": ["y", "x"], "shape": [12, 5]}

    learned = fit.to_dict()["fitted_state_summary"]
    assert learned["n_components"] == 3
    assert learned["n_observations"] == 12
    assert learned["n_features"] == 5
    assert learned["components"] == {
        "dtype": "float64",
        "reason": "learned_array_values",
        "shape": [3, 5],
        "size": 15,
        "status": "omitted",
    }
    assert learned["backend"]["reason"] == "fitted_model_not_serialized"
    assert fit.capture["status"] == "partial"

    requested_transform = _parameter_values(transform, "requested")
    assert requested_transform["call"] == {"n_components": 2}
    resolved_transform = _parameter_values(transform, "resolved")
    assert resolved_transform["call"] == {"n_components": 2}
    assert resolved_transform["result"] == {
        "dims": ["y", "k"],
        "shape": list(result.shape),
    }
    assert transform.capture["status"] == "complete"


def test_p2_p5_p4_p3_p6_chain_links_direct_scientific_objects():
    source = _dataset(rows=14)
    center = scp.CenterTransformer(dim="y")
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        selected = source[:7]
        center.fit(selected)
        centered = center.transform(selected)
        assembled = scp.concatenate(centered, centered, dims="y")
        combined = assembled + assembled
        estimator.fit(combined)
        scores = estimator.transform(combined)
        transposed = scores.transpose()

    records = capture.ledger.operation_records
    operation_ids = [record.operation_id for record in records]
    assert operation_ids == [
        _instrument.SLICE_OPERATION_ID,
        _instrument.CENTER_FIT_OPERATION_ID,
        _instrument.CENTER_TRANSFORM_OPERATION_ID,
        _instrument.CONCATENATE_OPERATION_ID,
        _instrument.BINARY_ADD_OPERATION_ID,
        _instrument.PCA_FIT_OPERATION_ID,
        _instrument.PCA_TRANSFORM_OPERATION_ID,
        _instrument.TRANSPOSE_OPERATION_ID,
    ]
    assembly, binary, fit, transform, transpose = records[3:]
    assert binary.inputs[0].reference == assembly.outputs[0].reference
    assert binary.inputs[1].reference == assembly.outputs[0].reference
    assert fit.inputs[1].reference == binary.outputs[0].reference
    assert transform.inputs[0].reference == fit.outputs[0].reference
    assert transform.inputs[1].reference == binary.outputs[0].reference
    assert transpose.inputs[0].reference == transform.outputs[0].reference
    assert transposed.shape == (2, 14)


def test_successful_refit_advances_one_estimator_identity():
    first = _dataset()
    second = _dataset(offset=3.0)
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        estimator.fit(first)
        estimator.fit(second)

    first_fit, second_fit = capture.ledger.operation_records
    assert second_fit.inputs[0].reference == first_fit.outputs[0].reference
    assert (
        second_fit.outputs[0].reference.object == first_fit.outputs[0].reference.object
    )
    assert second_fit.outputs[0].reference.version == 2


def test_failed_initial_fit_records_unchanged_unfitted_state():
    incompatible = _dataset(rows=1)
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture, pytest.raises(ValueError) as caught:
        estimator.fit(incompatible)

    assert "n_components=2" in str(caught.value)
    record = capture.ledger.operation_records[0]
    assert record.status == "failed"
    assert record.outputs == ()
    assert record.state_effect == {"effect": "unchanged", "reconstructable": False}


def test_failed_refit_invalidates_fitted_state_and_does_not_reuse_it():
    source = _dataset()
    incompatible = _dataset(rows=1)
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        estimator.fit(source)
        with pytest.raises(ValueError):
            estimator.fit(incompatible)
        with pytest.raises(NotFittedError):
            estimator.transform(source)

    successful_fit, failed_fit, failed_transform = capture.ledger.operation_records
    assert failed_fit.status == "failed"
    assert failed_fit.outputs == ()
    assert failed_fit.inputs[0].reference == successful_fit.outputs[0].reference
    assert failed_fit.state_effect == {"effect": "changed", "reconstructable": False}
    assert failed_transform.inputs[0].reference.object == (
        successful_fit.outputs[0].reference.object
    )
    assert failed_transform.inputs[0].reference.version == 2
    assert failed_transform.inputs[0].reference != successful_fit.outputs[0].reference


def test_transform_before_fit_records_failure_without_masking_exception():
    source = _dataset()
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture, pytest.raises(NotFittedError):
        estimator.transform(source)

    record = capture.ledger.operation_records[0]
    assert record.status == "failed"
    assert record.outputs == ()
    assert record.state_effect == {"effect": "unchanged", "reconstructable": False}


def test_incompatible_transform_preserves_fitted_state_and_exception():
    source = _dataset()
    incompatible = _dataset(columns=4)
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        estimator.fit(source)
        with pytest.raises(ValueError) as caught:
            estimator.transform(incompatible)
        result = estimator.transform(source)

    assert "features" in str(caught.value)
    fit, failed, succeeded = capture.ledger.operation_records
    assert result.shape == (12, 2)
    assert failed.status == "failed"
    assert failed.outputs == ()
    assert failed.inputs[0].reference == fit.outputs[0].reference
    assert failed.state_effect == {"effect": "unchanged", "reconstructable": False}
    assert succeeded.inputs[0].reference == fit.outputs[0].reference


@pytest.mark.parametrize("mutation", ["configuration", "learned_state"])
def test_unrecorded_estimator_changes_break_verified_continuity(mutation):
    source = _dataset()
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        estimator.fit(source)
        if mutation == "configuration":
            estimator.whiten = True
        else:
            estimator._pca.components_[0, 0] += 0.5
        estimator.transform(source)

    fit, transform = capture.ledger.operation_records
    assert transform.inputs[0].reference != fit.outputs[0].reference
    assert transform.inputs[0].reference.object == fit.outputs[0].reference.object
    assert transform.capture["status"] == "partial"
    assert any(
        omission["reason"] == "unrecorded_state_change"
        for omission in transform.capture["omissions"]
    )
    if mutation == "configuration":
        resolved = _parameter_values(transform, "resolved")
        assert resolved["configuration"]["whiten"] is False


def test_unrecorded_backend_whiten_change_breaks_verified_continuity():
    source = _dataset()
    control = scp.PCA(n_components=2, svd_solver="full")
    expected = control.fit(source).transform(source)
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        estimator.fit(source)
        estimator._pca.whiten = True
        result = estimator.transform(source)

    fit, transform = capture.ledger.operation_records
    assert not np.allclose(np.abs(result.data), np.abs(expected.data))
    assert transform.inputs[0].reference != fit.outputs[0].reference
    assert transform.inputs[0].reference.object == fit.outputs[0].reference.object
    assert transform.capture["status"] == "partial"
    assert any(
        omission["reason"] == "unrecorded_state_change"
        for omission in transform.capture["omissions"]
    )
    resolved = _parameter_values(transform, "resolved")
    assert resolved["configuration"]["whiten"] is True


def test_set_params_invalidation_is_not_reported_as_old_fitted_state():
    source = _dataset()
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        estimator.fit(source)
        estimator.set_params(n_components=1)
        with pytest.raises(NotFittedError):
            estimator.transform(source)

    fit, transform = capture.ledger.operation_records
    assert transform.status == "failed"
    assert transform.inputs[0].reference != fit.outputs[0].reference
    assert transform.inputs[0].reference.object == fit.outputs[0].reference.object
    assert transform.capture["status"] == "partial"


def test_fit_boundary_failure_starts_conservative_estimator_identity(monkeypatch):
    source = _dataset()
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        observe_state = capture._observe_state

        def fail_observation(value, fingerprint):
            raise RuntimeError("identity registry unavailable")

        monkeypatch.setattr(capture, "_observe_state", fail_observation)
        returned = estimator.fit(source)
        monkeypatch.setattr(capture, "_observe_state", observe_state)
        result = estimator.transform(source)

    assert returned is estimator
    assert result.shape == (12, 2)
    assert len(capture.ledger) == 1
    record = capture.ledger.operation_records[0]
    assert record.operation_id == _instrument.PCA_TRANSFORM_OPERATION_ID
    assert record.inputs[0].reference.version == 1
    assert record.capture["status"] == "partial"


def test_transform_write_failure_invalidates_result_for_following_p2(monkeypatch):
    source = _dataset()
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        append = capture.ledger.append
        calls = 0

        def fail_second(record):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("ledger unavailable")
            append(record)

        monkeypatch.setattr(capture.ledger, "append", fail_second)
        estimator.fit(source)
        result = estimator.transform(source)
        result.transpose()

    assert [record.operation_id for record in capture.ledger] == [
        _instrument.PCA_FIT_OPERATION_ID,
        _instrument.TRANSPOSE_OPERATION_ID,
    ]
    transpose = capture.ledger.operation_records[-1]
    assert transpose.capture["status"] == "partial"
    assert any(
        omission["reason"] == "unrecorded_state_change"
        for omission in transpose.capture["omissions"]
    )


def test_excluded_paths_emit_no_pca_records():
    source = _dataset()
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        estimator.fit_transform(source)
        _ = estimator.scores, estimator.loadings
        estimator.inverse_transform()
        array_estimator = scp.PCA(n_components=2, svd_solver="full")
        array_estimator.fit(source.data)
        array_estimator.transform(source.data)

    assert not _records(capture, _instrument.PCA_FIT_OPERATION_ID)
    assert not _records(capture, _instrument.PCA_TRANSFORM_OPERATION_ID)


def test_pca_subclass_is_not_implicitly_instrumented():
    class ExtendedPCA(scp.PCA):
        def _fit(self, X, Y=None):
            result = super()._fit(X, Y)
            self.extra_learned_ = np.sum(X)
            return result

    source = _dataset()
    estimator = ExtendedPCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as capture:
        returned = estimator.fit(source)
        result = estimator.transform(source)

    assert returned is estimator
    assert result.shape == (12, 2)
    assert len(capture.ledger) == 0


def test_pipeline_suppresses_pca_records_without_suppressing_p2():
    source = _dataset()
    pipeline = scp.Pipeline([("pca", scp.PCA(n_components=2, svd_solver="full"))])

    with ProvenanceCapture() as capture:
        selected = source[:10]
        pipeline.fit(selected)
        result = pipeline.transform(selected)

    assert result.shape == (10, 2)
    assert [record.operation_id for record in capture.ledger] == [
        _instrument.SLICE_OPERATION_ID
    ]


def test_nested_captures_and_fit_transform_suppression_do_not_leak():
    source = _dataset()
    estimator = scp.PCA(n_components=2, svd_solver="full")

    with ProvenanceCapture() as outer:
        estimator.fit_transform(source)
        with ProvenanceCapture() as inner:
            estimator.fit(source)
            estimator.transform(source)
        estimator.transform(source)

    assert [record.operation_id for record in inner.ledger] == [
        _instrument.PCA_FIT_OPERATION_ID,
        _instrument.PCA_TRANSFORM_OPERATION_ID,
    ]
    assert [record.operation_id for record in outer.ledger] == [
        _instrument.PCA_TRANSFORM_OPERATION_ID
    ]


def test_capture_releases_pca_datasets_result_and_backend_arrays():
    source = _dataset()
    estimator = scp.PCA(n_components=2, svd_solver="full")
    source_ref = weakref.ref(source)
    estimator_ref = weakref.ref(estimator)

    with ProvenanceCapture() as capture:
        estimator.fit(source)
        result = estimator.transform(source)
        result_ref = weakref.ref(result)
        components_ref = weakref.ref(estimator._pca.components_)

    del source, estimator, result
    gc.collect()
    assert capture._identity_size == 0
    assert source_ref() is None
    assert estimator_ref() is None
    assert result_ref() is None
    assert components_ref() is None
