# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Contract tests for the internal structured cross-validation result prototype."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

import spectrochempy as scp
from spectrochempy.analysis._cross_validation import _build_cross_validation_result
from spectrochempy.utils.exceptions import SpectroChemPyError


@pytest.fixture()
def validation_data():
    """Return aligned spectra, targets, and deterministic OOF predictions."""
    samples = scp.Coord(
        np.arange(5),
        labels=[f"sample-{index}" for index in range(5)],
        title="sample",
    )
    features = scp.Coord.linspace(1000.0, 1300.0, 4, units="cm^-1")
    targets = scp.Coord(
        [10, 20], labels=["concentration", "temperature"], title="target"
    )
    X = scp.NDDataset(
        np.arange(20.0).reshape(5, 4),
        coordset=[samples, features],
        dims=["y", "x"],
        units="absorbance",
        name="spectra",
        title="input spectra",
    )
    observed = scp.NDDataset(
        np.column_stack((np.arange(5.0), 2.0 * np.arange(5.0))),
        coordset=[samples.copy(), targets],
        dims=["y", "x"],
        units="mol/L",
        name="reference",
        title="observed targets",
    )
    predicted = observed.copy()
    predicted.name = "oof"
    predicted.title = "OOF predictions"
    predicted.data[-1, 0] += 3.0
    return X, observed, predicted


def _kfold_inputs(n_observations=5):
    splitter = KFold(n_splits=3)
    # The supplied execution order deliberately differs from observation order.
    folds = list(splitter.split(np.arange(n_observations)))[::-1]
    return splitter, folds


def _build(validation_data, **overrides):
    X, observed, predicted = validation_data
    splitter, folds = _kfold_inputs()
    arguments = {
        "estimator": scp.PLSRegression(n_components=1, scale=False),
        "splitter": splitter,
        "X": X,
        "observed": observed,
        "oof_predictions": predicted,
        "folds": folds,
    }
    arguments.update(overrides)
    return _build_cross_validation_result(**arguments)


def test_multivariate_result_preserves_identity_and_uses_global_oof_rmsecv(
    validation_data,
):
    X, observed, predicted = validation_data

    result = _build(
        validation_data,
        X=X.T,
        observed=observed.T,
        oof_predictions=predicted.T,
        warnings=["synthetic warning"],
    )

    assert (result.x_sample_axis, result.y_sample_axis) == (1, 1)
    assert result.n_splits == 3
    assert [fold.validation_positions.tolist() for fold in result.folds] == [
        [4],
        [2, 3],
        [0, 1],
    ]
    assert_allclose(result.metric("rmsecv").values.data, [np.sqrt(9.0 / 5.0), 0.0])
    assert_allclose(result.metric("bias").values.data, [0.6, 0.0])
    assert_allclose(result.metric("mae").values.data, [0.6, 0.0])
    assert_allclose(result.metric("r2").values.data, [0.1, 1.0])
    assert_allclose(result.residuals.data[:, -1], [-3.0, 0.0])
    assert result.observed.coordset == observed.T.coordset
    assert result.oof_predictions.units == observed.units
    assert result.metric("rmsecv").values.coordset["x"] == observed.coordset["x"]
    assert result.metric("rmsecv").values.units == observed.units
    assert result.metric("r2").values.units is None
    assert result.warnings == ("synthetic warning",)

    # RMSE is computed once over complete OOF predictions, not averaged by fold.
    mean_fold_rmse = np.mean(
        [fold.metric("rmse").values.data[0] for fold in result.folds]
    )
    assert mean_fold_rmse == pytest.approx(1.0)
    assert result.metric("rmsecv").values.data[0] != pytest.approx(mean_fold_rmse)


def test_univariate_result_uses_scalar_metric_outputs(validation_data):
    X, observed, predicted = validation_data
    observed = observed[:, 0].squeeze()
    predicted = predicted[:, 0].squeeze()

    result = _build(
        validation_data,
        observed=observed,
        oof_predictions=predicted,
    )

    assert result.metric("rmsecv").values.ndim == 0
    assert result.n_valid.ndim == 0
    assert result.residuals.dims == observed.dims
    assert result.observation_coordinate == observed.coordset["y"]


def test_invalid_prediction_is_explicit_per_target_and_preserves_mask(
    validation_data,
):
    _, _, predicted = validation_data
    mask = np.zeros(predicted.shape, dtype=bool)
    mask[1, 0] = True
    predicted.mask = mask

    result = _build(validation_data, oof_predictions=predicted)

    assert_allclose(result.n_valid.data, [4, 5])
    assert result.residuals.mask[1, 0]
    for name in ("rmsecv", "r2", "bias", "mae"):
        metric = result.metric(name)
        assert metric.defined == (False, True)
        assert metric.reasons == ("invalid_prediction", None)
        assert np.isnan(metric.values.data[0])
    assert {
        (record.scope, record.metric, record.target_index, record.reason)
        for record in result.undefined_metrics
        if record.scope == "global"
    } == {
        ("global", name, 0, "invalid_prediction")
        for name in ("rmsecv", "r2", "bias", "mae")
    }


def test_leave_one_out_records_undefined_fold_r2_but_defined_global_r2(
    validation_data,
):
    X, observed, _ = validation_data
    predicted = observed.copy()
    splitter = LeaveOneOut()
    folds = list(splitter.split(np.arange(5)))

    result = _build(
        validation_data,
        splitter=splitter,
        folds=folds,
        oof_predictions=predicted,
    )

    assert result.metric("r2").defined == (True, True)
    assert all(fold.metric("r2").defined == (False, False) for fold in result.folds)
    fold_r2_reasons = [
        record
        for record in result.undefined_metrics
        if record.scope == "fold" and record.metric == "r2"
    ]
    assert len(fold_r2_reasons) == 10
    assert {record.reason for record in fold_r2_reasons} == {"insufficient_valid_pairs"}


@pytest.mark.parametrize(
    ("folds", "message"),
    [
        (
            [
                (np.array([0, 1, 2]), np.array([3, 4])),
                (np.array([0, 1, 3, 4]), np.array([2])),
                (np.array([1, 2, 3, 4]), np.array([0])),
            ],
            "cover every observation",
        ),
        (
            [
                (np.array([0, 1, 2]), np.array([3, 4])),
                (np.array([0, 3, 4]), np.array([1, 2])),
                (np.array([1, 3, 4]), np.array([0, 2])),
            ],
            "repeat",
        ),
        (
            [
                (np.array([0, 1, 2]), np.array([2, 4])),
                (np.array([0, 1, 3, 4]), np.array([2])),
                (np.array([1, 2, 3, 4]), np.array([0])),
            ],
            "overlap",
        ),
    ],
)
def test_fold_coverage_and_disjointness_are_enforced(validation_data, folds, message):
    with pytest.raises(SpectroChemPyError, match=message):
        _build(validation_data, folds=folds)


def test_splitter_fold_count_and_prediction_geometry_are_enforced(validation_data):
    _, _, predicted = validation_data
    _, folds = _kfold_inputs()

    with pytest.raises(SpectroChemPyError, match="expects 3 folds"):
        _build(validation_data, folds=folds[:2])
    with pytest.raises(SpectroChemPyError, match="same shape"):
        _build(validation_data, oof_predictions=predicted[:, :1])
    with pytest.raises(SpectroChemPyError, match="coordinate"):
        _build(validation_data, oof_predictions=predicted[[1, 0, 2, 3, 4]])


def test_units_and_out_of_bounds_fold_positions_are_rejected(validation_data):
    _, _, predicted = validation_data
    incompatible = predicted.copy()
    incompatible.ito("s", force=True)
    with pytest.raises(SpectroChemPyError, match="not compatible"):
        _build(validation_data, oof_predictions=incompatible)

    _, folds = _kfold_inputs()
    folds[0] = (folds[0][0], np.array([5]))
    with pytest.raises(SpectroChemPyError, match="outside the dataset bounds"):
        _build(validation_data, folds=folds)


def test_group_summary_is_compact_and_group_usage_is_bounded(validation_data):
    X, observed, predicted = validation_data
    groups = np.array(["batch-a", "batch-a", "batch-b", "batch-c", "batch-c"])
    splitter = GroupKFold(n_splits=3)
    folds = list(splitter.split(np.arange(5), groups=groups))

    result = _build(
        validation_data,
        splitter=splitter,
        folds=folds,
        groups=groups,
    )

    assert result.groups.n_groups == 3
    assert [(item.value, item.count) for item in result.groups.counts] == [
        ("batch-a", 2),
        ("batch-b", 1),
        ("batch-c", 2),
    ]
    groups[0] = "changed"
    assert result.groups.counts[0].value == "batch-a"

    kfold, kfolds = _kfold_inputs()
    with pytest.raises(SpectroChemPyError, match="non-group splitter"):
        _build(
            validation_data,
            splitter=kfold,
            folds=kfolds,
            groups=np.arange(5),
        )
    with pytest.raises(SpectroChemPyError, match="requires groups"):
        _build(
            validation_data,
            splitter=GroupKFold(n_splits=3),
            folds=folds,
        )


def test_groupkfold_rejects_folds_that_leak_groups(validation_data):
    groups = np.array([0, 1, 0, 1, 2])
    splitter = GroupKFold(n_splits=3)
    leaking_folds = list(KFold(n_splits=3).split(np.arange(5)))

    with pytest.raises(
        SpectroChemPyError, match="group in both training and validation"
    ):
        _build(
            validation_data,
            splitter=splitter,
            folds=leaking_folds,
            groups=groups,
        )


def test_masked_numpy_groups_are_rejected(validation_data):
    group_values = np.array([0, 0, 1, 2, 2])
    groups = np.ma.array(group_values, mask=[False, True, False, False, False])
    splitter = GroupKFold(n_splits=3)
    folds = list(splitter.split(np.arange(5), groups=group_values))

    with pytest.raises(SpectroChemPyError, match="groups must not contain masked"):
        _build(
            validation_data,
            splitter=splitter,
            folds=folds,
            groups=groups,
        )


def test_result_snapshots_inputs_positions_configuration_and_containers(
    validation_data,
):
    X, observed, predicted = validation_data
    splitter, folds = _kfold_inputs()
    estimator = scp.PLSRegression(n_components=1, scale=False)
    warning_messages = ["before"]

    result = _build_cross_validation_result(
        estimator=estimator,
        splitter=splitter,
        X=X,
        observed=observed,
        oof_predictions=predicted,
        folds=folds,
        warnings=warning_messages,
    )
    observed.data[0, 0] = 999.0
    predicted.data[4, 0] = 999.0
    folds[0][0][0] = 4
    warning_messages[0] = "after"
    estimator.n_components = 2
    splitter.n_splits = 2

    assert result.observed.data[0, 0] == 0.0
    assert result.oof_predictions.data[4, 0] == 7.0
    assert result.folds[0].train_positions.tolist() == [0, 1, 2, 3]
    assert not result.folds[0].train_positions.flags.writeable
    assert result.warnings == ("before",)
    assert result.estimator.parameters["n_components"] == 1
    assert result.splitter.parameters["n_splits"] == 3
    with pytest.raises(TypeError):
        result.estimator.parameters["n_components"] = 2


def test_pipeline_is_recorded_as_step_configuration_without_estimator_objects(
    validation_data,
):
    pipeline = scp.Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=1)),
        ]
    )

    result = _build(validation_data, estimator=pipeline)

    assert result.estimator.class_name.endswith(".Pipeline")
    assert result.estimator.parameters == {}
    assert [step.name for step in result.estimator.steps] == ["center", "pls"]
    assert result.estimator.steps[-1].class_name.endswith(".PLSRegression")
    assert result.estimator.steps[-1].parameters["n_components"] == 1
    assert all(not hasattr(step, "fit") for step in result.estimator.steps)


def test_fold_estimators_are_opt_in_fitted_and_isolated(validation_data):
    estimator = scp.PLSRegression(n_components=1)
    result = _build(validation_data, estimator=estimator)
    assert result.fold_estimators is None

    fitted = []
    for value in range(3):
        fold_estimator = scp.PLSRegression(n_components=1)
        fold_estimator._fitted = True
        fold_estimator._retention_marker = np.array([value])
        fitted.append(fold_estimator)
    retained = _build(
        validation_data,
        estimator=estimator,
        retain_estimators=True,
        fold_estimators=fitted,
    )
    fitted[0]._retention_marker[0] = 99
    assert retained.fold_estimators[0]._retention_marker[0] == 0
    assert retained.fold_estimators[0] is not fitted[0]

    with pytest.raises(SpectroChemPyError, match="one fitted estimator per fold"):
        _build(validation_data, retain_estimators=True)
    with pytest.raises(SpectroChemPyError, match="not fitted"):
        _build(
            validation_data,
            retain_estimators=True,
            fold_estimators=[scp.PLSRegression(n_components=1)] * 3,
        )
    different = []
    for _ in range(3):
        fold_estimator = scp.PLSRegression(n_components=2)
        fold_estimator._fitted = True
        different.append(fold_estimator)
    with pytest.raises(SpectroChemPyError, match="different constructor"):
        _build(
            validation_data,
            retain_estimators=True,
            fold_estimators=different,
        )
    with pytest.raises(SpectroChemPyError, match="retain_estimators is False"):
        _build(validation_data, fold_estimators=fitted)


def test_operation_description_is_non_executable_and_result_is_public(
    validation_data,
):
    result = _build(validation_data)

    assert result.operation.operation == "cross_validate"
    assert [item.role for item in result.operation.inputs] == ["X", "y"]
    assert "non-executable" in result.operation.note
    assert "provenance" in result.operation.note
    assert isinstance(result, scp.CrossValidationResult)
    assert not hasattr(scp, "_CrossValidationResult")


@pytest.mark.parametrize("warnings", ["single string", ["valid", 2]])
def test_warning_container_is_validated(validation_data, warnings):
    with pytest.raises(SpectroChemPyError, match="warning"):
        _build(validation_data, warnings=warnings)
