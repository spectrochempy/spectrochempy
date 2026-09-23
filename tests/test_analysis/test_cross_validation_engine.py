# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Integration tests for the private supervised cross-validation engine."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

import spectrochempy as scp
from spectrochempy.analysis._cross_validation import _execute_cross_validation
from spectrochempy.utils._estimator import clone_unfitted
from spectrochempy.utils._estimator import is_fitted
from spectrochempy.utils.exceptions import SpectroChemPyError


@pytest.fixture()
def supervised_data():
    """Return deterministic spectra and two aligned nonlinear targets."""
    n_observations = 8
    latent = np.linspace(-2.5, 2.0, n_observations)
    X_values = np.column_stack(
        [latent, latent**2, np.sin(latent), np.cos(1.5 * latent), latent**3]
    )
    y_values = np.column_stack(
        [
            1.0 + latent + 0.7 * latent**3,
            2.0 - latent**2 + 0.2 * np.sin(3.0 * latent),
        ]
    )
    samples = scp.Coord(
        np.arange(100, 100 + n_observations),
        labels=[f"sample-{index}" for index in range(n_observations)],
        title="sample",
    )
    features = scp.Coord.linspace(
        1000.0,
        1400.0,
        X_values.shape[1],
        title="wavenumber",
        units="cm^-1",
    )
    targets = scp.Coord(
        [10, 20], labels=["concentration", "temperature"], title="target"
    )
    X = scp.NDDataset(
        X_values,
        coordset=[samples, features],
        dims=["y", "x"],
        units="absorbance",
        title="spectra",
    )
    X.meta.instrument = "synthetic"
    y = scp.NDDataset(
        y_values,
        coordset=[samples.copy(), targets],
        dims=["y", "x"],
        units="mmol/L",
        title="response",
    )
    y.meta.method = "synthetic"
    return X, y


def _manual_oof(template, X, y, folds):
    """Independent row-oriented loop used as the numerical oracle."""
    values = np.ma.masked_all(y.shape, dtype=float)
    fitted = []
    for train, validation in folds:
        estimator = clone_unfitted(template)
        estimator.fit(X[train, :], y[train, :])
        prediction = estimator.predict(X[validation, :])
        values[validation, :] = prediction.masked_data
        fitted.append(estimator)
    result = y.copy()
    result.data = values
    return result, fitted


def test_pls_engine_matches_independent_manual_loop_and_materializes_once(
    supervised_data,
):
    X, y = supervised_data
    template = scp.PLSRegression(n_components=2, scale=False).fit(X, y)
    template_coef = template._coef.copy()
    X_before = X.copy()
    y_before = y.copy()
    ordinary_folds = list(
        KFold(n_splits=3, shuffle=True, random_state=7).split(np.arange(8))
    )
    execution_folds = ordinary_folds[::-1]
    splitter = KFold(n_splits=3, shuffle=True, random_state=7)
    split_calls = 0

    def split_once(*_args, **_kwargs):
        nonlocal split_calls
        split_calls += 1
        return iter(execution_folds)

    splitter.split = split_once
    manual, _ = _manual_oof(template, X, y, execution_folds)

    result = _execute_cross_validation(
        template,
        X,
        y,
        splitter=splitter,
        return_estimators=True,
    )

    assert split_calls == 1
    assert [fold.validation_positions.tolist() for fold in result.folds] == [
        validation.tolist() for _, validation in execution_folds
    ]
    assert_allclose(result.oof_predictions.data, manual.data)
    errors = np.asarray(manual.data) - np.asarray(y.data)
    expected_rmsecv = np.sqrt(np.mean(errors**2, axis=0))
    assert_allclose(result.metric("rmsecv").values.data, expected_rmsecv)
    fold_rmse_mean = np.mean(
        [fold.metric("rmse").values.data for fold in result.folds], axis=0
    )
    assert not np.allclose(result.metric("rmsecv").values.data, fold_rmse_mean)

    assert is_fitted(template)
    assert_allclose(template._coef, template_coef)
    assert_allclose(X.data, X_before.data)
    assert X.coordset == X_before.coordset
    assert_allclose(y.data, y_before.data)
    assert y.coordset == y_before.coordset
    assert len(result.fold_estimators) == 3
    assert all(is_fitted(estimator) for estimator in result.fold_estimators)
    assert all(estimator is not template for estimator in result.fold_estimators)
    assert len({id(estimator) for estimator in result.fold_estimators}) == 3
    assert all(
        not np.shares_memory(left._coef, right._coef)
        for left, right in zip(
            result.fold_estimators,
            result.fold_estimators[1:],
            strict=False,
        )
    )


def test_pipeline_matches_manual_loop_and_fits_preprocessing_per_fold(
    supervised_data,
):
    X, y = supervised_data
    template = scp.Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=2, scale=False)),
        ]
    )
    folds = list(KFold(n_splits=3).split(np.arange(8)))
    manual, manual_estimators = _manual_oof(template, X, y, folds)
    global_mean = np.mean(X.data, axis=0, keepdims=True)

    result = _execute_cross_validation(
        template,
        X,
        y,
        splitter=KFold(n_splits=3),
        return_estimators=True,
    )

    assert_allclose(result.oof_predictions.data, manual.data)
    assert template._fitted is False
    assert all(step._fitted is False for _, step in template.steps)
    for fold, retained, expected in zip(
        result.folds,
        result.fold_estimators,
        manual_estimators,
        strict=True,
    ):
        retained_mean = retained.fitted_named_steps_["center"].mean_
        expected_mean = expected.fitted_named_steps_["center"].mean_
        assert_allclose(retained_mean, expected_mean)
        assert_allclose(
            retained_mean,
            np.mean(X.data[fold.train_positions], axis=0, keepdims=True),
        )
        assert not np.allclose(retained_mean, global_mean)
    assert len({id(estimator) for estimator in result.fold_estimators}) == 3


def test_transposed_multivariate_and_one_dimensional_targets_restore_geometry(
    supervised_data,
):
    X, y = supervised_data
    multivariate = _execute_cross_validation(
        scp.PLSRegression(n_components=2, scale=False),
        X.T,
        y.T,
        splitter=KFold(n_splits=3),
    )

    assert (multivariate.x_sample_axis, multivariate.y_sample_axis) == (1, 1)
    assert multivariate.oof_predictions.shape == y.T.shape
    assert multivariate.oof_predictions.dims == y.T.dims
    assert multivariate.oof_predictions.coordset == y.T.coordset
    assert multivariate.oof_predictions.units == y.units
    assert multivariate.fold_estimators is None

    univariate_y = y[:, 0].squeeze()
    univariate = _execute_cross_validation(
        scp.PLSRegression(n_components=2, scale=False),
        X,
        univariate_y,
        splitter=KFold(n_splits=3),
    )
    assert univariate.oof_predictions.shape == univariate_y.shape
    assert univariate.oof_predictions.dims == univariate_y.dims
    assert univariate.oof_predictions.coordset == univariate_y.coordset
    assert univariate.oof_predictions.units == univariate_y.units
    assert univariate.metric("rmsecv").values.ndim == 0


def test_groupkfold_and_leave_one_out_execute_their_bounded_contract(
    supervised_data,
):
    X, y = supervised_data
    groups = np.repeat(np.arange(4), 2)
    grouped = _execute_cross_validation(
        scp.PLSRegression(n_components=1, scale=False),
        X,
        y,
        splitter=GroupKFold(n_splits=4),
        groups=groups,
    )
    assert grouped.n_splits == 4
    assert grouped.groups.n_groups == 4
    for fold in grouped.folds:
        assert set(groups[fold.train_positions]).isdisjoint(
            groups[fold.validation_positions]
        )

    loo = _execute_cross_validation(
        scp.PLSRegression(n_components=1, scale=False),
        X,
        y[:, 0].squeeze(),
        splitter=LeaveOneOut(),
    )
    assert loo.n_splits == X.shape[0]
    assert loo.metric("r2").defined == (True,)
    assert all(fold.metric("r2").defined == (False,) for fold in loo.folds)


def test_invalid_partitions_are_rejected_before_fit(supervised_data, monkeypatch):
    X, y = supervised_data
    fit_calls = 0
    original_fit = scp.PLSRegression.fit

    def counting_fit(self, X_train, y_train):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, X_train, y_train)

    monkeypatch.setattr(scp.PLSRegression, "fit", counting_fit)
    incomplete = KFold(n_splits=2)
    incomplete.split = lambda *_args, **_kwargs: iter(
        [
            (np.array([2, 3, 4, 5, 6, 7]), np.array([0, 1])),
            (np.array([0, 1, 4, 5, 6, 7]), np.array([2, 3])),
        ]
    )
    with pytest.raises(SpectroChemPyError, match="cover every observation"):
        _execute_cross_validation(
            scp.PLSRegression(n_components=1),
            X,
            y,
            splitter=incomplete,
        )

    groups = np.tile(np.arange(4), 2)
    leaking = GroupKFold(n_splits=2)
    leaking.split = lambda *_args, **_kwargs: iter(
        KFold(n_splits=2).split(np.arange(8))
    )
    with pytest.raises(SpectroChemPyError, match="group in both"):
        _execute_cross_validation(
            scp.PLSRegression(n_components=1),
            X,
            y,
            splitter=leaking,
            groups=groups,
        )
    assert fit_calls == 0


@pytest.mark.parametrize("invalid", ["varying-mask", "X-nan", "y-mask", "y-inf"])
def test_invalid_input_values_are_rejected_before_fit(
    supervised_data, monkeypatch, invalid
):
    X, y = supervised_data
    fit_calls = 0

    def unexpected_fit(self, X_train, y_train):
        nonlocal fit_calls
        fit_calls += 1
        return self

    monkeypatch.setattr(scp.PLSRegression, "fit", unexpected_fit)
    if invalid == "varying-mask":
        mask = np.zeros(X.shape, dtype=bool)
        mask[0, 0] = True
        X.mask = mask
    elif invalid == "X-nan":
        X.data[0, 0] = np.nan
    elif invalid == "y-mask":
        mask = np.zeros(y.shape, dtype=bool)
        mask[0, 0] = True
        y.mask = mask
    else:
        y.data[0, 0] = np.inf

    with pytest.raises(SpectroChemPyError, match="mask|non-finite"):
        _execute_cross_validation(
            scp.PLSRegression(n_components=1),
            X,
            y,
            splitter=KFold(n_splits=2),
        )
    assert fit_calls == 0


@pytest.mark.parametrize("transpose_X", [False, True])
def test_consistently_masked_features_are_preserved(supervised_data, transpose_X):
    X, y = supervised_data
    X = X.T if transpose_X else X
    mask = np.zeros(X.shape, dtype=bool)
    feature_axis = 1 - X.dims.index("y")
    index = [slice(None)] * X.ndim
    index[feature_axis] = -1
    mask[tuple(index)] = True
    X.mask = mask

    result = _execute_cross_validation(
        scp.PLSRegression(n_components=1, scale=False),
        X,
        y,
        splitter=KFold(n_splits=2),
    )

    assert np.all(np.isfinite(result.oof_predictions.data))
    assert np.array_equal(X.mask, mask)


@pytest.mark.parametrize("operation", ["fit", "predict"])
def test_fold_errors_are_contextualized_and_preserve_their_cause(
    supervised_data, monkeypatch, operation
):
    X, y = supervised_data

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"synthetic {operation} failure")

    monkeypatch.setattr(scp.PLSRegression, operation, fail)
    with pytest.raises(
        SpectroChemPyError,
        match=rf"fold 0 failed during {operation}",
    ) as caught:
        _execute_cross_validation(
            scp.PLSRegression(n_components=1, scale=False),
            X,
            y,
            splitter=KFold(n_splits=2),
        )
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_engine_remains_private():
    assert not hasattr(scp, "_execute_cross_validation")
    assert not hasattr(scp, "cross_validate")
