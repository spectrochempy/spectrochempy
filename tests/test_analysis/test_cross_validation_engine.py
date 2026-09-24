# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Integration tests for the private supervised cross-validation engine."""

import inspect
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import GroupKFold as SklearnGroupKFold
from sklearn.model_selection import KFold as SklearnKFold
from sklearn.model_selection import LeaveOneOut as SklearnLeaveOneOut
from sklearn.model_selection import ShuffleSplit

import spectrochempy as scp
from spectrochempy.analysis._cross_validation import _execute_cross_validation
from spectrochempy.utils._estimator import clone_unfitted
from spectrochempy.utils._estimator import is_fitted
from spectrochempy.utils.exceptions import SpectroChemPyError

GroupKFold = SklearnGroupKFold
KFold = SklearnKFold
LeaveOneOut = SklearnLeaveOneOut


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


def test_public_exports_and_signature_match_the_contract():
    assert all(
        scp._LAZY_IMPORTS[name] == "spectrochempy.analysis.model_selection"
        for name in ("KFold", "GroupKFold", "LeaveOneOut")
    )
    assert scp.cross_validate.__name__ == "cross_validate"
    assert scp.CrossValidationResult.__name__ == "CrossValidationResult"
    assert scp.KFold is not SklearnKFold
    assert scp.GroupKFold is not SklearnGroupKFold
    assert scp.LeaveOneOut is not SklearnLeaveOneOut
    assert issubclass(scp.KFold, SklearnKFold)
    assert issubclass(scp.GroupKFold, SklearnGroupKFold)
    assert issubclass(scp.LeaveOneOut, SklearnLeaveOneOut)
    assert scp.KFold.__module__ == "spectrochempy.analysis.model_selection"
    assert scp.GroupKFold.__module__ == "spectrochempy.analysis.model_selection"
    assert scp.LeaveOneOut.__module__ == "spectrochempy.analysis.model_selection"
    assert str(inspect.signature(scp.KFold)) == (
        "(n_splits=5, *, shuffle=False, random_state=None)"
    )
    assert str(inspect.signature(scp.GroupKFold)) == "(n_splits=5)"
    assert str(inspect.signature(scp.LeaveOneOut)) == "()"
    signature = inspect.signature(scp.cross_validate)
    assert tuple(signature.parameters) == (
        "estimator",
        "X",
        "y",
        "cv",
        "groups",
        "sample_dim",
        "metrics",
        "return_estimators",
    )
    assert signature.parameters["cv"].default == 5
    assert signature.parameters["groups"].default is None
    assert signature.parameters["sample_dim"].default == "y"
    assert signature.parameters["metrics"].default == ("rmsecv", "r2")
    assert signature.parameters["return_estimators"].default is False
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in (
            "cv",
            "groups",
            "sample_dim",
            "metrics",
            "return_estimators",
        )
    )


def test_public_splitter_module_is_loaded_lazily():
    code = """
import sys
import spectrochempy as scp

module = "spectrochempy.analysis.model_selection"
assert module not in sys.modules
assert scp.KFold.__module__ == module
assert module in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_splitter_documentation_is_native_without_mutating_sklearn():
    sklearn_docs = {
        SklearnKFold: SklearnKFold.__doc__,
        SklearnGroupKFold: SklearnGroupKFold.__doc__,
        SklearnLeaveOneOut: SklearnLeaveOneOut.__doc__,
    }

    for adapter, original in (
        (scp.KFold, SklearnKFold),
        (scp.GroupKFold, SklearnGroupKFold),
        (scp.LeaveOneOut, SklearnLeaveOneOut),
    ):
        assert "SpectroChemPy adaptation" in adapter.__doc__
        assert "scp.cross_validate" in adapter.__doc__
        assert adapter.__doc__ != original.__doc__
        assert "SpectroChemPy adaptation" not in original.__doc__
        assert original.__module__.startswith("sklearn.")
        assert original.__doc__ == sklearn_docs[original]


@pytest.mark.parametrize(
    ("adapter", "original", "split_args"),
    [
        (
            scp.KFold(n_splits=4, shuffle=True, random_state=7),
            SklearnKFold(n_splits=4, shuffle=True, random_state=7),
            {},
        ),
        (
            scp.GroupKFold(n_splits=4),
            SklearnGroupKFold(n_splits=4),
            {"groups": np.repeat(np.arange(4), 2)},
        ),
        (scp.LeaveOneOut(), SklearnLeaveOneOut(), {}),
    ],
)
def test_public_splitter_adapters_match_sklearn_folds(adapter, original, split_args):
    positions = np.arange(8)
    adapter_folds = list(adapter.split(positions, **split_args))
    original_folds = list(original.split(positions, **split_args))

    assert len(adapter_folds) == len(original_folds)
    for adapter_fold, original_fold in zip(adapter_folds, original_folds, strict=True):
        assert all(
            np.array_equal(adapter_positions, original_positions)
            for adapter_positions, original_positions in zip(
                adapter_fold, original_fold, strict=True
            )
        )


def test_public_splitter_exports_execute_their_bounded_contract(supervised_data):
    X, y = supervised_data
    template = scp.PLSRegression(n_components=1, scale=False)

    shuffled = scp.KFold(n_splits=4, shuffle=True, random_state=7)
    shuffled_repeat = scp.KFold(n_splits=4, shuffle=True, random_state=7)
    first = scp.cross_validate(template, X, y, cv=shuffled)
    repeated = scp.cross_validate(template, X, y, cv=shuffled_repeat)
    assert [fold.validation_positions.tolist() for fold in first.folds] == [
        fold.validation_positions.tolist() for fold in repeated.folds
    ]

    groups = np.repeat(np.arange(4), 2)
    grouped = scp.cross_validate(
        template,
        X,
        y,
        cv=scp.GroupKFold(n_splits=4),
        groups=groups,
    )
    for fold in grouped.folds:
        assert set(groups[fold.train_positions]).isdisjoint(
            groups[fold.validation_positions]
        )

    loo = scp.cross_validate(template, X, y, cv=scp.LeaveOneOut())
    assert loo.n_splits == X.shape[0]
    assert all(len(fold.validation_positions) == 1 for fold in loo.folds)

    sklearn_loo = scp.cross_validate(template, X, y, cv=SklearnLeaveOneOut())
    assert sklearn_loo.splitter.class_name == (
        "sklearn.model_selection._split.LeaveOneOut"
    )
    assert_allclose(sklearn_loo.oof_predictions.data, loo.oof_predictions.data)


def test_integer_cv_matches_explicit_splitter_and_manual_oof(supervised_data):
    X, y = supervised_data
    template = scp.PLSRegression(n_components=2, scale=False)
    folds = list(KFold(n_splits=4, shuffle=False).split(np.arange(X.shape[0])))
    manual, _ = _manual_oof(template, X, y, folds)

    integer_result = scp.cross_validate(template, X, y, cv=np.int64(4))
    explicit_result = scp.cross_validate(template, X, y, cv=KFold(n_splits=4))

    assert isinstance(integer_result, scp.CrossValidationResult)
    assert integer_result.splitter.class_name == (
        "spectrochempy.analysis.model_selection.KFold"
    )
    assert explicit_result.splitter.class_name == (
        "sklearn.model_selection._split.KFold"
    )
    assert integer_result.splitter.parameters["n_splits"] == 4
    assert integer_result.splitter.parameters["shuffle"] is False
    assert_allclose(integer_result.oof_predictions.data, manual.data)
    assert_allclose(
        integer_result.oof_predictions.data,
        explicit_result.oof_predictions.data,
    )


def test_integer_cv_with_groups_matches_groupkfold(supervised_data):
    X, y = supervised_data
    groups = np.repeat(np.arange(4), 2)
    template = scp.PLSRegression(n_components=1, scale=False)

    integer_result = scp.cross_validate(template, X, y, cv=4, groups=groups)
    explicit_result = scp.cross_validate(
        template,
        X,
        y,
        cv=GroupKFold(n_splits=4),
        groups=groups,
    )

    assert integer_result.splitter.class_name == (
        "spectrochempy.analysis.model_selection.GroupKFold"
    )
    assert explicit_result.splitter.class_name == (
        "sklearn.model_selection._split.GroupKFold"
    )
    assert integer_result.groups.n_groups == 4
    assert_allclose(
        integer_result.oof_predictions.data,
        explicit_result.oof_predictions.data,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cv": True}, "cv must be an integer"),
        ({"cv": 2.5}, "floating-point"),
        ({"cv": 1}, "at least 2"),
        ({"cv": ShuffleSplit(n_splits=2)}, "KFold, GroupKFold, or LeaveOneOut"),
        ({"metrics": "rmsecv"}, "ordered sequence"),
        ({"metrics": {"rmsecv", "r2"}}, "ordered sequence"),
        ({"metrics": ()}, "At least one"),
        ({"metrics": ("r2", "r2")}, "must not be repeated"),
        ({"metrics": ("rmse",)}, "Use 'rmsecv'"),
        ({"metrics": ("unknown",)}, "Unsupported"),
        ({"return_estimators": 1}, "must be a boolean"),
    ],
)
def test_public_options_are_validated_before_fit(
    supervised_data,
    monkeypatch,
    overrides,
    message,
):
    X, y = supervised_data
    fit_calls = 0

    def unexpected_fit(self, X_train, y_train):
        nonlocal fit_calls
        fit_calls += 1
        return self

    monkeypatch.setattr(scp.PLSRegression, "fit", unexpected_fit)
    with pytest.raises(SpectroChemPyError, match=message):
        scp.cross_validate(
            scp.PLSRegression(n_components=1),
            X,
            y,
            **overrides,
        )
    assert fit_calls == 0


@pytest.mark.parametrize(
    "splitter",
    [
        type("CustomSklearnKFold", (SklearnKFold,), {})(n_splits=2),
        type("CustomSCPKFold", (scp.KFold,), {})(n_splits=2),
    ],
)
def test_arbitrary_splitter_subclasses_remain_out_of_scope(
    supervised_data, monkeypatch, splitter
):
    X, y = supervised_data
    fit_calls = 0

    def unexpected_fit(self, X_train, y_train):
        nonlocal fit_calls
        fit_calls += 1
        return self

    monkeypatch.setattr(scp.PLSRegression, "fit", unexpected_fit)
    with pytest.raises(SpectroChemPyError, match="KFold, GroupKFold, or LeaveOneOut"):
        scp.cross_validate(
            scp.PLSRegression(n_components=1),
            X,
            y,
            cv=splitter,
        )
    assert fit_calls == 0


def test_supplied_adapter_state_and_snapshot_identity_are_preserved(supervised_data):
    X, y = supervised_data
    random_state = np.random.RandomState(7)
    before = random_state.get_state()
    splitter = scp.KFold(n_splits=4, shuffle=True, random_state=random_state)

    result = scp.cross_validate(
        scp.PLSRegression(n_components=1, scale=False),
        X,
        y,
        cv=splitter,
    )

    after = random_state.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert result.splitter.class_name == (
        "spectrochempy.analysis.model_selection.KFold"
    )
    assert result.splitter.parameters["n_splits"] == 4
    assert result.splitter.parameters["shuffle"] is True


@pytest.mark.parametrize(
    ("cv", "groups", "message"),
    [
        (9, None, "n_splits=9"),
        (3, np.repeat([0, 1], 4), "number of groups"),
        (KFold(n_splits=2), np.repeat([0, 1], 4), "non-group splitter"),
        (GroupKFold(n_splits=2), None, "requires groups"),
    ],
)
def test_splitter_constraints_fail_before_fit(
    supervised_data,
    monkeypatch,
    cv,
    groups,
    message,
):
    X, y = supervised_data
    fit_calls = 0

    def unexpected_fit(self, X_train, y_train):
        nonlocal fit_calls
        fit_calls += 1
        return self

    monkeypatch.setattr(scp.PLSRegression, "fit", unexpected_fit)
    with pytest.raises(SpectroChemPyError, match=message):
        scp.cross_validate(
            scp.PLSRegression(n_components=1),
            X,
            y,
            cv=cv,
            groups=groups,
        )
    assert fit_calls == 0


def test_public_result_fields_metric_order_and_optional_estimators(supervised_data):
    X, y = supervised_data
    result = scp.cross_validate(
        scp.PLSRegression(n_components=1, scale=False),
        X,
        y,
        cv=4,
        metrics=("mae", "rmsecv", "bias"),
        return_estimators=True,
    )

    assert isinstance(result, scp.CrossValidationResult)
    assert tuple(metric.name for metric in result.global_metrics) == (
        "mae",
        "rmsecv",
        "bias",
    )
    assert tuple(metric.name for metric in result.folds[0].metrics) == (
        "mae",
        "rmse",
        "bias",
    )
    assert result.metric("rmsecv").values.units == y.units
    assert result.n_valid.shape == (y.shape[1],)
    assert result.observed.coordset == y.coordset
    assert result.oof_predictions.coordset == y.coordset
    assert result.residuals.coordset == y.coordset
    assert result.sample_dim == "y"
    assert (result.x_sample_axis, result.y_sample_axis) == (0, 0)
    assert result.observation_coordinate == y.y
    assert result.estimator.class_name.endswith(".PLSRegression")
    assert result.n_splits == 4
    assert len(result.folds) == 4
    assert len(result.fold_estimators) == 4
    assert result.operation.operation == "cross_validate"
    assert "non-executable" in result.operation.note


def test_public_pipeline_supports_univariate_target_and_distinct_axes(supervised_data):
    X, y = supervised_data
    univariate_y = y[:, 0].squeeze()
    template = scp.Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=1, scale=False)),
        ]
    )
    X_before = X.copy()
    y_before = univariate_y.copy()

    result = scp.cross_validate(template, X.T, univariate_y, cv=4)

    assert isinstance(result, scp.CrossValidationResult)
    assert result.oof_predictions.shape == univariate_y.shape
    assert (result.x_sample_axis, result.y_sample_axis) == (1, 0)
    assert template._fitted is False
    assert_allclose(X.data, X_before.data)
    assert X.coordset == X_before.coordset
    assert_allclose(univariate_y.data, y_before.data)
    assert univariate_y.coordset == y_before.coordset


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


def test_pipeline_with_msc_preserves_masked_feature_geometry(supervised_data):
    X, y = supervised_data
    mask = np.zeros(X.shape, dtype=bool)
    mask[:, -1] = True
    X.mask = mask
    reference = scp.NDDataset(
        np.mean(X.data, axis=0),
        coordset=[X.x.copy()],
        dims=["x"],
    )
    template = scp.Pipeline(
        [
            ("msc", scp.MSCTransformer(reference=reference, dim="y")),
            ("pls", scp.PLSRegression(n_components=1, scale=False)),
        ]
    )
    folds = list(KFold(n_splits=2).split(np.arange(X.shape[0])))
    manual, manual_estimators = _manual_oof(template, X, y, folds)

    result = scp.cross_validate(
        template,
        X,
        y,
        cv=KFold(n_splits=2),
        return_estimators=True,
    )

    assert_allclose(result.oof_predictions.data, manual.data)
    assert np.array_equal(X.mask, mask)
    for retained, expected in zip(
        result.fold_estimators,
        manual_estimators,
        strict=True,
    ):
        assert retained.fitted_named_steps_["msc"]._spectral_size_ == X.shape[1]
        assert_allclose(
            retained.fitted_named_steps_["msc"].reference_,
            expected.fitted_named_steps_["msc"].reference_,
        )


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

    result = scp.cross_validate(
        scp.PLSRegression(n_components=1, scale=False),
        X,
        y,
        cv=KFold(n_splits=2),
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
