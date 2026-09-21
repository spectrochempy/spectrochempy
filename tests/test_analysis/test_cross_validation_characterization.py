# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Characterize existing building blocks for future cross-validation support."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

import spectrochempy as scp
from spectrochempy.utils._estimator import clone_unfitted
from spectrochempy.utils._estimator import is_fitted
from spectrochempy.utils.exceptions import SpectroChemPyError


@pytest.fixture()
def cv_data():
    """Return deterministic spectra and an aligned univariate response."""
    n_samples = 9
    latent = np.linspace(-2.0, 2.0, n_samples)
    x_data = np.column_stack(
        [
            latent,
            latent**2,
            np.sin(latent),
            np.cos(latent),
            0.5 * latent + 2.0,
            latent**3 - latent,
        ]
    )
    y_data = (1.5 + 2.2 * latent - 0.4 * latent**2)[:, None]

    samples = scp.Coord(
        np.arange(100, 100 + n_samples),
        labels=[f"sample-{index}" for index in range(n_samples)],
        title="sample",
    )
    features = scp.Coord.linspace(
        1000.0,
        1500.0,
        x_data.shape[1],
        title="wavenumber",
        units="cm^-1",
    )
    target = scp.Coord([0], labels=["concentration"], title="property")

    X = scp.NDDataset(
        x_data,
        coordset=[samples, features],
        dims=["y", "x"],
        units="absorbance",
        title="spectra",
    )
    X.meta.instrument = "synthetic"
    y = scp.NDDataset(
        y_data,
        coordset=[samples.copy(), target],
        dims=["y", "x"],
        units="mmol/L",
        title="response",
    )
    y.meta.method = "synthetic"
    return X, y


@pytest.mark.parametrize(
    ("splitter", "groups"),
    [
        (KFold(n_splits=3, shuffle=True, random_state=7), None),
        (GroupKFold(n_splits=3), np.repeat(np.arange(3), 3)),
        (LeaveOneOut(), None),
    ],
    ids=["kfold", "group-kfold", "leave-one-out"],
)
def test_splitters_partition_nddataset_observations(cv_data, splitter, groups):
    X, y = cv_data
    folds = list(splitter.split(X, groups=groups))
    validation_positions = []

    for train, validation in folds:
        assert np.issubdtype(train.dtype, np.integer)
        assert np.issubdtype(validation.dtype, np.integer)
        assert np.intersect1d(train, validation).size == 0

        X_validation = X[validation, :]
        y_validation = y[validation, :]
        assert X_validation.shape[0] == validation.size
        assert y_validation.shape[0] == validation.size
        assert X_validation.coordset["y"] == y_validation.coordset["y"]

        if groups is not None:
            assert set(groups[train]).isdisjoint(groups[validation])
        validation_positions.extend(validation.tolist())

    assert sorted(validation_positions) == list(range(X.shape[0]))
    assert len(validation_positions) == len(set(validation_positions))
    if isinstance(splitter, LeaveOneOut):
        assert all(len(validation) == 1 for _, validation in folds)


@pytest.mark.parametrize("positions", [np.array([7, 1, 4]), np.array([3])])
def test_manual_slicing_preserves_alignment_and_metadata(cv_data, positions):
    X, y = cv_data

    X_subset = X[positions, :]
    y_subset = y[positions, :]
    X_transposed_subset = X.T[:, positions].T
    y_transposed_subset = y.T[:, positions].T

    expected_samples = X.coordset["y"][positions]
    assert X_subset.coordset["y"] == expected_samples
    assert y_subset.coordset["y"] == expected_samples
    assert X_subset.coordset["x"] == X.coordset["x"]
    assert y_subset.coordset["x"] == y.coordset["x"]
    assert X_subset.units == X.units
    assert y_subset.units == y.units
    assert X_subset.meta.instrument == "synthetic"
    assert y_subset.meta.method == "synthetic"

    assert X_transposed_subset.dims == X.dims
    assert y_transposed_subset.dims == y.dims
    assert X_transposed_subset.coordset["y"] == expected_samples
    assert y_transposed_subset.coordset["y"] == expected_samples
    assert_allclose(X_transposed_subset.data, X_subset.data)
    assert_allclose(y_transposed_subset.data, y_subset.data)


def test_manual_pls_oof_predictions_restore_original_observation_order(cv_data):
    X, y = cv_data
    template = scp.PLSRegression(n_components=2, scale=False)
    template_params = template.get_params(deep=False)
    splitter = KFold(n_splits=3, shuffle=True, random_state=7)
    oof_values = np.full(y.shape, np.nan)
    fold_positions = []
    fold_predictions = []
    fold_estimators = []

    for train, validation in splitter.split(X):
        estimator = clone_unfitted(template)
        estimator.fit(X[train, :], y[train, :])
        prediction = estimator.predict(X[validation, :])

        assert estimator.get_params(deep=False) == template_params
        assert prediction.coordset["y"] == y[validation, :].coordset["y"]
        assert prediction.coordset["x"] == y.coordset["x"]
        assert prediction.units == y.units

        oof_values[validation] = prediction.data
        fold_positions.append(validation)
        fold_predictions.append(prediction.data)
        fold_estimators.append(estimator)

    fold_order = np.concatenate(fold_positions)
    predictions_in_fold_order = np.concatenate(fold_predictions)
    oof = y.copy()
    oof.data = oof_values

    assert not np.array_equal(fold_order, np.arange(X.shape[0]))
    assert_allclose(oof.data[fold_order], predictions_in_fold_order)
    assert not np.allclose(oof.data, predictions_in_fold_order)
    assert np.all(np.isfinite(oof.data))
    assert oof.coordset["y"] == y.coordset["y"]
    assert oof.coordset["x"] == y.coordset["x"]
    assert oof.units == y.units

    assert is_fitted(template) is False
    assert all(estimator is not template for estimator in fold_estimators)
    assert all(is_fitted(estimator) for estimator in fold_estimators)
    assert all(
        not np.shares_memory(left._coef, right._coef)
        for left, right in zip(fold_estimators, fold_estimators[1:], strict=False)
    )


def test_pipeline_fit_uses_fresh_fold_local_preprocessing_state(cv_data):
    X, y = cv_data
    center_template = scp.CenterTransformer(dim="y")
    pls_template = scp.PLSRegression(n_components=2, scale=False)
    pipeline = scp.Pipeline([("center", center_template), ("pls", pls_template)])
    global_mean = np.mean(X.data, axis=0, keepdims=True)
    fitted_steps = []

    for train, _ in KFold(n_splits=3, shuffle=True, random_state=7).split(X):
        pipeline.fit(X[train, :], y[train, :])
        fitted_center = pipeline.fitted_named_steps_["center"]
        fitted_pls = pipeline.fitted_named_steps_["pls"]

        assert_allclose(
            fitted_center.mean_, np.mean(X.data[train], axis=0, keepdims=True)
        )
        assert not np.allclose(fitted_center.mean_, global_mean)
        fitted_steps.append((fitted_center, fitted_pls))

    assert center_template._fitted is False
    assert pls_template._fitted is False
    assert not hasattr(center_template, "mean_")
    assert all(center is not center_template for center, _ in fitted_steps)
    assert all(pls is not pls_template for _, pls in fitted_steps)
    assert all(
        left_center is not right_center and left_pls is not right_pls
        for (left_center, left_pls), (right_center, right_pls) in zip(
            fitted_steps, fitted_steps[1:], strict=False
        )
    )


@pytest.mark.xfail(
    strict=True,
    raises=SpectroChemPyError,
    reason="The accepted CV contract requires Pipeline-aware unfitted cloning.",
)
def test_clone_unfitted_pipeline_contract_gap():
    pipeline = scp.Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=2, scale=False)),
        ]
    )

    cloned = clone_unfitted(pipeline)

    assert cloned is not pipeline
    assert cloned.get_params(deep=False) == pipeline.get_params(deep=False)
