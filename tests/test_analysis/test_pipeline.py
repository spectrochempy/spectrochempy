# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the public SpectroChemPy Pipeline v1 contract."""

from types import MappingProxyType

import numpy as np
import pytest

import spectrochempy as scp
import spectrochempy.analysis.pipeline as pipeline_module
from spectrochempy.analysis.pipeline import Pipeline
from spectrochempy.processing.baselineprocessing.baselineprocessing import Baseline
from spectrochempy.utils.exceptions import NotFittedError
from spectrochempy.utils.exceptions import SpectroChemPyError


def _xy():
    x = scp.Coord.linspace(1000.0, 1010.0, 6, title="wavenumber", units="cm^-1")
    y = scp.Coord.arange(5, title="sample")
    X = scp.NDDataset(
        np.arange(30.0).reshape(5, 6) + 1.0,
        coordset=[y, x],
        dims=["y", "x"],
        units="absorbance",
        title="spectra",
        name="calibration",
    )
    X.meta.operator = "Alice"
    X.history = "created calibration data"
    target = scp.NDDataset(
        np.linspace(1.0, 5.0, 5),
        coordset=[y.copy()],
        dims=["y"],
        units="mol/L",
        title="concentration",
        name="target",
    )
    return X, target


def _test_x():
    X, _ = _xy()
    new = X.copy()
    new.data = X.data + 3.0
    new.name = "test"
    return new


def _assert_unfitted(pipeline):
    assert pipeline._fitted is False
    with pytest.raises(NotFittedError):
        _ = pipeline.fitted_steps_
    with pytest.raises(NotFittedError):
        _ = pipeline.fitted_named_steps_


def test_constructs_valid_one_step_transformer_pipeline():
    pipeline = Pipeline([("center", scp.CenterTransformer(dim="y"))])

    assert scp.Pipeline is Pipeline
    assert pipeline.steps[0][0] == "center"
    assert isinstance(pipeline.named_steps, MappingProxyType)
    assert pipeline.named_steps["center"] is pipeline.steps[0][1]


def test_constructs_valid_multi_step_transformer_pipeline():
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )

    assert [name for name, _ in pipeline.steps] == ["center", "pca"]


def test_constructs_valid_predictor_terminal_pipeline():
    pipeline = Pipeline(
        [
            ("scale", scp.AutoscaleTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=2)),
        ]
    )

    assert [name for name, _ in pipeline.steps] == ["scale", "pls"]


@pytest.mark.parametrize(
    "steps, match",
    [
        ([], "at least one"),
        (
            [("a", scp.CenterTransformer()), ("a", scp.AutoscaleTransformer())],
            "duplicated",
        ),
        ([("", scp.CenterTransformer())], "non-empty"),
        ([(1, scp.CenterTransformer())], "strings"),
        ([("a__b", scp.CenterTransformer())], "cannot contain"),
        ([("steps", scp.CenterTransformer())], "reserved"),
        ([("none", None)], "cannot be None"),
        ([("pass", "passthrough")], "cannot be a string"),
    ],
)
def test_constructor_rejects_invalid_step_structure(steps, match):
    with pytest.raises(SpectroChemPyError, match=match):
        Pipeline(steps)


@pytest.mark.parametrize(
    "steps, match",
    [
        ([("baseline", Baseline())], "not in the v1 allowlist"),
        ([("svd", scp.SVD())], "not in the v1 allowlist"),
        (
            [
                ("pca", scp.PCA(n_components=2)),
                ("center", scp.CenterTransformer()),
            ],
            "not an allowlisted intermediate transformer",
        ),
        (
            [
                ("pls", scp.PLSRegression(n_components=2)),
                ("center", scp.CenterTransformer()),
            ],
            "not an allowlisted intermediate transformer",
        ),
    ],
)
def test_constructor_rejects_unsupported_classes_and_positions(steps, match):
    with pytest.raises(SpectroChemPyError, match=match):
        Pipeline(steps)


def test_fit_clones_steps_and_leaves_templates_unfitted():
    X, _ = _xy()
    center = scp.CenterTransformer(dim="y")
    pca = scp.PCA(n_components=2)
    pipeline = Pipeline([("center", center), ("pca", pca)])

    fitted = pipeline.fit(X)

    assert fitted is pipeline
    assert center._fitted is False
    assert pca._fitted is False
    assert pipeline.fitted_steps_[0][1] is not center
    assert pipeline.fitted_steps_[1][1] is not pca
    assert pipeline.fitted_named_steps_["center"] is pipeline.fitted_steps_[0][1]
    assert isinstance(pipeline.fitted_named_steps_, MappingProxyType)
    with pytest.raises(TypeError):
        pipeline.fitted_named_steps_["center"] = center
    assert not hasattr(center, "mean_")
    assert hasattr(pipeline.fitted_named_steps_["center"], "mean_")


def test_each_fit_creates_fresh_fitted_instances():
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )

    pipeline.fit(X)
    first = pipeline.fitted_steps_
    pipeline.fit(X)
    second = pipeline.fitted_steps_

    assert first[0][1] is not second[0][1]
    assert first[1][1] is not second[1][1]


def test_mutable_constructor_parameters_are_isolated_from_fitted_steps():
    X, _ = _xy()
    reference = np.linspace(1.0, 2.0, 6)
    pipeline = Pipeline([("msc", scp.MSCTransformer(reference=reference, dim="y"))])

    pipeline.fit(X)
    fitted_reference = pipeline.fitted_named_steps_["msc"].reference

    assert fitted_reference is not pipeline.named_steps["msc"].reference
    fitted_reference[0] = 99.0
    assert pipeline.named_steps["msc"].reference[0] != 99.0


def test_transformer_only_fit_and_transform():
    X, _ = _xy()
    pipeline = Pipeline([("center", scp.CenterTransformer(dim="y"))])

    pipeline.fit(X)
    transformed = pipeline.transform(_test_x())

    assert isinstance(transformed, scp.NDDataset)
    assert transformed.dims == X.dims
    assert transformed.units == X.units
    assert transformed.coordset["x"] == X.coordset["x"]


def test_preprocessing_followed_by_pca_fit_transform_equivalence():
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )
    expected_pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )

    result = pipeline.fit_transform(X)
    expected = expected_pipeline.fit(X).transform(X)

    assert isinstance(result, scp.NDDataset)
    assert result.shape == (5, 2)
    assert np.allclose(result.data, expected.data)


def test_preprocessing_followed_by_pls_predict_and_score():
    X, y = _xy()
    pipeline = Pipeline(
        [
            ("scale", scp.AutoscaleTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=2)),
        ]
    )

    pipeline.fit(X, y)
    predicted = pipeline.predict(_test_x())
    score = pipeline.score(_test_x(), y)

    assert isinstance(predicted, scp.NDDataset)
    assert isinstance(score, float)


def test_estimator_final_fit_transform_is_unavailable_without_fitting():
    X, y = _xy()
    pipeline = Pipeline([("pls", scp.PLSRegression(n_components=2))])

    with pytest.raises(SpectroChemPyError, match="fit_transform is not available"):
        pipeline.fit_transform(X, y)

    _assert_unfitted(pipeline)


@pytest.mark.parametrize("final", [scp.LSTSQ(), scp.NNLS()])
def test_preprocessing_followed_by_linear_regression_terminal(final):
    X, y = _xy()
    pipeline = Pipeline([("center", scp.CenterTransformer(dim="y")), ("linear", final)])

    pipeline.fit(X, y)

    assert isinstance(pipeline.predict(_test_x()), scp.NDDataset)
    assert isinstance(pipeline.score(_test_x(), y), float)


def test_y_is_routed_only_to_supervised_final_estimators():
    X, y = _xy()
    transformer_pipeline = Pipeline([("center", scp.CenterTransformer(dim="y"))])
    estimator_pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=2)),
        ]
    )

    with pytest.raises(SpectroChemPyError, match="does not accept y"):
        transformer_pipeline.fit(X, y)
    with pytest.raises(SpectroChemPyError, match="requires y"):
        estimator_pipeline.fit(X)

    estimator_pipeline.fit(X, y)
    assert estimator_pipeline.fitted_named_steps_["center"]._fitted
    assert estimator_pipeline.fitted_named_steps_["pls"]._fitted


def test_incompatible_y_preserves_original_error_cause():
    X, _ = _xy()
    pipeline = Pipeline([("pls", scp.PLSRegression(n_components=2))])

    with pytest.raises(SpectroChemPyError, match="step 'pls'") as excinfo:
        pipeline.fit(X, scp.NDDataset(np.arange(4.0)))

    assert excinfo.value.__cause__ is not None
    _assert_unfitted(pipeline)


def test_intermediate_fit_failure_leaves_no_partial_state():
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("msc", scp.MSCTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )

    with pytest.raises(SpectroChemPyError, match="step 'msc'"):
        pipeline.fit(scp.NDDataset(np.arange(6.0)))

    _assert_unfitted(pipeline)


def test_intermediate_transform_failure_leaves_no_partial_state(monkeypatch):
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )
    original_clone = pipeline_module.clone_unfitted

    class BrokenTransform:
        _fitted = False

        def fit(self, X):
            self._fitted = True
            return self

        def transform(self, X):
            raise ValueError("broken transform")

    def clone_with_broken_transform(step):
        if isinstance(step, scp.CenterTransformer):
            return BrokenTransform()
        return original_clone(step)

    monkeypatch.setattr(pipeline_module, "clone_unfitted", clone_with_broken_transform)

    with pytest.raises(SpectroChemPyError, match="Pipeline transform failed"):
        pipeline.fit(X)

    _assert_unfitted(pipeline)


def test_intermediate_transform_must_return_nddataset(monkeypatch):
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )
    original_clone = pipeline_module.clone_unfitted

    class ArrayTransform:
        _fitted = False

        def fit(self, X):
            self._fitted = True
            return self

        def transform(self, X):
            return X.data

    def clone_with_array_transform(step):
        if isinstance(step, scp.CenterTransformer):
            return ArrayTransform()
        return original_clone(step)

    monkeypatch.setattr(pipeline_module, "clone_unfitted", clone_with_array_transform)

    with pytest.raises(SpectroChemPyError, match="expected NDDataset"):
        pipeline.fit(X)

    _assert_unfitted(pipeline)


def test_clone_failure_reports_step_context(monkeypatch):
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )
    original_clone = pipeline_module.clone_unfitted

    def clone_with_broken_pca(step):
        if isinstance(step, scp.PCA):
            raise ValueError("broken clone")
        return original_clone(step)

    monkeypatch.setattr(pipeline_module, "clone_unfitted", clone_with_broken_pca)

    with pytest.raises(
        SpectroChemPyError,
        match="Pipeline clone failed at step 'pca' \\(position 1, class PCA\\)",
    ) as excinfo:
        pipeline.fit(X)

    assert excinfo.value.__cause__ is not None
    _assert_unfitted(pipeline)


def test_final_fit_failure_leaves_no_partial_state():
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=99)),
        ]
    )

    with pytest.raises(SpectroChemPyError, match="step 'pca'"):
        pipeline.fit(X)

    _assert_unfitted(pipeline)


def test_failed_refit_clears_previous_fitted_state():
    X, _ = _xy()
    pipeline = Pipeline([("pca", scp.PCA(n_components=2))]).fit(X)

    assert pipeline._fitted
    pipeline.set_params(pca__n_components=99)
    with pytest.raises(SpectroChemPyError, match="step 'pca'"):
        pipeline.fit(X)

    _assert_unfitted(pipeline)


def test_public_operations_require_fit():
    X, y = _xy()
    transformer_pipeline = Pipeline([("center", scp.CenterTransformer(dim="y"))])
    estimator_pipeline = Pipeline([("pls", scp.PLSRegression(n_components=2))])

    with pytest.raises(NotFittedError):
        transformer_pipeline.transform(X)
    with pytest.raises(NotFittedError):
        estimator_pipeline.predict(X)
    with pytest.raises(NotFittedError):
        estimator_pipeline.score(X, y)
    with pytest.raises(NotFittedError):
        _ = transformer_pipeline.fitted_steps_
    with pytest.raises(NotFittedError):
        _ = transformer_pipeline.fitted_named_steps_


def test_incompatible_category_methods_fail_after_fit():
    X, y = _xy()
    transformer_pipeline = Pipeline([("pca", scp.PCA(n_components=2))]).fit(X)
    estimator_pipeline = Pipeline([("pls", scp.PLSRegression(n_components=2))]).fit(
        X, y
    )

    assert callable(transformer_pipeline.transform)
    assert callable(transformer_pipeline.predict)
    assert callable(estimator_pipeline.predict)
    assert callable(estimator_pipeline.transform)
    with pytest.raises(SpectroChemPyError, match="predict is not available"):
        transformer_pipeline.predict(X)
    with pytest.raises(SpectroChemPyError, match="score is not available"):
        transformer_pipeline.score(X, y)
    with pytest.raises(SpectroChemPyError, match="transform is not available"):
        estimator_pipeline.transform(X)
    with pytest.raises(SpectroChemPyError, match="score requires y"):
        estimator_pipeline.score(X)


def test_repeated_predict_does_not_refit_steps(monkeypatch):
    X, y = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=2)),
        ]
    ).fit(X, y)
    fitted_center = pipeline.fitted_named_steps_["center"]

    def fail_fit(X):
        raise AssertionError("fit should not be called")

    monkeypatch.setattr(fitted_center, "fit", fail_fit)

    pipeline.predict(_test_x())
    pipeline.predict(_test_x())


def test_get_params_exposes_template_configuration_only():
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )
    shallow = pipeline.get_params(deep=False)
    deep = pipeline.get_params(deep=True)

    assert shallow == {"steps": pipeline.steps}
    assert deep["center"] is pipeline.named_steps["center"]
    assert deep["pca"] is pipeline.named_steps["pca"]
    assert deep["center__dim"] == "y"
    assert deep["pca__n_components"] == 2

    X, _ = _xy()
    pipeline.fit(X)
    assert deep["center"] is pipeline.named_steps["center"]
    assert "fitted_steps_" not in pipeline.get_params(deep=True)


def test_set_params_nested_effective_and_equal_updates():
    X, _ = _xy()
    center = scp.CenterTransformer(dim="y")
    pca = scp.PCA(n_components=2)
    pipeline = Pipeline(
        [
            ("center", center),
            ("pca", pca),
        ]
    ).fit(X)
    fitted_steps = pipeline.fitted_steps_

    pipeline.set_params(pca__n_components=2)
    assert pipeline.fitted_steps_ is fitted_steps
    assert pipeline.named_steps["center"] is center
    assert pipeline.named_steps["pca"] is pca

    pipeline.set_params(pca__n_components=1)
    assert pipeline.named_steps["center"] is center
    assert pipeline.named_steps["pca"] is not pca
    assert pipeline.named_steps["pca"].n_components == 1
    _assert_unfitted(pipeline)


def test_set_params_nested_update_preserves_unmodified_step_instances():
    center = scp.CenterTransformer(dim="y")
    pca = scp.PCA(n_components=2)
    pipeline = Pipeline([("center", center), ("pca", pca)])

    pipeline.set_params(pca__n_components=1)

    assert pipeline.named_steps["center"] is center
    assert pipeline.named_steps["pca"] is not pca


def test_set_params_whole_step_replacement_and_steps_replacement():
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    )
    new_pca = scp.PCA(n_components=1)

    pipeline.set_params(pca=new_pca)
    assert pipeline.named_steps["pca"] is new_pca

    pipeline.set_params(steps=[("scale", scp.AutoscaleTransformer(dim="y"))])
    assert [name for name, _ in pipeline.steps] == ["scale"]


def test_set_params_identical_step_replacement_is_noop():
    X, _ = _xy()
    pca = scp.PCA(n_components=2)
    pipeline = Pipeline([("pca", pca)]).fit(X)
    original_steps = pipeline.steps
    original_fitted_steps = pipeline.fitted_steps_

    pipeline.set_params(pca=pca)

    assert pipeline.steps is original_steps
    assert pipeline.fitted_steps_ is original_fitted_steps


def test_set_params_equivalent_new_step_replacement_is_effective():
    X, _ = _xy()
    old_pca = scp.PCA(n_components=2)
    new_pca = scp.PCA(n_components=2)
    pipeline = Pipeline([("pca", old_pca)]).fit(X)

    pipeline.set_params(pca=new_pca)

    assert pipeline.named_steps["pca"] is new_pca
    _assert_unfitted(pipeline)


def test_set_params_equivalent_steps_replacement_is_effective():
    X, _ = _xy()
    old_pca = scp.PCA(n_components=2)
    new_pca = scp.PCA(n_components=2)
    pipeline = Pipeline([("pca", old_pca)]).fit(X)

    pipeline.set_params(steps=[("pca", new_pca)])

    assert pipeline.named_steps["pca"] is new_pca
    _assert_unfitted(pipeline)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"missing__dim": "x"}, "Invalid step name"),
        ({"center__missing": "x"}, "Invalid nested parameter"),
        ({"pca__svd_solver": "bad"}, "Invalid nested parameter"),
        ({"unknown": 1}, "Invalid parameter"),
    ],
)
def test_set_params_rejects_invalid_updates_transactionally(params, match):
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    ).fit(X)
    original_steps = pipeline.steps
    original_fitted_steps = pipeline.fitted_steps_

    with pytest.raises(SpectroChemPyError, match=match):
        pipeline.set_params(**params)

    assert pipeline.steps is original_steps
    assert pipeline.fitted_steps_ is original_fitted_steps
    assert pipeline.named_steps["pca"].n_components == 2


def test_set_params_mixed_update_is_transactional_when_later_update_fails():
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    ).fit(X)
    original_steps = pipeline.steps
    original_fitted_steps = pipeline.fitted_steps_

    with pytest.raises(SpectroChemPyError, match="Invalid nested parameter"):
        pipeline.set_params(center__dim="x", pca__svd_solver="bad")

    assert pipeline.steps is original_steps
    assert pipeline.named_steps["center"].dim == "y"
    assert pipeline.fitted_steps_ is original_fitted_steps


def test_set_params_rejects_replacement_that_violates_position():
    X, _ = _xy()
    pipeline = Pipeline(
        [
            ("center", scp.CenterTransformer(dim="y")),
            ("pca", scp.PCA(n_components=2)),
        ]
    ).fit(X)
    original_steps = pipeline.steps
    original_fitted_steps = pipeline.fitted_steps_

    with pytest.raises(SpectroChemPyError, match="not an allowlisted intermediate"):
        pipeline.set_params(center=scp.PCA(n_components=1))

    assert pipeline.steps is original_steps
    assert pipeline.fitted_steps_ is original_fitted_steps


def test_set_params_rejects_invalid_steps_replacement_transactionally():
    X, _ = _xy()
    pipeline = Pipeline([("pca", scp.PCA(n_components=2))]).fit(X)
    original_steps = pipeline.steps
    original_fitted_steps = pipeline.fitted_steps_

    with pytest.raises(SpectroChemPyError, match="at least one"):
        pipeline.set_params(steps=[])

    assert pipeline.steps is original_steps
    assert pipeline.fitted_steps_ is original_fitted_steps


def test_set_params_effective_whole_step_replacement_invalidates_fitted_state():
    X, _ = _xy()
    pipeline = Pipeline([("pca", scp.PCA(n_components=2))]).fit(X)

    pipeline.set_params(pca=scp.PCA(n_components=1))

    _assert_unfitted(pipeline)


def test_scientific_contracts_are_supplied_by_underlying_steps():
    X, _ = _xy()
    X[0, 0] = scp.MASKED
    X_test = _test_x()
    X_test[1, 2] = scp.MASKED
    pipeline = Pipeline([("center", scp.CenterTransformer(dim="y"))]).fit(X)

    transformed = pipeline.transform(X_test)

    assert isinstance(transformed, scp.NDDataset)
    assert transformed.dims == X.dims
    assert transformed.units == X.units
    assert transformed.coordset["x"] == X.coordset["x"]
    assert transformed.coordset["y"] == X.coordset["y"]
    assert np.ma.getmaskarray(transformed.masked_data)[1, 2]
    assert transformed.meta.operator == "Alice"
    assert "CenterTransformer applied" in transformed.history[-1]


def test_train_test_reuses_learned_preprocessing_statistics():
    X, _ = _xy()
    X_test = _test_x()
    pipeline = Pipeline([("center", scp.CenterTransformer(dim="y"))]).fit(X)
    fitted_center = pipeline.fitted_named_steps_["center"]

    transformed = pipeline.transform(X_test)
    expected = X_test.masked_data - fitted_center.mean_

    assert np.allclose(transformed.data, expected)
