# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Focused tests for the internal estimator contract used by future pipelines."""

from collections.abc import Mapping
from numbers import Number

import numpy as np
import pytest
import traitlets as tr

import spectrochempy as scp
from spectrochempy.analysis.crossdecomposition.pls import PLSRegression
from spectrochempy.analysis.curvefitting.linearregression import LSTSQ
from spectrochempy.analysis.curvefitting.linearregression import NNLS
from spectrochempy.analysis.decomposition.pca import PCA
from spectrochempy.analysis.decomposition.svd import SVD
from spectrochempy.processing.baselineprocessing.baselineprocessing import Baseline
from spectrochempy.processing.transformation.preprocessing_transformers import (
    AutoscaleTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    CenterTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    LogTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    MSCTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    NormalizeTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    ParetoScaleTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    RangeScaleTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    RobustScaleTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    SNVTransformer,
)
from spectrochempy.utils._estimator import _clone_constructor_parameter
from spectrochempy.utils._estimator import clone_unfitted
from spectrochempy.utils._estimator import is_fitted
from spectrochempy.utils._estimator import pipeline_v1_step_kind
from spectrochempy.utils.exceptions import NotFittedError
from spectrochempy.utils.exceptions import SpectroChemPyError


def _xy():
    x = scp.Coord.arange(6, title="features")
    y = scp.Coord.arange(5, title="samples")
    data = np.arange(30.0).reshape(5, 6) + 1.0
    X = scp.NDDataset(data, coordset=[y, x], units="absorbance")
    target = scp.NDDataset(
        np.column_stack([np.arange(5.0), np.arange(5.0) ** 2 + 1.0]),
        coordset=[scp.Coord.arange(5), scp.Coord.arange(2)],
    )
    return X, target


def _preprocessor_cases():
    ref = np.linspace(1.0, 2.0, 6)
    return [
        CenterTransformer(dim="y"),
        AutoscaleTransformer(dim="y"),
        ParetoScaleTransformer(dim="y"),
        RangeScaleTransformer(dim="y"),
        RobustScaleTransformer(dim="y"),
        SNVTransformer(),
        NormalizeTransformer(method="max", dim="y"),
        MSCTransformer(reference=ref, dim="y"),
        LogTransformer(method="log1p", eps=1.0e-10),
    ]


def _final_transformer_cases():
    return [*_preprocessor_cases(), PCA(n_components=2)]


def _final_estimator_cases():
    return [
        PLSRegression(n_components=2),
        LSTSQ(),
        NNLS(),
    ]


def _fit_estimator(estimator, X, Y):
    if isinstance(estimator, PLSRegression | LSTSQ | NNLS):
        return estimator.fit(X, Y[:, 0])
    return estimator.fit(X)


def _category_method(estimator):
    if isinstance(estimator, PLSRegression | LSTSQ | NNLS):
        return "predict"
    return "transform"


def _assert_random_state_equal(left, right):
    left_state = left.get_state()
    right_state = right.get_state()
    assert left_state[0] == right_state[0]
    assert np.array_equal(left_state[1], right_state[1])
    assert left_state[2:] == right_state[2:]


def _assert_cloned_parameter_matches_policy(original, cloned):
    if original is None or isinstance(original, str | bytes | bool | Number):
        assert cloned == original
        return
    if isinstance(original, np.random.RandomState):
        assert cloned is not original
        _assert_random_state_equal(cloned, original)
        return
    if isinstance(original, np.ma.MaskedArray):
        assert cloned is not original
        assert np.ma.allequal(cloned, original)
        return
    if isinstance(original, np.ndarray):
        assert cloned is not original
        assert np.array_equal(cloned, original, equal_nan=True)
        return
    if isinstance(original, tuple):
        assert cloned is not original
        assert type(cloned) is type(original)
        for original_item, cloned_item in zip(original, cloned, strict=True):
            _assert_cloned_parameter_matches_policy(original_item, cloned_item)
        return
    if isinstance(original, list):
        assert cloned is not original
        assert type(cloned) is type(original)
        for original_item, cloned_item in zip(original, cloned, strict=True):
            _assert_cloned_parameter_matches_policy(original_item, cloned_item)
        return
    if isinstance(original, set):
        assert cloned is not original
        assert type(cloned) is type(original)
        assert cloned == original
        return
    if isinstance(original, Mapping):
        assert cloned is not original
        assert type(cloned) is type(original)
        assert cloned.keys() == original.keys()
        for key, original_item in original.items():
            _assert_cloned_parameter_matches_policy(original_item, cloned[key])
        return
    if original.__class__.__module__.startswith("spectrochempy.core.dataset"):
        assert cloned is not original
        assert cloned == original
        return
    assert cloned is original


def _assert_pipeline_templates_cloned(original, cloned):
    assert cloned is not original
    assert cloned._fitted is False
    assert [name for name, _ in cloned.steps] == [name for name, _ in original.steps]
    for (original_name, original_step), (cloned_name, cloned_step) in zip(
        original.steps, cloned.steps, strict=True
    ):
        assert cloned_name == original_name
        assert type(cloned_step) is type(original_step)
        assert cloned_step is not original_step
        original_params = original_step.get_params(deep=False)
        cloned_params = cloned_step.get_params(deep=False)
        assert cloned_params.keys() == original_params.keys()
        for name, original_value in original_params.items():
            _assert_cloned_parameter_matches_policy(original_value, cloned_params[name])
        assert is_fitted(cloned_step) is False
    with pytest.raises(NotFittedError):
        _ = cloned.fitted_steps_


@pytest.mark.parametrize(
    "estimator", _final_transformer_cases() + _final_estimator_cases()
)
def test_is_fitted_tracks_allowlisted_estimator_lifecycle(estimator):
    X, Y = _xy()
    assert is_fitted(estimator) is False

    result = _fit_estimator(estimator, X, Y)

    assert result is estimator
    assert is_fitted(estimator) is True


@pytest.mark.parametrize(
    "estimator", _final_transformer_cases() + _final_estimator_cases()
)
def test_clone_unfitted_reconstructs_configuration_without_learned_state(estimator):
    X, Y = _xy()
    _fit_estimator(estimator, X, Y)

    cloned = clone_unfitted(estimator)

    assert cloned is not estimator
    assert type(cloned) is type(estimator)
    original_params = estimator.get_params(deep=False)
    cloned_params = cloned.get_params(deep=False)
    assert cloned_params.keys() == original_params.keys()
    for name, original_value in original_params.items():
        _assert_cloned_parameter_matches_policy(original_value, cloned_params[name])
    assert is_fitted(cloned) is False
    if isinstance(estimator, PCA):
        assert not hasattr(cloned._pca, "components_")
    elif isinstance(estimator, PLSRegression):
        assert not hasattr(cloned._plsregression, "x_weights_")
    elif isinstance(estimator, LSTSQ | NNLS):
        assert not hasattr(cloned._linear_regression, "coef_")
    elif hasattr(estimator, "_learned_attributes"):
        for attr in estimator._learned_attributes:
            assert not hasattr(cloned, attr)


def test_clone_unfitted_copies_mutable_array_parameters():
    reference = np.linspace(1.0, 2.0, 6)
    transformer = MSCTransformer(reference=reference, dim="y")

    cloned = clone_unfitted(transformer)

    assert cloned.reference is not transformer.reference
    assert np.array_equal(cloned.reference, transformer.reference)
    cloned.reference[0] = 99.0
    assert transformer.reference[0] != 99.0


def test_clone_unfitted_copies_spectrochempy_parameters():
    X, _ = _xy()
    reference = X[0].copy()
    transformer = MSCTransformer(reference=reference, dim="y")

    cloned = clone_unfitted(transformer)

    assert cloned.reference is not transformer.reference
    assert cloned.reference == transformer.reference


def test_clone_unfitted_copies_random_state_without_sharing_state():
    state = np.random.RandomState(123)
    pca = PCA(n_components=2, random_state=state)
    state.rand()

    cloned = clone_unfitted(pca)

    assert cloned.random_state is not pca.random_state
    _assert_random_state_equal(cloned.random_state, pca.random_state)
    cloned.random_state.rand()
    assert cloned.random_state.get_state()[2] != pca.random_state.get_state()[2]


def test_clone_unfitted_recursively_copies_container_parameters():
    reference = {
        "array": np.arange(3.0),
        "nested": [np.ma.array([1.0, 2.0], mask=[False, True])],
    }
    transformer = MSCTransformer(reference=reference, dim="y")

    cloned = clone_unfitted(transformer)

    _assert_cloned_parameter_matches_policy(transformer.reference, cloned.reference)
    cloned.reference["array"][0] = 99.0
    cloned.reference["nested"][0][0] = 42.0
    assert transformer.reference["array"][0] != 99.0
    assert transformer.reference["nested"][0][0] != 42.0


def test_clone_unfitted_reconstructs_pipeline_templates():
    center = CenterTransformer(dim="y")
    pls = PLSRegression(n_components=2, scale=False)
    pipeline = scp.Pipeline([("center", center), ("pls", pls)])

    cloned = clone_unfitted(pipeline)

    _assert_pipeline_templates_cloned(pipeline, cloned)
    assert pipeline.steps == (("center", center), ("pls", pls))
    assert pipeline._fitted is False


def test_pipeline_cloning_does_not_expand_the_fitted_estimator_allowlist():
    pipeline = scp.Pipeline(
        [
            ("center", CenterTransformer(dim="y")),
            ("pls", PLSRegression(n_components=1)),
        ]
    )

    cloned = clone_unfitted(pipeline)

    with pytest.raises(SpectroChemPyError, match="not supported"):
        is_fitted(pipeline)
    with pytest.raises(SpectroChemPyError, match="not supported"):
        is_fitted(cloned)


def test_clone_unfitted_discards_fitted_pipeline_state_without_mutating_original():
    X, y = _xy()
    pipeline = scp.Pipeline(
        [
            ("center", CenterTransformer(dim="y")),
            ("pls", PLSRegression(n_components=1, scale=False)),
        ]
    ).fit(X, y)
    original_templates = pipeline.steps
    original_fitted_steps = pipeline.fitted_steps_
    original_mean = pipeline.fitted_named_steps_["center"].mean_.copy()
    original_coef = pipeline.fitted_named_steps_["pls"]._coef.copy()

    cloned = clone_unfitted(pipeline)

    _assert_pipeline_templates_cloned(pipeline, cloned)
    assert pipeline._fitted is True
    assert pipeline.steps is original_templates
    assert pipeline.fitted_steps_ is original_fitted_steps
    assert np.array_equal(pipeline.fitted_named_steps_["center"].mean_, original_mean)
    assert np.array_equal(pipeline.fitted_named_steps_["pls"]._coef, original_coef)


def test_clone_unfitted_discards_learned_state_from_fitted_templates():
    X, y = _xy()
    center = CenterTransformer(dim="y").fit(X)
    pls = PLSRegression(n_components=1, scale=False).fit(X, y)
    original_mean = center.mean_.copy()
    original_coef = pls._coef.copy()
    pipeline = scp.Pipeline([("center", center), ("pls", pls)])

    cloned = clone_unfitted(pipeline)

    _assert_pipeline_templates_cloned(pipeline, cloned)
    assert is_fitted(center) is True
    assert is_fitted(pls) is True
    assert np.array_equal(center.mean_, original_mean)
    assert np.array_equal(pls._coef, original_coef)
    assert not hasattr(cloned.named_steps["center"], "mean_")
    assert not hasattr(cloned.named_steps["pls"]._plsregression, "x_weights_")


def test_pipeline_clones_fit_independently_on_different_calibrations():
    X, y = _xy()
    shifted = X.copy()
    shifted.data = X.data + 10.0
    template = scp.Pipeline(
        [
            ("center", CenterTransformer(dim="y")),
            ("pls", PLSRegression(n_components=1, scale=False)),
        ]
    )
    first = clone_unfitted(template).fit(X, y)
    first_mean = first.fitted_named_steps_["center"].mean_.copy()
    first_coef = first.fitted_named_steps_["pls"]._coef.copy()

    second = clone_unfitted(template).fit(shifted, y)

    assert template._fitted is False
    assert first._fitted is True
    assert second._fitted is True
    assert not np.array_equal(
        first.fitted_named_steps_["center"].mean_,
        second.fitted_named_steps_["center"].mean_,
    )
    assert np.array_equal(first.fitted_named_steps_["center"].mean_, first_mean)
    assert np.array_equal(first.fitted_named_steps_["pls"]._coef, first_coef)
    assert (
        first.fitted_named_steps_["center"] is not second.fitted_named_steps_["center"]
    )
    assert first.fitted_named_steps_["pls"] is not second.fitted_named_steps_["pls"]
    assert not np.shares_memory(
        first.fitted_named_steps_["center"].mean_,
        second.fitted_named_steps_["center"].mean_,
    )
    assert not np.shares_memory(
        first.fitted_named_steps_["pls"]._coef,
        second.fitted_named_steps_["pls"]._coef,
    )


def test_pipeline_clones_isolate_supported_mutable_parameters():
    reference = np.linspace(1.0, 2.0, 6)
    pipeline = scp.Pipeline(
        [
            ("msc", MSCTransformer(reference=reference, dim="y")),
            ("pls", PLSRegression(n_components=2, scale=False)),
        ]
    )

    first = clone_unfitted(pipeline)
    second = clone_unfitted(pipeline)

    original_reference = pipeline.named_steps["msc"].reference
    first_reference = first.named_steps["msc"].reference
    second_reference = second.named_steps["msc"].reference
    assert first_reference is not original_reference
    assert second_reference is not original_reference
    assert first_reference is not second_reference
    first_reference[0] = 99.0
    assert original_reference[0] != 99.0
    assert second_reference[0] != 99.0


def test_pipeline_clone_failure_reports_step_context_and_preserves_cause(
    monkeypatch,
):
    center = CenterTransformer(dim="y")
    pipeline = scp.Pipeline(
        [("center", center), ("pls", PLSRegression(n_components=2))]
    )

    def fail_get_params(*, deep=True):
        raise ValueError("broken step parameters")

    monkeypatch.setattr(center, "get_params", fail_get_params)

    with pytest.raises(
        SpectroChemPyError,
        match=(
            "Cannot clone Pipeline step 'center' at position 0 "
            "\\(class CenterTransformer\\)"
        ),
    ) as excinfo:
        clone_unfitted(pipeline)

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert str(excinfo.value.__cause__) == "broken step parameters"
    assert pipeline._fitted is False
    assert pipeline.named_steps["center"] is center


def test_constructor_parameter_clone_preserves_generator_position():
    generator = np.random.default_rng(123)
    generator.random(4)

    cloned = _clone_constructor_parameter(generator)

    assert cloned is not generator
    assert cloned.bit_generator.state == generator.bit_generator.state
    assert cloned.random() == generator.random()


@pytest.mark.parametrize("unsupported", [SVD(), Baseline()])
def test_clone_and_fitted_helpers_reject_unsupported_candidates(unsupported):
    with pytest.raises(SpectroChemPyError, match="not supported"):
        clone_unfitted(unsupported)
    with pytest.raises(SpectroChemPyError, match="not supported"):
        is_fitted(unsupported)


@pytest.mark.parametrize("transformer", _preprocessor_cases())
def test_pipeline_v1_step_kind_classifies_preprocessors(transformer):
    assert pipeline_v1_step_kind(transformer, final=False) == "intermediate"
    assert pipeline_v1_step_kind(transformer, final=True) == "transformer"


def test_pipeline_v1_step_kind_classifies_terminal_only_candidates():
    assert pipeline_v1_step_kind(PCA(n_components=2), final=False) == "unsupported"
    assert pipeline_v1_step_kind(PCA(n_components=2), final=True) == "transformer"
    assert (
        pipeline_v1_step_kind(PLSRegression(n_components=2), final=False)
        == "unsupported"
    )
    assert (
        pipeline_v1_step_kind(PLSRegression(n_components=2), final=True) == "estimator"
    )
    assert pipeline_v1_step_kind(SVD(), final=True) == "unsupported"


@pytest.mark.parametrize(
    "estimator", _final_transformer_cases() + _final_estimator_cases()
)
def test_allowlisted_methods_raise_canonical_not_fitted_error(estimator):
    X, Y = _xy()
    method = _category_method(estimator)
    args = (X,) if method == "predict" else (X,)

    with pytest.raises(NotFittedError):
        getattr(estimator, method)(*args)

    _fit_estimator(estimator, X, Y)
    output = estimator.predict(X) if method == "predict" else estimator.transform(X)
    assert isinstance(output, scp.NDDataset)


@pytest.mark.parametrize("estimator", _final_estimator_cases())
def test_allowlisted_score_raises_canonical_not_fitted_error(estimator):
    X, Y = _xy()

    with pytest.raises(NotFittedError):
        estimator.score(X, Y[:, 0])

    _fit_estimator(estimator, X, Y)
    assert isinstance(estimator.score(X, Y[:, 0]), float)


@pytest.mark.parametrize(
    "estimator",
    [PCA(n_components=2), PLSRegression(n_components=2), LSTSQ(), NNLS()],
)
def test_analysis_set_params_effective_change_invalidates_fitted_state(estimator):
    X, Y = _xy()
    _fit_estimator(estimator, X, Y)
    assert is_fitted(estimator)

    params = estimator.get_params(deep=False)
    if "n_components" in params:
        estimator.set_params(n_components=1)
    else:
        estimator.set_params(fit_intercept=not params["fit_intercept"])

    assert is_fitted(estimator) is False
    with pytest.raises(NotFittedError):
        getattr(estimator, _category_method(estimator))(X)
    if isinstance(estimator, PLSRegression | LSTSQ | NNLS):
        with pytest.raises(NotFittedError):
            estimator.score(X, Y[:, 0])


@pytest.mark.parametrize(
    "estimator",
    [PCA(n_components=2), PLSRegression(n_components=2), LSTSQ(), NNLS()],
)
def test_analysis_set_params_equal_update_preserves_fitted_state(estimator):
    X, Y = _xy()
    _fit_estimator(estimator, X, Y)

    estimator.set_params(**estimator.get_params(deep=False))

    assert is_fitted(estimator) is True


def test_analysis_set_params_invalid_name_is_transactional():
    X, Y = _xy()
    pca = PCA(n_components=2).fit(X)

    with pytest.raises(SpectroChemPyError, match="Invalid parameter"):
        pca.set_params(n_components=1, invalid_parameter=1)

    assert pca.n_components == 2
    assert is_fitted(pca) is True


def test_analysis_set_params_invalid_value_is_transactional():
    X, _ = _xy()
    pca = PCA(n_components=2).fit(X)

    with pytest.raises(tr.TraitError):
        pca.set_params(n_components=1, svd_solver="bad")

    assert pca.n_components == 2
    assert pca.svd_solver == "auto"
    assert is_fitted(pca) is True


@pytest.mark.parametrize(
    "estimator",
    [PCA(n_components=2), PLSRegression(n_components=2), LSTSQ(), NNLS()],
)
def test_failed_initial_fit_leaves_allowlisted_analysis_unfitted(estimator):
    X, Y = _xy()

    if isinstance(estimator, PCA):
        estimator.n_components = 99
        with pytest.raises(ValueError):
            estimator.fit(X)
    elif isinstance(estimator, PLSRegression):
        estimator.n_components = 99
        with pytest.raises(ValueError):
            estimator.fit(X, Y[:, 0])
    else:
        with pytest.raises(ValueError):
            estimator.fit(scp.NDDataset(np.arange(4.0)))

    assert is_fitted(estimator) is False


@pytest.mark.parametrize(
    "estimator",
    [PCA(n_components=2), PLSRegression(n_components=2), LSTSQ(), NNLS()],
)
def test_failed_refit_clears_allowlisted_analysis_state(estimator):
    X, Y = _xy()
    _fit_estimator(estimator, X, Y)
    assert is_fitted(estimator) is True

    if isinstance(estimator, PCA):
        estimator.n_components = 99
        with pytest.raises(ValueError):
            estimator.fit(X)
    elif isinstance(estimator, PLSRegression):
        estimator.n_components = 99
        with pytest.raises(ValueError):
            estimator.fit(X, Y[:, 0])
    else:
        with pytest.raises(ValueError):
            estimator.fit(scp.NDDataset(np.arange(4.0)))

    assert is_fitted(estimator) is False
    with pytest.raises(NotFittedError):
        getattr(estimator, _category_method(estimator))(X)
    if isinstance(estimator, PLSRegression | LSTSQ | NNLS):
        with pytest.raises(NotFittedError):
            estimator.score(X, Y[:, 0])


def test_svd_is_characterized_but_excluded_because_transform_is_not_implemented():
    X, _ = _xy()
    svd = SVD().fit(X)

    assert svd._fitted is True
    with pytest.raises(NotImplementedError):
        svd.transform(X)
    with pytest.raises(SpectroChemPyError, match="not supported"):
        clone_unfitted(svd)
