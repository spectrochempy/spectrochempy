# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Focused tests for the internal estimator contract used by future pipelines."""

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
from spectrochempy.utils._estimator import clone_unfitted
from spectrochempy.utils._estimator import is_fitted
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


@pytest.mark.parametrize("estimator", _final_transformer_cases() + _final_estimator_cases())
def test_is_fitted_tracks_allowlisted_estimator_lifecycle(estimator):
    X, Y = _xy()
    assert is_fitted(estimator) is False

    result = _fit_estimator(estimator, X, Y)

    assert result is estimator
    assert is_fitted(estimator) is True


@pytest.mark.parametrize("estimator", _final_transformer_cases() + _final_estimator_cases())
def test_clone_unfitted_reconstructs_configuration_without_learned_state(estimator):
    X, Y = _xy()
    _fit_estimator(estimator, X, Y)

    cloned = clone_unfitted(estimator)

    assert cloned is not estimator
    assert type(cloned) is type(estimator)
    assert cloned.get_params(deep=False).keys() == estimator.get_params(deep=False).keys()
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

    cloned = clone_unfitted(pca)

    assert cloned.random_state is not pca.random_state
    assert cloned.random_state.get_state()[1].tolist() == pca.random_state.get_state()[
        1
    ].tolist()
    cloned.random_state.rand()
    assert cloned.random_state.get_state()[1].tolist() != pca.random_state.get_state()[
        1
    ].tolist()


@pytest.mark.parametrize("unsupported", [SVD(), Baseline()])
def test_clone_and_fitted_helpers_reject_unsupported_candidates(unsupported):
    with pytest.raises(SpectroChemPyError, match="not supported"):
        clone_unfitted(unsupported)
    with pytest.raises(SpectroChemPyError, match="not supported"):
        is_fitted(unsupported)


@pytest.mark.parametrize("estimator", _final_transformer_cases() + _final_estimator_cases())
def test_allowlisted_methods_raise_canonical_not_fitted_error(estimator):
    X, Y = _xy()
    method = _category_method(estimator)
    args = (X,) if method == "predict" else (X,)

    with pytest.raises(NotFittedError):
        getattr(estimator, method)(*args)

    _fit_estimator(estimator, X, Y)
    output = estimator.predict(X) if method == "predict" else estimator.transform(X)
    assert isinstance(output, scp.NDDataset)


@pytest.mark.parametrize("estimator", [PCA(n_components=2), PLSRegression(n_components=2), LSTSQ(), NNLS()])
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


@pytest.mark.parametrize("estimator", [PCA(n_components=2), PLSRegression(n_components=2), LSTSQ(), NNLS()])
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


@pytest.mark.parametrize("estimator", [PCA(n_components=2), PLSRegression(n_components=2), LSTSQ(), NNLS()])
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


@pytest.mark.parametrize("estimator", [PCA(n_components=2), PLSRegression(n_components=2), LSTSQ(), NNLS()])
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


def test_svd_is_characterized_but_excluded_because_transform_is_not_implemented():
    X, _ = _xy()
    svd = SVD().fit(X)

    assert svd._fitted is True
    with pytest.raises(NotImplementedError):
        svd.transform(X)
    with pytest.raises(SpectroChemPyError, match="not supported"):
        clone_unfitted(svd)
