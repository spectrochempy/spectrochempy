# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Internal estimator-contract helpers for future pipeline support."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number

import numpy as np

from spectrochempy.utils.exceptions import SpectroChemPyError

_PIPELINE_V1_INTERMEDIATE_TRANSFORMERS = frozenset(
    {
        "spectrochempy.processing.transformation.preprocessing_transformers.CenterTransformer",
        "spectrochempy.processing.transformation.preprocessing_transformers.AutoscaleTransformer",
        "spectrochempy.processing.transformation.preprocessing_transformers.ParetoScaleTransformer",
        "spectrochempy.processing.transformation.preprocessing_transformers.RangeScaleTransformer",
        "spectrochempy.processing.transformation.preprocessing_transformers.RobustScaleTransformer",
        "spectrochempy.processing.transformation.preprocessing_transformers.SNVTransformer",
        "spectrochempy.processing.transformation.preprocessing_transformers.NormalizeTransformer",
        "spectrochempy.processing.transformation.preprocessing_transformers.MSCTransformer",
        "spectrochempy.processing.transformation.preprocessing_transformers.LogTransformer",
    }
)

_PIPELINE_V1_FINAL_TRANSFORMERS = _PIPELINE_V1_INTERMEDIATE_TRANSFORMERS | frozenset(
    {"spectrochempy.analysis.decomposition.pca.PCA"}
)

_PIPELINE_V1_FINAL_ESTIMATORS = frozenset(
    {
        "spectrochempy.analysis.crossdecomposition.pls.PLSRegression",
        "spectrochempy.analysis.curvefitting.linearregression.LSTSQ",
        "spectrochempy.analysis.curvefitting.linearregression.NNLS",
    }
)

PIPELINE_V1_SUPPORTED_ESTIMATORS = (
    _PIPELINE_V1_FINAL_TRANSFORMERS | _PIPELINE_V1_FINAL_ESTIMATORS
)


def _qualified_name(estimator):
    cls = estimator.__class__
    return f"{cls.__module__}.{cls.__name__}"


def is_pipeline_v1_supported(estimator):
    """Return whether *estimator* is in the normative v1 pipeline allowlist."""
    return _qualified_name(estimator) in PIPELINE_V1_SUPPORTED_ESTIMATORS


def pipeline_v1_step_kind(estimator, *, final):
    """Return the normative v1 Pipeline category for *estimator*."""
    qualified = _qualified_name(estimator)
    if qualified in _PIPELINE_V1_INTERMEDIATE_TRANSFORMERS:
        return "transformer" if final else "intermediate"
    if final and qualified in _PIPELINE_V1_FINAL_TRANSFORMERS:
        return "transformer"
    if final and qualified in _PIPELINE_V1_FINAL_ESTIMATORS:
        return "estimator"
    return "unsupported"


def is_fitted(estimator):
    """
    Return the canonical fitted-state predicate for supported estimators.

    The helper is intentionally internal and allowlist-based. It does not infer
    pipeline support from method names alone.
    """
    if not is_pipeline_v1_supported(estimator):
        raise SpectroChemPyError(
            f"{estimator.__class__.__name__} is not supported by the "
            "pipeline v1 estimator contract."
        )
    return bool(getattr(estimator, "_fitted", False))


def clone_unfitted(estimator):
    """
    Reconstruct a supported estimator from its constructor configuration.

    Parameter isolation is deliberately bounded: immutable scalar values are
    reused; NumPy arrays, masked arrays, SpectroChemPy objects, random-state
    objects, and built-in containers are copied; arbitrary objects are reused
    by reference.
    """
    if not is_pipeline_v1_supported(estimator):
        raise SpectroChemPyError(
            f"{estimator.__class__.__name__} is not supported by the "
            "pipeline v1 estimator contract."
        )
    if not hasattr(estimator, "get_params"):
        raise SpectroChemPyError(
            f"{estimator.__class__.__name__} does not expose get_params()."
        )

    params = estimator.get_params(deep=False)
    if not isinstance(params, dict):
        raise SpectroChemPyError(
            f"{estimator.__class__.__name__}.get_params(deep=False) must "
            "return a dictionary."
        )
    cloned_params = {
        name: _clone_constructor_parameter(value) for name, value in params.items()
    }
    try:
        cloned = estimator.__class__(**cloned_params)
    except Exception as exc:
        raise SpectroChemPyError(
            f"Cannot clone {estimator.__class__.__name__} from constructor "
            "parameters."
        ) from exc
    if is_fitted(cloned):
        raise SpectroChemPyError(
            f"Cloning {estimator.__class__.__name__} produced a fitted instance."
        )
    return cloned


def parameter_values_equal(old, new):
    """Return whether two constructor parameter values are effectively equal."""
    if old is new:
        return True
    if old is None or new is None:
        return False
    if _is_immutable_scalar(old) and _is_immutable_scalar(new):
        return old == new
    if _is_spectrochempy_object(old) or _is_spectrochempy_object(new):
        return False
    if isinstance(old, np.ma.MaskedArray) or isinstance(new, np.ma.MaskedArray):
        try:
            return bool(np.ma.allequal(old, new))
        except (TypeError, ValueError):
            return False
    if isinstance(old, np.ndarray) or isinstance(new, np.ndarray):
        try:
            return bool(np.array_equal(old, new, equal_nan=True))
        except (TypeError, ValueError):
            return False
    if isinstance(old, tuple) and isinstance(new, tuple):
        return len(old) == len(new) and all(
            parameter_values_equal(left, right)
            for left, right in zip(old, new, strict=True)
        )
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(
            parameter_values_equal(left, right)
            for left, right in zip(old, new, strict=True)
        )
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        return old.keys() == new.keys() and all(
            parameter_values_equal(old[key], new[key]) for key in old
        )
    return False


def _clone_constructor_parameter(value):
    if _is_immutable_scalar(value):
        return value
    if isinstance(value, np.ma.MaskedArray):
        return value.copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.random.RandomState):
        cloned = np.random.RandomState()
        cloned.set_state(value.get_state())
        return cloned
    if isinstance(value, np.random.Generator):
        bit_generator = value.bit_generator.__class__()
        bit_generator.state = value.bit_generator.state
        return np.random.Generator(bit_generator)
    if _is_spectrochempy_object(value):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_clone_constructor_parameter(item) for item in value)
    if isinstance(value, list):
        return [_clone_constructor_parameter(item) for item in value]
    if isinstance(value, set):
        return {_clone_constructor_parameter(item) for item in value}
    if isinstance(value, Mapping):
        return value.__class__(
            (key, _clone_constructor_parameter(item)) for key, item in value.items()
        )
    return value


def _is_immutable_scalar(value):
    return value is None or isinstance(value, str | bytes | bool | Number)


def _is_spectrochempy_object(value):
    module = getattr(value.__class__, "__module__", "")
    return module.startswith("spectrochempy.core.dataset")
