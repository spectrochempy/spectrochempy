# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Internal dataset preparation helpers for supervised cross-validation."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from numbers import Number
from types import MappingProxyType

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils._estimator import clone_unfitted
from spectrochempy.utils.exceptions import SpectroChemPyError

__all__ = ["CrossValidationResult", "cross_validate"]


@dataclass(frozen=True)
class _SampleGeometry:
    """Resolved observation geometry for an aligned X/y pair."""

    sample_dim: str
    x_sample_axis: int
    y_sample_axis: int
    n_observations: int


@dataclass(frozen=True)
class _TargetGeometry:
    """Original geometry of one validation target subset."""

    template: NDDataset
    sample_dim: str
    sample_axis: int


@dataclass(frozen=True)
class _FoldSubsets:
    """Row-oriented datasets and geometry needed for one CV fold."""

    X_train: NDDataset
    X_validation: NDDataset
    y_train: NDDataset
    y_validation: NDDataset
    target_geometry: _TargetGeometry
    train_positions: np.ndarray
    validation_positions: np.ndarray
    sample_geometry: _SampleGeometry


@dataclass(frozen=True)
class CrossValidationMetric:
    """Per-target values and explicit definedness for one CV metric."""

    name: str
    values: NDDataset
    defined: tuple[bool, ...]
    reasons: tuple[str | None, ...]


@dataclass(frozen=True)
class _RegressionMetricResult:
    """Private metric outputs with target identity and validity metadata."""

    residuals: NDDataset
    n_valid: NDDataset
    metrics: tuple[CrossValidationMetric, ...]

    def metric(self, name):
        """Return one requested metric by its internal kernel name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


@dataclass(frozen=True)
class _PipelineStepConfiguration:
    """Non-fitted configuration snapshot for one Pipeline template step."""

    name: str
    class_name: str
    parameters: Mapping


@dataclass(frozen=True)
class _EstimatorConfiguration:
    """Non-executable estimator configuration snapshot."""

    class_name: str
    parameters: Mapping
    steps: tuple[_PipelineStepConfiguration, ...] = ()


@dataclass(frozen=True)
class _SplitterConfiguration:
    """Non-executable splitter configuration snapshot."""

    class_name: str
    parameters: Mapping


@dataclass(frozen=True)
class _GroupCount:
    """One group identity and its observation count."""

    value: object
    count: int


@dataclass(frozen=True)
class _GroupsSummary:
    """Compact group summary without retaining the caller's array."""

    n_groups: int
    counts: tuple[_GroupCount, ...]


@dataclass(frozen=True)
class _DatasetDescription:
    """Minimal, non-executable description of one validation input."""

    role: str
    name: str | None
    title: str | None
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    units: str | None


@dataclass(frozen=True)
class _ValidationDescription:
    """Minimal operation description, deliberately not full provenance."""

    operation: str
    inputs: tuple[_DatasetDescription, ...]
    note: str


@dataclass(frozen=True)
class _UndefinedMetric:
    """One explicitly undefined metric associated with a target and scope."""

    scope: str
    metric: str
    target_index: int
    reason: str
    fold_index: int | None = None


@dataclass(frozen=True)
class CrossValidationFoldResult:
    """Validated fold positions and per-target metric records."""

    fold_index: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    metrics: tuple[CrossValidationMetric, ...]
    n_valid: NDDataset

    def metric(self, name):
        """Return one fold metric by its generic kernel name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


@dataclass(frozen=True)
class CrossValidationResult:
    """
    Structured result returned by :func:`cross_validate`.

    This frozen record owns isolated copies of its dataset inputs, fold
    positions, configuration snapshots, and tracking containers. Its
    `NDDataset` members remain mutable, so the result is not deeply immutable.
    It is not persistent and does not provide replay or automatic provenance.

    Parameters
    ----------
    estimator : object
        Non-executable estimator configuration snapshot. It exposes
        ``class_name``, ``parameters``, and, for a Pipeline, ordered ``steps``.
    splitter : object
        Non-executable splitter snapshot exposing ``class_name`` and
        ``parameters``.
    n_splits : int
        Number of validated folds.
    sample_dim : str
        Observation dimension name shared by X and y.
    x_sample_axis, y_sample_axis : int
        Original observation-axis positions in X and y.
    observation_coordinate : Coord or None
        Isolated observation coordinate when one is available.
    groups : object or None
        Compact group summary exposing ``n_groups`` and ordered ``counts``.
    observed, oof_predictions, residuals : NDDataset
        Isolated target datasets in the original y geometry and units.
    global_metrics : tuple of CrossValidationMetric
        Requested global metrics in caller order. Prefer :meth:`metric` for
        lookup by name.
    n_valid : NDDataset
        Number of finite, unmasked observed/predicted pairs per target.
    folds : tuple of CrossValidationFoldResult
        Fold records in splitter order. Each record exposes ``fold_index``,
        read-only ``train_positions`` and ``validation_positions``, ``n_valid``,
        and :meth:`CrossValidationFoldResult.metric`.
    warnings : tuple of str
        Validation warnings. V1 normally returns an empty tuple.
    undefined_metrics : tuple
        Explicit records exposing metric scope, target, reason, and optional
        fold index whenever a requested metric is undefined.
    fold_estimators : tuple or None
        Independent fitted fold estimators only when requested. Retaining them
        can substantially increase memory use because each estimator is copied.
    operation : object
        Minimal non-executable operation description. It is not provenance or
        a replay specification.

    See Also
    --------
    cross_validate : Evaluate a fixed supervised regression configuration.

    Examples
    --------
    >>> values = np.arange(24.0).reshape(8, 3)
    >>> X = scp.NDDataset(values, dims=["y", "x"])
    >>> y = scp.NDDataset((1.0 + values[:, 0])[:, None], dims=["y", "t"])
    >>> model = scp.PLSRegression(n_components=1, scale=False)
    >>> result = scp.cross_validate(model, X, y, cv=4)
    >>> isinstance(result, scp.CrossValidationResult)
    True
    """

    estimator: _EstimatorConfiguration
    splitter: _SplitterConfiguration
    n_splits: int
    sample_dim: str
    x_sample_axis: int
    y_sample_axis: int
    observation_coordinate: object
    groups: _GroupsSummary | None
    observed: NDDataset
    oof_predictions: NDDataset
    residuals: NDDataset
    global_metrics: tuple[CrossValidationMetric, ...]
    n_valid: NDDataset
    folds: tuple[CrossValidationFoldResult, ...]
    warnings: tuple[str, ...]
    undefined_metrics: tuple[_UndefinedMetric, ...]
    fold_estimators: tuple[object, ...] | None
    operation: _ValidationDescription

    def metric(self, name):
        """
        Return a requested global metric by its public name.

        Parameters
        ----------
        name : {"rmsecv", "r2", "bias", "mae"}
            Metric selected in the original :func:`cross_validate` call.

        Returns
        -------
        CrossValidationMetric
            Frozen record containing ``name``, an `NDDataset` in target
            geometry under ``values``, per-target boolean ``defined`` flags,
            and matching reason codes under ``reasons``. RMSECV, bias, and MAE
            carry target units; R² is unitless.

        Raises
        ------
        KeyError
            If *name* was not requested.
        """
        for metric in self.global_metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


@dataclass(frozen=True)
class _CrossValidationPlan:
    """Validated, materialized execution plan shared by engine and builder."""

    geometry: _SampleGeometry
    metric_names: tuple[str, ...]
    estimator_configuration: _EstimatorConfiguration
    splitter_configuration: _SplitterConfiguration
    groups_values: np.ndarray | None
    groups_summary: _GroupsSummary | None
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]


_REGRESSION_METRICS = ("rmse", "r2", "bias", "mae")
_NO_OBSERVATIONS = "no_observations"
_INVALID_PREDICTION = "invalid_prediction"
_INSUFFICIENT_VALID_PAIRS = "insufficient_valid_pairs"
_ZERO_OBSERVED_VARIANCE = "zero_observed_variance"
_NON_FINITE_RESULT = "non_finite_result"


def _coordinate_for_dimension(dataset, dim):
    if dataset.coordset is None:
        return None
    coordinate = dataset.coordset[dim]
    if isinstance(coordinate, CoordSet):
        raise SpectroChemPyError(
            f"Dimension {dim!r} uses multiple coordinates; cross-validation "
            "alignment currently requires a single coordinate."
        )
    if coordinate is None or (not coordinate.has_data and not coordinate.is_labeled):
        return None
    return coordinate


def _ordered_values_equal(left, right, *, converted=False):
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape:
        return False
    if not converted:
        return np.array_equal(left_values, right_values)
    if not (
        np.issubdtype(left_values.dtype, np.number)
        and np.issubdtype(right_values.dtype, np.number)
    ):
        return np.array_equal(left_values, right_values)

    precisions = [np.finfo(float).eps]
    for values in (left_values, right_values):
        if np.issubdtype(values.dtype, np.inexact):
            precisions.append(np.finfo(values.dtype).eps)
    scale = max(
        1.0,
        float(np.max(np.abs(left_values))),
        float(np.max(np.abs(right_values))),
    )
    tolerance = 8.0 * max(precisions) * scale
    return np.allclose(left_values, right_values, rtol=0.0, atol=tolerance)


def _validate_coordinate_alignment(left, right, *, context, allow_missing=True):
    """Validate shared coordinate information without changing either side."""
    if left is None and right is None:
        return
    if left is None or right is None:
        if allow_missing:
            # A missing observation coordinate leaves positional alignment as the only
            # available identity. Sample-count validation is performed by the caller.
            return
        raise SpectroChemPyError(
            f"{context} coordinates must be defined on both datasets or neither."
        )

    if left.has_data != right.has_data:
        raise SpectroChemPyError(
            f"{context} coordinates do not carry comparable values on both datasets."
        )

    if left.has_data:
        if left.has_units != right.has_units:
            raise SpectroChemPyError(
                f"{context} coordinate units must be defined on both datasets or neither."
            )
        if left.has_units:
            if not left.is_units_compatible(right):
                raise SpectroChemPyError(
                    f"{context} coordinate units are not compatible."
                )
            converted = right.units != left.units
            right_values = (
                right.to(left.units, inplace=False).data if converted else right.data
            )
        else:
            converted = False
            right_values = right.data
        if not _ordered_values_equal(left.data, right_values, converted=converted):
            raise SpectroChemPyError(
                f"{context} coordinate values or ordering do not match."
            )

    if (
        left.is_labeled
        and right.is_labeled
        and not np.array_equal(left.labels, right.labels)
    ):
        raise SpectroChemPyError(
            f"{context} coordinate labels or ordering do not match."
        )
    if not left.has_data and left.is_labeled != right.is_labeled:
        raise SpectroChemPyError(
            f"{context} coordinates do not carry comparable labels on both datasets."
        )


def _resolve_sample_geometry(X, y, *, sample_dim="y"):
    """Resolve and validate observation axes for a supervised X/y pair."""
    if not isinstance(X, NDDataset) or not isinstance(y, NDDataset):
        raise SpectroChemPyError("X and y must both be NDDataset instances.")
    if X.ndim != 2:
        raise SpectroChemPyError("X must be a two-dimensional NDDataset.")
    if y.ndim not in (1, 2):
        raise SpectroChemPyError("y must be a one- or two-dimensional NDDataset.")
    if not isinstance(sample_dim, str):
        raise SpectroChemPyError("sample_dim must be a dimension name.")
    if X.dims.count(sample_dim) == 0:
        raise SpectroChemPyError(f"sample_dim {sample_dim!r} is not present in X.")
    if y.dims.count(sample_dim) == 0:
        raise SpectroChemPyError(f"sample_dim {sample_dim!r} is not present in y.")
    if X.dims.count(sample_dim) != 1 or y.dims.count(sample_dim) != 1:
        raise SpectroChemPyError(
            f"sample_dim {sample_dim!r} must identify exactly one axis in X and y."
        )

    x_sample_axis = X.dims.index(sample_dim)
    y_sample_axis = y.dims.index(sample_dim)
    n_observations = X.shape[x_sample_axis]
    if y.shape[y_sample_axis] != n_observations:
        raise SpectroChemPyError(
            "X and y must contain the same number of observations."
        )

    _validate_coordinate_alignment(
        _coordinate_for_dimension(X, sample_dim),
        _coordinate_for_dimension(y, sample_dim),
        context="Observation",
    )
    return _SampleGeometry(
        sample_dim=sample_dim,
        x_sample_axis=x_sample_axis,
        y_sample_axis=y_sample_axis,
        n_observations=n_observations,
    )


def _validate_fold_positions(positions, *, n_observations, name):
    """Return validated one-dimensional, non-repeating CV positions."""
    values = np.asarray(positions)
    if values.ndim != 1:
        raise SpectroChemPyError(f"{name} positions must be one-dimensional.")
    if values.size == 0:
        raise SpectroChemPyError(f"{name} positions must not be empty.")
    if not np.issubdtype(values.dtype, np.integer):
        raise SpectroChemPyError(f"{name} positions must be integers.")
    if np.any(values < 0) or np.any(values >= n_observations):
        raise SpectroChemPyError(f"{name} positions are outside the dataset bounds.")
    if np.unique(values).size != values.size:
        raise SpectroChemPyError(f"{name} positions must not contain duplicates.")
    return values.astype(np.intp, copy=True)


def _slice_along_axis(dataset, positions, axis):
    index = [slice(None)] * dataset.ndim
    index[axis] = positions
    return dataset[tuple(index)].copy()


def _orient_observations_as_rows(dataset, sample_axis):
    if sample_axis == 0:
        return dataset.copy()
    return dataset.swapdims(sample_axis, 0)


def _prepare_fold_subsets(
    X,
    y,
    train_positions,
    validation_positions,
    *,
    sample_dim="y",
):
    """Slice aligned fold subsets and orient their observations as rows."""
    geometry = _resolve_sample_geometry(X, y, sample_dim=sample_dim)
    train = _validate_fold_positions(
        train_positions,
        n_observations=geometry.n_observations,
        name="Training",
    )
    validation = _validate_fold_positions(
        validation_positions,
        n_observations=geometry.n_observations,
        name="Validation",
    )
    if np.intersect1d(train, validation).size:
        raise SpectroChemPyError(
            "Training and validation positions must be disjoint within a fold."
        )

    return _prepare_validated_fold_subsets(X, y, train, validation, geometry)


def _prepare_validated_fold_subsets(
    X,
    y,
    train,
    validation,
    geometry,
):
    """Slice one fold whose geometry and positions were already validated."""

    X_train = _slice_along_axis(X, train, geometry.x_sample_axis)
    X_validation = _slice_along_axis(X, validation, geometry.x_sample_axis)
    y_train = _slice_along_axis(y, train, geometry.y_sample_axis)
    y_validation = _slice_along_axis(y, validation, geometry.y_sample_axis)
    target_geometry = _TargetGeometry(
        template=y_validation.copy(),
        sample_dim=geometry.sample_dim,
        sample_axis=geometry.y_sample_axis,
    )

    return _FoldSubsets(
        X_train=_orient_observations_as_rows(X_train, geometry.x_sample_axis),
        X_validation=_orient_observations_as_rows(X_validation, geometry.x_sample_axis),
        y_train=_orient_observations_as_rows(y_train, geometry.y_sample_axis),
        y_validation=_orient_observations_as_rows(y_validation, geometry.y_sample_axis),
        target_geometry=target_geometry,
        train_positions=train,
        validation_positions=validation,
        sample_geometry=geometry,
    )


def _restore_prediction_geometry(prediction, target_geometry):
    """Restore a row-oriented prediction to its validation target geometry."""
    if not isinstance(prediction, NDDataset):
        raise SpectroChemPyError("A fold prediction must be an NDDataset.")
    if not isinstance(target_geometry, _TargetGeometry):
        raise SpectroChemPyError("target_geometry must come from fold preparation.")

    template = target_geometry.template
    expected = _orient_observations_as_rows(template, target_geometry.sample_axis)
    if prediction.ndim != expected.ndim or prediction.shape != expected.shape:
        raise SpectroChemPyError(
            "Prediction shape does not match the validation target geometry."
        )
    if prediction.dims != expected.dims:
        raise SpectroChemPyError(
            "Prediction dimensions do not match the validation target geometry."
        )

    for dim in expected.dims:
        _validate_coordinate_alignment(
            _coordinate_for_dimension(expected, dim),
            _coordinate_for_dimension(prediction, dim),
            context=f"Prediction {dim!r}",
            allow_missing=dim == target_geometry.sample_dim,
        )

    if prediction.has_units != expected.has_units:
        raise SpectroChemPyError(
            "Prediction and validation target units must both be defined or absent."
        )
    if prediction.has_units:
        if not expected.is_units_compatible(prediction):
            raise SpectroChemPyError(
                "Prediction units are not compatible with validation target units."
            )
        prediction = prediction.to(expected.units, inplace=False)
    else:
        prediction = prediction.copy()

    if target_geometry.sample_axis != 0:
        prediction = prediction.swapdims(0, target_geometry.sample_axis)
    if prediction.shape != template.shape or prediction.dims != template.dims:
        raise SpectroChemPyError(
            "Prediction cannot be restored to the validation target geometry."
        )

    restored = template.copy()
    restored.data = prediction.masked_data.copy()
    return restored


def _stable_mean(values):
    """Return a scaled mean without integer arithmetic or avoidable overflow."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    scale = np.max(np.abs(values))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * (np.sum(values / scale, dtype=float) / values.size)


def _stable_mean_absolute(values):
    """Return a scaled mean absolute value."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    scale = np.max(np.abs(values))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * (np.sum(np.abs(values / scale), dtype=float) / values.size)


def _stable_root_mean_square(values):
    """Return a scaled root mean square."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    scale = np.max(np.abs(values))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = values / scale
        return scale * np.sqrt(np.sum(scaled * scaled, dtype=float) / values.size)


def _stable_sum_of_squares(values):
    """Represent a sum of squares as a scale and a finite scaled sum."""
    values = np.asarray(values, dtype=float)
    scale = np.max(np.abs(values))
    if scale == 0:
        return 0.0, 0.0
    scaled = values / scale
    return scale, np.sum(scaled * scaled, dtype=float)


def _scaled_errors(observed, predicted):
    """Return direct errors, scaling only when subtraction would overflow."""
    observed_array = np.asarray(observed)
    predicted_array = np.asarray(predicted)
    if np.issubdtype(observed_array.dtype, np.integer) and np.issubdtype(
        predicted_array.dtype, np.integer
    ):
        errors = np.fromiter(
            (
                int(predicted_value) - int(observed_value)
                for observed_value, predicted_value in zip(
                    observed_array.flat, predicted_array.flat, strict=True
                )
            ),
            dtype=float,
            count=observed_array.size,
        )
        return errors.reshape(observed_array.shape), 1.0

    observed_float = np.asarray(observed_array, dtype=float)
    predicted_float = np.asarray(predicted_array, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        direct_errors = predicted_float - observed_float
    if np.all(np.isfinite(direct_errors)):
        return direct_errors, 1.0

    scale = max(
        np.max(np.abs(observed_float)),
        np.max(np.abs(predicted_float)),
    )
    if scale == 0:
        return np.zeros(observed_float.shape, dtype=float), 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        return predicted_float / scale - observed_float / scale, scale


def _scaled_centered_observations(observed):
    """Center near the data, scaling only when direct subtraction overflows."""
    observed_array = np.asarray(observed)
    if np.issubdtype(observed_array.dtype, np.integer):
        origin = int(observed_array.flat[0])
        offsets = np.fromiter(
            (int(value) - origin for value in observed_array.flat),
            dtype=float,
            count=observed_array.size,
        ).reshape(observed_array.shape)
        return offsets - _stable_mean(offsets), 1.0

    observed_float = np.asarray(observed_array, dtype=float)
    origin = observed_float.flat[0]
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = observed_float - origin
    if np.all(np.isfinite(offsets)):
        return offsets - _stable_mean(offsets), 1.0

    scale = np.max(np.abs(observed_float))
    if scale == 0:
        return np.zeros(observed_float.shape, dtype=float), 1.0
    scaled = observed_float / scale
    scaled_offsets = scaled - scaled.flat[0]
    return scaled_offsets - _stable_mean(scaled_offsets), scale


def _log_sum_of_squares(values, external_scale):
    """Return log(sum((external_scale * values) ** 2)) without overflow."""
    inner_scale, scaled_sum = _stable_sum_of_squares(values)
    if inner_scale == 0 or scaled_sum == 0:
        return -np.inf
    return 2 * np.log(external_scale) + 2 * np.log(inner_scale) + np.log(scaled_sum)


def _rmse(observed, predicted):
    """Compute generic RMSE for one target from already valid pairs."""
    errors, scale = _scaled_errors(observed, predicted)
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * _stable_root_mean_square(errors)


def _bias(observed, predicted):
    """Compute mean predicted-minus-observed error for one target."""
    errors, scale = _scaled_errors(observed, predicted)
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * _stable_mean(errors)


def _mae(observed, predicted):
    """Compute mean absolute error for one target."""
    errors, scale = _scaled_errors(observed, predicted)
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * _stable_mean_absolute(errors)


def _r2(observed, predicted):
    """Compute R² for one target whose denominator is known to be nonzero."""
    errors, error_scale = _scaled_errors(observed, predicted)
    centered, centered_scale = _scaled_centered_observations(observed)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inner_error_scale, scaled_error_sum = _stable_sum_of_squares(errors)
        if inner_error_scale == 0 or scaled_error_sum == 0:
            return 1.0
        inner_centered_scale, scaled_centered_sum = _stable_sum_of_squares(centered)
        scale_ratio = (error_scale / centered_scale) * (
            inner_error_scale / inner_centered_scale
        )
        ratio = scale_ratio**2 * (scaled_error_sum / scaled_centered_sum)
        if np.isfinite(ratio):
            return 1.0 - ratio

        log_error_sum = _log_sum_of_squares(errors, error_scale)
        log_centered_sum = _log_sum_of_squares(centered, centered_scale)
        ratio = np.exp(log_error_sum - log_centered_sum)
        return 1.0 - ratio


def _as_finite_float(value):
    """Return a finite float or None when the kernel result is unusable."""
    if not np.isfinite(value) or abs(value) > np.finfo(float).max:
        return None
    return float(value)


def _validate_metric_names(metrics):
    if isinstance(metrics, str):
        raise SpectroChemPyError("metrics must be a non-empty sequence of names.")
    names = tuple(metrics)
    if not names:
        raise SpectroChemPyError("At least one regression metric must be requested.")
    if len(set(names)) != len(names):
        raise SpectroChemPyError("Regression metric names must not be repeated.")
    unsupported = [name for name in names if name not in _REGRESSION_METRICS]
    if unsupported:
        raise SpectroChemPyError(
            f"Unsupported internal regression metric: {unsupported[0]!r}."
        )
    return names


def _validate_metric_inputs(observed, predicted, *, sample_dim):
    """Validate paired target datasets and convert predictions to target units."""
    if not isinstance(observed, NDDataset) or not isinstance(predicted, NDDataset):
        raise SpectroChemPyError(
            "Observed targets and predictions must both be NDDataset instances."
        )
    if observed.ndim not in (1, 2) or predicted.ndim != observed.ndim:
        raise SpectroChemPyError(
            "Observed targets and predictions must have matching one- or "
            "two-dimensional geometry."
        )
    if observed.shape != predicted.shape:
        raise SpectroChemPyError(
            "Observed targets and predictions must have exactly the same shape."
        )
    if observed.dims != predicted.dims:
        raise SpectroChemPyError(
            "Observed targets and predictions must have exactly the same dimensions."
        )
    if observed.dims.count(sample_dim) != 1:
        raise SpectroChemPyError(
            f"sample_dim {sample_dim!r} must identify exactly one target axis."
        )
    if observed.shape[observed.dims.index(sample_dim)] == 0:
        raise SpectroChemPyError(
            "Observed targets must contain at least one observation."
        )

    for dim in observed.dims:
        observed_coordinate = _coordinate_for_dimension(observed, dim)
        predicted_coordinate = _coordinate_for_dimension(predicted, dim)
        _validate_coordinate_alignment(
            observed_coordinate,
            predicted_coordinate,
            context=f"Metric input {dim!r}",
            allow_missing=False,
        )
        if (
            observed_coordinate is not None
            and predicted_coordinate is not None
            and observed_coordinate.is_labeled != predicted_coordinate.is_labeled
        ):
            raise SpectroChemPyError(
                f"Metric input {dim!r} coordinate labels must be defined on "
                "both datasets or neither."
            )

    if observed.has_units != predicted.has_units:
        raise SpectroChemPyError(
            "Observed target and prediction units must both be defined or absent."
        )
    if observed.has_units:
        if not observed.is_units_compatible(predicted):
            raise SpectroChemPyError(
                "Prediction units are not compatible with observed target units."
            )
        return predicted.to(observed.units, inplace=False)
    return predicted.copy()


def _target_output(values, observed, *, sample_axis, units, title):
    """Build a scalar or target-indexed private output without inventing identity."""
    values = np.asarray(values)
    if observed.ndim == 1:
        return NDDataset(np.asarray(values[0]), units=units, title=title)

    target_axis = 1 - sample_axis
    target_dim = observed.dims[target_axis]
    target_coordinate = _coordinate_for_dimension(observed, target_dim)
    coordset = None if target_coordinate is None else [target_coordinate.copy()]
    return NDDataset(
        values,
        coordset=coordset,
        dims=[target_dim],
        units=units,
        title=title,
    )


def _evaluate_target_metrics(observed, predicted, *, metrics):
    """Evaluate requested metrics and reasons for a single target column."""
    pair_valid = np.isfinite(observed) & np.isfinite(predicted)
    n_valid = int(np.count_nonzero(pair_valid))
    invalid_prediction = n_valid != observed.size
    values = {}
    defined = {}
    reasons = {}

    for name in metrics:
        reason = None
        value = np.nan
        if observed.size == 0:
            reason = _NO_OBSERVATIONS
        elif invalid_prediction:
            reason = _INVALID_PREDICTION
        elif name == "r2" and n_valid < 2:
            reason = _INSUFFICIENT_VALID_PAIRS
        else:
            observed_valid = observed[pair_valid]
            predicted_valid = predicted[pair_valid]
            if name == "r2":
                centered, _ = _scaled_centered_observations(observed_valid)
                centered_scale, centered_sum = _stable_sum_of_squares(centered)
                if centered_scale == 0 or centered_sum == 0:
                    reason = _ZERO_OBSERVED_VARIANCE
                else:
                    value = _r2(observed_valid, predicted_valid)
            elif name == "rmse":
                value = _rmse(observed_valid, predicted_valid)
            elif name == "bias":
                value = _bias(observed_valid, predicted_valid)
            else:
                value = _mae(observed_valid, predicted_valid)

            if reason is None:
                value = _as_finite_float(value)
                if value is None:
                    value = np.nan
                    reason = _NON_FINITE_RESULT

        values[name] = value
        defined[name] = reason is None
        reasons[name] = reason

    return n_valid, values, defined, reasons


def _compute_regression_metrics(
    observed,
    predicted,
    *,
    sample_dim="y",
    metrics=_REGRESSION_METRICS,
):
    """Compute private per-target regression metrics for paired datasets."""
    metric_names = _validate_metric_names(metrics)
    converted_prediction = _validate_metric_inputs(
        observed, predicted, sample_dim=sample_dim
    )
    sample_axis = observed.dims.index(sample_dim)

    observed_masked = np.ma.asarray(observed.masked_data)
    observed_values = np.asarray(np.ma.getdata(observed_masked))
    observed_mask = np.ma.getmaskarray(observed_masked)
    if np.any(observed_mask) or not np.all(np.isfinite(observed_values)):
        raise SpectroChemPyError(
            "Observed targets must not contain masked or non-finite values."
        )

    predicted_masked = np.ma.asarray(converted_prediction.masked_data)
    predicted_values = np.asarray(np.ma.getdata(predicted_masked))
    predicted_mask = np.ma.getmaskarray(predicted_masked)
    prediction_valid = ~predicted_mask & np.isfinite(predicted_values)

    observed_rows = np.moveaxis(observed_values, sample_axis, 0)
    predicted_rows = np.moveaxis(predicted_values, sample_axis, 0)
    prediction_valid_rows = np.moveaxis(prediction_valid, sample_axis, 0)
    if observed.ndim == 1:
        observed_rows = observed_rows[:, np.newaxis]
        predicted_rows = predicted_rows[:, np.newaxis]
        prediction_valid_rows = prediction_valid_rows[:, np.newaxis]

    n_targets = observed_rows.shape[1]
    valid_counts = np.empty(n_targets, dtype=np.intp)
    metric_values = {
        name: np.full(n_targets, np.nan, dtype=float) for name in metric_names
    }
    metric_defined = {name: [] for name in metric_names}
    metric_reasons = {name: [] for name in metric_names}

    for target in range(n_targets):
        target_prediction = predicted_rows[:, target].astype(float, copy=True)
        target_prediction[~prediction_valid_rows[:, target]] = np.nan
        n_valid, values, defined, reasons = _evaluate_target_metrics(
            observed_rows[:, target], target_prediction, metrics=metric_names
        )
        valid_counts[target] = n_valid
        for name in metric_names:
            metric_values[name][target] = values[name]
            metric_defined[name].append(defined[name])
            metric_reasons[name].append(reasons[name])

    with np.errstate(over="ignore", invalid="ignore"):
        residual_values = np.asarray(observed_values, dtype=float) - np.asarray(
            predicted_values, dtype=float
        )
        residual_values = np.asarray(residual_values, dtype=float)
    residuals = observed.copy()
    residuals.data = np.ma.array(residual_values, mask=predicted_mask, copy=True)

    outputs = []
    for name in metric_names:
        units = None if name == "r2" else observed.units
        outputs.append(
            CrossValidationMetric(
                name=name,
                values=_target_output(
                    metric_values[name],
                    observed,
                    sample_axis=sample_axis,
                    units=units,
                    title=name,
                ),
                defined=tuple(metric_defined[name]),
                reasons=tuple(metric_reasons[name]),
            )
        )

    return _RegressionMetricResult(
        residuals=residuals,
        n_valid=_target_output(
            valid_counts,
            observed,
            sample_axis=sample_axis,
            units=None,
            title="n_valid",
        ),
        metrics=tuple(outputs),
    )


def _qualified_class_name(value):
    cls = value.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _readonly_array(value):
    copied = np.array(copy.deepcopy(value), copy=True, subok=True)
    copied.flags.writeable = False
    if isinstance(copied, np.ma.MaskedArray):
        copied.data.flags.writeable = False
        if isinstance(copied.mask, np.ndarray):
            copied.mask.flags.writeable = False
    return copied


def _snapshot_configuration_value(value):
    """Take an isolated, non-executable snapshot of a configuration value."""
    if value is None or isinstance(value, str | bytes | bool | Number):
        return value
    if isinstance(value, np.ma.MaskedArray | np.ndarray):
        return _readonly_array(value)
    if isinstance(value, np.random.RandomState):
        return MappingProxyType(
            {
                "kind": "RandomState",
                "state": _snapshot_configuration_value(value.get_state()),
            }
        )
    if isinstance(value, np.random.Generator):
        return MappingProxyType(
            {
                "kind": value.bit_generator.__class__.__name__,
                "state": _snapshot_configuration_value(value.bit_generator.state),
            }
        )
    if _qualified_class_name(value).startswith("spectrochempy.core.dataset"):
        return value.copy()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _snapshot_configuration_value(item) for key, item in value.items()}
        )
    if isinstance(value, tuple | list):
        return tuple(_snapshot_configuration_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_snapshot_configuration_value(item) for item in value)
    return repr(value)


def _snapshot_parameters(estimator):
    if not hasattr(estimator, "get_params"):
        raise SpectroChemPyError(
            f"{estimator.__class__.__name__} does not expose get_params()."
        )
    parameters = estimator.get_params(deep=False)
    if not isinstance(parameters, dict):
        raise SpectroChemPyError(
            f"{estimator.__class__.__name__}.get_params(deep=False) must "
            "return a dictionary."
        )
    return MappingProxyType(
        {
            name: _snapshot_configuration_value(value)
            for name, value in parameters.items()
        }
    )


def _snapshot_estimator_configuration(estimator):
    """Snapshot the bounded CV estimator configuration without learned state."""
    cloned = clone_unfitted(estimator)
    class_name = _qualified_class_name(cloned)
    pls_name = "spectrochempy.analysis.crossdecomposition.pls.PLSRegression"
    pipeline_name = "spectrochempy.analysis.pipeline.Pipeline"
    if class_name == pls_name:
        return _EstimatorConfiguration(
            class_name=class_name,
            parameters=_snapshot_parameters(cloned),
        )
    if class_name != pipeline_name:
        raise SpectroChemPyError(
            "Cross-validation supports only PLSRegression or an "
            "exact Pipeline ending in PLSRegression."
        )
    if not cloned.steps or _qualified_class_name(cloned.steps[-1][1]) != pls_name:
        raise SpectroChemPyError(
            "Cross-validation requires Pipeline to end in PLSRegression."
        )
    steps = tuple(
        _PipelineStepConfiguration(
            name=name,
            class_name=_qualified_class_name(step),
            parameters=_snapshot_parameters(step),
        )
        for name, step in cloned.steps
    )
    return _EstimatorConfiguration(
        class_name=class_name,
        parameters=MappingProxyType({}),
        steps=steps,
    )


def _snapshot_splitter_configuration(splitter):
    parameters = {
        name: _snapshot_configuration_value(value)
        for name, value in vars(splitter).items()
        if not name.startswith("_") and not callable(value)
    }
    return _SplitterConfiguration(
        class_name=_qualified_class_name(splitter),
        parameters=MappingProxyType(parameters),
    )


def _prepare_groups(groups, observed, *, sample_dim, n_observations):
    if groups is None:
        return None, None
    if isinstance(groups, NDDataset):
        if groups.ndim != 1 or groups.dims.count(sample_dim) != 1:
            raise SpectroChemPyError("groups must be one-dimensional along sample_dim.")
        if groups.shape[0] != n_observations:
            raise SpectroChemPyError(
                "groups must contain exactly one value per observation."
            )
        _validate_coordinate_alignment(
            _coordinate_for_dimension(observed, sample_dim),
            _coordinate_for_dimension(groups, sample_dim),
            context="Group observation",
        )
        masked = np.ma.asarray(groups.masked_data)
        if np.any(np.ma.getmaskarray(masked)):
            raise SpectroChemPyError("groups must not contain masked values.")
        values = np.asarray(np.ma.getdata(masked)).copy()
    else:
        masked = np.ma.asarray(groups)
        if masked.ndim != 1 or masked.size != n_observations:
            raise SpectroChemPyError(
                "groups must be one-dimensional with one value per observation."
            )
        if np.any(np.ma.getmaskarray(masked)):
            raise SpectroChemPyError("groups must not contain masked values.")
        values = np.asarray(np.ma.getdata(masked)).copy()

    counts = {}
    order = []
    for raw_value in values.tolist():
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if isinstance(value, Number) and not np.isfinite(value):
            raise SpectroChemPyError("groups must not contain non-finite values.")
        try:
            hash(value)
        except TypeError as exc:
            raise SpectroChemPyError("Each group identity must be hashable.") from exc
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1

    summary_counts = tuple(
        _GroupCount(
            value=_snapshot_configuration_value(value),
            count=counts[value],
        )
        for value in order
    )
    return values, _GroupsSummary(n_groups=len(summary_counts), counts=summary_counts)


def _validate_splitter_scope(splitter, *, groups):
    class_name = _qualified_class_name(splitter)
    supported = {
        "sklearn.model_selection._split.KFold",
        "sklearn.model_selection._split.GroupKFold",
        "sklearn.model_selection._split.LeaveOneOut",
    }
    if class_name not in supported:
        raise SpectroChemPyError(
            f"{splitter.__class__.__name__} is not a supported v1 CV splitter."
        )
    is_group_splitter = class_name.endswith(".GroupKFold")
    if is_group_splitter and groups is None:
        raise SpectroChemPyError("GroupKFold requires groups.")
    if not is_group_splitter and groups is not None:
        raise SpectroChemPyError("groups must not be supplied to a non-group splitter.")
    return class_name


def _validate_splitter(splitter, *, groups, n_observations, n_folds):
    _validate_splitter_scope(splitter, groups=groups)
    try:
        expected_folds = splitter.get_n_splits(
            np.empty((n_observations, 1)), groups=groups
        )
    except Exception as exc:
        raise SpectroChemPyError(
            "Cannot resolve the splitter's number of folds."
        ) from exc
    if expected_folds != n_folds:
        raise SpectroChemPyError(
            f"Splitter expects {expected_folds} folds but {n_folds} were supplied."
        )
    return _snapshot_splitter_configuration(splitter)


def _validate_cross_validation_input_values(X, observed, geometry):
    """Enforce the conservative v1 input mask and finite-value policy."""
    observed_masked = np.ma.asarray(observed.masked_data)
    if np.any(np.ma.getmaskarray(observed_masked)):
        raise SpectroChemPyError("Observed targets must not contain masked values.")
    try:
        observed_finite = np.isfinite(np.ma.getdata(observed_masked))
    except TypeError as exc:
        raise SpectroChemPyError(
            "Observed targets must contain numeric values."
        ) from exc
    if not np.all(observed_finite):
        raise SpectroChemPyError("Observed targets must not contain non-finite values.")

    X_masked = np.ma.asarray(X.masked_data)
    X_mask = np.ma.getmaskarray(X_masked)
    row_masks = np.moveaxis(X_mask, geometry.x_sample_axis, 0)
    if row_masks.shape[0] == 0:
        raise SpectroChemPyError("Cross-validation requires at least one observation.")
    if not np.array_equal(row_masks, np.broadcast_to(row_masks[0], row_masks.shape)):
        raise SpectroChemPyError(
            "X masks must select the same feature positions for every observation."
        )
    try:
        X_finite = np.isfinite(np.ma.getdata(X_masked))
    except TypeError as exc:
        raise SpectroChemPyError("X must contain numeric values.") from exc
    if not np.all(X_finite):
        raise SpectroChemPyError("X must not contain non-finite values.")
    feature_mask = np.asarray(row_masks[0], dtype=bool)
    if np.all(feature_mask):
        raise SpectroChemPyError("X must contain at least one unmasked feature.")


def _materialize_splitter_folds(splitter, *, groups, n_observations):
    """Consume one splitter iteration and isolate its returned position arrays."""
    try:
        execution_splitter = copy.deepcopy(splitter)
        generated = execution_splitter.split(
            np.empty((n_observations, 1)),
            groups=groups,
        )
        return tuple(
            (np.asarray(train).copy(), np.asarray(validation).copy())
            for train, validation in generated
        )
    except Exception as exc:
        raise SpectroChemPyError(
            f"Cannot materialize cross-validation folds from the splitter: {exc}"
        ) from exc


def _validate_cross_validation_folds(folds, *, geometry, groups):
    """Validate an already-materialized complete OOF partition."""
    supplied_folds = tuple(folds)
    if not supplied_folds:
        raise SpectroChemPyError("At least one CV fold must be supplied.")

    coverage = np.zeros(geometry.n_observations, dtype=np.intp)
    validated = []
    for fold_index, fold in enumerate(supplied_folds):
        if not isinstance(fold, tuple | list) or len(fold) != 2:
            raise SpectroChemPyError(
                "Each fold must contain training and validation positions."
            )
        train = _validate_fold_positions(
            fold[0],
            n_observations=geometry.n_observations,
            name=f"Fold {fold_index} training",
        )
        validation = _validate_fold_positions(
            fold[1],
            n_observations=geometry.n_observations,
            name=f"Fold {fold_index} validation",
        )
        if np.intersect1d(train, validation).size:
            raise SpectroChemPyError(
                f"Fold {fold_index} training and validation positions overlap."
            )
        if groups is not None:
            training_groups = set(groups[train].tolist())
            validation_groups = set(groups[validation].tolist())
            if training_groups & validation_groups:
                raise SpectroChemPyError(
                    f"Fold {fold_index} places a group in both training and "
                    "validation positions."
                )
        coverage[validation] += 1
        validated.append((_readonly_positions(train), _readonly_positions(validation)))

    if np.any(coverage == 0):
        raise SpectroChemPyError(
            "Validation folds do not cover every observation exactly once."
        )
    if np.any(coverage > 1):
        raise SpectroChemPyError(
            "Validation folds repeat one or more observation positions."
        )
    return tuple(validated)


def _prepare_cross_validation_plan(
    *,
    estimator,
    splitter,
    X,
    observed,
    folds=None,
    sample_dim="y",
    metrics=_REGRESSION_METRICS,
    groups=None,
):
    """Validate inputs and return a complete plan before any estimator fit."""
    geometry = _resolve_sample_geometry(X, observed, sample_dim=sample_dim)
    _validate_cross_validation_input_values(
        X,
        observed,
        geometry,
    )
    metric_names = _validate_metric_names(metrics)
    estimator_configuration = _snapshot_estimator_configuration(estimator)
    groups_values, groups_summary = _prepare_groups(
        groups,
        observed,
        sample_dim=sample_dim,
        n_observations=geometry.n_observations,
    )
    _validate_splitter_scope(splitter, groups=groups_values)

    supplied_folds = folds
    if supplied_folds is None:
        supplied_folds = _materialize_splitter_folds(
            splitter,
            groups=groups_values,
            n_observations=geometry.n_observations,
        )
    supplied_folds = tuple(supplied_folds)
    if not supplied_folds:
        raise SpectroChemPyError("At least one CV fold must be supplied.")
    splitter_configuration = _validate_splitter(
        splitter,
        groups=groups_values,
        n_observations=geometry.n_observations,
        n_folds=len(supplied_folds),
    )
    validated_folds = _validate_cross_validation_folds(
        supplied_folds,
        geometry=geometry,
        groups=groups_values,
    )
    if groups_values is not None:
        groups_values.flags.writeable = False

    return _CrossValidationPlan(
        geometry=geometry,
        metric_names=metric_names,
        estimator_configuration=estimator_configuration,
        splitter_configuration=splitter_configuration,
        groups_values=groups_values,
        groups_summary=groups_summary,
        folds=validated_folds,
    )


def _readonly_positions(values):
    copied = np.array(values, dtype=np.intp, copy=True)
    copied.flags.writeable = False
    return copied


def _renamed_metric(metric, name):
    values = metric.values.copy()
    values.title = name
    return CrossValidationMetric(
        name=name,
        values=values,
        defined=metric.defined,
        reasons=metric.reasons,
    )


def _undefined_metric_records(metrics, *, scope, fold_index=None):
    records = []
    for metric in metrics:
        for target_index, (defined, reason) in enumerate(
            zip(metric.defined, metric.reasons, strict=True)
        ):
            if not defined:
                records.append(
                    _UndefinedMetric(
                        scope=scope,
                        metric=metric.name,
                        target_index=target_index,
                        reason=reason,
                        fold_index=fold_index,
                    )
                )
    return records


def _dataset_description(dataset, role):
    return _DatasetDescription(
        role=role,
        name=getattr(dataset, "name", None),
        title=getattr(dataset, "title", None),
        shape=tuple(dataset.shape),
        dims=tuple(dataset.dims),
        units=str(dataset.units) if dataset.has_units else None,
    )


def _configuration_values_equal(left, right):
    """Compare isolated configuration snapshots, including array values."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ma.MaskedArray):
        return bool(np.ma.allequal(left, right))
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right, equal_nan=True))
    if isinstance(left, NDDataset):
        return bool(left == right)
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _configuration_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _configuration_values_equal(old, new)
            for old, new in zip(left, right, strict=True)
        )
    return bool(left == right)


def _estimator_configurations_equal(left, right):
    if left.class_name != right.class_name or len(left.steps) != len(right.steps):
        return False
    if not _configuration_values_equal(left.parameters, right.parameters):
        return False
    return all(
        old.name == new.name
        and old.class_name == new.class_name
        and _configuration_values_equal(old.parameters, new.parameters)
        for old, new in zip(left.steps, right.steps, strict=True)
    )


def _copy_fold_estimators(
    fold_estimators,
    *,
    retain_estimators,
    estimator_configuration,
    n_folds,
):
    if not retain_estimators:
        if fold_estimators is not None:
            raise SpectroChemPyError(
                "fold_estimators were supplied while retain_estimators is False."
            )
        return None
    if fold_estimators is None:
        raise SpectroChemPyError(
            "retain_estimators=True requires one fitted estimator per fold."
        )
    estimators = tuple(fold_estimators)
    if len(estimators) != n_folds:
        raise SpectroChemPyError(
            "The number of retained estimators must match the number of folds."
        )
    retained = []
    for fold_index, fitted in enumerate(estimators):
        if _qualified_class_name(fitted) != estimator_configuration.class_name:
            raise SpectroChemPyError(
                f"Retained estimator for fold {fold_index} has the wrong class."
            )
        if not bool(getattr(fitted, "_fitted", False)):
            raise SpectroChemPyError(
                f"Retained estimator for fold {fold_index} is not fitted."
            )
        fitted_configuration = _snapshot_estimator_configuration(fitted)
        if not _estimator_configurations_equal(
            fitted_configuration, estimator_configuration
        ):
            raise SpectroChemPyError(
                f"Retained estimator for fold {fold_index} has a different "
                "constructor configuration."
            )
        try:
            retained.append(copy.deepcopy(fitted))
        except Exception as exc:
            raise SpectroChemPyError(
                f"Cannot isolate retained estimator for fold {fold_index}."
            ) from exc
    return tuple(retained)


def _assemble_cross_validation_result(
    *,
    plan,
    X,
    observed,
    oof_predictions,
    warnings=(),
    retain_estimators=False,
    fold_estimators=None,
):
    """Assemble a result from a validated plan and complete OOF predictions."""
    geometry = plan.geometry
    metric_names = plan.metric_names
    sample_dim = geometry.sample_dim

    observed_snapshot = observed.copy()
    prediction_snapshot = _validate_metric_inputs(
        observed_snapshot,
        oof_predictions,
        sample_dim=sample_dim,
    )
    global_result = _compute_regression_metrics(
        observed_snapshot,
        prediction_snapshot,
        sample_dim=sample_dim,
        metrics=metric_names,
    )
    global_metrics = tuple(
        _renamed_metric(metric, "rmsecv") if metric.name == "rmse" else metric
        for metric in global_result.metrics
    )

    fold_results = []
    undefined_metrics = _undefined_metric_records(global_metrics, scope="global")
    for fold_index, (train, validation) in enumerate(plan.folds):
        fold_observed = _slice_along_axis(
            observed_snapshot, validation, geometry.y_sample_axis
        )
        fold_prediction = _slice_along_axis(
            prediction_snapshot, validation, geometry.y_sample_axis
        )
        fold_metric_result = _compute_regression_metrics(
            fold_observed,
            fold_prediction,
            sample_dim=sample_dim,
            metrics=metric_names,
        )
        fold_metrics = fold_metric_result.metrics
        fold_results.append(
            CrossValidationFoldResult(
                fold_index=fold_index,
                train_positions=_readonly_positions(train),
                validation_positions=_readonly_positions(validation),
                metrics=fold_metrics,
                n_valid=fold_metric_result.n_valid,
            )
        )
        undefined_metrics.extend(
            _undefined_metric_records(
                fold_metrics,
                scope="fold",
                fold_index=fold_index,
            )
        )

    if isinstance(warnings, str):
        raise SpectroChemPyError("warnings must be a sequence of strings.")
    warning_snapshot = tuple(warnings)
    if not all(isinstance(message, str) for message in warning_snapshot):
        raise SpectroChemPyError("Each warning must be a string.")

    retained_estimators = _copy_fold_estimators(
        fold_estimators,
        retain_estimators=retain_estimators,
        estimator_configuration=plan.estimator_configuration,
        n_folds=len(fold_results),
    )

    observation_coordinate = _coordinate_for_dimension(observed_snapshot, sample_dim)
    if observation_coordinate is not None:
        observation_coordinate = observation_coordinate.copy()
    input_descriptions = [
        _dataset_description(X, "X"),
        _dataset_description(observed_snapshot, "y"),
    ]
    if plan.groups_summary is not None:
        input_descriptions.append(
            _DatasetDescription(
                role="groups",
                name=None,
                title=None,
                shape=(geometry.n_observations,),
                dims=(sample_dim,),
                units=None,
            )
        )

    return CrossValidationResult(
        estimator=plan.estimator_configuration,
        splitter=plan.splitter_configuration,
        n_splits=len(fold_results),
        sample_dim=sample_dim,
        x_sample_axis=geometry.x_sample_axis,
        y_sample_axis=geometry.y_sample_axis,
        observation_coordinate=observation_coordinate,
        groups=plan.groups_summary,
        observed=observed_snapshot,
        oof_predictions=prediction_snapshot,
        residuals=global_result.residuals,
        global_metrics=global_metrics,
        n_valid=global_result.n_valid,
        folds=tuple(fold_results),
        warnings=warning_snapshot,
        undefined_metrics=tuple(undefined_metrics),
        fold_estimators=retained_estimators,
        operation=_ValidationDescription(
            operation="cross_validate",
            inputs=tuple(input_descriptions),
            note=(
                "Internal non-executable operation description; no automatic "
                "provenance capture, fingerprint, registry, or replay contract."
            ),
        ),
    )


def _build_cross_validation_result(
    *,
    estimator,
    splitter,
    X,
    observed,
    oof_predictions,
    folds,
    sample_dim="y",
    metrics=_REGRESSION_METRICS,
    groups=None,
    warnings=(),
    retain_estimators=False,
    fold_estimators=None,
):
    """Build a coherent internal structured CV result from complete OOF data."""
    plan = _prepare_cross_validation_plan(
        estimator=estimator,
        splitter=splitter,
        X=X,
        observed=observed,
        folds=folds,
        sample_dim=sample_dim,
        metrics=metrics,
        groups=groups,
    )
    return _assemble_cross_validation_result(
        plan=plan,
        X=X,
        observed=observed,
        oof_predictions=oof_predictions,
        warnings=warnings,
        retain_estimators=retain_estimators,
        fold_estimators=fold_estimators,
    )


def _execute_cross_validation(
    estimator,
    X,
    observed,
    *,
    splitter,
    groups=None,
    sample_dim="y",
    metrics=_REGRESSION_METRICS,
    return_estimators=False,
):
    """Execute bounded supervised CV without exposing a public API."""
    plan = _prepare_cross_validation_plan(
        estimator=estimator,
        splitter=splitter,
        X=X,
        observed=observed,
        sample_dim=sample_dim,
        metrics=metrics,
        groups=groups,
    )

    output_dtype = np.result_type(
        np.asarray(np.ma.getdata(np.ma.asarray(observed.masked_data))).dtype,
        np.float64,
    )
    oof_values = np.ma.masked_all(observed.shape, dtype=output_dtype)
    retained_estimators = [] if return_estimators else None

    for fold_index, (train, validation) in enumerate(plan.folds):
        fold = _prepare_validated_fold_subsets(
            X,
            observed,
            train,
            validation,
            plan.geometry,
        )
        try:
            fold_estimator = clone_unfitted(estimator)
        except Exception as exc:
            raise SpectroChemPyError(
                f"Cross-validation fold {fold_index} failed during estimator cloning."
            ) from exc
        try:
            fold_estimator.fit(fold.X_train, fold.y_train)
        except Exception as exc:
            raise SpectroChemPyError(
                f"Cross-validation fold {fold_index} failed during fit."
            ) from exc
        try:
            row_prediction = fold_estimator.predict(fold.X_validation)
        except Exception as exc:
            raise SpectroChemPyError(
                f"Cross-validation fold {fold_index} failed during predict."
            ) from exc
        try:
            prediction = _restore_prediction_geometry(
                row_prediction,
                fold.target_geometry,
            )
        except Exception as exc:
            raise SpectroChemPyError(
                f"Cross-validation fold {fold_index} returned an invalid prediction."
            ) from exc

        index = [slice(None)] * observed.ndim
        index[plan.geometry.y_sample_axis] = validation
        oof_values[tuple(index)] = prediction.masked_data
        if retained_estimators is not None:
            retained_estimators.append(fold_estimator)

    oof_predictions = observed.copy()
    oof_predictions.data = oof_values
    oof_predictions.title = "OOF predictions"
    return _assemble_cross_validation_result(
        plan=plan,
        X=X,
        observed=observed,
        oof_predictions=oof_predictions,
        retain_estimators=return_estimators,
        fold_estimators=retained_estimators,
    )


_PUBLIC_REGRESSION_METRICS = ("rmsecv", "r2", "bias", "mae")


def _resolve_public_metrics(metrics):
    """Validate public metric selectors and translate RMSECV for the kernel."""
    if metrics is None or isinstance(metrics, str | bytes | Mapping | set | frozenset):
        raise SpectroChemPyError(
            "metrics must be a non-empty ordered sequence of metric names."
        )
    try:
        names = tuple(metrics)
    except TypeError as exc:
        raise SpectroChemPyError(
            "metrics must be a non-empty ordered sequence of metric names."
        ) from exc
    if not names:
        raise SpectroChemPyError("At least one cross-validation metric is required.")
    if not all(isinstance(name, str) for name in names):
        raise SpectroChemPyError("Every cross-validation metric name must be a string.")
    if len(set(names)) != len(names):
        raise SpectroChemPyError("Cross-validation metric names must not be repeated.")
    if "rmse" in names:
        raise SpectroChemPyError(
            "Use 'rmsecv', not 'rmse', as the public cross-validation metric."
        )
    unsupported = [name for name in names if name not in _PUBLIC_REGRESSION_METRICS]
    if unsupported:
        raise SpectroChemPyError(
            f"Unsupported cross-validation metric: {unsupported[0]!r}."
        )
    return tuple("rmse" if name == "rmsecv" else name for name in names)


def _resolve_public_splitter(cv, *, groups):
    """Resolve the bounded public cv selector without consuming a splitter."""
    if isinstance(cv, bool | np.bool_):
        raise SpectroChemPyError("cv must be an integer of at least 2 or a splitter.")
    if isinstance(cv, Integral):
        n_splits = int(cv)
        if n_splits < 2:
            raise SpectroChemPyError("Integer cv must be at least 2.")
        if groups is None:
            return KFold(n_splits=n_splits, shuffle=False)
        return GroupKFold(n_splits=n_splits)
    if isinstance(cv, Number):
        raise SpectroChemPyError("cv must not be a floating-point value.")
    if type(cv) not in (KFold, GroupKFold, LeaveOneOut):
        raise SpectroChemPyError(
            "cv must be an integer or an explicit KFold, GroupKFold, or "
            "LeaveOneOut splitter."
        )
    return cv


def cross_validate(
    estimator,
    X,
    y,
    *,
    cv=5,
    groups=None,
    sample_dim="y",
    metrics=("rmsecv", "r2"),
    return_estimators=False,
):
    """
    Evaluate a fixed supervised regression configuration by cross-validation.

    Each fold receives a fresh unfitted estimator. Pipeline preprocessing is
    learned only from that fold's calibration observations, and predictions
    are assembled once in the original target order and geometry. The supplied
    estimator, datasets, groups, and explicit splitter remain unchanged.

    Parameters
    ----------
    estimator : PLSRegression or Pipeline
        Unfitted or fitted template to clone for each fold. A Pipeline must end
        in `PLSRegression`.
    X : NDDataset
        Two-dimensional predictor dataset.
    y : NDDataset
        One- or two-dimensional target dataset aligned with X along
        *sample_dim*.
    cv : int, KFold, GroupKFold, or LeaveOneOut, optional, default: 5
        Cross-validation design. An integer creates unshuffled `KFold` when
        *groups* is absent and `GroupKFold` otherwise. Integers must be at
        least 2; booleans and floating-point values are rejected.
    groups : array-like or NDDataset, optional
        One group identity per observation. Required by `GroupKFold` and
        forbidden for splitters that do not use groups.
    sample_dim : str, optional, default: "y"
        Dimension identifying observations independently in X and y.
    metrics : ordered iterable of str, optional
        Global metrics selected from ``"rmsecv"``, ``"r2"``, ``"bias"``, and
        ``"mae"``. Names must be unique and the order is preserved. ``"rmse"``
        is not a public selector; fold records use that generic internal name.
        An empty selection is rejected.
    return_estimators : bool, optional, default: False
        Retain independent fitted estimators for every fold. Enabling this may
        substantially increase memory use.

    Returns
    -------
    CrossValidationResult
        Structured OOF predictions, residuals, global and fold metrics,
        validated positions, configuration snapshots, and optional estimators.
        Dataset members are isolated copies but remain mutable.

    Raises
    ------
    SpectroChemPyError
        If the estimator, splitter, groups, geometry, masks, values, metrics,
        or complete unique OOF coverage violate the bounded v1 contract.

    See Also
    --------
    CrossValidationResult : Structured result returned by this function.

    Notes
    -----
    This function evaluates one fixed estimator configuration. It performs no
    hyperparameter search, nested cross-validation, final full-data fit,
    persistence, replay, or automatic provenance capture.

    Examples
    --------
    >>> values = np.arange(24.0).reshape(8, 3)
    >>> X = scp.NDDataset(values, dims=["y", "x"])
    >>> y = scp.NDDataset((1.0 + values[:, 0])[:, None], dims=["y", "t"])
    >>> model = scp.PLSRegression(n_components=1, scale=False)
    >>> result = scp.cross_validate(model, X, y, cv=4)
    >>> result.oof_predictions.shape
    (8, 1)
    >>> result.metric("rmsecv").values.shape
    (1,)
    """
    internal_metrics = _resolve_public_metrics(metrics)
    splitter = _resolve_public_splitter(cv, groups=groups)
    if not isinstance(return_estimators, bool | np.bool_):
        raise SpectroChemPyError("return_estimators must be a boolean.")
    return _execute_cross_validation(
        estimator,
        X,
        y,
        splitter=splitter,
        groups=groups,
        sample_dim=sample_dim,
        metrics=internal_metrics,
        return_estimators=bool(return_estimators),
    )
