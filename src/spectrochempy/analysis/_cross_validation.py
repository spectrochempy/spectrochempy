# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Internal dataset preparation helpers for supervised cross-validation."""

from dataclasses import dataclass

import numpy as np

from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.exceptions import SpectroChemPyError


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
class _PerTargetMetric:
    """Values and explicit definedness for one regression metric."""

    name: str
    values: NDDataset
    defined: tuple[bool, ...]
    reasons: tuple[str | None, ...]


@dataclass(frozen=True)
class _RegressionMetricResult:
    """Private metric outputs with target identity and validity metadata."""

    residuals: NDDataset
    n_valid: NDDataset
    metrics: tuple[_PerTargetMetric, ...]

    def metric(self, name):
        """Return one requested metric by its internal kernel name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


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

    X_train = _slice_along_axis(X, train, geometry.x_sample_axis)
    X_validation = _slice_along_axis(X, validation, geometry.x_sample_axis)
    y_train = _slice_along_axis(y, train, geometry.y_sample_axis)
    y_validation = _slice_along_axis(y, validation, geometry.y_sample_axis)
    target_geometry = _TargetGeometry(
        template=y_validation.copy(),
        sample_dim=sample_dim,
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
            _PerTargetMetric(
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
