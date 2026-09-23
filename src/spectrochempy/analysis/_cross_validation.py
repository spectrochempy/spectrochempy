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


def _ordered_values_equal(left, right):
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape:
        return False
    if np.issubdtype(left_values.dtype, np.number) and np.issubdtype(
        right_values.dtype, np.number
    ):
        return np.allclose(left_values, right_values, rtol=1.0e-12, atol=1.0e-12)
    return np.array_equal(left_values, right_values)


def _validate_coordinate_alignment(left, right, *, context, allow_missing=True):
    """Validate shared coordinate information without changing either side."""
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
            right_values = right.to(left.units, inplace=False).data
        else:
            right_values = right.data
        if not _ordered_values_equal(left.data, right_values):
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
