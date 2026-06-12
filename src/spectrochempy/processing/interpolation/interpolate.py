# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module implementing interpolation for NDDataset."""

__all__ = ["interpolate"]
__dataset_methods__ = __all__

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet


def _interp_along_axis(values, old_x, new_x, axis, method, fill_value):
    """
    Interpolate values along a single axis.

    Parameters
    ----------
    values : np.ndarray
        Data to interpolate.
    old_x : np.ndarray
        Original coordinates (1D).
    new_x : np.ndarray
        Target coordinates (1D).
    axis : int
        Axis along which to interpolate.
    method : str
        'linear' or 'pchip'.
    fill_value : any
        Value for extrapolation.

    Returns
    -------
    np.ndarray
        Interpolated data with same shape as input along other dimensions.
    """
    if method == "linear":
        interpolator = interp1d(
            old_x,
            values,
            axis=axis,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=True,
        )
    elif method == "pchip":
        if values.shape[axis] < 2:
            raise ValueError(
                f"PCHIP interpolation requires at least 2 points, got {values.shape[axis]}"
            )
        interpolator = PchipInterpolator(
            old_x,
            values,
            axis=axis,
            extrapolate=fill_value is not np.nan,
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return interpolator(new_x)


def _validate_monotonic(coord, assume_sorted, tol=1e-12):
    """
    Validate coordinate monotonicity with tolerance for floating point noise.

    Parameters
    ----------
    coord : Coord
        Coordinate to validate.
    assume_sorted : bool
        If True, skip validation.
    tol : float, optional
        Tolerance for duplicate detection. Default is 1e-12.

    Returns
    -------
    tuple: (sorted_data, sorted_indices, was_reversed)
    """
    data = coord.data

    if assume_sorted:
        return data, np.arange(len(data)), False

    diffs = np.diff(data)

    if np.any(np.abs(diffs) <= tol):
        raise ValueError(
            "Duplicate or near-duplicate coordinate values detected. "
            "Interpolation requires strictly monotonic coordinates. "
            "Use `assume_sorted=True` only if data is known to be strictly monotonic."
        )

    strictly_increasing = np.all(diffs > tol)
    strictly_decreasing = np.all(diffs < -tol)

    if strictly_increasing:
        return data, np.arange(len(data)), False
    if strictly_decreasing:
        return data[::-1], np.arange(len(data) - 1, -1, -1), True

    sorted_data = np.sort(data)
    sort_idx = np.argsort(data)
    return sorted_data, sort_idx, False


def _get_coord_data(coord):
    """Extract numeric data from coord."""
    if coord.has_data:
        return coord.data
    return None


def interpolate(
    dataset,
    dim=None,
    dims=None,
    coord=None,
    method="linear",
    fill_value=np.nan,
    assume_sorted=False,
    inplace=False,
):
    """
    Interpolate dataset onto new coordinates.

    Parameters
    ----------
    dataset : NDDataset
        Dataset to interpolate.
    dim : str or int, optional
        Single dimension name or index.
    dims : list of str or int, optional
        List of dimensions to interpolate. Higher priority than dim.
    coord : Coord, np.ndarray, NDDataset, or dict, optional
        Target coordinates. If dict, maps dim -> coord.
        For single dim: can be Coord, ndarray, or NDDataset.
        For multiple dims: dict {dim: coord} or sequential application.
    method : str, optional, default='linear'
        Interpolation method: 'linear' or 'pchip'.
    fill_value : any, optional, default=np.nan
        Value for points outside the original range.
    assume_sorted : bool, optional, default=False
        If True, skip monotonicity checks.
    inplace : bool, optional, default=False
        If True, modify the dataset in place.

    Returns
    -------
    NDDataset
        Dataset with interpolated data and updated coordinates.

    Raises
    ------
    TypeError
        If coordinate is not numeric.
    ValueError
        If coordinates are not strictly monotonic.

    Notes
    -----
    - Labels are NOT interpolated and are reset after interpolation.
    - For multiple coordinates per dimension, all are interpolated consistently.
    - Secondary coordinates are interpolated numerically. If they represent
      analytical transformations of the primary coordinate (e.g., wavelength = 1/wavenumber),
      the result may be approximate - consider recomputing them analytically after interpolation.
    - Unit conversion is performed if needed before interpolation.
    - Sequential interpolation is applied for multiple dimensions (not true N-D).
    """
    new = dataset if inplace else dataset.copy()

    axis_list, dim_list = new.get_axis(only_first=False, dim=dim, dims=dims)

    coord_map = {}
    if coord is not None:
        if isinstance(coord, dict):
            coord_map = coord
        elif len(axis_list) == 1:
            coord_map[dim_list[0]] = coord

    for ax, dim in zip(axis_list, dim_list, strict=False):
        target_coord = coord_map.get(dim)

        if target_coord is None:
            raise ValueError(f"No target coordinate provided for dimension '{dim}'")

        # Handle different target coordinate types
        from spectrochempy.core.dataset.nddataset import NDDataset

        if isinstance(target_coord, NDDataset):
            target_coord = target_coord.coord(dim)
        elif isinstance(target_coord, np.ndarray):
            target_coord = Coord(target_coord)
        elif not isinstance(target_coord, Coord):
            raise TypeError(
                f"coord must be Coord, np.ndarray, or NDDataset, got {type(target_coord)}"
            )

        old_coord = new.coord(dim)

        is_coordset = isinstance(old_coord, CoordSet)

        primary_old_coord = old_coord.default if is_coordset else old_coord

        if primary_old_coord is None or not primary_old_coord.has_data:
            raise ValueError(f"Dimension '{dim}' has no coordinate data to interpolate")

        old_data = _get_coord_data(primary_old_coord)
        if old_data is None:
            raise TypeError(f"Cannot interpolate on non-numeric coordinate '{dim}'")

        new_data = _get_coord_data(target_coord)
        if new_data is None:
            raise TypeError("Cannot interpolate to non-numeric coordinate")

        if primary_old_coord.has_units and target_coord.has_units:
            if not primary_old_coord.is_units_compatible(target_coord):
                raise ValueError(
                    f"Incompatible units: {primary_old_coord.units} vs {target_coord.units}"
                )
            if primary_old_coord.units != target_coord.units:
                target_coord = target_coord.copy()
                target_coord.ito(primary_old_coord.units)
                new_data = _get_coord_data(target_coord)

        sorted_old_x, sort_idx, _was_reversed = _validate_monotonic(
            primary_old_coord, assume_sorted
        )

        if sort_idx is not None and len(sort_idx) > 0:
            sorted_data = new._data[
                tuple(sort_idx if i == ax else slice(None) for i in range(new.ndim))
            ].copy()
        else:
            sorted_data = new._data

        if method == "linear":
            interpolator = interp1d(
                sorted_old_x,
                sorted_data,
                axis=ax,
                kind="linear",
                bounds_error=False,
                fill_value=fill_value,
                assume_sorted=True,
            )
        elif method == "pchip":
            if sorted_data.shape[ax] < 2:
                raise ValueError(
                    f"PCHIP interpolation requires at least 2 points, "
                    f"got {sorted_data.shape[ax]}"
                )
            interpolator = PchipInterpolator(
                sorted_old_x,
                sorted_data,
                axis=ax,
                extrapolate=False,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        interpolated_data = interpolator(new_data)

        if sort_idx is not None and len(sort_idx) > 0:
            new._data = interpolated_data
        else:
            new._data = interpolated_data

        if new.is_masked:
            # The mask must be reordered with the same ``sort_idx`` as the data
            # so it is interpolated against ``sorted_old_x`` in matching order;
            # otherwise a decreasing source coordinate flips the mask relative to
            # its samples (closely related to #1100).
            if sort_idx is not None and len(sort_idx) > 0:
                sorted_mask = new._mask[
                    tuple(sort_idx if i == ax else slice(None) for i in range(new.ndim))
                ]
            else:
                sorted_mask = new._mask
            mask_interpolator = interp1d(
                sorted_old_x,
                sorted_mask.astype(float),
                axis=ax,
                kind="linear",
                bounds_error=False,
                fill_value=True,
                assume_sorted=True,
            )
            new_mask = mask_interpolator(new_data) > 0.5
            new._mask = new_mask

        if new._coordset is None:
            new._coordset = CoordSet()

        # Build secondary-coordinate interpolator closure
        # Bind loop variables as defaults to capture values per iteration.
        def _interpolate_secondary(
            coord,
            _sorted_old_x=sorted_old_x,
            _new_data=new_data,
            _sort_idx=sort_idx,
        ):
            new_sec = coord.copy()
            if coord.has_data:
                old_sec_data = _get_coord_data(coord)
                if old_sec_data is not None and len(old_sec_data) == len(_sorted_old_x):
                    # Reorder the secondary coordinate with the primary's
                    # ``sort_idx`` so it aligns with ``sorted_old_x`` (matches the
                    # data/mask handling; needed when the primary is decreasing).
                    if _sort_idx is not None and len(_sort_idx) > 0:
                        old_sec_data = old_sec_data[_sort_idx]
                    sec_interpolator = interp1d(
                        _sorted_old_x,
                        old_sec_data,
                        axis=0,
                        kind="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                        assume_sorted=True,
                    )
                    new_sec._data = sec_interpolator(_new_data)
            new_sec._labels = None
            return new_sec

        new._coordset = new._coordset._interpolate_dim(
            dim,
            target_coord,
            interpolate_secondary=_interpolate_secondary,
        )

        # The interpolated data, mask and coordinates are produced in the order
        # of the target coordinate, so the result already follows the requested
        # ordering. Re-sorting to the *old* coordinate's direction (as was done
        # before) would silently flip a decreasing input onto an increasing
        # target back to decreasing, ignoring the user's requested order (#1100).

    new.history = (
        f"Interpolated along dims {dim_list} to {len(new_data)} points "
        f"using {method} method"
    )

    return new
