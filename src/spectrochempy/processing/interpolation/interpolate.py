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


def _match_indices(old_data, new_data):
    """
    Map each target point to an exactly matching original index, or ``-1``.

    Implements the point-wise label policy (#1098): a target point counts as
    "the same" as an original point only on exact value equality, so identity,
    reordering and subsetting carry labels over while genuinely resampled
    points (which fall between original values) do not. Matching is exact on
    purpose -- SpectroChemPy has no coordinate-matching tolerance convention,
    so a broader tolerance would be a separate, explicit design decision.
    """
    old_data = np.asarray(old_data)
    new_data = np.asarray(new_data)
    match_idx = np.full(len(new_data), -1, dtype=int)
    for j, value in enumerate(new_data):
        hits = np.flatnonzero(old_data == value)
        if hits.size:
            match_idx[j] = hits[0]
    return match_idx


def _carry_labels(old_labels, match_idx):
    """
    Carry labels onto interpolated points using precomputed match indices.

    Each target point that exactly matches an original coordinate value
    inherits that point's label(s); the others stay unlabelled (empty string).
    Returns ``None`` when there is nothing to carry, leaving the interpolated
    coordinate unlabelled (#1098). The label level structure is preserved by
    copying whole per-point rows (labels are stored with points on axis 0).
    """
    if old_labels is None:
        return None
    old_labels = np.asarray(old_labels)
    matched = match_idx >= 0
    if not np.any(matched):
        return None
    new_labels = np.full(
        (len(match_idx),) + old_labels.shape[1:], "", dtype=old_labels.dtype
    )
    new_labels[matched] = old_labels[match_idx[matched]]
    return new_labels


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
    fill_value : float or str, optional, default=np.nan
        Value used for points outside the original range. This applies to both
        the ``'linear'`` and ``'pchip'`` methods: a finite value fills
        out-of-range points with that constant, the default ``np.nan`` returns
        NaN outside the range, and ``"extrapolate"`` extrapolates instead.
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
    - Labels are not interpolated. A target point carries over the label of an
      original point only when it exactly matches that point's coordinate value
      (so identity, reordering and subsetting keep their labels); genuinely
      resampled points are left unlabelled (#1098).
    - For multiple coordinates per dimension, all are interpolated consistently.
    - Secondary coordinates are interpolated numerically. If they represent
      analytical transformations of the primary coordinate (e.g., wavelength = 1/wavenumber),
      the result may be approximate - consider recomputing them analytically after interpolation.
    - Unit conversion is performed if needed before interpolation.
    - Sequential interpolation is applied for multiple dimensions (not true N-D).
    """
    new = dataset if inplace else dataset.copy()

    # Only pass non-None dims to get_axis to avoid the dims=None shadowing dim=…
    # (_get_dims_from_args pops "dims" first, and if it exists (even as None)
    # the "dim" keyword is never consulted).
    kw = {}
    if dim is not None:
        kw["dim"] = dim
    if dims is not None:
        kw["dims"] = dims
    axis_list, dim_list = new.get_axis(only_first=False, **kw)

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

        target_from_array = False
        if isinstance(target_coord, NDDataset):
            target_coord = target_coord.coord(dim)
        elif isinstance(target_coord, np.ndarray):
            target_coord = Coord(target_coord)
            target_from_array = True
        elif not isinstance(target_coord, Coord):
            raise TypeError(
                f"coord must be Coord, np.ndarray, or NDDataset, got {type(target_coord)}"
            )

        old_coord = new.coord(dim)

        is_coordset = isinstance(old_coord, CoordSet)

        primary_old_coord = old_coord.default if is_coordset else old_coord

        if primary_old_coord is None or not primary_old_coord.has_data:
            raise ValueError(f"Dimension '{dim}' has no coordinate data to interpolate")

        if target_from_array:
            # A coordinate generated from a bare array carries no metadata. Keep
            # the semantics of the axis being interpolated by inheriting the
            # source coordinate's units and title (#1094). The array values are
            # assumed to already be expressed in the source coordinate's units,
            # so the units are *attached*, not converted.
            if primary_old_coord.has_units:
                target_coord.units = primary_old_coord.units
            target_coord.title = primary_old_coord.title

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

        # Point-wise label carry-over: for each target point, find the exactly
        # matching original index so labels can follow the data (#1098). Both
        # arrays are now expressed in the source coordinate's units.
        match_idx = _match_indices(old_data, new_data)

        sorted_old_x, sort_idx, _was_reversed = _validate_monotonic(
            primary_old_coord, assume_sorted
        )

        if sort_idx is not None and len(sort_idx) > 0:
            sorted_data = new._data[
                tuple(sort_idx if i == ax else slice(None) for i in range(new.ndim))
            ].copy()
        else:
            sorted_data = new._data

        # ``fill_value`` may be NaN (the default), a finite constant or the
        # string ``"extrapolate"``. ``interp1d`` (linear) handles all three
        # natively; ``PchipInterpolator`` only knows ``extrapolate`` True/False,
        # so the finite-constant case is applied by hand below so that
        # ``fill_value`` behaves consistently for both methods (#1093).
        extrapolate = isinstance(fill_value, str) and fill_value == "extrapolate"

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
                extrapolate=extrapolate,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        interpolated_data = interpolator(new_data)

        if (
            method == "pchip"
            and not extrapolate
            and not (isinstance(fill_value, float) and np.isnan(fill_value))
        ):
            # PCHIP leaves out-of-range points as NaN (``extrapolate=False``);
            # replace them with the requested constant so ``fill_value`` matches
            # the linear method. ``sorted_old_x`` is ascending, so this mask is
            # independent of the target-coordinate direction.
            out_of_range = (new_data < sorted_old_x[0]) | (new_data > sorted_old_x[-1])
            if out_of_range.any():
                interpolated_data[
                    tuple(
                        out_of_range if i == ax else slice(None)
                        for i in range(interpolated_data.ndim)
                    )
                ] = fill_value

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
            _match_idx=match_idx,
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
            # Carry the secondary coordinate's labels onto exactly-matching
            # target points, consistently with the primary coordinate (#1098).
            new_sec._labels = _carry_labels(coord._labels, _match_idx)
            return new_sec

        # Carry the primary coordinate's labels onto exactly-matching target
        # points (#1098); copy first so a user-supplied target is not mutated.
        target_coord = target_coord.copy()
        target_coord._labels = _carry_labels(primary_old_coord._labels, match_idx)

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
