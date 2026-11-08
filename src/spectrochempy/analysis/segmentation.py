# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Connected segmentation on the native sample grid."""

import numpy as np
from pint import Quantity as PintQuantity
from pint.errors import PintError
from pint.errors import UndefinedUnitError
from scipy import ndimage

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity

__all__ = ["segment_regions"]
__dataset_methods__ = ["segment_regions"]


def _label_components(foreground, connectivity):
    """Label a Boolean grid in order of first occurrence in native indices."""
    rank = 1 if connectivity == "face" else foreground.ndim
    structure = ndimage.generate_binary_structure(foreground.ndim, rank)
    labels, count = ndimage.label(foreground, structure=structure, output=np.intp)
    if count == 0:
        return labels, (), ()

    flat_labels = labels.ravel()
    occupied = np.flatnonzero(flat_labels)
    first = np.full(count + 1, flat_labels.size, dtype=np.intp)
    np.minimum.at(first, flat_labels[occupied], occupied)
    order = np.argsort(first[1:]) + 1
    mapping = np.zeros(count + 1, dtype=np.intp)
    mapping[order] = np.arange(1, count + 1)
    labels = mapping[labels]
    return labels, np.bincount(labels.ravel())[1:], ndimage.find_objects(labels)


def _threshold_value(value):
    """Validate a real threshold magnitude without casting to float64."""
    if isinstance(value, (bool, np.bool_, np.timedelta64)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("threshold must be a finite real scalar")
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("threshold must be a finite real scalar")
    return value


def _above_threshold(data, threshold):
    """Compare intensities without rounding an integer operand across the threshold."""
    if data.dtype.kind in "iu":
        limit = (
            int(np.floor(threshold))
            if isinstance(threshold, (float, np.floating))
            else int(threshold)
        )
        bounds = np.iinfo(data.dtype)
        if limit < bounds.min:
            return np.ones(data.shape, dtype=bool)
        if limit >= bounds.max:
            return np.zeros(data.shape, dtype=bool)
        return data > data.dtype.type(limit)
    if isinstance(threshold, (int, np.integer)):
        limit = int(threshold)
        largest = int(np.finfo(data.dtype).max)
        if limit < -largest:
            return np.ones(data.shape, dtype=bool)
        if limit >= largest:
            return np.zeros(data.shape, dtype=bool)
        cutoff = data.dtype.type(limit)
        if int(cutoff) > limit:
            cutoff = np.nextafter(cutoff, data.dtype.type(-np.inf))
        return data > cutoff
    return data > np.atleast_1d(threshold)


def _import_threshold(quantity):
    """Transfer a Pint scalar without re-scaling an equivalent unit."""
    magnitude, units = quantity.to_tuple()
    foreign_quantity = type(quantity)
    try:
        local = Quantity.from_tuple((magnitude, units))
        for reference in (0, 1, 2):
            foreign_root, foreign_units = (
                foreign_quantity.from_tuple((reference, units))
                .to_root_units()
                .to_tuple()
            )
            local_root, local_units = (
                Quantity.from_tuple((reference, units)).to_root_units().to_tuple()
            )
            if foreign_root != local_root or dict(foreign_units) != dict(local_units):
                break
        else:
            return local
    except UndefinedUnitError:
        pass
    return Quantity.from_tuple(quantity.to_root_units().to_tuple())


def _threshold_record(value, units):
    """Record a threshold magnitude and its unit in structured history."""
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        value = str(value)
    return {"value": value, "units": None if units is None else str(units)}


def segment_regions(dataset, *, threshold, dim=None, axis=None, connectivity="face"):
    """
    Label connected regions above a threshold in a real 1D or 2D spectrum.

    Parameters
    ----------
    dataset : NDDataset
        Nonempty real numeric spectrum. Masked and non-finite cells are excluded.
    threshold : float or Quantity
        Finite scalar threshold. Only values strictly above it are included.
        Plain numbers use the dataset intensity unit. A Pint quantity requires
        compatible dataset units.
    dim, axis : str, int or sequence
        Exactly one selector must be supplied and cover every dataset axis.
        Names and integer indices can be mixed. The output follows native axis order.
    connectivity : {"face", "full"}, optional
        Immediate axial neighbors by default. "full" includes diagonals in 2D.

    Returns
    -------
    labels : NDDataset
        Integer region labels with the input shape and coordinates and no
        intensity unit. Zero marks background. Excluded cells are zero and masked.
        Positive labels are ordered by each region's first native grid index.
    regions : tuple of dict
        One row per label, with keys "label", "size" (cell count), and "bounds".
        Bounds map each dimension name to a half-open pair of sample indices.

    Notes
    -----
    Connectivity is defined in index space, including at the grid boundaries.
    Bounding boxes can overlap. Use the label map to recover exact membership.

    Examples
    --------
    Two samples that touch only diagonally form separate regions with the
    default connectivity.

    >>> import spectrochempy as scp
    >>> spectrum = scp.NDDataset([[3, 0], [0, 2]])
    >>> labels, regions = spectrum.segment_regions(threshold=1, dim=("y", "x"))
    >>> labels.data.tolist()
    [[1, 0], [0, 2]]
    >>> regions[1]["bounds"]
    {'y': (1, 2), 'x': (1, 2)}

    Include diagonal neighbors to merge the samples into one region.

    >>> labels, regions = spectrum.segment_regions(
    ...     threshold=1, dim=("y", "x"), connectivity="full"
    ... )
    >>> labels.data.tolist()
    [[1, 0], [0, 1]]
    >>> regions[0]["size"]
    2
    """
    if dataset.ndim not in (1, 2) or dataset.size == 0:
        raise ValueError("segment_regions requires a nonempty 1D or 2D dataset")
    if (dim is None) == (axis is None):
        raise ValueError("Supply exactly one of dim or axis")
    selector = dim if dim is not None else axis
    requested = tuple(selector) if isinstance(selector, (list, tuple)) else (selector,)
    if len(requested) != dataset.ndim:
        raise ValueError("The selector must cover all dataset axes")
    if any(
        isinstance(item, (bool, np.bool_, np.timedelta64))
        or not isinstance(item, (str, int, np.integer))
        for item in requested
    ):
        raise TypeError("Select each axis by name or integer index")
    requested = tuple(
        int(item) if isinstance(item, np.integer) else item for item in requested
    )
    axes = dataset._get_dims_index(requested)
    if set(axes) != set(range(dataset.ndim)):
        raise ValueError("The selector must cover all dataset axes exactly once")
    if not isinstance(connectivity, str) or connectivity not in ("face", "full"):
        raise ValueError("connectivity must be 'face' or 'full'")

    data = np.asarray(dataset.data)
    if data.dtype.kind not in "iuf":
        raise ValueError("segment_regions requires real numeric data")
    if isinstance(threshold, PintQuantity):
        requested_value = _threshold_value(threshold.magnitude)
        requested_units = threshold.units
        if dataset.units is None:
            raise ValueError("A unit-bearing threshold requires dataset units")
        try:
            if not isinstance(threshold, Quantity):
                threshold = _import_threshold(threshold)
            effective_value = _threshold_value(threshold.to(dataset.units).magnitude)
        except PintError as exc:
            raise ValueError(
                "Threshold units must match the dataset intensity"
            ) from exc
    else:
        requested_value = effective_value = _threshold_value(threshold)
        requested_units = dataset.units

    invalid = np.asarray(dataset.mask, dtype=bool) | ~np.isfinite(data)
    foreground = ~invalid & _above_threshold(data, effective_value)
    values, sizes, boxes = _label_components(foreground, connectivity)
    regions = tuple(
        {
            "label": label,
            "size": int(size),
            "bounds": {
                name: (span.start, span.stop)
                for name, span in zip(dataset.dims, box, strict=True)
            },
        }
        for label, (size, box) in enumerate(zip(sizes, boxes, strict=True), start=1)
    )
    labels = NDDataset(
        values,
        dtype=values.dtype,
        dims=list(dataset.dims),
        mask=invalid,
        title="segmentation labels",
        coordset=dataset.coordset.copy() if dataset.coordset is not None else None,
        history=dataset.history_entries,
    )
    labels._append_history_entry(
        operation="segment_regions",
        parameters={
            "selector": "dim" if dim is not None else "axis",
            "requested": list(requested),
            "axes": list(axes),
            "dims": [dataset.dims[index] for index in axes],
            "threshold": {
                "requested": _threshold_record(requested_value, requested_units),
                "effective": _threshold_record(effective_value, dataset.units),
            },
            "comparison": ">",
            "connectivity": connectivity,
            "invalid_policy": "exclude_masked_nonfinite",
            "n_regions": len(regions),
        },
        message=f"Segmented {len(regions)} connected region(s)",
    )
    return labels, regions
