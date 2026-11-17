# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module defining functions related to NDDataset alignment."""

__all__ = ["align"]
__dataset_methods__ = __all__

from decimal import Decimal

import numpy as np

from spectrochempy.application.application import error_
from spectrochempy.application.application import warning_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import ur
from spectrochempy.utils import exceptions
from spectrochempy.utils.constants import NOMASK
from spectrochempy.utils.numutils import get_n_decimals


def can_merge_or_align(coord1, coord2):
    """
    Check if two coordinates can be merged or aligned.

    Parameters
    ----------
    coord1, coord2 :  `Coord`
        Coordinates to merge or align.

    Returns
    -------
    can_merge, can_align : tuple of bools
        Two flags about merge and alignment possibility.

    """
    if coord1 == coord2:
        # same coordinates
        can_merge = True  # merge is obvious
        can_align = True  # of course as it is the same coordinate

    else:
        # no the same coordinates
        can_merge = False  # we need to do alignment to merge
        # can align only if data exists, units compatibles, and title are
        # the same
        can_align = True
        can_align &= not coord1.is_empty
        can_align &= not coord2.is_empty
        can_align &= coord1.title == coord2.title
        if can_align and (coord1.has_units or coord2.has_units):
            if coord1.has_units:
                can_align &= coord1.is_units_compatible(coord2)
            else:
                can_align &= coord2.is_units_compatible(coord1)

    return can_merge, can_align


def align(dataset, *others, **kwargs):
    r"""
    Align individual `NDDataset` along given dimensions using various methods.

    Parameters
    ----------
    dataset : `NDDataset`
        Dataset on which we want to align other objects.
    *others : `NDDataset`
        Objects to align.
    dim : str. Optional, default='x'
        Along which axis to perform the alignment.
    dims : list of str, optional, default=None
        Align along all dims defined in dims (if dim is also
        defined, then dims have higher priority).
    method : enum ['outer', 'inner', 'first', 'last', 'interpolate'], optional, default='outer'
        Which method to use for the alignment.

        If align is defined :

        * 'outer' means that a union of the different coordinates is
          achieved (missing values are masked).
        * 'inner' means that the intersection of the coordinates is used.
        * 'first' uses the first dataset's coordinates, masking positions
          with no matching coordinate.
        * 'last' does the same using the last dataset's coordinates.
        * 'interpolate' interpolates onto the first dataset's coordinates.
          Points outside the source range are masked.

        Outer and reference joins match with at least five significant digits
        at each coordinate's scale. If any input coordinate disables rounding,
        those joins use full precision.

    interpolate_method : enum ['linear','pchip']. Optional, default='linear'
        Method of interpolation to performs for the alignment.
    interpolate_sampling : {'auto'}, optional, default='auto'
        Sampling of the target coordinate when ``method='interpolate'``.
        ``'auto'`` uses the target grid unchanged. Pass ``coord`` to supply
        an explicit grid. A numeric sampling interval is unsupported.

    coord :  `Coord` , optional, default=None
        Explicit target for interpolation along a single dimension. A
        unitless target is interpreted in the reference coordinate's units.
    copy : bool, optional, default=True
        If False, avoid extra copies where possible. Alignment can still
        allocate data buffers and does not guarantee memory sharing.

    Returns
    -------
    tuple of `NDDataset`
        Same objects as datasets with dimensions aligned.

    Raises
    ------
    ValueError
        Issued when the dimensions given in `dim` or `dims` argument are not
        compatibles (units, titles, etc.), or an inner join finds no shared
        coordinates or encounters duplicate coordinates after rounding.

    NotImplementedError
        Issued when ``interpolate_sampling`` is not ``'auto'``.

    """
    # TODO: Perform an alignment along numeric labels
    # TODO: add example in docs

    # copy objects?
    copy = kwargs.pop("copy", True)

    # make a single list with dataset and the remaining object
    objects = [dataset] + list(others)

    # should we align on given external coordinates
    extern_coord = kwargs.pop("coord", None)

    # what's the method to use (by default='outer')
    method = kwargs.pop("method", "outer")

    # Only the existing first-coordinate target is implemented. Constructing
    # a new grid from a numeric sampling request needs an explicit policy for
    # bounds, units and irregular source coordinates.
    interpolate_sampling = kwargs.pop("interpolate_sampling", "auto")
    if not (isinstance(interpolate_sampling, str) and interpolate_sampling == "auto"):
        raise NotImplementedError(
            "align() only supports interpolate_sampling='auto', which uses the "
            "target coordinate unchanged. To use a different grid, pass an "
            "explicit coord with method='interpolate'."
        )

    # trivial cases where alignment is not possible or unnecessary
    if not objects:
        warning_("No object provided for alignment!")
        return None

    if (
        len(objects) == 1
        and objects[0]._implements("NDDataset")
        and extern_coord is None
    ):
        # no necessary alignment
        return objects

    # evaluate on which axis we align
    axis, dims = dataset.get_axis(only_first=False, **kwargs)

    if extern_coord is not None:
        if method != "interpolate" or len(dims) != 1:
            raise ValueError(
                "coord requires method='interpolate' and one alignment dimension."
            )
        if not isinstance(extern_coord, Coord):
            raise TypeError("The explicit alignment target must be a Coord.")
        if (
            extern_coord.data is None
            or extern_coord.data.size == 0
            or extern_coord.data.dtype.kind not in "biuf"
            or not np.isfinite(extern_coord.data).all()
            or extern_coord.is_masked
        ):
            raise ValueError(
                "The explicit alignment target must be finite, unmasked and numeric."
            )

    # check compatibility of the dims and prepare the dimension for alignment
    for _axis, dim in zip(axis, dims, strict=False):
        # get all objects to align
        _objects = {}
        _nobj = 0

        for idx, object in enumerate(objects):
            if not object._implements("NDDataset"):
                error_(
                    f"Bad object(s) found: {object}. Note that only NDDataset "
                    f"objects are accepted "
                    f"for alignment",
                )
                return None

            coord = object.coord(dim)
            if (
                coord is None
                or coord.data is None
                or coord.data.size == 0
                or coord.data.dtype.kind not in "biuf"
                or not np.isfinite(coord.data).all()
                or coord.is_masked
            ):
                raise ValueError(
                    f"Alignment requires finite, unmasked numeric coordinates "
                    f"along dimension {dim!r}."
                )

            working = object.copy()
            if (
                method == "inner"
                and object.data.dtype.kind in "biufc"
                and working.data.dtype != object.data.dtype
            ):
                working._dtype = object.data.dtype
                working._data = object.data
            _objects[_nobj] = {
                "obj": working,
                "idx": idx,
            }
            _nobj += 1

        _last = _nobj - 1

        # get the reference object (by default the first, except if method if
        # set to 'last'
        ref_obj_index = 0
        if method == "last":
            ref_obj_index = _last

        ref_obj = _objects[ref_obj_index]["obj"]

        # as we will sort their coordinates at some point, we need to know
        # if the coordinates need to be reversed at
        # the end of the alignment process
        reversed = ref_obj.coordset[dim].reversed
        if reversed and method in {"inner", "outer"}:
            ref_obj.sort(descend=False, dim=dim, inplace=True)

        # get the coordset corresponding to the reference object
        ref_obj_coordset = ref_obj.coordset

        # get the coordinate for the reference dimension
        ref_coord = ref_obj_coordset[dim]

        # as we will sort their coordinates at some point, we need to know
        # if the coordinates need to be reversed at
        # the end of the alignment process
        reversed = ref_coord.reversed

        # prepare a new Coord object to store the final new dimension
        new_coord = ref_coord.copy()

        ndec = get_n_decimals(ref_coord.data.max(), 5) if method == "inner" else None
        reference_join = method in {"outer", "first", "last"}
        input_coords = [item["obj"].coordset[dim] for item in _objects.values()]
        rounded = all(coord._rounding for coord in input_coords)
        precision = max(5, *(coord.sigdigits for coord in input_coords))
        source_grids = {}
        contextual_grids = []

        # loop on all object
        for _index, object in _objects.items():
            obj = object["obj"]

            if obj is not ref_obj and reversed and method in {"inner", "outer"}:
                obj.sort(descend=False, dim=dim, inplace=True)

            coord = obj.coordset[dim]
            if not coord.is_units_compatible(ref_coord):
                raise exceptions.UnitsCompatibilityError(
                    exceptions.format_incompatible_units_message(
                        "align datasets",
                        coord.units,
                        ref_coord.units,
                        dim=dim,
                    ),
                )

            values = coord.data
            reconcile_roundoff = False
            if coord.units != ref_coord.units:
                source_unit = coord.units or ur.dimensionless
                target_unit = ref_coord.units or ur.dimensionless
                coord.ito(ref_coord)
                converted = coord.data
                if reference_join and not rounded:
                    if (
                        source_unit.dimensionality == target_unit.dimensionality
                        and source_unit.dimensionality
                    ):
                        decimal_values = np.fromiter(
                            (
                                float(
                                    ur.Quantity(Decimal(str(point)), source_unit)
                                    .to(target_unit)
                                    .magnitude
                                )
                                for point in values
                            ),
                            dtype=np.float64,
                            count=coord.size,
                        )
                        if np.allclose(decimal_values, converted, rtol=1e-14, atol=0):
                            converted = decimal_values
                        else:
                            reconcile_roundoff = True
                    else:
                        reconcile_roundoff = True
                values = converted

            keys = values
            if ndec is not None:
                keys = np.around(values, ndec)
            elif reference_join and rounded:
                keys = np.fromiter(
                    (float(f"{point:.{precision}g}") for point in values),
                    dtype=np.float64,
                    count=coord.size,
                )

            if not np.isfinite(keys).all():
                raise ValueError(
                    f"Alignment produced non-finite coordinates along dimension {dim!r}."
                )
            if method != "interpolate" and np.unique(keys).size != keys.size:
                reason = (
                    "Coordinate rounding makes"
                    if method == "inner"
                    else "Duplicate matching coordinates make"
                )
                raise ValueError(
                    f"{reason} {method} alignment ambiguous along dimension {dim!r}."
                )
            source_grids[_index] = keys, values
            if reconcile_roundoff:
                contextual_grids.append(_index)

        if contextual_grids:
            if method == "outer":
                anchors = {
                    point
                    for index, (keys, _) in source_grids.items()
                    if index not in contextual_grids
                    for point in keys
                }
            else:
                anchors = set(source_grids[ref_obj_index][0])
            for index in contextual_grids:
                keys, values = source_grids[index]
                ordered = np.asarray(sorted(anchors))
                if ordered.size:
                    pos = np.searchsorted(ordered, keys)
                    left = ordered[np.maximum(pos - 1, 0)]
                    right = ordered[np.minimum(pos, ordered.size - 1)]
                    tolerance = 4 * np.abs(np.spacing(keys))
                    near_left = np.abs(keys - left) <= tolerance
                    near_right = np.abs(keys - right) <= tolerance
                    ambiguous = near_left & near_right & (left != right)
                    if np.any(ambiguous & (keys != left) & (keys != right)):
                        raise ValueError(
                            f"Unit conversion makes {method} alignment ambiguous "
                            f"along dimension {dim!r}."
                        )
                    keys = np.where(
                        (keys == left) | (near_left & ~near_right),
                        left,
                        np.where(near_right, right, keys),
                    )
                    if np.unique(keys).size != keys.size:
                        raise ValueError(
                            f"Unit conversion makes {method} alignment ambiguous "
                            f"along dimension {dim!r}."
                        )
                source_grids[index] = keys, values
                if method == "outer":
                    anchors.update(keys)

        join_grid = source_grids[ref_obj_index][0]
        if method == "inner":
            join_keys = set(join_grid)
            for keys, _ in source_grids.values():
                join_keys.intersection_update(keys)
            if not join_keys:
                raise ValueError(f"No shared coordinates along dimension {dim!r}.")
            join_grid = np.asarray(sorted(join_keys))
            new_coord._data = join_grid
            new_coord._rounding = False
        elif method == "outer":
            representatives = {}
            for index, (keys, values) in source_grids.items():
                if index in contextual_grids:
                    values = keys
                for key, value in zip(keys, values, strict=True):
                    representatives.setdefault(key, value)
            join_grid = np.asarray(sorted(representatives))
            new_coord._data = np.asarray([representatives[key] for key in join_grid])
            new_coord._rounding = False
        elif method not in {"first", "last", "interpolate"}:
            raise NotImplementedError(f"The method {method} is unknown!")

        # For interpolate method, use reference coordinate as target
        # instead of the union
        if method == "interpolate":
            if extern_coord is None:
                target_coord = ref_coord.copy()
            else:
                target_coord = extern_coord.copy()
                if target_coord.units is None:
                    target_coord.units = ref_coord.units
                elif not target_coord.is_units_compatible(ref_coord):
                    raise exceptions.UnitsCompatibilityError(
                        exceptions.format_incompatible_units_message(
                            "align to explicit coordinates",
                            target_coord.units,
                            ref_coord.units,
                            dim=dim,
                        ),
                    )
                elif target_coord.units != ref_coord.units:
                    target_coord.ito(ref_coord.units)
                if not np.isfinite(target_coord.data).all():
                    raise ValueError(
                        f"Alignment target produced non-finite coordinates "
                        f"along dimension {dim!r}."
                    )
            target_coord._labels = None
        else:
            target_coord = new_coord

        # Now perform alignment of all objects on the new coordinates
        for index, object in _objects.items():
            obj = object["obj"]

            if method == "interpolate":
                # Perform actual interpolation onto reference coordinates
                interpolate_method = kwargs.get("interpolate_method", "linear")
                obj = obj.interpolate(
                    dim=dim,
                    coord=target_coord,
                    method=interpolate_method,
                    fill_value=np.nan,
                    inplace=False,
                )
            else:
                dim_index = obj.dims.index(dim)
                coord = obj.coordset[dim]
                new_obj = obj.copy() if copy and method != "inner" else obj
                new_coordset = obj.coordset.copy()

                if method == "inner":
                    _, source_indices, target_indices = np.intersect1d(
                        source_grids[index][0],
                        join_grid,
                        return_indices=True,
                    )
                    source_indices = source_indices[np.argsort(target_indices)]
                    mask = (
                        np.take(obj.mask, source_indices, axis=dim_index)
                        if obj.is_masked
                        else NOMASK
                    )
                    values = np.take(obj.data, source_indices, axis=dim_index)
                    if values.dtype.kind not in "biufc":
                        values = values.astype(np.float64)
                        new_obj._dtype = None
                    new_obj._data = values
                    new_obj._mask = mask
                    output_coord = new_coord.copy()
                    output_coord._labels = (
                        coord.labels[source_indices] if coord.is_labeled else None
                    )
                else:
                    _, source_indices, target_indices = np.intersect1d(
                        source_grids[index][0], join_grid, return_indices=True
                    )
                    shape = list(obj.shape)
                    shape[dim_index] = len(new_coord)
                    dtype = (
                        np.result_type(obj.data.dtype, np.float64)
                        if obj.data.dtype.kind == "c"
                        else np.float64
                    )
                    values = np.full(shape, np.nan, dtype=dtype)
                    mask = np.ones(shape, dtype=bool)
                    loc = (*([slice(None)] * dim_index), target_indices)
                    values[loc] = np.take(obj.data, source_indices, axis=dim_index)
                    mask[loc] = (
                        np.take(obj.mask, source_indices, axis=dim_index)
                        if obj.is_masked
                        else False
                    )
                    new_obj._data = values
                    new_obj._mask = mask if mask.any() else NOMASK

                    output_coord = new_coord.copy()
                    output_coord._labels = None
                    if coord.is_labeled:
                        labels = coord.labels[source_indices]
                        if target_indices.size == len(new_coord):
                            output_coord._labels = labels[np.argsort(target_indices)]
                        else:
                            label_shape = (len(new_coord), *labels.shape[1:])
                            output_coord._labels = np.full(
                                label_shape, "--", dtype=np.object_
                            )
                            output_coord._labels[target_indices] = labels

                setattr(new_coordset, dim, output_coord)
                new_obj._coordset = new_coordset

                # reversed?
                if reversed and method in {"inner", "outer"}:
                    # we must reverse the given coordinates
                    new_obj.sort(descend=reversed, dim=dim, inplace=True)

                obj = new_obj

            # update the _objects
            _objects[index]["obj"] = obj

        # the new transformed object must be in the same order as the passed
        # objects
        # and the missing values must be masked (for the moment they are defined to NaN

        for _index, object in _objects.items():
            obj = object["obj"]
            # NOTE: NaN is deliberately replaced with a sentinel value
            # rather than MASKED (np.ma.masked).  The mask is already set
            # correctly above, so .masked_data is accurate.  However,
            # __eq__ compares raw .data directly (IEEE 754 NaN != NaN),
            # and ~30 processing sites access .data instead of
            # .masked_data.  A sentinel avoids silent breakage in both
            # cases.  See audit/~align-sentinel-investigation.md.
            data = obj.data
            missing = np.isnan(data)
            if missing.any():
                if method == "interpolate":
                    obj._mask = np.logical_or(obj.mask, missing)
                if data.dtype.kind in "fc":
                    dtype = np.promote_types(data.dtype, np.float64)
                    if dtype != data.dtype:
                        obj._dtype = dtype
                        obj._data = data.astype(dtype)
                        data = obj.data
                data[missing] = 99999999999999.0
            idx = int(object["idx"])
            objects[idx] = obj

            # we also transform into linear coord if possible ?
            # TODO:

    # Now return

    return tuple(objects)
