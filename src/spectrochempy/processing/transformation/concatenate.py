# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["concatenate", "stack"]

__dataset_methods__ = __all__

from copy import deepcopy

import numpy as np

from spectrochempy.core.dataset.basearrays.ndarray import DEFAULT_DIM_NAME
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils import exceptions
from spectrochempy.utils.datetimeutils import utcnow
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.meta import Meta
from spectrochempy.utils.objects import OrderedSet


def concatenate(*datasets, **kwargs):
    r"""
    Concatenation of `NDDataset` objects along a given axis.

    Any number of `NDDataset` objects can be concatenated (by default
    the last on the last dimension). For this operation
    to be defined the following must be true :

        #. all inputs must be valid `NDDataset` objects;
        #. units of data must be compatible
        #. concatenation is along the axis specified or the last one;
        #. along the non-concatenated dimensions, shapes must match.

    Parameters
    ----------
    *datasets : positional `NDDataset` arguments
        The dataset(s) to be concatenated to the current dataset. The datasets
        must have the same shape, except in the dimension corresponding to axis
        (the last, by default).
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    -------
    out
        A `NDDataset` created from the contenations of the `NDDataset` input objects.

    Other Parameters
    ----------------
    dims : str, optional, default='x'
        The dimension along which the operation is applied.

    axis : int, optional
        The axis along which the operation is applied.

        For 1D datasets, ``axis=1`` promotes inputs to a 2D dataset and
        concatenates them as columns.

    See Also
    --------
    stack : Stack of `NDDataset` objects along a new dimension.

    Examples
    --------
    >>> A = scp.read('irdata/nh4y-activation.spg', protocol='omnic')
    >>> B = scp.read('irdata/nh4y-activation.scp')
    >>> C = scp.concatenate(A[10:], B[3:5], A[:10], axis=0)
    >>> A[10:].shape, B[3:5].shape, A[:10].shape, C.shape
    ((45, 5549), (2, 5549), (10, 5549), (57, 5549))

    or

    >>> D = A.concatenate(B, B, axis=0)
    >>> A.shape, B.shape, D.shape
    ((55, 5549), (55, 5549), (165, 5549))

    >>> E = A.concatenate(B, axis=1)
    >>> A.shape, B.shape, E.shape
    ((55, 5549), (55, 5549), (55, 11098))

    """
    # check use
    if "force_stack" in kwargs:
        deprecated("force_stack", replace="method stack()", removed="0.13.0")
        return stack(datasets)

    operation = kwargs.pop("_metadata_operation", "concatenate")
    metadata_sources = kwargs.pop("_metadata_sources", None)
    if operation not in {"concatenate", "stack"}:
        raise ValueError(f"Unsupported metadata operation context: {operation}")

    source_datasets = _normalize_datasets(datasets)
    source_datasets = [_reject_coord_input(dataset) for dataset in source_datasets]
    if metadata_sources is None:
        metadata_sources = source_datasets

    # get a copy of input datasets in order that input data are not modified
    datasets = [dataset.copy() for dataset in source_datasets]

    if _should_promote_1d_column_concatenation(datasets, kwargs):
        return _stack_1d_profiles_as_columns(
            datasets,
            metadata_sources=metadata_sources,
            operation=operation,
        )

    # get axis from arguments
    axis, dim = datasets[0].get_axis(**kwargs)

    # check shapes, except for dim along which concatenation will be done
    shapes = {ds.shape[:axis] + ds.shape[axis + 1 :] for ds in datasets}
    if len(shapes) != 1:
        raise exceptions.DimensionsCompatibilityError(
            "all input arrays must have the same shape",
        )

    # check units
    units = tuple({ds.units for ds in datasets})
    if len(units) == 1:
        units = datasets[0].units
    else:
        # check compatibility
        for i, u1 in enumerate(units[:-1]):
            for u2 in units[i + 1 :]:
                if u1.dimensionality != u2.dimensionality:
                    raise exceptions.UnitsCompatibilityError(
                        f"Units of the data are {[str(u) for u in units]}. The datasets can't be concatenated",
                    )
        # should be compatible, so convert
        units = datasets[0].units
        for ds in datasets[1:]:
            if ds.units != units:
                ds.ito(units)

    # concatenate or stack the data array + mask
    # --------------------------------------------
    sss = []

    # Extract metadata coordinates for domain-specific datasets
    # (e.g. variable-temperature TopSpin parameters for NMR).
    metacoords: dict[str, list] = {}
    from spectrochempy.plugins import manager as manager_module  # noqa: PLC0415

    extract = manager_module.plugin_manager.registry.get_handler(
        "concatenate.extract_metadata"
    )
    if extract is not None:
        result = extract(datasets)
        if result is not None:
            metacoords = result

    for _i, dataset in enumerate(datasets):
        d = dataset.masked_data
        sss.append(d)

    sconcat = np.ma.concatenate(sss, axis=axis)

    data = np.asarray(sconcat)
    mask = sconcat.mask

    # now manage coordinates and labels
    coords = datasets[0].coordset

    if coords is not None and not coords[dim].is_empty:
        coords = coords._concatenate_dim(dim, [ds.coordset for ds in datasets])

    out = dataset.copy()
    out._data = data
    if coords is not None:
        out._coordset[dim] = coords[dim]

    # Let plugins post-process the concatenation result.
    handler = manager_module.plugin_manager.registry.get_handler(
        "concatenate.postprocess"
    )
    if handler is not None:
        result = handler(out, datasets, metacoords=metacoords)
        if result is not None:
            out = result

    out._mask = mask
    out._units = units

    _apply_combination_metadata(out, metadata_sources, operation=operation)

    return out


def stack(*datasets, **kwargs):
    """
    Stack of `NDDataset` objects along a new dimension.

    Any number of `NDDataset` objects can be stacked. For this operation
    to be defined the following must be true :

    #. all inputs must be valid dataset objects,
    #. units of data and axis must be compatible (rescaling is applied
       automatically if necessary).

    Parameters
    ----------
    *datasets : a series of `NDDataset`
        The dataset to be stacked to the current dataset.
    **kwargs
        Optional keyword parameters.

    Other Parameters
    ----------------
    axis : int, optional, default=0
        Stacking axis.

        - ``axis=0`` preserves the historical behavior and stacks datasets
          along a new leading dimension.
        - ``axis=1`` is supported for 1D datasets and promotes them to a 2D
          dataset with profiles stacked as columns.

    Returns
    -------
    out
        A `NDDataset` created from the stack of the `datasets` datasets.

    See Also
    --------
    concatenate : Concatenate `NDDataset` objects along a given dimension.

    Examples
    --------
    >>> A = scp.read('irdata/nh4y-activation.spg', protocol='omnic')
    >>> B = scp.read('irdata/nh4y-activation.scp')
    >>> C = scp.stack(A, B)
    >>> print(C)
    NDDataset: [float64] a.u. (shape: (z:2, y:55, x:5549))

    Stack 1D profiles as columns in a 2D dataset

    >>> t = scp.linspace(0, 1, 5)
    >>> profiles = [np.exp(-((t - c) ** 2) / 0.05) for c in (0.2, 0.5)]
    >>> C = scp.stack(profiles, axis=1)
    >>> C.shape
    (5, 2)

    Create 1D datasets from a coordinate and stack them as columns:

    >>> time = scp.Coord.linspace(0, 1, 200, title="time")
    >>> data1 = scp.exp(-0.5 * ((time.data - 0.25) / 0.10) ** 2)
    >>> data2 = 0.8 * scp.exp(-0.5 * ((time.data - 0.55) / 0.12) ** 2)
    >>> c1 = scp.NDDataset(data1, coordset=[time], title="C1")
    >>> c2 = scp.NDDataset(data2, coordset=[time], title="C2")
    >>> profiles = scp.stack([c1, c2], axis=1)
    >>> profiles.shape
    (200, 2)

    """
    source_datasets = _normalize_datasets(datasets)
    source_datasets = [_reject_coord_input(dataset) for dataset in source_datasets]
    datasets = [dataset.copy() for dataset in source_datasets]
    # Reject Coord inputs — Coord must be explicitly wrapped in NDDataset
    axis = kwargs.pop("axis", 0)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"stack() got unexpected keyword argument(s): {unexpected}")

    if axis in (1, -1):
        return _stack_1d_profiles_as_columns(
            datasets,
            metadata_sources=source_datasets,
            operation="stack",
        )
    if axis not in (0, None):
        raise NotImplementedError("stack() currently supports only axis=0 or axis=1")

    shapes = {ds.shape for ds in datasets}
    if len(shapes) != 1:
        raise exceptions.DimensionsCompatibilityError(
            "all input arrays must have the same shape",
        )

    # prepend a new dimension
    for i, dataset in enumerate(datasets):
        dataset._data = dataset.data[np.newaxis]
        dataset._mask = dataset.mask[np.newaxis]
        newcoord = Coord([i], labels=[dataset.name])
        newcoord.name = (OrderedSet(DEFAULT_DIM_NAME) - dataset._dims).pop()
        dataset.add_coordset(newcoord)
        dataset.dims = [newcoord.name] + dataset.dims

    return concatenate(
        *datasets,
        dims=0,
        _metadata_operation="stack",
        _metadata_sources=source_datasets,
    )


def _stack_1d_profiles_as_columns(datasets, *, metadata_sources=None, operation):
    if not datasets:
        raise ValueError("stack() requires at least one dataset")

    # Reject Coord inputs — Coord must be explicitly wrapped in NDDataset
    datasets = [_reject_coord_input(d) for d in datasets]
    if metadata_sources is None:
        metadata_sources = datasets

    shapes = {ds.shape for ds in datasets}
    if len(shapes) != 1:
        raise exceptions.DimensionsCompatibilityError(
            "all input arrays must have the same shape",
        )

    ndim = {ds.ndim for ds in datasets}
    if ndim != {1}:
        raise NotImplementedError(
            "stack(axis=1) is currently implemented only for 1D datasets",
        )

    source_dims = {ds.dims[0] for ds in datasets}
    if len(source_dims) != 1:
        raise exceptions.DimensionsCompatibilityError(
            "all input 1D datasets must use the same dimension name",
        )

    promoted = []
    source_dim = datasets[0].dims[0]
    for i, dataset in enumerate(datasets):
        xcoord = dataset.coord(source_dim)
        newdim = (OrderedSet(DEFAULT_DIM_NAME) - dataset._dims).pop()
        masked = dataset.masked_data
        dataset._data = np.asarray(masked)[:, np.newaxis]
        dataset._mask = np.ma.getmaskarray(masked)[:, np.newaxis]
        dataset.dims = [source_dim, newdim]
        column_coord = Coord([i], labels=[dataset.name])
        column_coord.name = newdim
        dataset.set_coordset({source_dim: xcoord, newdim: column_coord})
        promoted.append(dataset)

    return concatenate(
        *promoted,
        dims=newdim,
        _metadata_operation=operation,
        _metadata_sources=metadata_sources,
    )


def _should_promote_1d_column_concatenation(datasets, kwargs):
    if not datasets:
        return False
    if any(ds.ndim != 1 for ds in datasets):
        return False

    axis = kwargs.get("axis")
    dims = kwargs.get("dims")
    dim = kwargs.get("dim")

    if axis in (1, -1):
        return True
    if dims == 1 or dim == 1:
        return True
    return False


# utility functions
# --------------------
def _reject_coord_input(obj):
    """
    Reject Coord inputs in stack/concatenate.

    Parameters
    ----------
    obj : object
        If a Coord, raises TypeError with an explicit message.
        Otherwise the object is returned unchanged.
    """
    if isinstance(obj, Coord):
        raise TypeError(
            "Coord inputs are ambiguous in stack/concatenate. "
            "Convert them explicitly to NDDataset with the intended coordinate axis.",
        )
    return obj


def _normalize_datasets(datasets):
    if isinstance(datasets, tuple) and isinstance(datasets[0], list | tuple):
        datasets = datasets[0]
    return list(datasets)


def _apply_combination_metadata(out, datasets, *, operation):
    names = _render_names(datasets)

    out._name = ""
    out._title = _consensus_text(dataset.title for dataset in datasets)
    out.description = _make_description(operation, names)
    out.author = _merge_authors(datasets)
    out.origin = _merge_origins(datasets)
    out.filename = None
    out._meta = _combine_meta(datasets)
    out._acquisition_date = _consensus_acquisition_date(datasets)
    out.history = [_make_history_entry(operation, names)]

    operation_time = utcnow()
    out._date = operation_time
    out._created = operation_time
    out._modified = operation_time


def _effective_text(value):
    if value in (None, ""):
        return None
    return str(value)


def _consensus_text(values):
    effective = [_effective_text(value) for value in values]
    if any(value is None for value in effective):
        return None
    first = effective[0]
    if all(value == first for value in effective[1:]):
        return first
    return None


def _deduplicated_texts(values):
    texts = []
    for value in values:
        text = _effective_text(value)
        if text is None or text in texts:
            continue
        texts.append(text)
    return texts


def _render_names(datasets):
    names = []
    for dataset in datasets:
        name = _effective_text(dataset._name)
        names.append(name if name is not None else "<unnamed>")
    return names


def _make_description(operation, names):
    label = "Concatenation" if operation == "concatenate" else "Stack"
    rendered = ", ".join(names)
    return f"{label} of {len(names)} datasets:\n( {rendered} )"


def _merge_authors(datasets):
    authors = _deduplicated_texts(dataset.author for dataset in datasets)
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    return " & ".join(authors)


def _merge_origins(datasets):
    origins = [_effective_text(dataset.origin) for dataset in datasets]
    if origins and all(origin is not None for origin in origins):
        first = origins[0]
        if all(origin == first for origin in origins[1:]):
            return first

    distinct = _deduplicated_texts(origins)
    if not distinct:
        return ""
    if len(distinct) == 1:
        return distinct[0]
    return f"multiple: {' | '.join(distinct)}"


def _combine_meta(datasets):
    first_meta = datasets[0].meta
    if all(dataset.meta == first_meta for dataset in datasets[1:]):
        return deepcopy(first_meta)
    return Meta()


def _consensus_acquisition_date(datasets):
    first = datasets[0]._acquisition_date
    if first is None:
        return None
    if all(dataset._acquisition_date == first for dataset in datasets[1:]):
        return deepcopy(first)
    return None


def _make_history_entry(operation, names):
    rendered = ", ".join(names)
    return f"Created by {operation} from {len(names)} datasets: {rendered}"
