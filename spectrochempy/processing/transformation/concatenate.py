# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["concatenate", "stack"]

__dataset_methods__ = __all__

from warnings import warn

import numpy as np

from spectrochempy.core.dataset.baseobjects.ndarray import DEFAULT_DIM_NAME
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.utils import exceptions
from spectrochempy.utils.datetimeutils import utcnow
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.orderedset import OrderedSet


def concatenate(*datasets, **kwargs):
    """
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
    \*datasets : positional `NDDataset` arguments
        The dataset(s) to be concatenated to the current dataset. The datasets
        must have the same shape, except in the dimension corresponding to axis
        (the last, by default).
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    out
        A `NDDataset` created from the contenations of the `NDDataset` input objects.

    Other Parameters
    ----------------
    dims : str, optional, default='x'
        The dimension along which the operation is applied.

    axis : int, optional
        The axis along which the operation is applied.

    See Also
    ---------
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

    # check uise
    if "force_stack" in kwargs:
        deprecated("force_stack", replace="method stack()")
        return stack(datasets)

    # get a copy of input datasets in order that input data are not modified
    datasets = _get_copy(datasets)

    # get axis from arguments
    axis, dim = datasets[0].get_axis(**kwargs)

    # check shapes, except for dim along which concatenation will be done
    shapes = {ds.shape[:axis] + ds.shape[axis + 1 :] for ds in datasets}
    if len(shapes) != 1:
        raise exceptions.DimensionsCompatibilityError(
            "all input arrays must have the same shape"
        )

    # check units
    units = tuple(set(ds.units for ds in datasets))
    if len(units) == 1:
        units = datasets[0].units
    else:
        # check compatibility
        for i, u1 in enumerate(units[:-1]):
            for u2 in units[i + 1 :]:
                if u1.dimensionality != u2.dimensionality:
                    raise exceptions.UnitsCompatibilityError(
                        f"Units of the data are {[str(u) for u in units]}. The datasets can't be concatenated"
                    )
        # should be compatible, so convert
        units = datasets[0].units
        for ds in datasets[1:]:
            if ds.units != units:
                ds.ito(units)

    # concatenate or stack the data array + mask
    # --------------------------------------------
    sss = []

    if datasets[0].origin == "topspin":
        # we can use metadata to create new coordinates
        metacoords = {}
        meta0 = datasets[0].meta
        for i, dataset in enumerate(datasets):
            if i == 0:
                continue
            meta = dataset.meta
            for item in meta0:
                if item in ["file_size", "pprog", "phc0", "phc1", "nsold"]:
                    continue
                keepitem = item if item != "date" else "timestamp"
                if np.any(meta0[item][-1] != meta[item][-1]):
                    if hasattr(meta0[item][-1], "size") and meta0[item][-1].size > 1:
                        # case of pulse length or delays for instance
                        for i in range(meta0[item][-1].size):
                            if np.any(meta0[item][-1][i] == meta[item][-1][i]):
                                continue
                            itemi = f"{item}{i}"
                            if itemi not in metacoords:
                                metacoords[itemi] = [
                                    meta0[item][-1][i],
                                    meta[item][-1][i],
                                ]
                            else:
                                metacoords[itemi].append(meta[item][-1][i])
                        continue
                    if keepitem not in metacoords:
                        metacoords[keepitem] = [meta0[item][-1], meta[item][-1]]
                    else:
                        metacoords[keepitem].append(meta[item][-1])

    for i, dataset in enumerate(datasets):
        d = dataset.masked_data
        sss.append(d)

    sconcat = np.ma.concatenate(sss, axis=axis)

    data = np.asarray(sconcat)
    mask = sconcat.mask

    # now manage coordinates and labels
    coords = datasets[0].coordset

    if coords is not None:
        if not coords[dim].is_empty:
            labels = []
            if coords[dim].is_labeled:
                for ds in datasets:
                    labels.append(ds.coordset[dim].labels)

            if coords[dim]._implements() in ["Coord"]:
                coords[dim] = Coord(coords[dim])
                if labels != []:
                    coords[dim]._labels = np.concatenate(labels)
            elif coords[dim]._implements("CoordSet"):
                if labels != []:
                    labels = np.array(labels, dtype=object)
                    for i, coord in enumerate(coords[dim]._coords):
                        try:
                            labels_not_none = np.all(
                                labels[:, i] != [None] * len(labels[:, i])
                            )
                        except ValueError:
                            labels_not_none = True
                        if labels_not_none:
                            coord._labels = np.concatenate(
                                [label for label in labels[:, i]]
                            )
            coord_data_tuple = tuple((ds.coordset[dim].data for ds in datasets))
            none_coord = len([x for x in coord_data_tuple if x is None])
            if not none_coord:
                coords[dim]._data = np.concatenate(coord_data_tuple)
            else:
                warn(f"Some dataset(s) coordinates in the {dim} dimension are None.")

    out = dataset.copy()
    out._data = data
    if coords is not None:
        out._coordset[dim] = coords[dim]

    # for topspin data, we can create new coordinates from metadata
    if datasets[0].origin == "topspin" and metacoords != {}:
        c = []
        for item in metacoords:
            c.append(Coord(metacoords[item], title=item))
        out.y = CoordSet(c)

    out._mask = mask
    out._units = units

    out.description = f"Concatenation of {len(datasets)}  datasets:\n"
    out.description += "( {}".format(datasets[0].name)
    out.title = datasets[0].title
    authortuple = (datasets[0].author,)

    for dataset in datasets[1:]:
        if out.title != dataset.title:
            warn("Different data title => the title is that of the 1st dataset")

        if not (dataset.author in authortuple):
            authortuple = authortuple + (dataset.author,)

        out.author = " & ".join([str(author) for author in authortuple])

        out.description += ", {}".format(dataset.name)

    out.description += " )"
    out._date = out._modified = utcnow()
    out.history = ["Created by concatenate"]

    return out


def stack(*datasets):
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

    Returns
    --------
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
    """

    datasets = _get_copy(datasets)

    shapes = {ds.shape for ds in datasets}
    if len(shapes) != 1:
        raise exceptions.DimensionsCompatibilityError(
            "all input arrays must have the same shape"
        )

    # prepend a new dimension
    for i, dataset in enumerate(datasets):
        dataset._data = dataset.data[np.newaxis]
        dataset._mask = dataset.mask[np.newaxis]
        newcoord = Coord([i], labels=[dataset.name])
        newcoord.name = (OrderedSet(DEFAULT_DIM_NAME) - dataset._dims).pop()
        dataset.add_coordset(newcoord)
        dataset.dims = [newcoord.name] + dataset.dims

    return concatenate(*datasets, dims=0)


# utility functions
# --------------------
def _get_copy(datasets):
    # get a copy of datasets from the input
    if isinstance(datasets, tuple):
        if isinstance(datasets[0], (list, tuple)):
            datasets = datasets[0]
    return [ds.copy() for ds in datasets]
