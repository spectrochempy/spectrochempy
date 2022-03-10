# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

__all__ = ["concatenate", "stack"]

__dataset_methods__ = __all__

import numpy as np
import datetime as datetime
from warnings import warn
from orderedset import OrderedSet

# from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.ndarray import DEFAULT_DIM_NAME
from spectrochempy.utils import SpectroChemPyWarning, DimensionsCompatibilityError


def concatenate(*datasets, **kwargs):
    """
    Concatenation of |NDDataset| objects along a given axis.

    Any number of |NDDataset| objects can be concatenated (by default
    the last on the last dimension). For this operation
    to be defined the following must be true :

        #. all inputs must be valid |NDDataset| objects;
        #. units of data and axis must be compatible
        #. concatenation is along the axis specified or the last one;
        #. along the non-concatenated dimensions, shape and units coordinates
           must match.

    Parameters
    ----------
    *datasets : positional |NDDataset| arguments
        The dataset(s) to be concatenated to the current dataset. The datasets
        must have the same shape, except in the dimension corresponding to axis
        (the last, by default).
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    out
        A |NDDataset| created from the contenations of the |NDDataset| input objects.

    Other Parameters
    ----------------
    dims : str, optional, default='x'
        The dimension along which the operation is applied.
    force_stack : bool, optional, default=False
        If True, the dataset are stacked instead of being concatenated. This means that a new dimension is prepended
        to each dataset before being stacked, except if one of the dimension is of size one. If this case the datasets
        are squeezed before stacking. The stacking is only possible is the shape of the various datasets are identical.
        This process is equivalent of using the method `stack`.

    See Also
    ---------
    stack : Stack of |NDDataset| objects along the first dimension.

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

    Stacking of datasets:
    for nDimensional datasets (with the same shape), a new dimension is added

    >>> F = A.concatenate(B, force_stack=True)
    >>> A.shape, B.shape, F.shape
    ((55, 5549), (55, 5549), (2, 55, 5549))

    If one of the dimensions is of size one, then this dimension is removed before stacking

    >>> G = A[0].concatenate(B[0], force_stack=True)
    >>> A[0].shape, B[0].shape, G.shape
    ((1, 5549), (1, 5549), (2, 5549))
    """

    # ------------------------------------------------------------------------
    # checks dataset validity
    # ------------------------------------------------------------------------

    # We must have a list of datasets
    if isinstance(datasets, tuple):
        if isinstance(datasets[0], (list, tuple)):
            datasets = datasets[0]

    # make a copy of the objects (in order that input data are not modified)
    datasets = [ds.copy() for ds in datasets]

    # a flag to force stacking of dataset instead of the default concatenation
    force_stack = kwargs.get("force_stack", False)

    # try assuming homogeneous NDDataset
    try:
        if force_stack:
            # when stacking, we add a new first dimension except if this dimension is of size one,
            # in this case we use this dimension for stacking

            if datasets[0].shape[0] != 1:
                for i, dataset in enumerate(datasets):
                    datasets[i] = _prepend_dim(dataset, i)

            axis, dim = datasets[0].get_axis(dim=0)

        else:
            # get axis from arguments (or set it to the default)
            axis, dim = datasets[0].get_axis(**kwargs)

        units = datasets[0].units
        coords = datasets[0].coordset

        if coords is not None:
            # check labels, coords type and concatenate them

            labels = []
            if coords[dim].implements() in ["Coord", "LinearCoord"]:
                if coords[dim].is_labeled:
                    for ds in datasets:
                        labels.append(ds.coordset[dim].labels)
                    coords[dim]._labels = np.concatenate(labels)

            if coords[dim].implements("CoordSet"):
                for coord in coords[dim]:
                    if coord.is_labeled:
                        for ds in datasets:
                            labels.append(ds.coordset[dim].labels)

                    for i, coord in enumerate(coords[dim]):
                        coord._labels = np.concatenate(labels[i :: len(coords[dim])])

            coords[dim] = Coord(coords[dim], linear=False)
            coords[dim]._data = np.concatenate(
                tuple((ds.coordset[dim].data for ds in datasets))
            )

        # concatenate or stack the data array + mask
        # ------------------------------------------------------------------------

        sss = []
        for i, dataset in enumerate(datasets):
            d = dataset.masked_data
            sss.append(d)

        try:
            sconcat = np.ma.concatenate(sss, axis=axis)

        except ValueError as e:
            print(f"---err sconcat = np.ma.concatenate(sss, axis=axis): {e}")
            # probably a dimension mismatch
            # get the shapes and ndims for comparison
            rshapes = []
            rndims = []
            for item in datasets:
                sh = list(item.shape)
                rshapes.append(sh)
                rndims.append(len(sh))

            # The number of dimensions is expected to be the same for all datasets
            if len(list(set(rndims))) > 1:
                raise DimensionsCompatibilityError(
                    "Only NDDataset with the same number of dimensions can be concatenated."
                )

            rcompat = list(map(list, list(map(set, list(zip(*rshapes))))))

            if force_stack:
                # in
                if len(set(list(map(len, rcompat)))) != 1:
                    warn(
                        "These datasets have not the same shape, so they cannot be stacked. By default they will be "
                        "concatenated along the first dimension.",
                        category=SpectroChemPyWarning,
                    )
                    axis, dim = datasets[0].get_axis(dim=0)

            # now check if data shapes are compatible (all dimension must have the same size
            # except the one to be concatenated)
            for i, item in enumerate(zip(*rshapes)):
                if i != axis and len(set(item)) > 1:
                    raise DimensionsCompatibilityError(
                        "Datasets must have the same shape for all dimensions except the one along which the"
                        " concatenation is performed"
                    )

            # retry
            sconcat = np.ma.concatenate(sss, axis=axis)

        data = np.asarray(sconcat)
        mask = sconcat.mask

        out = dataset.copy()
        out._data = data
        if coords is not None:
            out._coordset[dim] = coords[dim]
        out._mask = mask
        out._units = units

        thist = "Stack" if axis == 0 else "Concatenation"

        out.description = "{} of {}  datasets:\n".format(thist, len(datasets))
        out.description += "( {}".format(datasets[0].name)
        out.title = datasets[0].title
        authortuple = (datasets[0].author,)

        for dataset in datasets[1:]:

            if out.title != dataset.title:
                warn("Different data title => the title is that of the 1st dataset")

            if not (dataset.author in authortuple):
                authortuple = authortuple + (dataset.author,)
                out.author = out.author + " & " + dataset.author

            out.description += ", {}".format(dataset.name)

        out.description += " )"
        out._date = out._modified = datetime.datetime.now(datetime.timezone.utc)
        out._history = [str(out.date) + ": Created by %s" % thist]

        return out

    except Exception as e:
        print(f"oups : {e}")

    # except Exception:
    #     # there has been an error, could be due incompatible units, missing labels...
    #     # more checks are carried out
    #

    #     # Check unit compatibility
    #     # ------------------------------------------------------------------------
    #     units = datasets[0].units
    #     for dataset in datasets:
    #         if not dataset.is_units_compatible(datasets[0]):
    #             raise ValueError(
    #                 "units of the datasets to concatenate are not compatible"
    #             )
    #         # if needed transform to the same unit
    #         dataset.ito(units)
    #     # TODO: make concatenation of heterogeneous data possible by using labels
    #
    #     # Check coordinates compatibility
    #     # ------------------------------------------------------------------------
    #
    #     # coordinates units of NDDatasets must be compatible in all dimensions
    #     # get the coordss
    #
    #     coordss = [dataset.coordset for dataset in datasets]
    #     if set(coordss) == {None}:
    #         coordss = None
    #
    #     # concatenate coords if they exists
    #     # ------------------------------------------------------------------------
    #
    #     if coordss is None or (len(coordss) == 1 and coordss.pop() is None):
    #         # no coords
    #         coords = None
    #     else:
    #         # we take the coords of the first dataset, and extend the coord along the concatenate axis
    #         coords = coordss[0].copy()
    #
    #         try:
    #             coords[dim] = Coord(
    #                 coords[dim], linear=False
    #             )  # de-linearize the coordinates
    #             coords[dim]._data = np.concatenate(
    #                 tuple((c[dim].data for c in coordss))
    #             )
    #         except (KeyError, ValueError):
    #             pass
    #
    #     # check labeling
    #     # concatenation of the labels (first check the presence of at least one labeled coordinates)
    #
    #     is_labeled = False
    #     for i, c in enumerate(coordss):
    #         if c[dim].implements() in ["Coord", "LinearCoord"]:
    #             # this is a coord
    #             if c[dim].is_labeled:
    #                 # at least one of the coord is labeled
    #                 is_labeled = True
    #                 break
    #         if c[dim].implements("CoordSet"):
    #             # this is a coordset
    #             for coord in c[dim]:
    #                 if coord.is_labeled:
    #                     # at least one of the coord is labeled
    #                     is_labeled = True
    #                     break
    #
    #     if is_labeled:
    #         labels = []
    #         # be sure that now all the coordinates have a label, or create one
    #         for i, c in enumerate(coordss):
    #             if c[dim].implements() in ["Coord", "LinearCoord"]:
    #                 # this is a coord
    #                 if c[dim].is_labeled:
    #                     labels.append(c[dim].labels)
    #                 else:
    #                     labels.append(str(i))
    #             if c[dim].implements("CoordSet"):
    #                 # this is a coordset
    #                 for coord in c[dim]:
    #                     if coord.is_labeled:
    #                         labels.append(c[dim].labels)
    #                     else:
    #                         labels.append(str(i))
    #
    #         if isinstance(coords[dim], Coord):
    #             coords[dim]._labels = np.concatenate(labels)
    #         if coords[dim].implements("CoordSet"):
    #             for i, coord in enumerate(coords[dim]):
    #                 coord._labels = np.concatenate(labels[i :: len(coords[dim])])
    #
    #         coords[dim]._labels = np.concatenate(labels)
    #


def stack(*datasets, **kwargs):
    """
    Stack of |NDDataset| objects along the first dimension.

    Any number of |NDDataset| objects can be stacked. For this operation
    to be defined the following must be true :

    #. all inputs must be valid dataset objects,
    #. units of data and axis must be compatible (rescaling is applied
       automatically if necessary).

    The remaining dimension sizes must match along all dimension but the first.

    Parameters
    ----------
    *datasets : a series of |NDDataset|
        The dataset to be stacked to the current dataset.

    Returns
    --------
    out
        A |NDDataset| created from the stack of the `datasets` datasets.

    Examples
    --------

    >>> A = scp.read('irdata/nh4y-activation.spg', protocol='omnic')
    >>> B = scp.read('irdata/nh4y-activation.scp')
    >>> C = scp.stack(A, B)
    >>> print(C)
    NDDataset: [float64] a.u. (shape: (z:2, y:55, x:5549))
    """

    return concatenate(*datasets, force_stack=True, **kwargs)


def _prepend_dim(dataset, coord_value):
    """prepend a new dimension with coords"""
    dataset._data = dataset.data[np.newaxis]
    dataset._mask = dataset.mask[np.newaxis]
    newcoord = Coord([coord_value], labels=[dataset.name])
    newcoord.name = (OrderedSet(DEFAULT_DIM_NAME) - dataset._dims).pop()
    dataset.add_coordset(newcoord)
    dataset.dims = [newcoord.name] + dataset.dims
    return dataset


if __name__ == "__main__":
    pass
