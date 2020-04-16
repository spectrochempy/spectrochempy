# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ['concatenate', 'stack']

__dataset_methods__ = __all__

import numpy as np
import datetime as datetime
from warnings import warn

from ..dataset.nddataset import NDDataset
from ..dataset.ndcoordset import CoordSet
from ..dataset.ndcoord import Coord
from ...utils import is_sequence, docstrings

@docstrings.dedent
def concatenate(*datasets, **kwargs):
    """
    Concatenation of |NDDataset| objects along a given axis (by default
    the first)

    Any number of |NDDataset| objects can be concatenated. For this operation
    to be defined the following must be true :

        #. all inputs must be valid |NDDataset| objects;
        #. units of data and axis must be compatible (rescaling is applied
           automatically if necessary);
        #. concatenation is along the axis specified or the first one;
        #. along the non-concatenated dimensions, shape and units coordinates
           must match.

    Parameters
    ----------
    *datasets : positional |NDDataset| arguments
        The dataset(s) to be concatenated to the current dataset. The datasets
        must have the same shape, except in the dimension corresponding to axis
        (the last, by default).
    dims : str, optional, default='x'
        The dimension along which the operation is applied
    axis : int, optional, default=None
        Alternative to the the dim keyword. Direct specification of the axis index to use for concatenation.

    Returns
    --------
    out : |NDDataset|
        A dataset created from the contenations of the |NDDataset| input
        objects

    Examples
    --------
    >>> from spectrochempy import * # doctest: +ELLIPSIS
    ...
    >>> A = NDDataset.load('spec.spg', protocol='omnic')
    >>> B = NDDataset.load('mydataset.scp')
    >>> C = NDDataset.concatenate( A[10:], B[3:5], A[:10], axis=0)
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

    # ------------------------------------------------------------------------------------------------------------------
    # checks dataset validity
    # ------------------------------------------------------------------------------------------------------------------

    # We must have a list of datasets
    if isinstance(datasets, tuple):
        if isinstance(datasets[0], (list, tuple)):
            datasets = datasets[0]

    # try to cast of dataset to NDDataset
    for dataset in datasets:
        if not isinstance(dataset, NDDataset):
            try:
                dataset = NDDataset(dataset)
            except:
                raise TypeError(f"Only instance of NDDataset can be concatenated, not {type(dataset).__name__}, "
                                f"but casting to this type failed. ")

    # a flag to force stacking of dataset instead of the default concatenation
    force_stack = kwargs.get('force_stack', False)
    if force_stack:
        axis, dim = datasets[0].get_axis(0)
    else:
        # get axis from arguments
        if kwargs:
            axis, dim = datasets[0].get_axis(**kwargs)
        else:
            axis, dim = datasets[0].get_axis('x')

    # check if data shapes are compatible (all dimension must have the same size
    # except the one to be concatenated)
    rshapes = []
    for dataset in datasets:
        sh = list(dataset.shape)
        if len(sh) > 1:
            del sh[axis]
        rshapes.append(sh)

    for item in zip(*rshapes):
        if len(set(item)) > 1:
            raise ValueError("Datasets must have the same shape for all dimensions except the one along which the"
                             " concatenation is performed")

    if force_stack:
        # for this stack operation, we need the a dimension y of at least length 1
        # except if it is already of size 1
        for i, dataset in enumerate(datasets):
            if dataset.shape[-1] >= 1:
                continue
            dataset._data = dataset.data[np.newaxis]
            dataset._mask = dataset.mask[np.newaxis]
            dataset.set_coords(y=Coord(labels=[str(i)]))

    # Check unit compatibility
    # ------------------------------------------------------------------------------------------------------------------

    units = datasets[0].units
    for dataset in datasets:
        if not dataset.is_units_compatible(datasets[0]):
            raise ValueError(
                'units of the datasets to concatenate are not compatible')
        # if needed transform to the same unit
        dataset.ito(units)
    # TODO: make concatenation of heterogeneous data possible by using labels

    # Check coordinates compatibility
    # ------------------------------------------------------------------------------------------------------------------

    # coordinates units of NDDatasets must be compatible in all dimensions
    # get the coordss
    coordss = [dataset.coords for dataset in datasets]

    def check_coordinates(coordss, force_stack):

        # We will call this only in case of problems because it takes a lot of time

        # how many different coordss
        coordss = set(coordss)
        if len(coordss) == 1 and force_stack:
            # nothing to do (all datasets have the same coords and so are
            # perfectly compatibles for stacking)
            pass

        else:
            for i, cs in enumerate(zip(*coordss)):

                axs = set(cs)
                axref = axs.pop()
                for ax in axs:
                    # we expect compatible units
                    if not ax.is_units_compatible(axref):
                        raise ValueError(
                            "units of the dataset's axis are not compatible"
                        )
                    if i != axis and ax.size != axref.size:
                        # and same size for the non-concatenated axis
                        raise ValueError(
                            "size of the non-concatenated dimension must be "
                            "identical"
                        )

    # concatenate or stack the data array + mask
    # ------------------------------------------------------------------------------------------------------------------

    sss = []
    for i, dataset in enumerate(datasets):
        d = dataset.masked_data
        sss.append(d)

    sconcat = np.ma.concatenate(sss, axis=axis)
    data = np.asarray(sconcat)
    mask = sconcat.mask

    # concatenate coords if they exists
    # ------------------------------------------------------------------------------------------------------------------

    if len(coordss) == 1 and coordss.pop() is None:
        # no coords
        coords = None
    else:
        # we take the coords of the first dataset, and extend the coord along the concatenate axis
        coords = coordss[0].copy()

        try:
            coords[dim]._data = np.concatenate(tuple((c[dim].data for c in coordss)))
        except ValueError:
            pass

        # concatenation of the labels (fist check the presence of at least one labeled coordinates)
        is_labeled = False
        for i, c in enumerate(coordss):
            if c[dim].is_labeled:
                # at least one of the coord is labeled
                is_labeled = True
                break
                

        if is_labeled:
            labels = []
            # be sure that now all the coordinates have a label, or create one
            for i, c in enumerate(coordss):
                if c[dim].is_labeled:
                    labels.append(c[dim].labels)
                else:
                    labels.append(str(i))

            coords[dim]._labels = np.concatenate(labels)

    out = NDDataset(data, coords=coords, mask=mask, units=units)

    thist = 'Stack' if axis == 0 else 'Concatenation'

    out.description = '{} of {}  datasets:\n'.format(thist, len(datasets))
    out.description += '( {}'.format(datasets[0].name)
    out.title = datasets[0].title
    authortuple = (datasets[0].author,)

    for dataset in datasets[1:]:

        if out.title != dataset.title:
            warn('Different data title => the title is that of the 1st dataset')

        if not (dataset.author in authortuple):
            authortuple = authortuple + (dataset.author,)
            out.author = out.author + ' & ' + dataset.author

        out.description += ', {}'.format(dataset.name)
        
    out.description += ' )'
    out._date = out._modified = datetime.datetime.now()
    out._history = [str(out.date) + ': Created by %s' % thist]

    return out


def stack(*datasets):
    """
    Stack of |NDDataset| objects along the last dimension

    Any number of |NDDataset| objects can be stacked. For this operation
    to be defined the following must be true :

    #. all inputs must be valid dataset objects,
    #. units of data and axis must be compatible (rescaling is applied
       automatically if necessary)

    The remaining dimension sizes must match along all dimension but the first.

    Parameters
    ----------
    *datasets : a series of |NDDataset|
        The dataset to be stacked to the current dataset

    Returns
    --------
    out : |NDDataset|
        A dataset created from the stack of the `datasets` datasets

    Examples
    --------
    >>> from spectrochempy import * # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    ...
    >>> A = NDDataset.load('spec.spg', protocol='omnic')
    >>> B = NDDataset.load('mydataset.scp')
    >>> C = NDDataset.stack( A, B)
    >>> print(C) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
          name/id : NDDataset...

    """

    return concatenate(*datasets, force_stack=True)


if __name__ == '__main__':
    pass
