# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

__all__ = ['concatenate','stack']

__dataset_methods__ = __all__


import numpy as np
import datetime as datetime
from warnings import warn

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.dataset.ndcoords import Coord,  CoordSet
from spectrochempy.utils import (is_sequence, docstrings)
from spectrochempy.extern.uncertainties import unumpy as unp


def concatenate(*datasets, axis=-1, **kwargs):
    """
    Concatenation of |NDDataset| objects along a given axis (by default
    the first)

    Any number of |NDDataset| objects can be concatenated. For this operation
    to be defined the following must be true:

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
    axis : int, optional, default = -1
        The axis along which the operation is applied

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


    Notes
    -----
    This function is similar to :class:`numpy.ndarray` concatenation,
    but each attribute is treated specifically as follows:

        :out.name: concatenation of dataset names

        :out.title: title of the fist dataset which have a title

        :out.author: concatenation of dataset authors  (each distinct
                     author appear once)

        :out.date: date of concatenation

        :out.moddate: date of concatenation

        :out.data: numpy.concatenate(a.data,b.data,c.data,...,)

        :out.label: label sets are concatenated for the concatenated axis
                    and label sets are created for all other dims

        :out.axis: dim 'concatdim' axis sets are concatenated, and
                   axis sets are created for all other dims

        :out.description: concatenates all descriptions

    """

    # ------------------------------------------------------------------------
    # checks dataset validity
    # ------------------------------------------------------------------------

    # We must have a list of datasets
    for dataset in datasets:
        if is_sequence(dataset):  # numpy style of passing args
            datasets = dataset

    # try to cast of dataset to NDDataset
    for dataset in datasets:
        if not isinstance(dataset, NDDataset):
            try:
                dataset = NDDataset(dataset)
            except:
                raise TypeError("Only instance of NDDataset can be "
                                "concatenated, not %s, but casting to this "
                                "type failed. " % type(dataset).__name__)

    # a flag to force stacking of dataset instead of the drfault concatenation
    force_stack = kwargs.get('force_stack', False)
    if force_stack:
        axis = 0

    if axis < 0:
        axis = datasets[0].ndim + axis

    # check if data shapes are compatible (all dimension size must be the same
    # except the one to be concatenated)
    rshapes = []
    for dataset in datasets:
        sh = list(dataset.shape)
        del sh[axis]
        rshapes.append(sh)

    for item in zip(*rshapes):
        if len(set(item))>1:
            raise ValueError("Datasets must have the same shape for"
                             " all dimensions except the one along which the"
                             " concatenation is performed")

    # are data uncertain?
    isuncertain = any([dataset.is_uncertain for dataset in datasets])

    if force_stack:
        # for we need a first additional dimension of lendth 1
        # for this stack operation
        for i, dataset in enumerate(datasets):
            dataset._data = dataset.data[np.newaxis]
            dataset._mask = dataset.mask[np.newaxis]
            if isuncertain:
                dataset._uncertainty = dataset.uncertainty[np.newaxis]
            dataset.coordset._coords=[Coord(labels=[str(i)])]+dataset.coordset._coords

    # Check unit compatibility
    # -------------------------
    units = datasets[0].units
    for dataset in datasets:
        if not dataset.is_units_compatible(datasets[0]):
            raise ValueError(
                    'units of the datasets to concatenate are not compatible')
        # if needed transform to the same unit
        dataset.ito(units)
    # TODO: make concatenation of heterogeneous data possible by using labels


    # Check coordinates compatibility
    # -------------------------------

    # coordinates units of NDDatasets must be compatible in all dimensions

    # get the coordsets
    coordsets = [dataset.coordset for dataset in datasets]

    # how many different coordsets
    coordsets = set(coordsets)
    if len(coordsets)==1 and force_stack:
        # nothing to do (all datasets have the same coordset and so are
        # perfectly compatibles for stacking)
        pass

    else:
        for i, cs in enumerate(zip(*coordsets)):

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
    sss = []
    for i, dataset in enumerate(datasets):
        d = dataset._data
        if isuncertain:
            d = dataset._uarray(d, dataset._uncertainty)
        # masks ?
        d = dataset._umasked(d, dataset._mask)
        sss.append(d)

    sconcat = np.ma.concatenate(sss, axis=axis)
    if isuncertain:
        data = unp.nominal_values(np.asarray(sconcat))
        uncertainty = unp.std_devs(np.asarray(sconcat))
    else:
        data = np.asarray(sconcat)
        uncertainty = None
    mask = sconcat.mask

    # concatenate coordset if they exists:
    if len(coordsets) == 1 and coordsets.pop() is None:
        # no coordset
        coordset = None
    else:
        # we take the coordset of the first dataset, en extend the coord
        # along the concatenate axis
        coordset = datasets[0].copy().coordset

        c2arr = lambda x: x if isinstance(x, np.ndarray) else np.array([x])
        coordset[axis]._data = np.concatenate(
           tuple((c2arr(dataset.coordset[axis].data) for dataset in datasets)))
        coordset[axis]._labels = np.concatenate(
            tuple((c2arr(dataset.coordset[axis].labels)
                                             for dataset in datasets)),axis=-1)


    out = NDDataset(data, coordset=coordset, mask=mask, uncertainty=uncertainty,
                    units = units)

    thist = 'Stack' if axis==0 else 'Concatenation'

    out.description = '{} of {}  datasets :\n'.format(thist, len(datasets))
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
    out._history = [str(out.date) + ':created by %s'%thist]

    return out


def stack(*datasets):
    """
    Stack of |NDDataset| objects along the first dimension

    Any number of |NDDataset| objects can be stacked. For this operation
    to be defined the following must be true:

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
          name/id: NDDataset...

    """

    return concatenate(*datasets, force_stack=True)


if __name__ == '__main__':

    pass