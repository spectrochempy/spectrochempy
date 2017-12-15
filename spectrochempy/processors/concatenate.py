# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================



import numpy as np
import datetime as datetime
from warnings import warn

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.utils import (is_sequence)
from spectrochempy.extern.uncertainties import unumpy as unp

__all__ = ['concatenate','stack']


def concatenate(*sources, axis=None, **kwargs):
    """Concatenation of DataSet objects along a given axis (by default the fisrt)

    Any number of DataSet objects can be concatenated. For this operation
    to be defined the following must be true:

    #. all inputs must be valid dataset objects,
    #. units of data and axis must be compatible (rescaling is applied
       automatically if necessary)
    #. concatenation is along the axis specified or the first one.

    The remaining dimension sizes must match.

    Parameters
    ----------

    *sources : a series of :class:`~spectrochempy.dataset.nddataset.NDDataset`
        The dataset to be concatenated to the current dataset


    axis : int, optional, default = 0
        The axis along which the datasets are concatenated

    Returns
    --------

    out : :class:`~spectrochempy.dataset.nddataset.NDDataset`
        A dataset created from the contenations of the `sources` datasets

    Examples
    --------

    >>> from spectrochempy.api import * # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
            ...
    >>> A = NDDataset.load('spec.spg', protocol='omnic')
    >>> B = NDDataset.load('mydataset.scp')
    >>> C = NDDataset.concatenate( A, B, axis=0)
    >>> C # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    NDDataset: ...

    or

    >>> A = NDDataset.load('spec.spg', protocol='omnic')
    >>> B = NDDataset.load('mydataset.scp')
    >>> C = A.concatenate(B, axis=0)

    Notes
    -----
    This function is similar to :class:`numpy.ndarray` concatenation,
    but each attribute is treated specifically as follows:

    - ``out.name``       : concatenation of dataset names

    - ``out.title``        : title of the fist dataset

    - ``out.author``     : concatenation of dataset authors
                            (each distinct author appear once)

    - ``out.date``       : date of concatenation

    - ``out.moddate``    : date of concatenation

    - ``out.data``       : numpy.concatenate(a.data,b.data,c.data,...,concatdim)

    - ``out.label``      : dim 'concatdim' label sets are concatenated, and \
                          label sets are created for all other dims

    - ``out.axis``  : dim 'concatdim' axis sets are concatenated, and \
                          axis sets are created for all other dims

    - ``out.description``: concatenates all descriptions

    """

    # checks dataset validity

    shapes = []

    for source in sources:

        if is_sequence(source):  # numpy style of passing args
            sources = source

    units = sources[0].units

    for source in sources:

        if not isinstance(source, NDDataset):
            raise TypeError(
                    "Only instance of NDDataset can be concatenated, not: " + type(
                            source).__name__)

        if source.ndim != sources[0].ndim:
            raise ValueError(
                    "All datasets must have the same number of dims")

        if not source.is_units_compatible(sources[0]):
            raise ValueError(
                    'units of the datasets to concatenate are not compatible')
        source.to(units)

        sax = sources[0].coordset
        for i, ax in enumerate(source.coordset):
            if not ax.is_units_compatible(sax[i]):
                raise ValueError(
                        "units of the dataset's axis are not compatible")
            ax.to(sax[i].units)

        shapes.append(source.shape)

    if axis is None:
        try:
            # should work whatever the number of dimensions
            axis = [pair[0] == pair[1] for pair in zip(*shapes)].index(False)
        except ValueError:
            warn('concatenation axis was set to 0 (default)')
            axis = 0

    if axis < 0:
        axis = sources[0].ndim + axis

    # concatenate or stack the data array + mask and uncertainty
    sss = []
    for source in sources:
        sss.append(source._uncert_data)

    sconcat = np.ma.concatenate(sss, axis=axis)
    data = unp.nominal_values(np.asarray(sconcat))
    mask = sconcat.mask  # np.array(self._mask[keys])
    uncertainty = unp.std_devs(np.asarray(sconcat))

    # data = np.concatenate(sources, axis=axis)
    # # for the concatenation to work we need to take the real _mask
    # #twod = lambda x: x #if x.ndim>1 else np.array([x])
    # mask = np.concatenate(tuple((source._mask
    #                              for source in sources)), axis=axis)
    # uncertainty = np.concatenate(tuple((source._uncertainty)
    #                                     for source in sources), axis=axis)

    # concatenate coordset
    stack = kwargs.get('force_stack', False)
    coordset = sources[0].copy().coordset

    c2arr = lambda x: x if isinstance(x, np.ndarray) else np.array([x])
    coordset[axis]._data = np.concatenate(
            tuple((c2arr(source.coordset[axis].data) for source in sources)))
    coordset[axis]._mask = np.concatenate(
            tuple((c2arr(source.coordset[axis].mask) for source in sources)))
    coordset[axis]._labels = np.concatenate(
            tuple((c2arr(source.coordset[axis].labels) for source in sources)),axis=-1)


    out = NDDataset(data, coordset=coordset, mask=mask, uncertainty=uncertainty,
                    units = units)

    t = 'Stack' if axis==0 else 'Concatenation'

    out.description = '{} of {}  datasets :\n'.format(t,len(sources))
    out.description += '( {}'.format(sources[0].name)
    out.title = sources[0].title
    authortuple = (sources[0].author,)

    for source in sources[1:]:

        if out.title != source.title:
            warn('Different data title => the title is that of the 1st dataset')

        if not (source.author in authortuple):
            authortuple = authortuple + (source.author,)
            out.author = out.author + ' & ' + source.author

        out.description += ', {}'.format(source.name)
    out.description += ' )'
    out._date = out._modified = datetime.datetime.now()
    out._history = [str(out.date) + ':created by concatenate()']

    return out


def stack(*sources):
    """Stack of DataSet objects along the fisrt dimension

    Any number of DataSet objects can be stacked. For this operation
    to be defined the following must be true:

    #. all inputs must be valid dataset objects,
    #. units of data and axis must be compatible (rescaling is applied
       automatically if necessary)

    The remaining dimension sizes must match along all dimension but the first.

    Parameters
    ----------

    *sources : a series of :class:`~spectrochempy.dataset.nddataset.NDDataset`
        The dataset to be stacked to the current dataset

    Returns
    --------

    out : :class:`~spectrochempy.dataset.nddataset.NDDataset`
        A dataset created from the stack of the `sources` datasets

    Examples
    --------

    >>> from spectrochempy.api import * # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
            SpectroChemPy's API...
    >>> A = NDDataset.load('spec.spg', protocol='omnic')
    >>> B = NDDataset.load('mydataset.scp')
    >>> C = NDDataset.stack( A, B)
    >>> print(C) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
    --------------------------------------------------------------------------------
          name/id: ...
                   title: Wavenumbers
                    data: [5999.556 5998.591 ...,  650.868  649.904] cm^-1
    --------------------------------------------------------------------------------
    <BLANKLINE>

    """

    return concatenate(*sources, axis=0, force_stack=True)


if __name__ == '__main__':

    from spectrochempy.api import *
    A = NDDataset.load('spec.spg', protocol='omnic')
    B = NDDataset.load('mydataset.scp')
    C = concatenate( A, B, axis=0)
    print(C)


    A = NDDataset.load('spec.spg', protocol='omnic')
    B = NDDataset.load('mydataset.scp')
    C = A.concatenate(B, axis=0)