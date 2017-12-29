# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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
from spectrochempy.utils import (is_sequence)
from spectrochempy.extern.uncertainties import unumpy as unp



def concatenate(*datasets, axis=None, **kwargs):
    """Concatenation of |NDDataset| objects along a given axis (by default the
    first)

    Any number of |NDDataset| objects can be concatenated. For this operation
    to be defined the following must be true:

    #. all inputs must be valid dataset objects;
    #. units of data and axis must be compatible (rescaling is applied
       automatically if necessary);
    #. concatenation is along the axis specified or the first one;
    #. along the non-concatenated dimensions, any dataset (or array-like
       objects) without axis coordinates will be concatenated silently
       assuming compatible dimensions and units coordinates in those
       dimensions, as far as the dimension sizes match.

    Parameters
    ----------
    *datasets : a series of |NDDataset|
        The dataset(s) to be concatenated to the current dataset
    axis : int, optional, default = 0
        The axis along which the datasets are concatenated

    Returns
    --------
    out : |NDDataset|
        A dataset created from the contenations of the `datasets` datasets

    Examples
    --------
    >>> from spectrochempy import * # doctest: +ELLIPSIS
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

    - ``out.data``       : numpy.concatenate(a.data,b.data,c.data,...,
      concatdim)

    - ``out.label``      : dim 'concatdim' label sets are concatenated, and
      label sets are created for all other dims

    - ``out.axis``  : dim 'concatdim' axis sets are concatenated, and
      axis sets are created for all other dims

    - ``out.description``: concatenates all descriptions

    """

    # checks dataset validity

    shapes = []

    for dataset in datasets:

        if is_sequence(dataset):  # numpy style of passing args
            datasets = dataset

        #
    units = datasets[0].units


    for dataset in datasets:

        if not isinstance(dataset, NDDataset):
            raise TypeError(
                    "Only instance of NDDataset can be concatenated, not: " + type(
                            dataset).__name__)

        # check if dimension are compatibles
        if dataset.ndim != datasets[0].ndim:
            raise ValueError(
                    "All datasets must have the same number of dims")

        # check if units are compatibles
        if not dataset.is_units_compatible(datasets[0]):
            raise ValueError(
                    'units of the datasets to concatenate are not compatible')
        dataset.to(units)

        # check if coordinates are compatible
        sax = datasets[0].coordset
        if sax is not None:
            # we expect that coordinates are the same!
            for i, ax in enumerate(dataset.coordset):
                if not ax.is_units_compatible(sax[i]):
                    raise ValueError(
                            "units of the dataset's axis are not compatible")
                ax.to(sax[i].units)
        else:
            # there is no coordinates on the first dataset.
            # so we don't care.
            # OK, but what about the case there is some dataset with
            # coordinates and other withuot. #TODO: try to handle this case
            pass

        shapes.append(dataset.shape)

    if axis is None:
        try:
            # should work whatever the number of dimensions
            axis = [pair[0] == pair[1] for pair in zip(*shapes)].index(False)
        except ValueError:
            warn('concatenation axis was set to 0 (default)')
            axis = 0

    if axis < 0:
        axis = datasets[0].ndim + axis

    # concatenate or stack the data array + mask and uncertainty
    sss = []
    for dataset in datasets:
        sss.append(dataset._uncert_data)

    sconcat = np.ma.concatenate(sss, axis=axis)
    data = unp.nominal_values(np.asarray(sconcat))
    mask = sconcat.mask  # np.array(self._mask[keys])
    uncertainty = unp.std_devs(np.asarray(sconcat))      #TODO: check first
    # the exstence of uncertianty to accelerate this process

    # concatenate coordset
    stack = kwargs.get('force_stack', False)
    coordset = datasets[0].copy().coordset

    c2arr = lambda x: x if isinstance(x, np.ndarray) else np.array([x])
    coordset[axis]._data = np.concatenate(
            tuple((c2arr(dataset.coordset[axis].data) for dataset in datasets)))
    coordset[axis]._mask = np.concatenate(
            tuple((c2arr(dataset.coordset[axis].mask) for dataset in datasets)))
    coordset[axis]._labels = np.concatenate(
            tuple((c2arr(dataset.coordset[axis].labels) for dataset in datasets)),axis=-1)


    out = NDDataset(data, coordset=coordset, mask=mask, uncertainty=uncertainty,
                    units = units)

    t = 'Stack' if axis==0 else 'Concatenation'

    out.description = '{} of {}  datasets :\n'.format(t, len(datasets))
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
    out._history = [str(out.date) + ':created by concatenate()']

    return out


def stack(*datasets):
    """
    Stack of |NDDataset| objects along the fisrt dimension

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
    <BLANKLINE>
    --------------------------------------------------------------------------------
          name/id: ...
                   title: Wavenumbers
                    data: [5999.556 5998.591 ...,  650.868  649.904] cm^-1
    --------------------------------------------------------------------------------
    <BLANKLINE>

    """

    return concatenate(*datasets, axis=0, force_stack=True)


if __name__ == '__main__':

    pass