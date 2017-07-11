# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

import datetime as datetime
from warnings import warn
import numpy as np

from ..dataset import NDDataset
from spectrochempy.utils import is_sequence

__all__ = []

def concatenate(*sources, axis=None):
    '''Concatenation of DataSet objects along a given axis (by default the fisrt)

    Any number of DataSet objects can be concatenated. For this operation
    to be defined the following must be true:

    #. all inputs must be valid dataset objects,
    #. units of data and axis must be compatible (rescaling is applied
       automatically if necessary)
    #. concatenation is along the axis specified or the first one.

    The remaining dimension sizes must match.

    Parameters
    ----------

    *sources : a series of :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
        The dataset to be concatenated to the current dataset


    axis : `ìnt`, optional, default = 0
        The axis along which the datasets are concatenated

    Returns
    --------

    out : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
        A dataset created from the contenations of the `sources` datasets

    Examples
    --------

    >>> import spectrochempy as scp
    >>> A = scp.load_dataset('spec.spg', protocol='omnic')
    >>> B = scp.load_dataset('spec2.scp')
    >>> C = scp.concatenate( A, B, axis=0)

    or

    >>> A = scp.load_dataset('spec.spg', protocol='omnic')
    >>> B = scp.load_dataset('spec2.scp')
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

    '''

    # checks dataset validity

    shapes = []

    for source in sources:

        if is_sequence(source):  # numpy style of passing args
            sources = source

    for source in sources:

        if not isinstance(source, NDDataset):
            raise TypeError(
                "Only instance of NDDataset can be concatenated, not: " + type(
                            source).__name__)

        if source.ndim != sources[0].ndim:
            raise ValueError("All datasets must have the same number of dims")

        if not source.is_units_compatible(sources[0]):
            raise TypeError(
                    'units of the datasets to concatenate are not compatible')
        source.to(sources[0].units)

        sax = sources[0].axes
        for i, ax in enumerate(source.axes):
            if not ax.is_units_compatible(sax[i]):
                raise TypeError(
                        "units of the dataset's axis are not compatible")
            ax.to(sax[i].units)

        shapes.append(source.shape)

    # guess ndim
    if axis == None:
        try:
            # should work whatever the number of dimensions
            axis = [pair[0] == pair[1] for pair in zip(*shapes)].index(False)
        except ValueError:
            warn('concatenation axis was set to 0 (default)')
            axis = 0

    if axis < 0:
        axis = sources[0].ndim + axis

    # initialize dataset based on the 1st passed dataset
    out = sources[0].copy()
    outaxis = out.axes[axis]  # axis for concatenation

    out.description = 'Concatenation of ' + str(len(sources)) + ' datasets :\n'
    out.description += ' ->' + sources[0].name + ' : ' + sources[
        0].description + ' \n'

    authortuple = (sources[0].author,)

    for source in sources[1:]:

        out.name = out.name + ' & ' + source.name

        if out.title != source.title:
            warn('Different data title => the title is that of the 1st dataset')

        if not (source.author in authortuple):
            authortuple = authortuple + (source.author,)
            out.author = out.author + ' & ' + source.author

        # concatenate the data array
        out._data = np.concatenate((out.data, source.data), axis=axis)
        out._mask = np.concatenate((out.mask, source.mask), axis=axis)
        out._uncertainty = np.concatenate((out.uncertainty, source.uncertainty),
                                          axis=axis)

        # concatenate axes

        axes = out.axes
        for i, saxe in enumerate(source.axes):

            if saxe.title != axes[i].title:
                warn(
'Different axis title [%s] => the axis name is that of the 1st dataset [%s]' % (
                            outaxis.title,
                            saxe.title))

            if i == axis:  # concatenation axis

                outaxis._data = np.concatenate((outaxis.data, saxe.data))
                outaxis._mask = np.concatenate((outaxis.mask, saxe.mask))
                outaxis._labels = np.concatenate((outaxis.labels, saxe.labels),
                                                 axis=-1)

        # TODO: not sure yet what to do here
        # for i, labels in enumerate(out.axes[i].labels):
        #     if labels.name != .labels[i].name:
        #         print('warning: different labels names =>
        # the labels name is that of the 1st dataset')
        #     # labels.labels =
        # labels.labels.append(X.dims[ndim].labels[i].labels)
        #     labels.labels.append(source.dims[ndim].labels[i].labels)

        out.description = out.description + ' \n->' + source.title + ' : ' + \
                          source.description + ' \n'

    out._date = datetime.datetime.now()
    out._history = [str(out.date) + ':created by concatenate()']

    return out
