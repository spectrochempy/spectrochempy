# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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


"""
This module implements the base `NDDataset` class.

"""

# =============================================================================
# Standard python imports
# =============================================================================

import copy
import itertools
import logging
import textwrap
from datetime import datetime
from warnings import warn

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame

# =============================================================================
# third-party imports
# =============================================================================
from traitlets import (List, Unicode, Instance,  Bool, All, Float,
                      validate, observe, default)

# =============================================================================
# Local imports
# =============================================================================
from spectrochempy.core.units import Quantity
from spectrochempy.utils import (SpectroChemPyWarning,
                                 is_sequence, is_number,
                                 numpyprintoptions, get_user_and_node)
from spectrochempy.extern.traittypes import Array
from spectrochempy.core.dataset.ndarray import NDArray, CoordSet
from spectrochempy.core.dataset.ndcoords import Coord, CoordError
from spectrochempy.core.dataset.ndmath import NDMath, set_operators
# from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.dataset.ndio import NDIO
from spectrochempy.application import log

# =============================================================================
# Constants
# =============================================================================

__all__ = ['NDDataset',
           'NDDatasetError',
           'NDDatasetWarning',
           # dataset
           'squeeze',
           'sort',
           'swapaxes',
           'transpose',
           'abs',
           'conj',
           'imag',
           'real',
           ]

_classes = [
    'NDDataset',
    'NDDatasetError',
    'NDDatasetWarning'
]

# =============================================================================
# numpy print options
# =============================================================================

numpyprintoptions()


# =============================================================================
# NDDataset class definition
# =============================================================================

class NDDatasetError(ValueError):
    """
    An exception that is raised when something is wrong with the NDDataset`
    definitions.
    """


class NDDatasetWarning(SpectroChemPyWarning):
    """
    A warning that is raised when something is wrong with the `NDDataset`
    definitions but do not necessarily need to raise an error.
    """


class NDDataset(
        NDIO,
        NDMath,
        NDArray,
):
    """
    The main N-dimensional dataset class used by |scp|.

    Parameters
    -----------
    data : :class:`~numpy.ndarray`-like object.

        If possible, the provided data will not be copied for `data` input,
        but will be
        passed by reference, so you should make a copy
        the :attr:`data` before passing it in if that's the desired behavior
        or set of the `iscopy` parameter to True.
        Any size or shape of data is accepted.

    mask : :class:`~numpy.ndarray`-like, optional

        Mask for the data. The values must be `False` where
        the data is *valid* and `True` when it is not (like Numpy
        masked arrays). If `data` is already a :class:`~numpy.ma.MaskedArray`,
        or
        any array object (such as a `NDDataset`), providing
        a `mask` here will causes the mask from the masked array to be
        ignored.

    coords : An instance of :class:`~spectrochempy.core.dataset.ndcoords.CoordSet`,
    optional

        `coords` contains the coordinates for the different
        dimensions of the `data`. if `coords` is provided, it must specified
        the `coord` and `labels` for all dimensions of the `data`.
        Multiple coord can be specified in an CoordSet instance for each dimension.

    uncertainty : :class:`~numpy.ndarray`, optional

        standard deviation on the `data`. Handling of uncertainty
        use a fork of the
        `uncertainties <http://pythonhosted.org/uncertainties/>`_
        package (BSD Licence) which is embedded in |scp|.

    units : an instance of :class:`~spectrochempy.core.units.Unit` or string,
    optional

        The units of the data. If `data` is a `Quantity` then
        `units` is set to the units of the `data`; if a `unit`
        is also explicitly
        provided an error is raised. Handling of `units` use a fork of the
        `pint <https://pint.readthedocs.org/en/0.6>`_ (BSD Licence) package
        which is embedded in |scp|)

    meta : :class:`~spectrochempy.core.dataset.ndmeta.Meta` object, optional

        Metadata for this object.

    is_copy :  `bool`, optional, default = `False`.

        `False` means that the initializer try to keep reference
        to the passed `data`


    Notes
    -----
    The underlying array in a `NDDataset` object can be accessed
    through the `data`
    attribute, which will return a conventional :class:`~numpy.ndarray`.

    Examples
    --------

    Usage by an end-user:

    >>> from spectrochempy.api import NDDataset
    >>> x = NDDataset([1,2,3])
    >>> print(x.data) # doctest : +NORMALIZE_WHITESPACE
    [       1        2        3])

    """
    author = Unicode(get_user_and_node(),
                     desc='Name of the author of this dataset',
                     config=True)

    # private metadata in addition to those of the base NDArray class
    _modified = Instance(datetime)
    _description = Unicode
    _history = List(Unicode())

    _coordset = Instance(CoordSet, allow_none=True)

    _modeldata = Array(Float(), allow_none=True)

    _copy = Bool(False)
    _labels_allowed = Bool(False)  # no labels for NDDataset

    def __init__(self, data=None,
                 coordset=None,
                 coordunits=None,
                 coordtitles=None,
                 **kwargs):

        if 'coords' in kwargs:
            raise KeyError()

        super(NDDataset, self).__init__(data, **kwargs)

        self._modified = self._date
        self._description = ''
        self._history = []

        self.coordset = coordset
        self.coordtitles = coordtitles
        self.coordunits = coordunits

    #
    # Default values
    #
    @default('_coordset')
    def _coordset_default(self):
        return None  # CoordSet([None for dim in self.shape])


    @default('_copy')
    def _copy_default(self):
        return False

    @default('_modeldata')
    def _modeldata_default(self):
        return None

    # --------------------------------------------------------------------------
    # additional properties (not in the NDArray base class)
    # --------------------------------------------------------------------------
    @property
    def description(self):
        """
        `str`,

        Provides a description of the underlying data

        """
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def history(self):
        """
        List of strings

        Describes the history of actions made on this array

        """
        return self._history

    @history.setter
    def history(self, value):
        self._history.append(value)

    @validate('_coordset')
    def _coordset_validate(self, proposal):
        coordset = proposal['value']
        if coordset is None:
            return
        for i, item in enumerate(coordset):
            if isinstance(item, NDArray) and \
                                    not isinstance(item, Coord):
                coordset[i] = Coord(item)
        return coordset


    @property
    def coordset(self):
        """
        :class:`~spectrochempy.core.dataset.ndarray.CoordSet` instance

        Contain the coordinates of the various dimension of the dataset

        """
        return self._coordset

    @coordset.setter
    def coordset(self, value):
        if value is not None:
            if self._coordset is not None:
                log.info("Overwriting NDDataset's current "
                         "coordset with one specified")

            for i, coord in enumerate(value):
                if isinstance(coord, CoordSet):
                    size = coord.sizes[i]
                else:
                    size = coord.size
                if size != self.shape[i]:
                    raise CoordError(
                            'the size of each coordinates array must '
                            'be equal to that of the respective data dimension')

            if not isinstance(value, CoordSet):
                self._coordset = CoordSet(value)
            else:
                self._coordset = value

    @property
    def coordtitles(self):
        """
        `List` - A list of the :class:`~spectrochempy.core.dataset.ndcoords.Coord`
        titles.

        """
        if self.coordset is not None:
            return self.coordset.titles

    @coordtitles.setter
    def coordtitles(self, value):
        if self.coordset is not None:
            self.coordset.titles = value

    @property
    def coordunits(self):
        """
        `List`- A list of the :class:`~spectrochempy.core.dataset.ndcoords.Coord`
        units
        """
        if self.coordset is not None:
            return self.coordset.units

    @coordunits.setter
    def coordunits(self, value):

        if self.coordset is not None:
            self.coordset.units = value

    @property
    def modeldata(self):
        """
        models data eventually generated by modelling of the data

        """
        return self._modeldata

    @modeldata.setter
    def modeldata(self, data):
        self._modeldata = data

    @property
    def T(self):
        """
        Same type - Transposed array.

        The object is returned if `ndim` is less than 2.

        """
        return self.transpose()

    @property
    def x(self):
        """
        Read-pnly properties

        Return the x coord, i.e. coordset(-1)

        """
        return self.coordset[-1]

    @x.setter
    def x(self, value):
        self.coordset[-1] = value

    @property
    def y(self):
        """
        Return the y coord, i.e. coordset(-2) for 2D dataset.

        """
        if self.ndim > 1:
            return self.coordset[-2]

    @y.setter
    def y(self, value):
        self.coordset[-2] = value

    @property
    def z(self):
        """
        Read-Only properties

        Return the z coord, i.e. coordset(-3) fpr 3D dataset

        """
        if self.ndim > 2:
            return self.coordset[-3]

    @z.setter
    def z(self, value):
        self.coordset[-3] = value

    @property
    def date(self):
        """
        Read-Only properties

        Date of the dataset creation
        """
        return self._date

    @property
    def modified(self):
        """
        Read-Only properties

        Date of modification

        """
        return self._modified

    # -------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docs)
    # some of the property of NDArray has to be hidden because they are not
    # usefull for this Coord class
    # -------------------------------------------------------------------------

    @property
    def labels(self):
        # not valid for NDDataset
        raise NDDatasetError("There is no label for nd-dataset")

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------

    def coord(self, axis=-1):
        """
        This method return the the coordinates along the given axis

        Parameters
        ----------
        axis : `int` or `unicode`

            An axis index or name, default=-1 for the last axis

        Returns
        -------
        coord : :class:`~numpy.ndarray`

        """
        return self.coordset[axis]

    def transpose(self, axes=None, inplace=False):
        """
        Permute the dimensions of a NDDataset.

        Parameters
        ----------
        axes : list of ints, optional

            By default, reverse the dimensions, otherwise permute the coordset
            according to the values given.

        inplace : `bool`, optional, default = `False`.

            By default a new dataset is returned.
            Change to `True` to chnage data inplace.

        Returns
        -------
        transposed_dataset : same type

            The nd-dataset or a new nd-dataset (inplace=False)
            is returned with coords
            transposed

        See Also
        --------
        :meth:`swapaxes`

        """
        if not inplace:
            new = self.copy()
        else:
            new = self

        if self.ndim < 2:  # cannot transpose 1D data
            return new

        if axes is None:
            axes = list(range(self.ndim - 1, -1, -1))

        new._data = np.transpose(new._data, axes)
        if new.is_masked:
            new._mask = np.transpose(new._mask, axes)
        if new.is_uncertain:
            new._uncertainty = np.transpose(new._uncertainty, axes)

        new._coordset._transpose(axes)
        new._is_complex = [new._is_complex[axis] for axis in axes]

        return new

    def swapaxes(self, axis1, axis2, inplace=False):
        """
        Interchange two dimension of a NDDataset.

        Parameters
        ----------
        axis1 : int

            First axis.

        axis2 : int

            Second axis.

        inplace : bool, optional, default = False

            if False a new object is returned

        Returns
        -------
        swapped_dataset : same type

            The object or a new object (inplace=False) is returned with coordset
            swapped

        See Also
        --------
        :meth:`transpose`

        """

        new = super(NDDataset, self).swapaxes(axis1, axis2, inplace)

        # in addition to what has been done in NDArray base class
        # we need to also swap coordset.

        if new._coordset:
            new._coordset[axis1], new._coordset[axis2] = \
                new._coordset[axis2], new._coordset[axis1]

        return new

    def sort(self, axis=0, pos=None, by='axis', descend=False, inplace=False):
        """
        Returns a copy of the dataset sorted along a given axis
        using the numeric or label values.

        Parameters
        ----------
        axis : `int` , optional, default = 0.

            axis id along which to sort.

        pos: `int` , optional

            If labels are multidimensional  - allow to sort on a define
            row of labels: labels[pos]. Experimental: Not yet checked

        by : `str` among ['axis', 'label'], optional, default = ``axis``.

            Indicate if the sorting is following the order of labels or
            numeric axis values.

        descend : `bool`, optional, default = ``False``.

        inplace : bool, optional, default = ``False``.

            if False a new object is returned,
            else the data are modified inline.

        Returns
        -------
        sorted_dataset : same type

            The object or a new object (inplace=False) is returned with coordset
            sorted

        """

        if not inplace:
            new = self.copy()
        else:
            new = self

        if axis == -1:
            axis = self.ndim - 1

        indexes = []
        for i in range(self.ndim):
            if i == axis:
                args = self.coordset[axis]._argsort(by=by, pos=pos, descend=descend)
                new.coordset[axis] = self.coordset[axis]._take(args)
                indexes.append(args)
            else:
                indexes.append(slice(None))

        new._data = new._data[indexes]
        if new.is_masked:
            new._mask = new._mask[indexes]
        if new.is_uncertain:
            new._uncertainty = new._uncertainty[indexes]

        return new

    def set_complex(self, axis=-1):
        """
        Make a dimension complex

        Parameters
        ----------
        axis : `int`, optional, default = -1

            The axis to make complex

        """
        # override the ndarray function because we must care about the axis too.

        if self.data.shape[axis] % 2 == 0:
            # we have a pair number of element along this axis.
            # It can be complex
            # data are then supposed to be interlaced (real, imag, real, imag ..
            self._is_complex[axis] = True
        else:
            raise ValueError('The odd size along axis {} is not compatible with'
                             ' complex interlaced data'.format(axis))

        if self.coordset:
            new_axis = self.coordset[axis][::2]
            self.coordset[axis] = new_axis

    # Create the returned values of functions should be same class as input.
    # The units should have been handled by __array_wrap__ already

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    def __dir__(self):
        return NDIO().__dir__() + ['data', 'mask', 'units', 'uncertainty',
                                   'meta', 'name', 'title', 'is_complex',
                                   'coordset', 'description', 'history', 'date',
                                   'modified', 'modeldata'
                                   ]

    def __str__(self):
        # Display the metadata of the object and partially the data

        # print field names/values (class/sizes)
        # data.name, .author, .date,
        out = '\n' + '-' * 80 + '\n'
        out += '      name/id: {}\n'.format(self.name)
        out += '       author: {}\n'.format(self.author)
        out += '      created: {}\n'.format(self._date)
        out += 'last modified: {}\n'.format(self._modified)

        wrapper1 = textwrap.TextWrapper(initial_indent='',
                                        subsequent_indent=' ' * 15,
                                        replace_whitespace=True)

        pars = self.description.strip().splitlines()

        out += '  description: '
        if pars:
            out += '{}\n'.format(wrapper1.fill(pars[0]))
        for par in pars[1:]:
            out += '{}'.format(textwrap.indent(par, ' ' * 15))

        if not out.endswith('\n'):
            out += '\n'

        if self._history != ['']:
            pars = self.history
            out += '      history: '
            if pars:
                out += '{}\n'.format(wrapper1.fill(pars[0]))
            for par in pars[1:]:
                out += '{}'.format(textwrap.indent(par, ' ' * 15))

            if not out.endswith('\n'):
                out += '\n'

        #if uncertainty is not nouncertainty:
        #    uncertainty = "(+/-%s)" % self.uncertainty

        # units = '{:~K}'.format(
        #         self.units) if self.units is not None else 'unitless'

        sh = ' size' if self.ndim < 2 else 'shape'
        shapecplx = (x for x in
                     itertools.chain.from_iterable(
                             list(zip(self.shape,
                                 [False]*self.ndim
                                 if not self.is_complex else self.is_complex))))
        shape = (' x '.join(['{}{}'] * len(self.shape))).format(
                *shapecplx).replace(
                'False', '').replace('True', '(complex)')
        size = self.size
        sizecplx = '' if not self.is_complex else " (complex)"

        out += '   data title: {}\n'.format(self.title)
        out += '    data size: {}{}\n'.format(size, sizecplx) if self.ndim < 2 \
            else '   data shape: {}\n'.format(shape)

        #out += '   data units: {}\n'.format(units)
        #data_str = str(
        #        self._uarray(self._data, self._uncertainty)).replace('\n\n',
        #                                                 '\n')

        data_str = super(NDDataset, self).__str__()
        out += '  data values:\n'
        out += '{}\n'.format(textwrap.indent(str(data_str), ' ' * 9))

        if self.coordset is not None:
            for i, coord in enumerate(self.coordset):
                coord_str = str(coord).replace('\n\n', '\n')
                out += 'coordinates {}:\n'.format(i)
                out += textwrap.indent(coord_str, ' ' * 9)
                out += '\n'

        if not out.endswith('\n'):
            out += '\n'
        out += '-' * 80
        out += '\n'

        return out

    def __getattr__(self, item):
        # when the attribute was not found

        if item in ["__numpy_ufunc__"] or '_validate' in item or \
                        '_changed' in item:
            # raise an error so that masked array will be handled correctly
            # with arithmetic operators and more
            raise AttributeError


    def __getitem__(self, item):
        # we need coordset (but they might be not present...
        # in this case coordset are simply the indexes

        #if self.coordset is None:
        #    # create coordset from indexes
        #    coordset = CoordSet([Coord(np.arange(l)) for l in self._data.shape])
        #else:
        #    coordset = self.coordset

        # transform the passed index (if necessary) to integer indexes
        keys, internkeys = self._make_index(item)

        new = self.copy()

        # slicing by index of all internal array
        new._data = np.array(self._data[internkeys])
        new._is_complex = self._is_complex

        if self.is_masked:
            new._mask = np.array(self._mask[keys])

        if self.is_uncertain:
            new._uncertainty = np.array(self._uncertainty[keys])

        if new._data.size == 0:
            raise IndexError("Empty array of shape {} resulted from slicing.\n"
                             "Check the indexes and make "
                             "sure to use floats for "
                             "location slicing".format(str(new._data.shape)))

        if self._coordset is not None:
            new_coordset = self.coordset.copy()
            for i, coord in enumerate(new_coordset):
                new_coordset[i] = coord[keys[i]]
            new._coordset = new_coordset

        new._name = '*' + self._name

        return new.squeeze()

    def __eq__(self, other, attrs=None):
        attrs = self.__dir__()
        for attr in ('name', 'description', 'history', 'date', 'modified'):
            attrs.remove(attr)
        #some attrib are not important for equality
        return super(NDDataset, self).__eq__(other, attrs)

    # def __iter__(self):
    #     if self.ndim == 0:
    #         raise TypeError('iteration over a 0-d array')
    #     for n in range(len(self)):
    #         yield self[n]

    # # the following methods are to give NDArray based class
    # # a behavior similar to np.ndarray regarding the ufuncs
    # def __array_prepare(self, *args, **kwargs):
    #     pass
    #
    # def __array_wrap__(self, *args):
    #     # called when element-wise ufuncs are applied to the array
    #
    #     f, objs, huh = args[1]
    #
    #     # case of complex dataset
    #     if f.__name__ in ['real', 'imag', 'conjugate', 'absolute']:
    #         return getattr(objs[0], f.__name__)()
    #
    #     if  self.iscomplex[-1]:
    #         if f.__name__ in ["fabs",]:
    #  fonction not available for complex data
    #             raise ValueError("{} does not accept complex data ".format(f))
    #
    #     data, uncertainty, units, mask = self._op(f, objs, ufunc=True)
    #     return self._op_result(data, uncertainty, units, mask)

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    def _repr_html_(self):
        tr = "<tr style='border-bottom: 1px solid lightgray;" \
             "border-top: 1px solid lightgray;'>" \
             "<td style='padding-right:5px'><strong>{}</strong></td>" \
             "<td>{}</td><tr>\n"

        out = '<table>\n'

        out += tr.format("Id/Name", self.name)
        out += tr.format("Author", self.author)
        out += tr.format("Created", str(self.date))
        out += tr.format("Last Modified", self.modified)

        wrapper1 = textwrap.TextWrapper(initial_indent='',
                                        subsequent_indent=' ' * 15,
                                        replace_whitespace=True)

        out += tr.format("Description", wrapper1.fill(self.description))

        if self.history:
            pars = self.history
            hist = ""
            if pars:
                hist += '{}'.format(wrapper1.fill(pars[0]))
            for par in pars[1:]:
                hist += '{}'.format(textwrap.indent(par, ' ' * 15))
            out += tr.format("History", hist)

        uncertainty = "(+/-%s)" % self.uncertainty \
            if self.uncertainty is not None else ""
        units = '{:~T}'.format(
                self.units) if self.units is not None else 'unitless'

        sh = ' size' if self.ndim < 2 else 'shape'
        shapecplx = (x for x in
                     itertools.chain.from_iterable(
                             list(zip(self.shape, self.is_complex))))

        shape = (' x '.join(['{}{}'] * len(self.shape))).format(
                *shapecplx).replace('False', '').replace('True', '(complex)')

        size = self.size
        sizecplx = '' if not self.is_complex else " (complex)"
        size = '{}{}'.format(size, sizecplx) \
            if self.ndim < 2 else '{}'.format(shape)

        data = '<table>\n'
        data += tr.format("Title", self.title)
        data += tr.format("Size", size)
        data += tr.format("Units", units)
        data_str = str(self._uarray(self._data, self._uncertainty))
        data_str = data_str.replace('\n\n', '\n')
        data += tr.format("Values", textwrap.indent(str(data_str), ' ' * 9))
        data += '</table>\n'  # end of row data

        out += tr.format('data', data)

        if self.coordset is not None:
            for i, axis in enumerate(self.coordset):
                axis_str = axis._repr_html_().replace('\n\n', '\n')
                out += tr.format("axis %i" % i,
                                 textwrap.indent(axis_str, ' ' * 9))

        out += '</table><br/>\n'

        return out

    def _loc2index(self, loc, axis):
        # Return the index of a location (label or coordinates) along the axis

        coord = self.coordset[axis]
        return coord._loc2index(loc)

    # -------------------------------------------------------------------------
    # events
    # -------------------------------------------------------------------------
    @observe(All)
    def _anytrait_changed(self, change):

        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }

        if change['name'] in ["_date", "_modified", "trait_added"]:
            return

        # changes in data -> update dates
        if change['name'] == '_data' and self._date == datetime(1970, 1, 1, 0, 0):
            self._date = datetime.now()
            self._modified = datetime.now()

        # change to complex
        # change type of data to complex
        # require modification of the coordset, if any
        if change['name'] == '_is_complex':
            pass

        # all the time -> update modified date
        self._modified = datetime.now()

        return


# make some function also accessiibles from the module
squeeze = NDDataset.squeeze
sort = NDDataset.sort
swapaxes = NDDataset.swapaxes
transpose = NDDataset.transpose
abs = NDDataset.abs
conj = NDDataset.conj
imag = NDDataset.imag
real = NDDataset.real

# =============================================================================
# Set the operators
# =============================================================================

set_operators(NDDataset, priority=50)
