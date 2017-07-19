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


#

"""This module provides Axis related classes

"""

import copy
import uuid
import warnings

import numpy as np
from pandas import Index
from six import string_types
from traitlets import (HasTraits, List, Bool, Unicode, default, Instance)

from spectrochempy.application import log

from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndmath import NDMath, set_operators
from spectrochempy.core.units import Quantity
# from ...utils import create_traitsdoc

from spectrochempy.utils import (is_sequence, numpyprintoptions,
                      SpectroChemPyWarning)
from spectrochempy.utils.traittypes import Range

__all__ = ['Axes',
           'Axis',
           'AxisRange',
           'AxisRangeError',
           'AxisError',
           'AxisError',
           'AxisWarning']
_classes = __all__[:]


# =============================================================================
# set numpy print options
# =============================================================================

numpyprintoptions()


# =============================================================================
#  Errors and warnings
# =============================================================================

class AxisRangeError(ValueError):
    """An exception that is raised when something is wrong with the axisrange.
    """


class AxisError(ValueError):
    """An exception that is raised when something is wrong with the axis or
    axes.
    """

class AxisWarning(SpectroChemPyWarning):
    """A warning that is raised when something is wrong with the axis or
    axes definitions but do not necessarily need to raise an error.
    """


# =============================================================================
# Axis
# =============================================================================

class Axis(NDMath, NDArray):
    """A class describing the axes of the data along a given axis.

    Parameters
    -----------
    coords : :class:`~numpy.ndarray`, :class:`~numpy.ndarray`-like,
        or another `Axis` object.

        The actual data contained in this `Axis` object.

    labels : :class:`~numpy.ndarray`, :class:`~numpy.ndarray`-like of the
        same length as coords, optional

        It contains the axes labels. If only labels are provided during
        initialisation of an axis, a numerical axis is automatically created
        with the labels indices

    units : :class:`pint.unit.Unit` instance or str, optional

        The units of the data. If data is a :class:`pint.quantity.Quantity` then
        `unit` is set to the unit of the data; if a unit is also explicitly
        provided an error is raised.

    title : unicode.

        The title of the axis. It will be used for instance to label the axe
        in plots

    name : unicode.

        The name of the axis. Default is set automatically.

    meta : `dict`-like object, optional.

        Additional metadata for this object. Must be dict-like but no further
        restriction is placed on meta.

    iscopy : bool

        Perform a copy of the passed object. By default, objects are not
        copied if possible

    Notes
    -----
    The data in a `Axis` object should be accessed through the coords (
    which is an alias of data) attribute.

    For example::

    >>> from spectrochempy.api import Axis


    >>> x = Axis([1,2,3], title='time on stream', units='hours')
    >>> x.coords # doctest: +NORMALIZE_WHITESPACE
    array([ 1, 2, 3])


    """
    _iscopy = Bool

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self,
                 coords=None,
                 labels=None,
                 mask=None,
                 units=None,
                 title=None,
                 name=None,
                 meta=None,
                 **kwargs):

        _iscopy = kwargs.pop('iscopy', False)

        super(Axis, self).__init__(**kwargs)

        if _iscopy:
            self._iscopy = True
            coords = copy.deepcopy(coords)
            labels = copy.deepcopy(labels)
            mask = copy.deepcopy(mask)
            units = copy.copy(units)  # deepcopy not working for units?
            meta = copy.deepcopy(meta)

        if coords is not None:
            self.coords = coords
            if labels is not None:
                self.labels = labels

        else:
            if labels is not None:
                self.coords = range(len(labels))
                self.labels = labels

        if mask is not None:
            if self._data_passed_with_mask and self._mask != mask:
                log.info("Axis was created with a masked array, and a "
                            "mask was explicitly provided to Axis. The  "
                            "explicitly passed-in mask will be used and the "
                            "masked array's mask will be ignored.")
            self.mask = mask

        if units is not None:
            if self._data_passed_is_quantity and self._units != units:
                raise ValueError(
                        "Cannot use the units argument when passed data "
                        "is a Quantity")

            self.units = units

        self.title = title
        self.name = name
        self.meta = meta

    # -------------------------------------------------------------------------
    # properties
    # -------------------------------------------------------------------------
    @default('_name')
    def _get_name_default(self):
        return u"Axis_"+str(uuid.uuid1()).split('-')[0]  # a unique id


    @property
    def is_reversed(self):
        """`bool`, read-only property - Whether the axis is ascending or reversed.

        return a correct result only if the data are sorted

        """
        return bool(self.data[0] > self.data[-1])

    @property
    def coords(self):
        """:class:`~numpy.ndarray`-like object - The axis coordinates.
        (alias of `data`)

        """
        return self._data

    @coords.setter
    def coords(self, data):
        # property.setter for data

        if data is None:
            self._data = np.array([]).astype(float)
            log.debug("init axis with an empty array of type float")
            return

        elif isinstance(data, Axis):
            log.debug("init axis with data from another Axis")
            # No need to check the data because data must have successfully
            # initialized.
            self._name = "copy of {}".format(data._name) if self._iscopy \
                else data._name
            self._title = data._title
            self._mask = data._mask
            self._data = data._data
            self._units = data._units
            self._meta = data._meta
            self._labels = data._labels


        elif isinstance(data, Index):  # pandas Index object
            self._data = data.values
            if data.name is not None:
                self._title = data.name

        elif isinstance(data, Quantity):
            log.debug("init data with data from a Quantity object")
            self._data_passed_is_quantity = True
            self._data = np.array(data.magnitude, subok=True, copy=self._iscopy)
            self.units = data.units  # we use the property setter for checking

        elif hasattr(data, 'mask'):  # an object with data and mask attributes
            log.debug("init mask from the passed data")
            self._data_passed_with_mask = True
            self._data = np.array(data.data, subok=True, copy=self._iscopy)
            self.mask = data.mask

        elif (not hasattr(data, 'shape') or
                  not hasattr(data, '__getitem__') or
                  not hasattr(data, '__array_struct__')):
            # Data doesn't look like a numpy array, try converting it to
            # one.
            log.debug("init axis with a non numpy-like array object")
            self._data = np.array(data, subok=True, copy=False)
            # Quick check to see if what we got out looks like an array
            # rather than an object (since numpy will convert a
            # non-numerical input to an array of objects).
            if self._data.dtype == 'O':
                raise AxisError(
                        "Could not convert data to numpy array.")
        else:
            log.debug("init data axis a numpy array")
            self._data = np.array(data, subok=True, copy=self._iscopy)



    # hidden properties (for the documentation, only - we remove the docs)
    # some of the property of NDArray has to be hidden because they are not
    # usefull for this Axis class
    @property
    def is_complex(self):
        return None  # always real

    @property
    def ndim(self):
        return 1

    @property
    def uncertainty(self):
        return None

    @property
    def T(self):  # no transpose
        return self

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape


    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    # def _argsort(self, by='axis', pos=None, descend=False):
    #     # found the indices sorted by axes or labels
    #
    #     if not self.is_labeled:
    #         by = 'axis'
    #         pos = None
    #         warnings.warn('no label to sort, use `axis` by default',
    #                       AxisWarning)
    #
    #     if by == 'axis':
    #         args = np.argsort(self._data)
    #
    #     elif by == 'label':
    #         labels = self._labels
    #         if len(self._labels.shape) > 1:
    #             # multidimentional labels
    #             if not pos: pos = 0
    #             labels = self._labels[pos]  # TODO: this must be checked
    #         args = np.argsort(labels)
    #
    #     else:
    #         by = 'axis'
    #         warnings.warn(
    #                 'parameter `by` should be set to `axis` or `label`, use `axis` by default',
    #                 AxisWarning)
    #         args = np.argsort(self._data)
    #
    #     if descend:
    #         args = args[::-1]
    #
    #     return args
    #
    # def _sort(self, by='axis', pos=None, descend=False, inplace=False):
    #     # sort axis in place using data or label values
    #
    #     if not inplace:
    #         new = self.copy()
    #     else:
    #         new = self
    #
    #     args = self._argsort(by, pos, descend)
    #
    #     return self._take(args)
    #
    # def _take(self, indices):
    #     # get an axis with indices
    #
    #     new = self.copy()
    #     new._data = new._data[indices]
    #     new._labels = new._labels[..., indices]
    #     new._mask = new._mask[indices]
    #
    #     return new

    # def _loc2index(self, loc):
    #     # Return the index of a location (label or coordinates) along the axis
    #
    #     if isinstance(loc, string_types):
    #         # it's probably a label
    #         indexes = np.argwhere(self._labels == loc).flatten()
    #         if indexes.size > 0:
    #             return indexes[0]
    #         else:
    #             raise ValueError('Could not find this label: {}'.format(loc))
    #
    #     elif isinstance(loc, datetime):
    #         # not implemented yet
    #         return None  # TODO: date!
    #
    #     elif is_number(loc):
    #         index = (np.abs(self._data - loc)).argmin()
    #         if loc > self._data.max() or loc < self._data.min():
    #             warnings.warn(
    #                     '\nThis coordinate ({}) is outside the axis limits.\n'
    #                     'The closest limit index is returned'.format(loc),
    #                     AxisWarning)
    #         return index
    #
    #     else:
    #         raise ValueError('Could not find this location: {}'.format(loc))

    def _repr_html_(self):

        units = '{:~T}'.format(self._units) \
            if self._units is not None else 'unitless'
        out = '<table>'
        out += '<tr><td>title</td><td>%s</td></tr>\n' % (
            self.title.capitalize())
        out += '<tr><td>coordinates</td><td>%s</td></tr>\n' % np.array2string(
                self.coords, separator=' ')
        out += '<tr><td>units</td><td>%s</td></tr>\n' % units
        if self.is_labeled:
            out += '<tr border=0><td>labels</td><td>%s</td></tr>\n' % self.labels
        out += '</table>\n'
        return out

    # TODO: _repr_latex

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    def __dir__(self):
        # with remove some methods with respect to the full NDArray
        # as they are not usefull for Axis.
        return ['data', 'mask', 'labels', 'units',
                'meta', 'name', 'title']

    def __str__(self):
        units = '{:~K}'.format(self._units) \
            if self._units is not None else 'unitless'
        out = '      title: %s\n' % (self.title.capitalize())
        out += 'coordinates: %s\n' % np.array2string(self.coords, separator=' ')
        out += '      units: %s\n' % units
        if self.is_labeled:
            out += '     labels: %s\n' % str(self.labels)
        out += '\n'
        return out

    def __repr__(self):
        units = '{:~K}'.format(self._units) \
            if self._units is not None else 'unitless'
        prefix = self.__class__.__name__ + '('
        body = np.array2string(self.coords, separator=', ', prefix=prefix)
        return ''.join([prefix, body, ') {}'.format(units)])


    # def __lt__(self, other):
    #     # hack to make axis sortable
    #     this = self.data
    #     if hasattr(other, '_data'):
    #         other = other._data
    #     try:
    #         return self[0] < other[0]
    #     except IndexError:
    #         return this < other


# =============================================================================
# Axes
# =============================================================================

class Axes(HasTraits):
    """A collection of axes for a dataset with a validation method.

    Parameters
    ----------
    axes : list of Axis objects.

    issamedim : bool, optional, default=False

        if true, all axis describes a single dimension.
        By default, each item describes a different dimension.

    """

    # Hidden attributes containing the collection of Axis instance
    _axes = List(Instance(Axis), allow_none=True)

    # Hidden name of the object
    _name = Unicode

    @default('_name')
    def _get_name_default(self):
        return u"Axes_"+str(uuid.uuid1()).split('-')[0]  # a unique id


    # Hidden attribute to specify if the collection is for a single dimension
    _issamedim = Bool

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self, *axes, **kwargs):

        _iscopy = kwargs.pop('iscopy', False)

        super(Axes, self).__init__(**kwargs)

        self._axes = []

        if all([isinstance(axes[i],(Axis,Axes)) for i in range(len(axes))]):
            axes = list(axes)
        elif len(axes)==1:
            # this a set of axis or axes passed as a list
            axes = axes[0]
        else:
            # not implemented yet -
            # a list of list of object have been passed
            #TODO: try to ipmplement this
            raise AxisError(
     'a list of list of object have been passed - this not yet implemented')

        if len(axes)==1 and isinstance(axes[0], Axes):
            if _iscopy:
                axes = copy.deepcopy(axes)
            self._axes = axes[0]._axes

        else:
            for item in axes:

                if not isinstance(item, (Axis, Axes)):

                    item = Axis(item,  iscopy=_iscopy)
                                                 # full validation of the item
                                                 # will be done in Axis
                if self._validation(item):
                    self._axes.append(item)

        # check if we have single dimension axis

        for item in self._axes:
            if isinstance(item, Axes):
                # it must be a single dimension axis
                item._issamedim = True
                # in this case we must have same length axes
                siz = item[0].size
                if np.any([elt.size!=siz for elt in item._axes]):
                    raise AxisError('axis must be of the same size '
                                'for a dimension with multiple axis dimension')

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def name(self):
        return self._name

    @property
    def names(self):
        """`list`, read-only property - Get the list of axis names.
        """
        if len(self._axes)<1:
            return []
        try:
            return [item.name for item in self._axes]
        except:
            log.critical(self._axes)

    @property
    def titles(self):
        """`list` - Get/Set a list of axis titles.

        """
        _titles = []
        for item in self._axes:
            if isinstance(item, Axis):
                _titles.append(item.title if item.title else item.name)
            elif isinstance(item, Axes):
                _titles.append([el.title if el.title else el.name
                                       for el in item])
            else:
                raise AxisError('Something wrong with the titles!')

        return _titles

    @titles.setter
    def titles(self, value):
        # Set the titles at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._axes[i].title = item

    @property
    def labels(self):
        """`list` - Get/Set a list of axis labels.

        """
        return [item.label for item in self._axes]

    @labels.setter
    def labels(self, value):
        # Set the labels at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._axes[i].label = item

    @property
    def units(self):
        """`list` - Get/Set a list of axis units.

        """
        return [item.units for item in self._axes]

    @units.setter
    def units(self, value):
        if is_sequence(value):
            for i, item in enumerate(value):
                self._axes[i].units = item

    @property
    def isempty(self):
        """`bool`, read-only property - `True` if there is no axes defined.

        """
        return len(self._axes) == 0

    @property
    def issamedim(self):
        """`bool`, read-only property -
        `True` if the axes define a single dimension.

        """
        return self._issamedim

    @property
    def sizes(self):
        """`int`, read-only property -
        gives the size of the axis or axes for each dimention"""
        _sizes = []
        for i, item in enumerate(self._axes):
            if isinstance(item, Axis):
                _sizes.append(item.size)
            elif isinstance(item, Axes):
                _sizes.append(item.sizes[i][0])
        return _sizes

    @property
    def coords(self):
        """:class:`~numpy.ndarray`-like object - The first axis coordinates.

        """
        return self[0]._data

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------
    def copy(self):
        """Make a disconnected copy of the current axes.

        Returns
        -------
        axes : same type
            an exact copy of the current object

        """
        return self.__copy__()

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------
    def _transpose(self, axes=None):
        # in principle it is not directly called by the user as it is intimately
        # linked to a dataset
        if self._issamedim :
            # not applicable for same dimension axes
            warnings.warn('Axes for a single dimentsion are not transposable',
                          AxisWarning)
            return
        if axes is None:
            self._axes.reverse()
        else:
            self._axes = [self._axes[axis] for axis in axes]

    def _validation(self, item):
        # To be valid any added axis must have a different name

        if not isinstance(item, (Axis, Axes) ):
            raise AxisError('The elements of must be Axis or '
                                      'Axes objects only!')

        if item._name in self.names:
            raise AxisError('The axis name must be unique!')

        if isinstance(item, Axis) and item.ndim > 1:
            raise AxisError('An axis should be a 1D array!')

        # TODO: add more validation for Axes objects

        return True

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    def __dir__(self):
        return ['_axes']

    def __call__(self, *args):
        # allow the following syntax: axes(0,2), or axes(axis=(0,2))
        axes = []
        if args:
            for idx in args:
                axes.append(self._axes[idx])
        if len(axes) == 1:
            return axes[0]
        else:
            return Axes(axes)

    def __len__(self):
        return len(self._axes)

    def __getitem__(self, index):

        if isinstance(index, string_types):
            if index in self.titles:
                # selection by axis title
                return self._axes.__getitem__(self.titles.index(index))
            # may be it is in a multiple axis
            for item in self._axes:
                if isinstance(item, Axes) and index in item.titles:
                    # selection by subaxis title
                    return item.__getitem__(item.titles.index(index))

        res = self._axes.__getitem__(index)
        if isinstance(index, slice):
            return Axes(res)
        else:
            return res

    def __setitem__(self, index, axes):
        self._axes[index] = axes

    def __iter__(self):
        for item in self._axes:
            yield item

    def __repr__(self):
        out = ("Axes object <" + ', '.join(['<Axis object {}>']
                                           * len(self._axes)) + ">")
        out = out.format(*self.names)
        return out

    def __str__(self):
        out = "(" + ', '.join(['[{}]'] * len(self._axes)) + ")"
        out = out.format(*self.titles)
        return out

    def __deepcopy__(self, memo):
        return self.__class__([copy.deepcopy(ax, memo) for ax in self])

    def __copy__(self):
        return self.__class__([copy.copy(ax) for ax in self])

    def __eq__(self, other):
        return self._axes == other._axes  # TODO: check the case of compatible units

    def __ne__(self, other):
        return not self.__eq__(other)

# =============================================================================
# AxisRange
# =============================================================================

class AxisRange(HasTraits):
    """An axisrange is a set of ordered, non intersecting intervals,\
    e.g. [[a, b], [c, d]] with a < b < c < d or a > b > c > d.

    Parameters
    -----------

   ranges :  iterable,  an interval or a set of intervals.
            an interval or a set of intervals.
            set of  intervals. If none is given, the axisrange \
            will be a set of an empty interval [[]]. The interval limits do not \
            need to be ordered, and the intervals do not need to be distincts.

   reversed : (`bool`) the intervals are ranked by decreasing order if True \
     or increasing order if False.

   nranges:  int
        number of distinct ranges

   """
    # TODO: May use also units ???

    ranges = List(Range)
    reversed = Bool

    def __init__(self, *ranges, **kwargs):
        """ Constructs Axisrange with default values

        """
        self.reversed = kwargs.get('reversed', False)

        if len(ranges) == 0:
            # first case: no argument passed, returns an empty range

            self.ranges = []

        elif len(ranges) == 2 and all(
                isinstance(elt, (int, float)) for elt in ranges):
            # second case: a pair of scalars has been passed
            # using the Interval class, we have autochecking of the interval validity

            self.ranges = [list(ranges)]

        else:
            # third case: a set of pairs of scalars has been passed

            self.ranges = self._cleanranges(ranges)

        if self.reversed:
            self.reverse()

    def __iter__(self):
        """
        return an iterator

        """
        for x in self.ranges:
            yield x

    # Properties

    @property
    def nranges(self):
        return len(self.ranges)

    # public methods

    def reverse(self):
        """ Reverse the order of the axis range

        """
        for range in self.ranges:
            range.reverse()
        self.ranges.reverse()

    # private methods

    def _cleanranges(self, ranges):
        ''' sort and merge overlaping ranges

        works as follows:
         1. orders each interval
         2. sorts intervals
         3. merge overlapping intervals
         4. reverse the orders if required

        '''

        # transforms each pairs into valid interval
        # should generate an error if a pair is not valid

        ranges = [list(range) for range in ranges]

        # order the ranges

        ranges = sorted(ranges, key=lambda r: min(r[0], r[1]))

        cleaned_ranges = [ranges[0]]

        for range in ranges[1:]:
            if range[0] <= cleaned_ranges[-1][1]:
                if range[1] >= cleaned_ranges[-1][1]:
                    cleaned_ranges[-1][1] = range[1]
            else:
                cleaned_ranges.append(range)

        return cleaned_ranges


# =============================================================================
# Set the operators
# =============================================================================

set_operators(Axis, priority=50)

# =============================================================================
# Modify the doc to include Traits
# =============================================================================
#create_traitsdoc(Axes)
#create_traitsdoc(Axis)

if __name__ == '__main__':
    pass