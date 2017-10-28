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


#

"""This module provides Coord related classes

"""

import copy
import uuid
import warnings

import numpy as np
from pandas import Index
from six import string_types
from traitlets import (HasTraits, List, Bool, Unicode, default, Instance)

from spectrochempy.application import log

from spectrochempy.core.dataset.ndarray import (NDArray)
from spectrochempy.core.dataset.ndmath import NDMath, set_operators
from spectrochempy.core.units import Quantity
# from ...utils import create_traitsdoc

from spectrochempy.utils import (is_sequence, numpyprintoptions,
                                 SpectroChemPyWarning)
from spectrochempy.utils.traittypes import Range

__all__ = ['CoordSet',
           'Coord',
           'CoordsRange',
           'CoordsRangeError',
           'CoordsError',
           'CoordsError',
           'CoordsWarning']
_classes = __all__[:]

# =============================================================================
# set numpy print options
# =============================================================================

numpyprintoptions()


# =============================================================================
#  Errors and warnings
# =============================================================================

class CoordsRangeError(ValueError):
    """An exception that is raised when something is wrong with the CoordsRange.
    """


class CoordsError(ValueError):
    """An exception that is raised when something is wrong with the Coord or
    CoordSet.
    """


class CoordsWarning(SpectroChemPyWarning):
    """A warning that is raised when something is wrong with the Coord or
    CoordSet definitions but do not necessarily need to raise an error.
    """


# =============================================================================
# Coord
# =============================================================================

class Coord(NDMath, NDArray):
    """A class describing the coords of the data along a given axis.

    Parameters
    -----------
    data : :class:`~numpy.ndarray`, :class:`~numpy.ndarray`-like,
        or another `Coord` object.

        The actual data contained in this `Coord` object.

    labels : :class:`~numpy.ndarray`, :class:`~numpy.ndarray`-like of the
        same length as coords, optional

        It contains the coords labels. If only labels are provided during
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
    The data in a `Coord` object should be accessed through the coords (
    which is an alias of data) attribute.

    For example::

    >>> from spectrochempy.api import Coord # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
            SpectroChemPy's API
            Version   : ...
    <BLANKLINE>

    >>> x = Coord([1,2,3], title='time on stream', units='hours')
    >>> print(x) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
              title: Time on stream
    coordinates: [       1        2        3]
          units: hr

    >>> print(x.data) # doctest: +NORMALIZE_WHITESPACE
    [       1        2        3]

    """
    _copy = Bool

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):

        super(Coord, self).__init__(data, **kwargs)

        # some checking
        if self._data.ndim >1:
            raise CoordsError("Number of dimension for coordinate's array "
                              "should be 1!")


        # if self._copy:
        #     self._copy = True
        #     data = copy.deepcopy(data)
        #     labels = copy.deepcopy(labels)
        #     mask = copy.deepcopy(mask)
        #     units = copy.copy(units)  # deepcopy not working for units?
        #     meta = copy.deepcopy(meta)
        #
        # if data is not None:
        #     self.data = data
        #     if labels is not nolabel:
        #         self.labels = labels
        #
        # else:
        #     if labels is not nolabel:
        #         self.coords = range(len(labels))
        #         self.labels = labels
        #
        # if mask is not None:
        #     if self._data_passed_with_mask and self._mask != mask:
        #         log.info("Coord was created with a masked array, and a "
        #                  "mask was explicitly provided to Coord. The  "
        #                  "explicitly passed-in mask will be used and the "
        #                  "masked array's mask will be ignored.")
        #     self.mask = mask
        #
        # if units is not None:
        #     if self._data_passed_is_quantity and self._units != units:
        #         raise ValueError(
        #                 "Cannot use the units argument when passed data "
        #                 "is a Quantity")
        #
        #     self.units = units
        #
        # self.title = title
        # self.name = name
        # self.meta = meta

    # -------------------------------------------------------------------------
    # properties
    # -------------------------------------------------------------------------
    @default('_name')
    def _get_name_default(self):
        return u"Coords_" + str(uuid.uuid1()).split('-')[0]  # a unique id

    @property
    def is_reversed(self):
        """`bool`, read-only property - Whether the axis is ascending or reversed.

        return a correct result only if the data are sorted

        """
        return bool(self.data[0] > self.data[-1])

    # @data.setter
    # def coords(self, data):
    #     # property.setter for data
    #
    #     if data is None:
    #         self._data = np.array([]).astype(float)
    #         log.debug("init axis with an empty array of type float")
    #         return
    #
    #     elif isinstance(data, Coord):
    #         log.debug("init axis with data from another Coord")
    #         # No need to check the data because data must have successfully
    #         # initialized.
    #         self._name = "copy of {}".format(data._name) if self._iscopy \
    #             else data._name
    #         self._title = data._title
    #         self._mask = data._mask
    #         self._data = data._data
    #         self._units = data._units
    #         self._meta = data._meta
    #         self._labels = data._labels
    #
    #
    #     elif isinstance(data, Index):  # pandas Index object
    #         self._data = data.values
    #         if data.name is not None:
    #             self._title = data.name
    #
    #     elif isinstance(data, Quantity):
    #         log.debug("init data with data from a Quantity object")
    #         self._data_passed_is_quantity = True
    #         self._data = np.array(data.magnitude, subok=True, copy=self._iscopy)
    #         self.units = data.units  # we use the property setter for checking
    #
    #     elif hasattr(data, 'mask'):  # an object with data and mask attributes
    #         log.debug("init mask from the passed data")
    #         self._data_passed_with_mask = True
    #         self._data = np.array(data.data, subok=True, copy=self._iscopy)
    #         self._mask = data.mask
    #
    #     elif (not hasattr(data, 'shape') or
    #               not hasattr(data, '__getitem__') or
    #               not hasattr(data, '__array_struct__')):
    #         # Data doesn't look like a numpy array, try converting it to
    #         # one.
    #         log.debug("init axis with a non numpy-like array object")
    #         self._data = np.array(data, subok=True, copy=False)
    #         # Quick check to see if what we got out looks like an array
    #         # rather than an object (since numpy will convert a
    #         # non-numerical input to an array of objects).
    #         if self._data.dtype == 'O':
    #             raise CoordsError(
    #                     "Could not convert data to numpy array.")
    #     else:
    #         log.debug("init data axis a numpy array")
    #         self._data = np.array(data, subok=True, copy=self._iscopy)

    ########
    # hidden properties (for the documentation, only - we remove the docs)
    # some of the property of NDArray has to be hidden because they are not
    # usefull for this Coord class

    @property
    def is_complex(self):
        return False  # always real

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
    def shape(self):
        return self._data.shape

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    def _repr_html_(self):

        tr = "<tr style='border-bottom: 1px solid lightgray;" \
             "border-top: 1px solid lightgray;'>" \
             "<td style='padding-right:5px'><strong>{}</strong></td>" \
             "<td>{}</td><tr>\n"

        units = '{:~T}'.format(self._units) \
            if self._units is not None else 'unitless'
        out = '<table>\n'
        out += tr.format("Title", self.title.capitalize())
        out += tr.format("Coordinates",
                         np.array2string(self.data, separator=' '))
        out += tr.format("Units", units)
        if self.is_labeled:
            out += tr.format("Labels", self.labels)
        out += '</table><br/>\n'
        return out

    # TODO: _repr_latex

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    def __dir__(self):
        # with remove some methods with respect to the full NDArray
        # as they are not usefull for Coord.
        return ['data', 'mask', 'labels', 'units', 'meta', 'name', 'title']

    def __str__(self):
        units = '{:~K}'.format(self._units) \
            if self._units is not None else 'unitless'
        out = '      title: %s\n' % (self.title.capitalize())
        out += 'coordinates: %s\n' % np.array2string(self.data, separator=' ')
        out += '      units: %s\n' % units
        if self.is_labeled:
            out += '     labels: %s\n' % str(self.labels)
        #out += '\n'
        if out[-1]=='\n':
            out = out[:-1]
        return out

    # def __repr__(self):
    #     units = '{:~f}'.format(self._units) \
    #         if self._units is not None else 'unitless'
    #     prefix = self.__class__.__name__ + '('
    #     body = np.array2string(self.data, separator=', ', prefix=prefix)
    #     return ''.join([prefix, body, ') {}'.format(units)])
    #

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
# CoordSet
# =============================================================================

class CoordSet(HasTraits):
    """A collection of Coord for a dataset with a validation method.

    Parameters
    ----------
    coords : list of Coord objects.

    issamedim : bool, optional, default=False

        if true, all axis describes a single dimension.
        By default, each item describes a different dimension.

    """

    # Hidden attributes containing the collection of Coord instance
    _coords = List(Instance(Coord), allow_none=True)

    # Hidden name of the object
    _name = Unicode

    @default('_name')
    def _get_name_default(self):
        return u"CoordSet_" + str(uuid.uuid1()).split('-')[0]  # a unique id

    # Hidden attribute to specify if the collection is for a single dimension
    _issamedim = Bool

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------
    def __init__(self, *coords, **kwargs):

        _iscopy = kwargs.pop('iscopy', False)

        super(CoordSet, self).__init__(**kwargs)

        self._coords = []

        if all([isinstance(coords[i], (Coord, CoordSet)) for i in range(len(coords))]):
            coords = list(coords)
        elif len(coords) == 1:
            # this a set of CoordsSet or Coord passed as a list
            coords = coords[0]
        else:
            # not implemented yet -
            # a list of list of object have been passed
            # TODO: try to ipmplement this
            raise CoordsError(
                    'a list of list of object have been passed - this not yet implemented')

        if len(coords) == 1 and isinstance(coords[0], CoordSet):
            if _iscopy:
                coords = copy.deepcopy(coords)
            self._coords = coords[0]._coords

        else:
            for item in coords:

                if not isinstance(item, (Coord, CoordSet)):
                    item = Coord(item, iscopy=_iscopy)
                    # full validation of the item
                    # will be done in Coord
                if self._validation(item):
                    self._coords.append(item)

        # check if we have single dimension axis

        for item in self._coords:
            if isinstance(item, CoordSet):
                # it must be a single dimension axis
                item._issamedim = True
                # in this case we must have same length coords
                siz = item[0].size
                if np.any([elt.size != siz for elt in item._coords]):
                    raise CoordsError('axis must be of the same size '
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
        if len(self._coords) < 1:
            return []
        try:
            return [item.name for item in self._coords]
        except:
            log.critical(self._coords)

    @property
    def titles(self):
        """`list` - Get/Set a list of axis titles.

        """
        _titles = []
        for item in self._coords:
            if isinstance(item, Coord):
                _titles.append(item.title if item.title else item.name)
            elif isinstance(item, CoordSet):
                _titles.append([el.title if el.title else el.name
                                for el in item])
            else:
                raise CoordsError('Something wrong with the titles!')

        return _titles

    @titles.setter
    def titles(self, value):
        # Set the titles at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].title = item

    @property
    def labels(self):
        """`list` - Get/Set a list of axis labels.

        """
        return [item.label for item in self._coords]

    @labels.setter
    def labels(self, value):
        # Set the labels at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].label = item

    @property
    def units(self):
        """`list` - Get/Set a list of axis units.

        """
        return [item.units for item in self._coords]

    @units.setter
    def units(self, value):
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].units = item

    @property
    def isempty(self):
        """`bool`, read-only property - `True` if there is no coords defined.

        """
        return len(self._coords) == 0

    @property
    def issamedim(self):
        """`bool`, read-only property -
        `True` if the coords define a single dimension.

        """
        return self._issamedim

    @property
    def sizes(self):
        """`int`, read-only property -
        gives the size of the axis or coords for each dimention"""
        _sizes = []
        for i, item in enumerate(self._coords):
            if isinstance(item, Coord):
                _sizes.append(item.size)
            elif isinstance(item, CoordSet):
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
        """Make a disconnected copy of the current coords.

        Returns
        -------
        coords : same type
            an exact copy of the current object

        """
        return self.__copy__()

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------
    def _transpose(self, coords=None):
        # in principle it is not directly called by the user as it is intimately
        # linked to a dataset
        if self._issamedim:
            # not applicable for same dimension coords
            warnings.warn('CoordSet for a single dimentsion are not transposable',
                          CoordsWarning)
            return
        if coords is None:
            self._coords.reverse()
        else:
            self._coords = [self._coords[axis] for axis in coords]

    def _validation(self, item):
        # To be valid any added axis must have a different name

        if not isinstance(item, (Coord, CoordSet)):
            raise CoordsError('The elements of must be Coord or '
                            'CoordSet objects only!')

        if item._name in self.names:
            raise CoordsError('The axis name must be unique!')

        if isinstance(item, Coord) and item.ndim > 1:
            raise CoordsError('An axis should be a 1D array!')

        # TODO: add more validation for CoordSet objects

        return True

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    @staticmethod
    def __dir__():
        return ['_coords']

    def __call__(self, *args):
        # allow the following syntax: coords(0,2), or coords(axis=(0,2))
        coords = []
        if args:
            for idx in args:
                coords.append(self._coords[idx])
        if len(coords) == 1:
            return coords[0]
        else:
            return CoordSet(coords)

    def __len__(self):
        return len(self._coords)

    def __getitem__(self, index):

        if isinstance(index, string_types):
            if index in self.titles:
                # selection by axis title
                return self._coords.__getitem__(self.titles.index(index))
            # may be it is in a multiple axis
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subaxis title
                    return item.__getitem__(item.titles.index(index))

        res = self._coords.__getitem__(index)
        if isinstance(index, slice):
            return CoordSet(res)
        else:
            return res

    def __setitem__(self, index, coords):
        self._coords[index] = coords

    def __iter__(self):
        for item in self._coords:
            yield item

    def __repr__(self):
        out = ("CoordSet object <" + ', '.join(['<Coord object {}>']
                                           * len(self._coords)) + ">")
        out = out.format(*self.names)
        return out

    def __str__(self):
        out = "(" + ', '.join(['[{}]'] * len(self._coords)) + ")"
        out = out.format(*self.titles)
        return out

    def __deepcopy__(self, memo):
        return self.__class__([copy.deepcopy(ax, memo=memo) for ax in self])

    def __copy__(self):
        return self.__class__([copy.copy(ax) for ax in self])

    def __eq__(self, other):
        return self._coords == other._coords  # TODO: check the case of compatible units

    def __ne__(self, other):
        return not self.__eq__(other)


# =============================================================================
# CoordsRange
# =============================================================================

class CoordsRange(HasTraits):
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
        """ Constructs Coordsrange with default values

        """
        super(CoordsRange, self).__init__(**kwargs)

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

    @staticmethod
    def _cleanranges(ranges):
        """ sort and merge overlaping ranges

        works as follows:
         1. orders each interval
         2. sorts intervals
         3. merge overlapping intervals
         4. reverse the orders if required

        """

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

set_operators(Coord, priority=50)

# =============================================================================
# Modify the doc to include Traits
# =============================================================================
# create_traitsdoc(CoordSet)
# create_traitsdoc(Coord)

if __name__ == '__main__':
    pass
