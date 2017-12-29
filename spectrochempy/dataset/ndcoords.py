# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
This module implements three classes |Coord|, |CoordSet| and |CoordRange|.


"""

__all__ = ['Coord', 'CoordSet', 'CoordRange']


# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------

import copy
import uuid
import warnings
from datetime import datetime

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np
from traitlets import (HasTraits, List, Bool, Unicode, default, Instance)

# ----------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------

from .ndarray import NDArray
from ..application import log
from .ndmath import NDMath
from ..utils import (set_operators, is_number, is_sequence,
                    numpyprintoptions, docstrings, SpectroChemPyWarning)
from ..utils.traittypes import Range

# ============================================================================
# set numpy print options
# ============================================================================

numpyprintoptions()


# ============================================================================
# Coord
# ============================================================================

class Coord(NDMath, NDArray):
    """Coordinates for a dataset along a given axis.

    The coordinates of a |NDDataset| can be created using the |Coord| object.
    This is a single dimension array with either numerical (float) values or
    labels (str, `Datetime` objects, or any other kind of objects) to
    represent the coordinates. Only a one numerical axis can be defined,
    but labels can be multiple.

    """

    _copy = Bool

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------
    docstrings.delete_params('NDArray.parameters', 'data', 'mask',
                             'uncertainty')

    @docstrings.dedent
    def __init__(self, data=None, **kwargs):
        """
        Parameters
        -----------
        data : array of floats
            The actual data array contained in the |Coord| object.
            The given array (with a single dimension) can be a list,
            a tuple, a |ndarray|, or a |ndarray|-like object.
            If an object is passed that contains labels, or units,
            these elements will be used to accordingly set those of the
            created object.
            If possible, the provided data will not be copied for `data` input,
            but will be passed by reference, so you should make a copy the
            `data` before passing it in the object constructor if that's the
            desired behavior or set the `copy` argument to True.
        %(NDArray.parameters.no_data|mask|uncertainty)s


        Examples
        --------
        We first import the object from the main scp:

        >>> from spectrochempy.scp import Coord # doctest: +ELLIPSIS
        SpectroChemPy's scp - v.0.1...

        We then create a numpy |ndarray| and use it as the numerical `data`
        axis of our new |Coord| object.

        >>> arr = np.arange(1,12,2)
        >>> c0 = Coord(data=arr, title='frequency', units='Hz')
        >>> c0     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Coord: [       1,        3,        5,        7,        9,       11] Hz

        We can take a series of str to create a non numerical but labelled
        axis:

        >>> tarr = list('abcdef')
        >>> tarr
        ['a', 'b', 'c', 'd', 'e', 'f']
        >>> c1 = Coord(labels=tarr, title='mylabels')
        >>> c1   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Coord: [a, b, c, d, e, f]
        >>> print(c1) # doctest: +NORMALIZE_WHITESPACE
        title: Mylabels
        labels: [a b c d e f]

        Some other examples will found in the :ref:`userguide`.

        """
        super(Coord, self).__init__(data, **kwargs)

        # some checking
        if self.ndim > 1:
            raise ValueError("Number of dimension for coordinate's array "
                             "should be 1!")

    def implements(self, name=None):
        # For compatibility with pyqtgraph
        # Rather than isinstance(obj, NDDataset) use object.implements(
        # 'NDDataset')
        # This is useful to check type without importing the module
        if name is None:
            return ['Coord']
        else:
            return name == 'Coord'

    # ------------------------------------------------------------------------
    # readonly property
    # ------------------------------------------------------------------------

    @property
    def is_reversed(self):
        """bool - Whether the axis is ascending or reversed (readonly
        property).

        Return a correct result only if the data are sorted

        """
        if "wavenumber" in self.title.lower() or "ppm" in self.title.lower():
            return True
        return False

        return bool(self.data[0] > self.data[-1])

    # ------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docs)
    # some of the property of NDArray has to be hidden because they are not
    # useful for this Coord class
    # ------------------------------------------------------------------------

    @property
    def is_complex(self):
        return [False, False]  # always real (the first dimension is of size 1)

    @property
    def ndim(self):
        return 1

    @property
    def uncertainty(self):
        return np.zeros_like(self._data, dtype=float)

    @uncertainty.setter
    def uncertainty(self, uncertainty):
        pass

    @property
    def T(self):  # no transpose
        return self

    @property
    def shape(self):
        return self._data.shape

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------

    # ........................................................................
    def get_labels(self, level=0):
        """Get the labels at a given level

        Used to replace `data` when only labels are provided, and/or for
        labeling axis in plots

        Parameters
        ----------
        level : int, optional, default:0

        Returns
        -------
        |ndarray|
            The labels at the desired level or None

        """
        if not self.is_labeled:
            return None

        if level > self._labels.ndim-1:
            warnings.warn("There is no such level in the existing labels",
                          SpectroChemPyWarning)
            return None

        if self._labels.ndim > 1:
            return self._labels[level]
        else:
            return self._labels

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    def _loc2index(self, loc, axis=None):
        # Return the index of a location (label or coordinates) along the axis
        # axis is not use here as wa are already ib a coord axis.

        # underlying axis array and labels
        data = self._data
        labels = self._labels

        if isinstance(loc, str) and labels is not None:
            # it's probably a label
            indexes = np.argwhere(labels == loc).flatten()
            if indexes.size > 0:
                return indexes[0]
            else:
                raise IndexError('Could not find this label: {}'.format(loc))

        elif isinstance(loc, datetime):
            # not implemented yet
            return None  # TODO: date!

        elif is_number(loc):
            # get the index of this coordinate
            if loc > data.max() or loc < data.min():
                warnings.warn('This coordinate ({}) is outside the axis limits '
                              '({}-{}).\nThe closest limit index is '
                              'returned'.format(loc, data.min(), data.max()),
                              SpectroChemPyWarning)
            index = (np.abs(data - loc)).argmin()
            return index

        else:
            raise IndexError('Could not find this location: {}'.format(loc))

    def _repr_html_(self):

        tr = "<tr style='border: 1px solid lightgray;'>" \
             "<td style='padding-right:5px; width:100px'><strong>{" \
             "}</strong></td>" \
             "<td style='text-align:left'>{}</td><tr>\n"

        out = "<table style='width:100%'>\n"
        out += tr.format("Title", self.title.capitalize())
        if self.data is not None:
            data_str = super(Coord, self)._repr_html_()
            out += tr.format("Data", data_str)

        if self.is_labeled:
            out += tr.format("Labels", self.labels)

        out += '</table>\n'
        return out

    # TODO: _repr_latex

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __dir__(self):
        # with remove some methods with respect to the full NDArray
        # as they are not usefull for Coord.
        return ['data', 'mask', 'labels', 'units', 'meta', 'name', 'title']

    def __str__(self):
        out = '      title: %s\n' % (self.title.capitalize())
        if not self.is_empty:
            data_str = super(Coord, self).__str__()
            out += '       data: %s\n' % data_str
        elif not self.is_labeled:
            out = 'None'

        if self.is_labeled:
            out += '     labels: %s\n' % str(self.labels)
        if out[-1] == '\n':
            out = out[:-1]
        return out


# ============================================================================
# CoordSet
# ============================================================================

class CoordSet(HasTraits):
    """
    A collection of Coord objects for a NDArray object with a validation
    method.

    """

    # Hidden attributes containing the collection of objects
    _coords = List(Instance(NDArray), allow_none=True)

    # Hidden name of the object
    _name = Unicode()

    # .........................................................................
    @default('_name')
    def _get_name_default(self):
        return "CoordSet_" + str(uuid.uuid1()).split('-')[0]  # a unique id

    # Hidden attribute to specify if the collection is for a single dimension
    _is_same_dim = Bool

    # -------------------------------------------------------------------------
    # initialization
    # -------------------------------------------------------------------------

    # .........................................................................
    def __init__(self, *coords, **kwargs):
        """
        Parameters
        ----------
        coords : array-like, |NDarray|, |NDArray| subclass or |CoordSet| objects
           If an instance of CoordSet is found, instead of an array, this means
           that all coordinates in this coordset describe the same axis.
        is_same_dim : bool, optional, default:False
            if true, all elements of coords describes a single dimension.
            By default, this is false, which means that each item describes
            a different dimension.


        """
        _copy = kwargs.pop('copy', False)

        super(CoordSet, self).__init__(**kwargs)

        self._coords = []

        if all([isinstance(coords[i], (NDArray, CoordSet)) for i in
                range(len(coords))]):
            # Any instance of a NDArray can be accepted as coordinates for a
            # dimension.
            # If an instance of CoordSet is found, this means that all
            # coordinates in this set describe the same axis

            coords = list(coords)

        elif len(coords) == 1:
            # this a set of CoordSet or NDArray passed as a list
            coords = coords[0]

        else:
            # not implemented yet -
            # a list of list of object have been passed
            # TODO: try to ipmplement this
            raise NotImplementedError(
                    'a list of list of object have been passed - '
                    'this not yet implemented')

        if len(coords) == 1 and isinstance(coords[0], CoordSet):
            if _copy:
                coords = copy.deepcopy(coords)
            self._coords = coords[0]._coords

        else:
            for item in coords:

                if item is not None and not isinstance(item, (NDArray, \
                        CoordSet)):
                    item = NDArray(item, copy=_copy)
                    # full validation of the item
                    # will be done in NDArray.__init__

                if self._validation(item):
                    self._coords.append(item)

        # check if we have single dimension axis

        for item in self._coords:
            if isinstance(item, CoordSet):
                # it must be a single dimension axis
                item._is_same_dim = True
                # in this case we must have same length for all coordinates
                siz = item[0].size
                if np.any([elt.size != siz for elt in item._coords]):
                    raise ValueError('Coordinates must be of the same size '
                                     'for a dimension with multiple '
                                     'coordinates')

    def implements(self, name=None):
        # For compatibility with pyqtgraph
        # Rather than isinstance(obj, NDDataset) use
        # object.implements(
        # 'NDDataset')
        # This is useful to check type without importing the module
        if name is None:
            return ['CoordSet']
        else:
            return name == 'CoordSet'

    # ------------------------------------------------------------------------
    # Readonly Properties
    # ------------------------------------------------------------------------

    # .........................................................................
    @property
    def names(self):
        """list - Names of the coords in the current coordset (readonly
        property).

        """
        if len(self._coords) < 1:
            return []
        try:
            return [item.name for item in self._coords]
        except:
            log.critical(self._coords)

    # .........................................................................
    @property
    def isempty(self):
        """bool - True if there is no coords defined (readonly
        property).

        """
        return len(self._coords) == 0

    # .........................................................................
    @property
    def is_same_dim(self):
        """bool - True if the coords define a single dimension (readonly
        property).

        """
        return self._is_same_dim

    # .........................................................................
    @property
    def sizes(self):
        """int - Size of the coord object for each dimention (readonly
        property).

        """
        _sizes = []
        for i, item in enumerate(self._coords):
            if isinstance(item, NDArray):
                _sizes.append(item.size)
            elif isinstance(item, CoordSet):
                _sizes.append(item.sizes[i][0])
        return _sizes

    # .........................................................................
    @property
    def coords(self):
        """list - list of the Coord objects in the current coordset (readonly
        property).

        """
        return self._coords

    # ------------------------------------------------------------------------
    # Readonly Properties
    # ------------------------------------------------------------------------

    # .........................................................................
    @property
    def name(self):
        """str - Name of the coordset

        """
        return self._name

    # .........................................................................
    @name.setter
    def name(self, value):
        self._name = value

    # .........................................................................
    @property
    def titles(self):
        """list - Titles of the coords in the current coordset

        """
        _titles = []
        for item in self._coords:
            if isinstance(item, NDArray):
                _titles.append(item.title if item.title else item.name)
            elif isinstance(item, CoordSet):
                _titles.append(
                        [el.title if el.title else el.name for el in item])
            else:
                raise ValueError('Something wrong with the titles!')

        return _titles

    # .........................................................................
    @titles.setter
    def titles(self, value):
        # Set the titles at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].title = item

    # .........................................................................
    @property
    def labels(self):
        """list - Labels of the coords in the current coordset

        """
        return [item.label for item in self._coords]

    # .........................................................................
    @labels.setter
    def labels(self, value):
        # Set the labels at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].label = item

    # .........................................................................
    @property
    def units(self):
        """list - Units of the coords in the current coordset

        """
        return [item.units for item in self._coords]

    # .........................................................................
    @units.setter
    def units(self, value):
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].units = item

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------

    # .........................................................................
    def copy(self):
        """Make a disconnected copy of the current coordset.

        Returns
        -------
        object
            an exact copy of the current object

        """
        return self.__copy__()

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    # .........................................................................
    def _transpose(self, coords=None):
        # in principle it is not directly called by the user as it is intimately
        # linked to a dataset
        if self._is_same_dim:
            # not applicable for same dimension coords
            warnings.warn(
                    'CoordSet for a single dimension are not transposable',
                    SpectroChemPyWarning)
            return
        if coords is None:
            self._coords.reverse()
        else:
            self._coords = [self._coords[axis] for axis in coords]

    # .........................................................................
    def _validation(self, item):
        # To be valid any added axis must have a different name

        if item is None:
            return True

        if not isinstance(item, (NDArray, CoordSet)):
            raise ValueError('The elements of must be NDArray or '
                             'CoordSet objects only!')

        if item._name in self.names:
            raise ValueError('The Coord name must be unique!')

        if isinstance(item, NDArray) and item.ndim > 1:
            raise ValueError('A Coord should be a 1D array!')

        # TODO: add more validation for CoordSet objects

        return True

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    # .........................................................................
    @staticmethod
    def __dir__():
        return ['_coords']

    # .........................................................................
    def __call__(self, *args, **kwargs):
        # allow the following syntax:
        #              coordset(), coordset(0,2) or coordset(axis=(0,2))
        coords = []
        axis = kwargs.get('axis', None)

        if args:
            for idx in args:
                coords.append(self[idx])
        elif axis is not None:
            if not is_sequence(axis):
                axis = [axis]
            for i in axis:
                coords.append(self[i])
        else:
            coords = self._coords

        if len(coords) == 1:
            return coords[0]
        else:
            return CoordSet(coords)

    # .........................................................................
    def __len__(self):
        return len(self._coords)

    # .........................................................................
    def __getitem__(self, index):
        if isinstance(index, str):
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

    # .........................................................................
    def __setitem__(self, index, coords):
        self._coords[index] = coords

    # .........................................................................
    def __iter__(self):
        for item in self._coords:
            yield item

    # .........................................................................
    def __repr__(self):
        out = ("CoordSet object <" + ', '.join(
            ['<object {}>'] * len(self._coords)) + ">")
        out = out.format(*self.names)
        return out

    # .........................................................................
    def __str__(self):
        out = "[" + ', '.join(['{}'] * len(self._coords)) + "]"
        s = []
        for item in self._coords:
            if isinstance(item, CoordSet):
                s.append(item.__str__())
            else:
                s.append(item.title)
        out = out.format(*s)
        return out

    # .........................................................................
    def __deepcopy__(self, memo):
        return self.__class__([copy.deepcopy(ax, memo=memo) for ax in self])

    # .........................................................................
    def __copy__(self):
        return self.__class__([copy.copy(ax) for ax in self])

    # .........................................................................
    def __eq__(self, other):
        return self._coords == other._coords
        # TODO: check the case of compatible units

    # .........................................................................
    def __ne__(self, other):
        return not self.__eq__(other)


# =============================================================================
# CoordRange
# =============================================================================

class CoordRange(HasTraits):
    """Set of ordered, non intersecting intervals

    e.g. [[a, b], [c, d]] with a < b < c < d or a > b > c > d.

    """
    # TODO: May use also units ???

    ranges = List(Range)
    reversed = Bool

    def __init__(self, *ranges, **kwargs):
        """
        Parameters
        -----------
        ranges :  iterable
            An interval or a set of intervals.
            set of  intervals. If none is given, the range
            will be a set of an empty interval [[]]. The interval limits do not
            need to be ordered, and the intervals do not need to be distincts.
        reversed : bool, optional.
            The intervals are ranked by decreasing order if True
            or increasing order if False.
        nranges :  int
            Number of distinct ranges


        """
        super(CoordRange, self).__init__(**kwargs)

        self.reversed = kwargs.get('reversed', False)

        if len(ranges) == 0:
            # first case: no argument passed, returns an empty range

            self.ranges = []

        elif len(ranges) == 2 and all(
                isinstance(elt, (int, float)) for elt in ranges):
            # second case: a pair of scalars has been passed
            # using the Interval class, we have autochecking of the interval
            # validity

            self.ranges = [list(map(float, ranges))]

        else:
            # third case: a set of pairs of scalars has been passed

            self.ranges = self._cleanranges(ranges)

        if self.reversed:
            self.reverse()

        if self.ranges:
            self.ranges = self._cleanranges(self.ranges)

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __iter__(self):
        for x in self.ranges:
            yield x

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def nranges(self):
        """int - Number of interval in the current |CoordRange| (readonly
        property).

        """
        return len(self.ranges)

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------

    def reverse(self):
        """Reverse the order of the range

        """
        for range in self.ranges:
            range.reverse()
        self.ranges.reverse()

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    @staticmethod
    def _cleanranges(ranges):
        """Sort and merge overlscpng ranges

        It works as follows::

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


# ============================================================================
# Set the operators
# ============================================================================

set_operators(Coord, priority=50)

# ============================================================================
if __name__ == '__main__':

    pass