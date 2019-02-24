# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implements three classes |Coord|, |CoordSet| and |CoordRange|.
"""

__all__ = ['Coord', 'CoordSet', 'CoordRange']

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------
import copy
import warnings
from datetime import datetime
import textwrap
import uuid
import re

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from traitlets import (HasTraits, List, Bool, Unicode, observe, All, validate,
                       TraitType, TraitError, class_of, default, Instance)

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------
from .ndarray import NDArray, DEFAULT_DIM_NAME
from .ndmath import NDMath
from spectrochempy.core import log
from spectrochempy.utils import (set_operators, is_number, is_sequence,
                                 docstrings, SpectroChemPyWarning,
                                 colored_output, NOMASK)


# ======================================================================================================================
# Coord
# ======================================================================================================================
class Coord(NDMath, NDArray):
    """Coordinates for a dataset along a given axis.

    The coordinates of a |NDDataset| can be created using the |Coord| object.
    This is a single dimension array with either numerical (float) values or
    labels (str, `Datetime` objects, or any other kind of objects) to
    represent the coordinates. Only a one numerical axis can be defined,
    but labels can be multiple.

    """
    _copy = Bool

    _html_output = False

    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------
    docstrings.delete_params('NDArray.parameters', 'data', 'mask')

    # ..................................................................................................................
    @docstrings.dedent
    def __init__(self, data=None, **kwargs):
        """
        Parameters
        -----------
        data : ndarray, tuple or list
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
        %(NDArray.parameters.no_data|mask)s
        Examples
        --------
        We first import the object from the api:
        >>> from spectrochempy import *
        We then create a numpy |ndarray| and use it as the numerical `data`
        axis of our new |Coord| object.
        >>> arr = np.arange(1.,12.,2.)
        >>> c0 = Coord(data=arr, title='frequency', units='Hz')
        >>> c0     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Coord: [   1.000,    3.000,    5.000,    7.000,    9.000,   11.000] Hz
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

        assert self.ndim <= 1

    # ..................................................................................................................
    def implements(self, name=None):
        # Rather than isinstance(obj, NDDataset) use object.implements(
        # 'NDDataset')
        # This is useful to check type without importing the module
        if name is None:
            return ['Coord']
        else:
            return name == 'Coord'

    # ------------------------------------------------------------------------------------------------------------------
    # readonly property
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    @property
    def reversed(self):
        """bool - Whether the axis is reversed (readonly
        property).
        """
        if "wavenumber" in self.title.lower() or "ppm" in self.title.lower():
            return True
        return False
        ## Return a correct result only if the data are sorted
        # return bool(self.data[0] > self.data[-1])

    # ------------------------------------------------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docstring)
    # some of the property of NDArray has to be hidden because they
    # are not useful for this Coord class
    # ------------------------------------------------------------------------------------------------------------------

    # NDarray methods

    # ..................................................................................................................
    @property
    def is_complex(self):
        return False  # always real

    # ..................................................................................................................
    @property
    def ndim(self):
        ndim = super().ndim
        if ndim > 1:
            raise ValueError("Coordinate's array should be 1-dimensional!")
        return 1

    # ..................................................................................................................
    @property
    def T(self):  # no transpose
        return self

    # ..................................................................................................................
    @property
    def date(self):
        return None

    # ..................................................................................................................
    #@property
    #def values(self):
    #    return super().values

    # ..................................................................................................................
    @property
    def masked_data(self):
        return super().masked_data

    # ..................................................................................................................
    @property
    def is_masked(self):
        return False

    # ..................................................................................................................
    @property
    def mask(self):
        return super().mask

    # ..................................................................................................................
    @mask.setter
    def mask(self, val):
        # Coordinates cannot be masked. Set mask always to NOMASK
        self._mask = NOMASK

    # NDmath methods

    # ..................................................................................................................
    def cumsum(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def mean(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def pipe(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def remove_masks(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def std(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def sum(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def swapaxes(self, **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def __copy__(self):
        res = self.copy(deep=False)  # we keep name of the coordinate by default
        res.name = self.name
        return res

    # ..................................................................................................................
    def __deepcopy__(self, memo=None):
        res = self.copy(deep=True, memo=memo)
        res.name = self.name
        return res

    # ..................................................................................................................
    def __dir__(self):
        # remove some methods with respect to the full NDArray
        # as they are not usefull for Coord.
        # dtype must stay first item
        return ['data', 'labels', 'units', 'meta', 'title', 'name']

    # ..................................................................................................................
    def __getitem__(self, items, return_index=False):
        # we need to keep the names when copying coordinates to avoid later problems
        res = super().__getitem__(items, return_index=return_index)
        res.name = self.name
        return res

    # ..................................................................................................................
    def __str__(self):
        return repr(self)

    def _cstr(self, header='  coordinates: ... \n', print_size=True):

        out = ''
        if not self.is_empty and print_size:
            out += f'{self._str_shape().rstrip()}\n'
        out += f'        title: {self.title}\n' if self.title else ''
        if self.has_data:
            out += '{}\n'.format(self._str_value(header=header))
        elif self.is_empty and not self.is_labeled:
            out += header.replace('...', '\0Undefined\0')

        if self.is_labeled:
            header = '       labels: ... \n'
            text = str(self.labels).strip()
            if '\n' not in text:  # single line!
                out += header.replace('...', '\0\0{}\0\0'.format(text))
            else:
                out += header
                out += '\0\0{}\0\0'.format(textwrap.indent(text.strip(), ' ' * 9))

        if out[-1] == '\n':
            out = out[:-1]

        if not self._html_output:
            return colored_output(out)
        else:
            return out

    # ..................................................................................................................
    def __repr__(self):
        out = self._repr_value().rstrip()
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    @observe(All)
    def _anytrait_changed(self, change):
        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }
        log.debug(f'changes in Coord: {change.name}')


# ======================================================================================================================
# CoordSet
# ======================================================================================================================
class CoordSet(HasTraits):
    """
    A collection of Coord objects for a NDArray object with a validation
    method.
    """
    # Hidden attributes containing the collection of objects
    _id = Unicode()
    _coords = List(allow_none=True)
    _parent = Instance('spectrochempy.core.dataset.nddataset.NDDataset', allows_none=True)

    _updated = Bool(False)
    # Hidden id and name of the object
    _id = Unicode()
    _name = Unicode()

    # Hidden attribute to specify if the collection is for a single dimension
    _is_same_dim = Bool(False)
    _copy = Bool(False)

    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def __init__(self, *coords, **kwargs):
        """
        Parameters
        ----------
        coords : |NDarray|, |NDArray| subclass or |CoordSet| sequence of objects.
            If an instance of CoordSet is found, instead of an array, this means
            that all coordinates in this coords describe the same axis.
            It is assumed that the coordinates are passed in the order of the
            dimensions of a nD numpy array (
            `row-major<https://docs.scipy.org/doc/numpy-1.14.1/glossary.html#term-row-major>`_
            order), i.e., for a 3d object: 'z', 'y', 'x'.
        x : |NDarray|, |NDArray| subclass or |CoordSet|
            A single coordinate associated to the 'x'-dimension.
            If a coord was already passed in the argument, this will overwrite
            the previous. It is thus not recommended to simultaneously use
            both way to initialize the coordinates to avoid such conflicts.
        y, z, u, ...: |NDarray|, |NDArray| subclass or |CoordSet|
            Same as `x` for the others dimensions.
        is_same_dim : bool, optional, default:False
            if true, all elements of coords describes a single dimension.
            By default, this is false, which means that each item describes
            a different dimension.
        """

        self._copy = kwargs.pop('copy', False)

        # initialise the coordinate list
        # max number of dimensions in a coordset = len(DEFAULT_DIM_NAME)
        _coords = [None] * len(DEFAULT_DIM_NAME)

        # First evaluate passed args
        if coords:
            # we consider that they are passed in the same order as the array dimensions (x, y, z ...)
            if all([(isinstance(coords[i], (np.ndarray, NDArray, CoordSet)) or coords[i] is None)
                    for i in range(len(coords))]):
                # Any instance of a NDArray can be accepted as coordinates for a
                # dimension.
                # If an instance of CoordSet is found, this means that all
                # coordinates in this set describe the same axis
                coords = list(coords)

            elif is_sequence(coords) and len(coords) == 1:
                coords = coords[0]
                if isinstance(coords, dict):
                    # we have a dict
                    kwargs.update(coords)
                    coords = None

            else:
                raise ValueError('Did not understand the inputs')

            if coords:

                if len(coords) == 1 and isinstance(coords[0], CoordSet):
                    if self._copy:
                        coords = copy.deepcopy(coords)
                    _coords = coords[0]._coords
                else:
                    for idx, coord in enumerate(coords[::-1]):  # we fill from the end of the list (in reverse order)
                        id = len(_coords) - 1 - idx
                        if not isinstance(coord, CoordSet):
                            coord = Coord(coord)  # be sure to cast to the correct type
                        # set the name from the position except if we copy a coordset
                        keepnames = kwargs.get('keepnames', False)
                        if not keepnames:
                            coord.name = DEFAULT_DIM_NAME[id]  # get the name in fonction of the position
                        _coords[id] = coord

        # now evaluate keywords argument
        for key, coord in kwargs.items():
            # prepare values to Coord or CoordSet
            if isinstance(coord, (list, tuple)):  # make sure it's a coordset
                coord = CoordSet(coord)
            elif isinstance(coord, np.ndarray) or coord is None:
                coord = Coord(coord)

            # populate the coords with coord and coord's name.
            if isinstance(coord, (NDArray, Coord, CoordSet)):
                if key in DEFAULT_DIM_NAME:
                    # ok we can find it as a canonical name:
                    # this will overwrite any already defined value
                    coord.name = key
                    _coords[DEFAULT_DIM_NAME.index(key)] = coord

        # store the item (validation will be performed)
        self._coords = _coords

        # inform the parent about the update
        self._updated = True

        # set a notifier on the name traits name of each coordinates
        for coord in self._coords:
            if coord is not None:
                HasTraits.observe(coord, self._coords_update, '_name')

        # initialize the base class with the eventual remaining arguments
        super(CoordSet, self).__init__(**kwargs)

    # ..................................................................................................................
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

    # ------------------------------------------------------------------------------------------------------------------
    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    @validate('_coords')
    def _coords_validate(self, proposal):
        coords = proposal['value']
        for id, coord in enumerate(coords):
            if coord and not isinstance(coord, (Coord, CoordSet)):
                coord = Coord(coord)
                # full validation of the item
                # will be done in Coord.__init__
            coords[id] = coord

        # reduce to the minimal number of coordinates
        id = 0
        # we start from the left side of the list!
        while coords[id] is None:
            id += 1
        coords = coords[id:]

        for coord in coords:
            if isinstance(coord, CoordSet):
                # it must be a single dimension axis
                # in this case we must have same length for all coordinates
                coord._is_same_dim = True
                # change the internal names
                coord.names = [f"_{i}" for i in range(len(coord))]

        return coords

    # ..................................................................................................................
    @default('_id')
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    # ------------------------------------------------------------------------------------------------------------------
    # Readonly Properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def id(self):
        """
        str - Object identifier (Readonly property).
        """
        return self._id

    # ..................................................................................................................
    @property
    def is_empty(self):
        """bool - True if there is no coords defined (readonly
        property).
        """
        return len(self._coords) == 0

    # ..................................................................................................................
    @property
    def is_same_dim(self):
        """bool - True if the coords define a single dimension (readonly
        property).
        """
        return self._is_same_dim

    # ..................................................................................................................
    @property
    def sizes(self):
        """int or tuple of int - Sizes of the coord object for each dimension
        (readonly property). If the set is for a single dimension return a
        single size as all coordinates must have the same.
        """
        _sizes = []
        for i, item in enumerate(self._coords):
            if isinstance(item, NDArray):
                _sizes.append(item.size)
            elif isinstance(item, CoordSet):
                _sizes.append(item.sizes[i][0])

        if self.is_same_dim:
            _sizes = list(set(_sizes))
            if len(_sizes) > 1:
                raise ValueError('Coordinates must be of the same size '
                                 'for a dimension with multiple '
                                 'coordinates')
            return _sizes[0]
        return _sizes

    # alias
    size = sizes

    # ..................................................................................................................
    @property
    def coords(self):
        """list - list of the Coord objects in the current coords (readonly
        property).
        """
        return self._coords

    # ------------------------------------------------------------------------------------------------------------------
    # Mutable Properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return "<unnamed>"

    @name.setter
    def name(self, value):
        self._name = value

    # ..................................................................................................................
    @property
    def names(self):
        """list - Names of the coords in the current coords
        """
        _names = []
        for item in self._coords:
            _names.append(item.name)
        return _names

    # ..................................................................................................................
    @names.setter
    def names(self, value):
        # Set the names at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].name = item

    # ..................................................................................................................
    @property
    def titles(self):
        """list - Titles of the coords in the current coords
        """
        _titles = []
        for item in self._coords:
            if isinstance(item, NDArray):
                _titles.append(item.title if item.title else item.name)  # TODO:name
            elif isinstance(item, CoordSet):
                _titles.append(
                    [el.title if el.title else el.name for el in item])  # TODO:name
            else:
                raise ValueError('Something wrong with the titles!')
        return _titles

    # ..................................................................................................................
    @titles.setter
    def titles(self, value):
        # Set the titles at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].title = item

    # ..................................................................................................................
    @property
    def labels(self):
        """list - Labels of the coords in the current coords
        """
        return [item.label for item in self._coords]

    # ..................................................................................................................
    @labels.setter
    def labels(self, value):
        # Set the labels at once
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].label = item

    # ..................................................................................................................
    @property
    def units(self):
        """
        list - Units of the coords in the current coords

        """
        return [item.units for item in self._coords]

    # ..................................................................................................................
    @units.setter
    def units(self, value):
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].units = item

    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def copy(self):
        """
        Make a disconnected copy of the current coords.

        Returns
        -------
        object
            an exact copy of the current object

        """
        return self.__copy__()

    # ..................................................................................................................
    def to_index(self):
        """Convert all index coordinates into a `pandas.Index`
        Returns
        -------
        pandas.Index
            Index subclass corresponding to the outer-product of all dimension
            coordinates. This will be a MultiIndex if this object is has more
            than more dimension.
        """
        import pandas as pd
        if len(self) == 0:
            raise ValueError('no valid index for a 0-dimensional object')
        elif len(self) == 1:
            coord = self.coords[0]
            return pd.Index(coord.values, name=coord.title)  # TODO: keep the units
        else:
            return pd.MultiIndex.from_product(self.coords, names=self.titles)

    def update(self, **kwargs):
        """
        Update a specific coordinates in the CoordSet.

        Parameters
        ----------
        \*\*kwarg : Only keywords among the CoordSet.names are allowed - they denotes the name of a dimension.

        """
        dims = kwargs.keys()
        for dim in list(dims)[:]:
            if dim in self.names:
                # we can replace the given coordinates
                idx = self.names.index(dim)
                self.coords[idx] = Coord(kwargs.pop(dim), name=dim)

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    @staticmethod
    def __dir__():
        return ['_coords', 'is_same_dim', 'name']

    # ..................................................................................................................
    def __call__(self, *args, **kwargs):
        # allow the following syntax:
        #              coords(), coords(0,2) or coords(axis=(0,2))
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

    # ..................................................................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return hash(tuple(self._coords))

    # ..................................................................................................................
    def __len__(self):
        return len(self._coords)

    # ..................................................................................................................
    def __getattr__(self, item):
        # when the attribute was not found
        if '_validate' in item or '_changed' in item:
            raise AttributeError

        elif item in self.names:  # syntax such as ds.x, ds.y, etc...
            idx = self.names.index(item)
            return self._coords[idx]

    # ..................................................................................................................
    def __getitem__(self, index):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                return self._coords.__getitem__(idx)

            # ok we did not find it!
            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                if self.titles.count(index) > 1:
                    warnings.warn(f"Getting a coordinate from its title. However `{index}` occurs several time. Only"
                                  f" the first occurence is returned!")
                return self._coords.__getitem__(self.titles.index(index))

            # may be it is a title in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    return item.__getitem__(item.titles.index(index))  # TODO: check this

            try:
                # let try with the canonical dimension names defined by  DEFAULT_DIM_NAME
                if index[0] in self.names:
                    # ok we can find it a a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == '_' and isinstance(c, CoordSet):
                        c = c.__getitem__(index[1:])
                    return c
            except IndexError:
                pass

            log.error(f"Could not find `{index}` in coordinates names or titles")
            return None

        res = self._coords.__getitem__(index)
        if isinstance(index, slice):
            return CoordSet(res, keepnames=True)
        else:
            return res

    # ..................................................................................................................
    def __setitem__(self, index, coord):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                self._coords.__setitem__(idx, coord)
                return

            # ok we did not find it!
            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                if self.titles.count(index) > 1:
                    warnings.warn(f"Getting a coordinate from its title. However `{index}` occurs several time. Only"
                                  f" the first occurence is returned!")
                self._coords.__setitem__(self.titles.index(index), coord)
                return

            # may be it is a title in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    item.__setitem__(item.titles.index(index), coord)
                    return
            try:
                # let try with the canonical dimension names defined by  DEFAULT_DIM_NAME
                if index[0] in self.names:
                    # ok we can find it a a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == '_' and isinstance(c, CoordSet):
                        c.__setitem__(index[1:], coord)
                    return

            except IndexError:
                pass

            raise IndexError(f"Could not find `{index}` in coordinates names or titles")

        self._coords[index] = coord

    # ..................................................................................................................
    def __delitem__(self, index):
        del self._coords[index]

    # ..................................................................................................................
    def __iter__(self):
        for item in self._coords:
            yield item

    # ..................................................................................................................
    def __repr__(self):
        out = "CoordSet: [" + ', '.join(['{}'] * len(self._coords)) + "]"
        s = []
        for item in self._coords:
            if isinstance(item, CoordSet):
                s.append(f"{item.name}:" + repr(item).replace('CoordSet: ', ''))
            else:
                s.append(f"{item.name}:{item.title}")
        out = out.format(*s)
        return out

    # ..................................................................................................................
    def __str__(self):
        return repr(self)

    # ..................................................................................................................
    def _cstr(self, header='   Multicoord: ... \n'):
        out = ""
        for i, item in enumerate(self._coords):
            if i == 0:
                out += f'{item._str_shape().rstrip()}\n'
                out += header
            txt = f'{item._cstr(print_size=False)}\n'
            out += format(textwrap.indent(txt, ' ' * 4))
            out = out.replace('        title', f'({item.name}){" " * (6 - len(item.name))}title')

        return out

    # ..................................................................................................................
    def __deepcopy__(self, memo):
        coords = self.__class__([copy.deepcopy(ax, memo=memo) for ax in self], keepnames=True)
        coords.name = self.name
        return coords

    # ..................................................................................................................
    def __copy__(self):
        coords = self.__class__([copy.copy(ax) for ax in self], keepnames=True)
        coords.name = self.name
        return coords

        # ..................................................................................................................

    def __eq__(self, other):
        if other is None:
            return False
        try:
            return self._coords == other._coords
        except:
            return False

    # ..................................................................................................................
    def __ne__(self, other):
        return not self.__eq__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def _coords_update(self, change):
        # when notified that a coord name have been updated
        self._updated = True

    # ..................................................................................................................
    @observe(All)
    def _anytrait_changed(self, change):
        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }
        log.debug('changes in CoordSet: %s to %s' % (change.name, change.new))
        if change.name == '_updated' and change.new:
            self._updated = False  # reset


# ======================================================================================================================
# Range trait type
# ======================================================================================================================

class Range(List):
    """
    Create a trait with two values defining an ordered range of values,
    with an optional sampling parameters

    Parameters
    ----------

    trait : TraitType [ optional ]
        the type for restricting the contents of the Container.
        If unspecified, types are not checked.

    default_value : SequenceType [ optional ]
        The default value for the Trait.  Must be list/tuple/set, and
        will be cast to the container type.


    Examples
    --------

    >>> class MyClass(HasTraits):
    ...     r = Range([10,5])  # Initialized with some default values

    >>> c = MyClass()
    >>> print(c.r) # the list is ordered
    [5, 10]
    >>> c.r = [1, 3, 5]
    Traceback (most recent call last):
     ...
    traitlets.traitlets.TraitError: The 'r' trait of a type instance must be of length 2 exactly, but a value of [1, 3, 5] was specified.

    """
    klass = list
    _cast_types = (tuple,)

    # Describe the trait type
    info_text = 'an ordered interval trait'
    allow_none = True

    def __init__(self, default_value=None, **kwargs):

        super(Range, self).__init__(trait=None, default_value=default_value,
                                    **kwargs)
        pass

    def length_error(self, obj, value):
        e = "The '%s' trait of %s instance must be of length 2 exactly," \
            " but a value of %s was specified." \
            % (self.name, class_of(obj), value)
        raise TraitError(e)

    def validate_elements(self, obj, value):
        if value is None or len(value) == 0:
            return
        length = len(value)
        if length != 2:
            self.length_error(obj, value)
        value.sort()
        value = super(Range, self).validate_elements(obj, value)
        return value

    def validate(self, obj, value):

        value = super(Range, self).validate(object, value)
        value = self.validate_elements(obj, value)

        return value


# ======================================================================================================================
# CoordRange
# ======================================================================================================================
class CoordRange(HasTraits):
    """Set of ordered, non intersecting intervals
    e.g. [[a, b], [c, d]] with a < b < c < d or a > b > c > d.
    """
    # TODO: May use also units ???
    ranges = List(Range)
    reversed = Bool

    # ..................................................................................................................
    def __call__(self, *ranges, **kwargs):
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
        """
        # super(CoordRange, self).__init__(**kwargs)
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
            self._cleanranges(ranges)
        if self.ranges:
            self._cleanranges(self.ranges)
        return self.ranges

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def _cleanranges(self, ranges):
        """Sort and merge overlapping ranges
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
        self.ranges = cleaned_ranges
        if self.reversed:
            for range in self.ranges:
                range.reverse()
            self.ranges.reverse()


CoordRange = CoordRange()

# ======================================================================================================================
# Set the operators
# ======================================================================================================================
set_operators(Coord, priority=50)

# ======================================================================================================================
if __name__ == '__main__':
    pass
