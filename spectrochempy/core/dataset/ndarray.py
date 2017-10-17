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


"""This module implements the basic `NDArray` class  which is an array
container.  :class:`~spectrochempy.core.dataset.ndaxes.Axis` and
:class:`~spectrochempy.core.dataset.nddataset.NDDataset` classes are derived
from it.

"""
# =============================================================================
# standard imports
# =============================================================================

import copy
import uuid
import warnings
from datetime import datetime

import numpy as np
# from ...utils import create_traitsdoc
from pint.errors import DimensionalityError, UndefinedUnitError
from six import string_types
from traitlets import List, Unicode, Instance, Bool, HasTraits, default
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
# =============================================================================
# local imports
# =============================================================================
from spectrochempy.application import log

from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.units import Unit, ur, Quantity

from spectrochempy.utils import EPSILON, is_number
from spectrochempy.utils import SpectroChemPyWarning, deprecated
from spectrochempy.utils.traittypes import Array


# =============================================================================
# Third party imports
# =============================================================================

# =============================================================================
# Constants
# =============================================================================

__all__ = ['NDArray']
_classes = __all__[:]

# =============================================================================
# NDArray class
# =============================================================================

class NDArray(HasTraits):
    """A read-only NDArray object (This is the base class for SpectroChemPy
    array-like object, intended to be subclassed)

    The key distinction from raw numpy arrays is the presence of
    optional properties such as mask, axes, uncertainties, units and/or
    extensible metadata dictionary.

    Warnings
    --------
    This class needs to be subclassed as it provides only minimal
    functionalities. See for example the
    :class:`~spectrochempy.core.dataset.ndaxes.Axis` and
    :class:`~spectrochempy.core.dataset.nddataset.NDDataset` which both inherit
    from this object.

    Examples
    --------
    First we create a subclass with a setter for data because it is not provided
    by this basic :class:`~spectrochempy.core.dataset.ndarray.NDArray` class.

    >>> class MinimalSubclass(NDArray):
    ...
    ...     def _set_data(self, values):
    ...         self._data = values
    ...

    Then, we create an instance and populate its underlying `data`

    >>> ndd = MinimalSubclass()
    >>> ndd.data = np.random.random((10, 10))

    Let's see the string representation of this newly created `ndd` object.

    >>> print(ndd)  #doctest: +ELLIPSIS
    NDArray:
    array([...])


    """

    _data = Array
    _is_complex = List(Bool)
    _mask = Array
    _units = Instance(Unit, allow_none=True)
    _uncertainty = Array
    _name = Unicode
    _title = Unicode
    _meta = Instance(Meta, allow_none=True)
    _date = Instance(datetime)
    _labels = Array
    _fig = Instance(plt.Figure, allow_none=True)
    _ax = Instance(plt.Axes, allow_none=True)

    # _scaling = Float(1.)

    # private flags
    _data_passed_with_mask = Bool(transient=True)     # TODO: transient is a residue of Traits
    _data_passed_is_quantity = Bool(transient=True)   # we need to adapt something similar for traitlets

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, **kwargs):
        super(NDArray, self).__init__(**kwargs)

    # --------------------------------------------------------------------------
    # Defaults
    # --------------------------------------------------------------------------
    # @default('_data')
    # def _get_data_default(self):
    #     return None

    @default('_name')
    def _get_name_default(self):
        return str(uuid.uuid1()).split('-')[0]  # a unique id

    @default('_mask')
    def _get_mask_default(self):
        return np.zeros(self._data.shape, dtype=bool)

    @default('_units')
    def _get_units_default(self):
        return None #ur.dimensionless

    @default('_uncertainty')
    def _get_uncertainty_default(self):
        return np.zeros(self._data.shape, dtype=float)

    @default('_meta')
    def _get_meta_default(self):
        return Meta()

    @default('_is_complex')
    def _get_is_complex_default(self):
        return list([False for _ in self._data.shape])

    @default('_date')
    def _get_date_default(self):
        return datetime(1, 1, 1, 0, 0)

    @default('_labels')
    def _get_labels_default(self):
        return np.array([''] * self._data.size)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def data(self):
        """:class:`~numpy.ndarray`-like object - The actual array data
        contained in this object.

        """
        return self._data

    @data.setter
    def data(self, data):

        self._data = data

    @property
    def values(self):
        """:class:`~numpy.ndarray`-like object - The actual values (data, units,
        + uncertainties) contained in this object.

        """
        return self._uarray(self._data, self._uncertainty, self._units)

    @property
    def name(self):
        """`str` - An user friendly name for this object (often not necessary,
        as the title may be used for the same purpose).

        If the name is not provided, the object
        will automatically create a unique name.
        For most usage, the object name needs to be unique.

        """
        return self._name

    @name.setter
    def name(self, name):
        # property.setter for name
        if name is not None:
            self._name = name


    @property
    def title(self):
        """`str` - An user friendly title for this object.

        Unlike the :attr:`name`, the title doesn't need to be unique.
        When the title is provided, it can be used for labelling the object,
        e.g. axe title in a matplotlib plot.

        """
        return self._title

    @title.setter
    def title(self, title):
        # property.setter for title
        if title is not None:
            if self._title is not None:
                log.info(
                        "Overwriting Axis's current title with specified title")
            self._title = title

    @property
    def mask(self):
        """:class:`~numpy.ndarray`-like - Mask for the data.

        The values must be `False` where
        the data is *valid* and `True` when it is not (like Numpy
        masked arrays). If `data` is a numpy masked array, providing
        `mask` here will causes the mask from the masked array to be
        ignored.

        """

        return self._mask

    @mask.setter
    def mask(self, mask):
        # property.setter for mask

        if mask is not None:
            if self.is_masked and np.any(mask != self._mask):
                log.info("Overwriting Axis's current "
                            "mask with specified mask")
            self._mask = mask

    @property
    def units(self):
        """An instance of `Unit` or `str` - The units of the data.

        If data is a
        :class:`~pint.units.Quantity` then
        :class:`~pint.units.Unit` is set to the unit of the data;
        if a unit is also explicitly
        provided an error is raised.

        """
        return self._units

    @units.setter
    def units(self, units):

        if units is None:
            return

        try:
            if isinstance(units, string_types):
                units = ur.Unit(units)
            elif isinstance(units, Quantity):
                raise TypeError("Units or string representation "
                                "of unit is expected, not Quantity")

        except DimensionalityError:
            raise DimensionalityError

        except UndefinedUnitError as und:
            raise UndefinedUnitError(und.unit_names)

        if self._units is not None and units != self._units:
            # first try to cast
            try:
                self.to(units)
            except:
                raise ValueError(
                    "Unit provided in initializer does not match data units.\n "
                    "To force a change - use the change_units() method")

        self._units = units


    @property
    def uncertainty(self):
        """:class:`~numpy.ndarray` -  Uncertainty (std deviation) on the data.

        """
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, uncertainty):
        raise NotImplementedError()

    @property
    def meta(self):
        """:class:`~spectrochempy.core.dataset.ndmeta.Meta` instance object -
        Additional metadata for this object.

        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        # property.setter for meta
        if meta is not None:
            self._meta.update(meta)

    @property
    def labels(self):
        """:class:`~numpy.ndarray` - An array of objects of any type (but most
        generally string).

        The array contains the labels for the axis (if any)
        which complements the coordinates

        """
        if self.is_labeled:
            return self._labels
        else:
            return np.empty_like(self._data, dtype='str')

    @labels.setter
    def labels(self, labels):
        # Property setter for labels
        if labels is None:
            return
        elif isinstance(labels, np.ndarray):
            self._labels = labels
        else:
            self._labels = np.array(labels, subok=True,
                                    copy=self._iscopy).astype(object)

    @property
    def is_labeled(self):
        """`bool`, read-only property - Whether the axis has labels or not.

        """
        if self._labels is None:
             return False
        elif self._labels.size == 0:
            return False
        elif np.any(self._labels != ''):
            return True
        return False

    # -------------------------------------------------------------------------
    # read-only properties / attributes
    # -------------------------------------------------------------------------

    @property
    def shape(self):
        """`tuple`, read-only property - A `tuple` with the size of each axis.

        i.e., the number of data element on each axis (possibly complex).

        """
        # read the actual shape of the underlying array
        shape = list(self._data.shape)

        # take into account that the data may be complex,
        # so that the real and imag data are stored sequentially
        if self._is_complex is not None:
            for dim, is_complex in enumerate(self._is_complex):
                if is_complex:
                    # here we divide by 2 tha apparent shape
                    shape[dim] //= 2

        return tuple(shape)

    @property
    def size(self):
        """`int`, read-only property - Size of the underlying `ndarray`.

        i.e., the total number of data element
        (possibly complex or hyper-complex in the array).

        """
        size = self._data.size
        if self._is_complex is not None:
            for is_complex in self._is_complex:
                if is_complex:
                    size //= 2
        return size

    @property
    def ndim(self):
        """`int`, read-only property - The number of dimensions of
        the underlying array.

        """
        return self._data.ndim

    @property
    def dtype(self):
        """`dtype`, read-only property - data type of the underlying array

        """
        if np.sum(self.is_complex) > 0:
            return np.complex
        else:
            return self._data.dtype

    @property
    def is_empty(self):
        """`bool`, read-only property - Whether the array is empty (size==0)
        or not.

        """
        return self._data.size == 0

    @property
    def is_masked(self):
        """`bool`, read-only property - Whether the array is masked or not.

        """
        if self._mask.size == 0:
            return False

        if np.any(self._mask):
            return True

        return False

    @property
    def is_uncertain(self):
        """`bool`, read-only property - Whether the array has uncertainty
        or not.

        """
        if self._uncertainty.size == 0:
            return False

        if np.any(self._uncertainty > EPSILON):
            return True
        return False

    @property
    def is_untitled(self):
        """`bool`, read-only property - Whether the array has `title` or not.

        """
        if self.title:
            return False
        return True

    @property
    def unitless(self):
        """`bool`, read-only property - Whether the array has `units` or not.

        """
        return self._units is None

    @property
    def dimensionless(self):
        """`bool`, read-only property - Whether the array is dimensionless
        or not.

        Equal to `True` if the `data` is dimensionless
        (warning : different of unitless, which means no unit).

        """
        if self.unitless:
            return False

        return self._units.dimensionless

    @property
    def is_complex(self):
        """`tuple` of `bool` - Indicate if any dimension is is_complex.

        If a dimension is is_complex, real and imaginary part are interlaced
        in the `data` array.

        """
        try:
            if len(self._data)==0: # self._data.any():
                self._is_complex = None
                return None
        except:
            if self._data.size == 0: # self._data.any():
                self._is_complex = None
                return None

        if self._is_complex is None:
            self._is_complex = list([False for _ in self._data.shape])

        return self._is_complex


    @property
    def has_complex_dims(self):
        """`bool` - Check if any of the dimension is complex

        """
        if self._is_complex is not None:
            return np.sum(self.is_complex) > 0

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def set_complex(self, axis=-1):
        """Make a dimension complex

        Parameters
        ----------
        axis : `int`, optional, default = -1
            The axis to make complex

        """
        if self.data.shape[axis] % 2 == 0:
            # we have a pair number of element along this axis. It can be complex
            # data are then supossed to be interlaced (real, imag, real, imag ..
            self._is_complex[axis] = True
        else:
            raise ValueError('The odd size along axis {} is not compatible with'
                             ' complex interlaced data'.format(axis))

    @deprecated('use `to` instead')
    def ito(self, other):
        """Rescale the current object data to different units.

        (same as :attr:`to` with inplace=`True`).

        Parameters
        ----------
        other : `Quantity` or `str`.
            destination units.

        Returns
        -------
        object : same type
            same object with new units.

        See Also
        --------
        to

        """
        return self.to(other, inplace=True)

    def to(self, other, inplace=True):
        """Return the object with data rescaled to different units.

        Parameters
        ----------
        other : :class:`Quantity` or `str`.
            destination units.

        inplace : `bool`, optional, default = `True`.
            if inplace is True, the object itself is returned with
            the new units. If `False` a copy is created.

        Returns
        -------
        object : same type
            same object or a copy depending on `ìnplace` with new units.


        """
        if self._units is not None:
            q = Quantity(1., self._units).to(other)
            scale = q.magnitude
            if inplace:
                new = self
            else:
                new = self.copy()
            new._data = new._data * scale #new * scale #
            if new.uncertainty is not None:
                new._uncertainty = new._uncertainty * scale
            new._units = q.units
            return new
        else:
            warnings.warn("There is no units for this NDArray!", SpectroChemPyWarning)

        return self

    def change_units(self, units):
        """
        Force a chnage of units

        Parameters
        ----------
        units : cccccc

        Returns
        -------

        """
        try:
            if isinstance(units, string_types):
                units = ur.Unit(units)
            elif isinstance(units, Quantity):
                raise TypeError("Units or string representation "
                                "of unit is expected, not Quantity")

        except DimensionalityError:
            raise DimensionalityError

        except UndefinedUnitError as und:
            raise UndefinedUnitError(und.unit_names)

        try:
            self.to(units)
        except DimensionalityError:
            self._units = units
            log.info('units forced to change')


    def is_units_compatible(self, other):
        """
        Check the compatibility of units with another NDArray

        Parameters
        ----------
        other : NDArray

        Returns
        -------
        compat : `bool`

        """
        _other = other.copy()

        try:
            _other.to(self.units)
        except:
            return False

        return True

    def copy(self, deep=True, memo=None):
        """Make a disconnected copy of the current object.

        Parameters
        ----------
        deep : `bool`, optional.

            If `True` a deepcopy is performed which is the default behavior

        memo : Not really used.

            This parameter ensure compatibility with deepcopy() from the copy
            package.

        Returns
        -------
        object : same type.
            an exact copy of the current object.

        """
        if deep:
            do_copy = copy.deepcopy
        else:
            do_copy = copy.copy

        new = type(self)()
        new._data = do_copy(self._data)
        for attr in self.__dir__():

            try:
                setattr(new, "_" + attr, do_copy(getattr(self, attr)))
            except:
                # if deep copy do not work
                setattr(new, "_" + attr, copy.copy(getattr(self,
                                                       attr)))
            # if attr not in ['data', 'units']:
            #     # we set directly the hidden attribute as no checking
            #     # is necessary for such copy
            #     setattr(new, "_" + attr, do_copy(getattr(self, attr)))
            # elif attr in ['units']:
            #     setattr(new, "_" + attr, copy.copy(getattr(self,
            #                                                attr)))  # deepcopy not working (and not necessary)
        new._name = str(uuid.uuid1()).split('-')[0]
        new._date = datetime.now()
        return new

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    def __repr__(self):  #TODO: display differently if no uncertainty
        txt = "NDArray: \n" + repr(self._uarray(self._data, self._uncertainty))
        if self.units is not None:
            txt += repr(self.units)
        return txt

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.shape[0]

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        return self.copy(deep=True, memo=memo)

    def __dir__(self):
        return     ['data', 'mask', 'units', 'uncertainty', 'labels',
                    'meta', 'name', 'title', 'is_complex']

    def __getitem__(self, item):

        # to avoid slicing error when there is only one element
        if item == slice(None, None, None) and self.size == 1:
            return self.__copy__()

        # slicing by index of all internal array
        new_coords = np.array(self._data[item])
        new_mask = np.array(self._mask[item])
        if self.is_labeled:
            new_labels = np.array(self._labels[..., item])
        else:
            new_labels = None

        return self.__class__(new_coords,
                              labels=new_labels,
                              mask=new_mask,
                              units=self.units,
                              meta=self.meta,
                              title=self.title)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):

        otherlabels = None
        otherislabeled = False
        if hasattr(other, "_data"):
            otherdata = other._data
            otherunits = other._units
            otherlabels = other._labels
            otherislabeled = other.is_labeled
        elif isinstance(other, Quantity):
            otherdata = other.magnitude
            otherunits = other.units
        elif isinstance(other, (float, int, np.ndarray)):
            otherdata = other
            otherunits = None
        else:
            raise TypeError("cannot compare with type: " % type(other))

        if self._units is None and otherunits is None:
            eq = np.all(self._data == otherdata)
        elif self._units is not None and otherunits is not None:
            eq = np.all(self._data * self._units == otherdata * otherunits)
        else:
            return False

        if self.is_labeled and otherislabeled:  # label in both.
            eq &= (np.all(self._labels == otherlabels))

        if hasattr(other, '_meta'):
            eq &= (self._meta == other._meta)

        return eq

    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return 1234509876

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d array')
        for n in range(len(self)):
            yield self[n]

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------
    @staticmethod
    def _umasked(data, mask):
        # This ensures that a masked array is returned if self is masked.

        if mask is not None and np.any(mask):
            data = np.ma.masked_array(data, mask)
        else:
            data = np.array(data)
        return data

    @staticmethod
    def _uarray(data, uncertainty, units=None):
        # return the array with uncertainty and units if any

        if uncertainty is None or np.all(uncertainty <= EPSILON):
            uar= np.array(data)
        else:
            uar= unp.uarray(data, uncertainty)

        if units:
            return Quantity(uar, units)
        else:
            return uar

    def _argsort(self, by='value', pos=None, descend=False):
        # found the indices sorted by axes or labels

        if not self.is_labeled:
            by = 'value'
            pos = None
            warnings.warn('no label to sort, use `axis` by default')

        if by == 'value':
            args = np.argsort(self._data)

        elif by == 'label':
            labels = self._labels
            if len(self._labels.shape) > 1:
                # multidimentional labels
                if not pos: pos = 0
                labels = self._labels[pos]  # TODO: this must be checked
            args = np.argsort(labels)

        else:
            by = 'value'
            warnings.warn(
                    'parameter `by` should be set to `value` or `label`, '
                    'use `value` by default')
            args = np.argsort(self._data)

        if descend:
            args = args[::-1]

        return args

    def _sort(self, by='value', pos=None, descend=False, inplace=False):
        # sort an ndarray in place using data or label values

        if not inplace:
            new = self.copy()
        else:
            new = self

        args = self._argsort(by, pos, descend)

        return self._take(args)

    def _take(self, indices):
        # get a ndarray with passed indices

        new = self.copy()
        new._data = new._data[indices]
        if new.is_labeled:
            new._labels = new._labels[..., indices]
        new._mask = new._mask[indices]

        return new

    def _loc2index(self, loc, axis):
        # Return the index of a location (label or coordinates) along the axis

        if isinstance(loc, string_types):
            # it's probably a label
            indexes = np.argwhere(self._labels == loc).flatten()
            if indexes.size > 0:
                return indexes[0]
            else:
                raise ValueError('Could not find this label: {}'.format(loc))

        elif isinstance(loc, datetime):
            # not implemented yet
            return None  # TODO: date!

        elif is_number(loc):
            index = (np.abs(self._data - loc)).argmin()
            if loc > self._data.max() or loc < self._data.min():
                warnings.warn(
                        '\nThis coordinate ({}) is outside the axis limits.\n'
                        'The closest limit index is returned'.format(loc),)
                        # AxisWarning)
            return index

        else:
            raise ValueError('Could not find this location: {}'.format(loc))
