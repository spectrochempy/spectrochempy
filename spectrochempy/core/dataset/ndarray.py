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

# =============================================================================
# Third party imports
# =============================================================================

import numpy as np
from pint.errors import DimensionalityError, UndefinedUnitError
from traitlets import (Union, List, Unicode, Instance, Bool, HasTraits, default,
                       Any, Integer, Sentinel)
from uncertainties import unumpy as unp
import pandas as pd
from pandas.core.generic import NDFrame
import matplotlib.pyplot as plt

# =============================================================================
# local imports
# =============================================================================

from spectrochempy.application import log

from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.units import Unit, ur, Quantity, Measurement

from spectrochempy.utils import EPSILON, is_number, is_sequence, numpyprintoptions
from spectrochempy.utils import SpectroChemPyWarning, deprecated
from spectrochempy.utils.traittypes import Array, HyperComplexArray

# =============================================================================
# Constants
# =============================================================================

__all__ = ['NDArray']
_classes = __all__[:]

# =============================================================================
# Some initializations
# =============================================================================

numpyprintoptions()  # set up the numpy print format

# =============================================================================
# The basic NDArray class
# =============================================================================

class NDArray(HasTraits):
    """A NDArray object

    The key distinction from raw numpy arrays is the presence of
    optional properties such as labels, mask, uncertainties, units and/or
    extensible metadata dictionary.

    Warnings
    --------
    This class generally needs to be subclassed as it provides only minimal
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

    Then, we create an instance and populate its underlying `data` with some
    random data

    >>> ndd = MinimalSubclass()
    >>> np.random.seed(12345)
    >>> ndd.data = np.random.random((10, 10))

    Let's see the string representation of this newly created `ndd` object.

    >>> print(ndd)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    MinimalSubclass: [[    0.93,    0.316, ...,    0.749,    0.654],
                      [   0.748,    0.961, ...,    0.965,    0.724],
                      ...,
                      [   0.945,    0.533, ...,    0.651,    0.313],
                      [   0.769,    0.782, ...,    0.898,   0.0427]]

    """

    _ax = Instance(plt.Axes, allow_none=True)
    _data = HyperComplexArray(allow_none=True)
    _date = Instance(datetime)
    _fig = Instance(plt.Figure, allow_none=True)
    #_is_complex = List(Bool(), allow_none=True)
    _labels = Array(allow_none=True)
    _mask = Array(allow_none=True)
    _meta = Instance(Meta, allow_none=True)
    _name = Unicode()
    _title = Unicode()
    _uncertainty = Array(allow_none=True)
    _units = Instance(Unit, allow_none=True)

    # _scaling = Float(1.)

    # private flags
    _data_passed_is_measurement = Bool()
    _data_passed_is_quantity = Bool()
    _data_passed_with_mask = Bool()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self,
                 data=None,
                 mask=None,
                 uncertainty=None,
                 units=None,
                 meta=None,
                 name=None,
                 title=None,
                 #is_complex=None,
                 **kwargs):

        #super(NDArray, self).__init__(**kwargs)

        #if is_complex is not None:
        #    self._is_complex = is_complex

        self.data = data

        self.name = name

        self.meta = meta

        self.title = title

        if mask is not None:
            if self._data_passed_with_mask and self._mask != mask:
                log.info("NDDataset was created with a masked array, and a "
                         "mask was explicitly provided to Axis. The  "
                         "explicitly passed-in mask will be used and the "
                         "masked array's mask will be ignored.")
            self.mask = mask

        if units is not None:
            if self._data_passed_is_quantity and self._units != units:
                raise ValueError(
                        "Cannot use the units argument "
                        "when passed data is a Quantity")
            self.units = units

        # This must come after self's units has been set so that the units
        # of the uncertainty, if any, can be converted to the units of self.
        if uncertainty is not None:
            if self._data_passed_is_measurement \
                    and self._uncertainty != uncertainty:
                raise ValueError(
                        "Cannot use the uncertainty argument "
                        "when passed data is already with uncertainty")
            self.uncertainty = uncertainty

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        return self.copy(deep=True, memo=memo)

    def __dir__(self):
        return ['data', 'mask', 'units', 'uncertainty', 'labels', \
                'meta', 'name', 'title'] #, 'is_complex']

    def __eq__(self, other, attrs=None):
        if not (other.__hash__()==self.__hash__()):
            return False
        eq = True
        if attrs is None:
            attrs = self.__dir__()
        for attr in attrs:
            if hasattr(other, "_%s" % attr):
                eq &= np.all(
                        getattr(self, "_%s" % attr) == getattr(other,
                                                               "_%s" % attr))
                if not eq:
                    print("attributes '%s' are not equals "
                          "or one is missing" % attr)
                    return False
        return eq

    def _make_index(self, key):

        if isinstance(key, np.ndarray) and key.dtype == np.bool:
            # this is a boolean selection
            # we can proceed directly
            return key

        # we need to have a list of slice for each argument
        # or a single slice acting on the axis=0
        # the given key can be a single argument
        # or a single slice

        # we need a list in all cases
        if not is_sequence(key):
            keys = [key, ]
        else:
            keys = list(key)

        # Ellipsis
        while Ellipsis in keys:
            i = keys.index(Ellipsis)
            keys.pop(i)
            for j in range(self.ndim - len(keys)):
                keys.insert(i, slice(None))

        if len(keys) > self.ndim:
            raise IndexError("invalid index")

        # pad the list with additional dimensions
        for i in range(len(keys), self.ndim):
            keys.append(slice(None))

        for i, key in enumerate(keys):
            if is_number(key) and self.is_complex[i]:
                keys[i]= key*2

        return tuple(keys)

    def __getitem__(self, items):

        # to avoid slicing error when there is only one element
        if items == slice(None, None, None) and self.size == 1:
            return self.__copy__()

        new = self.copy()

        # The actual index depends on the complexity of the dimension
        keys = self._make_index(items)

        # slicing by index of all internal array
        new._data = np.array(self._data[keys])

        if self.is_masked:
            new._mask = np.array(self._mask[keys])

        if self.is_uncertain:
            new._uncertainty = np.array(self._uncertainty[keys])

        if self.is_labeled:
            new._labels = np.array(self._labels[..., keys])

        return new

    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return str(type(self)) + "1234567890"

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d array')
        for n in range(len(self)):
            yield self[n]

    def __len__(self):
        return self.shape[0]

    def __ne__(self, other, attrs=None):
        return not self.__eq__(other, attrs)

    def __repr__(self):
        prefix = type(self).__name__ + ': '
        data = self.uncert_data
        if self.is_masked:
            data = data.astype(object)
        body = np.array2string( \
                data, separator=', ', \
                prefix=prefix)  # this allow indentation of len of the prefix

        if self.units:
            units = ' {:~.3f}'.format(self.units)
        else:
            units = ''
        return ''.join([prefix, body, units])

    def __str__(self):
        return self.__repr__()

    # --------------------------------------------------------------------------
    # Defaults
    # --------------------------------------------------------------------------

    @default('_date')
    def _get_date_default(self):
        return datetime(1, 1, 1, 0, 0)

    #@default('_is_complex')
    #def _get_is_complex_default(self):
    #    return None # list([False for _ in self._data.shape])

    @default('_labels')
    def _get_labels_default(self):
        return np.empty_like(self._data, dtype='str')

    @default('_mask')
    def _get_mask_default(self):
        return np.zeros(self._data.shape, dtype=bool)

    @default('_meta')
    def _get_meta_default(self):
        return Meta()

    @default('_name')
    def _get_name_default(self):
        return str(uuid.uuid1()).split('-')[0]  # a unique id

    @default('_uncertainty')
    def _get_uncertainty_default(self):
        return np.zeros(self._data.shape, dtype=float)

    @default('_units')
    def _get_units_default(self):
        return None  # ur.dimensionless


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
        # property.setter for data

        if data is None:
            self._data = np.array([]).astype(float)  # reinit data
            log.debug("init data with an empty ndarray of type float")

        elif isinstance(data, NDArray):
            log.debug("init data with data from another NDArray")
            # No need to check the validity of the data
            # because the data must have been already
            # successfully initialized for the passed NDArray.data
            for attr in self.__dir__():
                val = getattr(data, "_%s"%attr)
                val = copy.deepcopy(val)
                setattr(self, "_%s"%attr, val)

            self._name = "copy of {}".format(data._name)

        elif isinstance(data, NDFrame):  # pandas object
            log.debug("init data with data from pandas NDFrame object")
            self._validate(data.values)
            self.axes = data.axes

        elif isinstance(data, pd.Index):  # pandas index object
            log.debug("init data with data from a pandas Index")
            self._validate(np.array(data.values, subok=True,
                                    copy=True))

        elif isinstance(data, Quantity):
            log.debug("init data with data from a Quantity object")
            self._data_passed_is_quantity = True
            self._validate(np.array(data.magnitude, subok=True,
                                    copy=True))
            self._units = data.units

        elif hasattr(data, 'mask'):  # an object with data and mask attributes
            log.debug("init mask from the passed data")
            self._data_passed_with_mask = True
            self._validate(np.array(data.data, subok=True,
                                    copy=True))
            if isinstance(data.mask, np.ndarray) and \
                            data.mask.shape == data.data.shape:
                self._mask = np.array(data.mask, dtype=np.bool_, copy=False)
            else:
                self._data_passed_with_mask = False  # not succesfull

        elif (not hasattr(data, 'shape') or
                  not hasattr(data, '__getitem__') or
                  not hasattr(data, '__array_struct__')):
            log.debug("init data with a non numpy-like array object")
            # Data doesn't look like a numpy array, try converting it to
            # one. Non-numerical input are converted to an array of objects.
            self._validate(np.array(data, subok=True, copy=False))

        else:
            log.debug("init data with a numpy array")
            self._validate(np.array(data, subok=True,
                                    copy=True))

    @property
    def date(self):
        """A datetime object containing date information
        about the ndarray object, for example a creation date
        or a modification date"""
        return self._date

    @date.setter
    def date(self, date):
        if isinstance(date, datetime):
            self._date = date
        elif isinstance(date, str):
            try:
                self._date = datetime.strptime(date, "%Y/%m/%d")
            except ValueError:
                self._date = datetime.strptime(date, "%d/%m/%Y")

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

    @property
    def labels(self):
        """:class:`~numpy.ndarray` - An array of objects of any type (but most
        generally string).

        The array contains the labels for the axis (if any)
        which complements the coordinates

        """
        return self._labels

    @labels.setter
    def labels(self, labels):
        # Property setter for labels
        if labels is None:
            return
        elif isinstance(labels, np.ndarray):
            self._labels = labels
        else:
            self._labels = np.array(labels, subok=True,
                                    copy=True).astype(object)

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
            if self._mask is not None:
                log.info("Overwriting {} ".format(type(self).__name__) +
                         "current mask with specified mask")

            # Check that value is not either type of null mask.
            if mask is not np.ma.nomask:
                mask = np.array(mask, dtype=np.bool_, copy=False)
                if mask.shape != self.shape:
                    raise ValueError("dimensions of mask do not match data")
                else:
                    self._mask = mask
        else:
            # internal representation should be one numpy understands
            self._mask = np.ma.nomask

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
                        "Overwriting ndarray current title with specified title")
            self._title = title

    @property
    def uncertainty(self):
        """:class:`~numpy.ndarray` -  Uncertainty (std deviation) on the data.

        """
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, uncertainty):
        # property setter for uncertainty

        if uncertainty is not None:
            if self.is_uncertain and np.any(uncertainty != self._uncertainty):
                log.info("Overwriting {} ".format(type(self).__name__) +
                         "current uncertainty with specified uncertainty")

            if not isinstance(uncertainty, np.ndarray):
                raise ValueError('Uncertainty must be specified as a ndarray')
                # TODO: make this a little less strict
                # so it accept other list structure

            if uncertainty.shape != self._data.shape:

                if not self.has_complex_dims:
                    raise ValueError(
                            'uncertainty shape does not match array data shape')
                else:  # complex data
                    pass

            self._uncertainty = uncertainty

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
            if isinstance(units, str):
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
    def values(self):
        """:class:`~numpy.ndarray`-like object - The actual values (data, units,
        + uncertainties) contained in this object.

        """
        return self._uarray(self._data, self._uncertainty, self._units)

    # -------------------------------------------------------------------------
    # read-only properties / attributes
    # -------------------------------------------------------------------------

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
    def dtype(self):
        """`dtype`, read-only property - data type of the underlying array

        """
        if np.sum(self._data.is_complex) > 0:
            return np.complex
        else:
            return self._data.dtype

    @property
    def has_complex_dims(self):
        """`bool` - Check if any of the dimension is complex

        """
        return np.sum(self._data.is_complex) > 0

    @property
    def is_complex(self):
        """`tuple` of `bool` - Indicate if any dimension is is_complex.

        If a dimension is is_complex, real and imaginary part are interlaced
        in the `data` array.

        """
        return self._data.is_complex

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
        if self._mask is None or self._mask.size == 0:
            return False

        if np.any(self._mask):
            return True

        return False

    @property
    def is_uncertain(self):
        """`bool`, read-only property - Whether the array has uncertainty
        or not.

        """
        if self._uncertainty is None or self._uncertainty.size == 0:
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
    def masked_data(self):
        """:class:`~numpy.ndarray`-like object - The actual masked array of data
        contained in this object.

        """
        return self._umasked(self._data, self._mask)

    @property
    def ndim(self):
        """`int`, read-only property - The number of dimensions of
        the underlying array.

        """
        return self._data.ndim

    @property
    def shape(self):
        """`tuple`, read-only property - A `tuple` with the size of each axis.

        i.e., the number of data element on each axis (possibly complex).

        """
        # read the actual shape of the underlying array
        # shape = list(self._data.shape)
        #
        # # take into account that the data may be complex,
        # # so that the real and imag data are stored sequentially
        # if self._is_complex is not None:
        #     for dim, is_complex in enumerate(self._is_complex):
        #         if is_complex:
        #             # here we divide by 2 tha apparent shape
        #             shape[dim] //= 2
        #return tuple(shape)
        return self._data.trueshape


    @property
    def size(self):
        """`int`, read-only property - Size of the underlying `ndarray`.

        i.e., the total number of data element
        (possibly complex or hyper-complex in the array).

        """
        size = self._data.truesize
        # if self._is_complex is not None:
        #     for is_complex in self._is_complex:
        #         if is_complex:
        #             size //= 2
        return size

    @property
    def uncert_data(self):
        """:class:`~numpy.ndarray`-like object - The actual array with
        uncertainty of data
        contained in this object.

        """
        return self._uarray(self.masked_data, self._uncertainty)

    @property
    def unitless(self):
        """`bool`, read-only property - Whether the array has `units` or not.

        """
        return self._units is None

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def change_units(self, units):
        """
        Force a chnage of units

        Parameters
        ----------
        units : cccccc

        Returns
        -------

        """
        if self._units is None:
            self.units = units
            return

        try:
            if isinstance(units, str):
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

    def copy(self, deep=True, memo=None):
        """Make a disconnected copy of the current object.

        Parameters
        ----------
        deep : `bool`, optional.

            If `True` a deepcopy is performed which is the default behavior

        memo : Not used.

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
                # ensure that if deepcopy do not work, a shadow copy can be done
                setattr(new, "_" + attr, copy.copy(getattr(self,
                                                           attr)))
        new._name = str(uuid.uuid1()).split('-')[0]
        new._date = datetime.now()
        return new

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

    def make_complex(self, axis=-1):
        """Make a dimension complex

        Parameters
        ----------
        axis : `int`, optional, default = -1
            The axis to make complex

        """
        self._data.make_complex(axis)

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
            new._data = new._data * scale  # new * scale #
            if new._uncertainty is not None:
                new._uncertainty = new._uncertainty * scale
            new._units = q.units
            return new
        else:
            warnings.warn("There is no units for this NDArray!",
                          SpectroChemPyWarning)

        return self


    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------
    def _validate(self, data):

        try:
            if len(data) == 0:  # self._data.any():
                return
        except:
            if data.size == 0:  # self._data.any():
                return

        self._data = data
        return


    def _argsort(self, by='value', pos=None, descend=False):
        # found the indices sorted by values or labels

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
            uar = data
        else:
            uar = unp.uarray(data, uncertainty)

        if units:
            return Quantity(uar, units)
        else:
            return uar








                # def _loc2index(self, loc, axis):
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
        #                     'The closest limit index is returned'.format(loc), )
        #             # AxisWarning)
        #         return index
        #
        #     else:
        #         raise ValueError('Could not find this location: {}'.format(loc))
