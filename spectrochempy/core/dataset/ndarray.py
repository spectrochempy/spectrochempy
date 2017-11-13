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
container.  :class:`~spectrochempy.core.dataset.ndcoords.Coord` and
:class:`~spectrochempy.core.dataset.nddataset.NDDataset` classes are derived
from it.

"""
# =============================================================================
# standard imports
# =============================================================================

import copy
import uuid
import warnings
import re
from datetime import datetime

# =============================================================================
# Third party imports
# =============================================================================

import numpy as np
from traitlets import (List, Unicode, Instance, Bool, HasTraits, default,
                       Any, Float, validate, observe)

# =============================================================================
# local imports
# =============================================================================

from spectrochempy.application import log

from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.units import Unit, ur, Quantity, Measurement

from spectrochempy.utils import (EPSILON,
                                 is_sequence,
                                 numpyprintoptions,
                                 deprecated,
                                 interleaved2complex,
                                 interleave,
                                 SpectroChemPyWarning)

from spectrochempy.extern.traittypes import Array
from spectrochempy.extern.pint.errors import (DimensionalityError,
                                              UndefinedUnitError)
from spectrochempy.extern.uncertainties import unumpy as unp
from pandas.core.generic import NDFrame, Index


# =============================================================================
# Constants
# =============================================================================

__all__ = ['NDArray', 'CoordSet']

_classes = __all__[:]

# =============================================================================
# Some initializations
# =============================================================================

numpyprintoptions()  # set up the numpy print format

gt_eps = lambda arr: np.any(arr > EPSILON)


# =============================================================================
# The basic NDArray class
# =============================================================================

class NDArray(HasTraits):
    """A NDArray object

    The key distinction from raw numpy arrays is the presence of
    optional properties such as labels, mask, uncertainties, units and/or
    extensible metadata dictionary.

    Parameters
    ----------

    data : array

    Examples
    --------

    >>> ndd = NDArray()


    >>> np.random.seed(12345)
    >>> ndd.data = np.random.random((10, 10))
    ...

    Let's see the string representation of this newly created `ndd` object.

    >>> print(ndd)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [[   0.930    0.316 ...,    0.749    0.654]
    [   0.748    0.961 ...,    0.965    0.724]
    ...,
    [   0.945    0.533 ...,    0.651    0.313]
    [   0.769    0.782 ...,    0.898    0.043]]

    NDArray can be also created using keywords arguments. Here is a masked array:
    >>> ndd = NDArray( data = np.random.random((3, 3)),
    ...                mask = [[True, False, True],
    ...                        [False, True, False],
    ...                        [False, False, True]],
    ...                units = 'absorbance')
    >>> print(ndd)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [[  --    0.295   --]
    [   0.086   --    0.516]
    [   0.689    0.857   --]] dimensionless

    """

    _data = Array(Float(), allow_none=True)
    _coordset = Instance(List, allow_none=True)
    _mask = Array(Bool(), allow_none=True)
    _uncertainty = Array(Float(), allow_none=True)
    _labels = Array(Any(), allow_none=True)
    _units = Instance(Unit, allow_none=True)
    _is_complex = List(Bool(), allow_none=True)

    _date = Instance(datetime)


    _meta = Instance(Meta, allow_none=True)
    _title = Unicode(allow_none=True)

    _id = Unicode()
    _name = Unicode()
    _copy = Bool()

    _labels_allowed = Bool(True)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    # .........................................................................
    def __init__(self, data=None, **kwargs):

        self._copy = kwargs.pop('copy', False)  # by default
        # we try to keep the same data

        self._is_complex = kwargs.pop('is_complex', None)

        self.data = data

        self.title = kwargs.pop('title', None)

        # a unique id / name
        self.name = kwargs.pop('name', self._id)

        self.mask = kwargs.pop('mask', None)

        if self._labels_allowed:
            self.labels = kwargs.pop('labels', None)

        self.units = kwargs.pop('units', None)

        # Setup of uncertainties must come after setting units of the NDArray
        # so that the units of the uncertainty, if any, can be converted
        # to the current units.

        self.uncertainty = kwargs.pop('uncertainty', None)

        self.meta = kwargs.pop('meta', None)

        # process eventual kwargs, adressing HasTrait class
        super(NDArray, self).__init__(**kwargs)

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    # .........................................................................
    def __copy__(self):

        return self.copy(deep=False)

    # .........................................................................
    def __deepcopy__(self, memo=None):

        return self.copy(deep=True, memo=memo)

    # .........................................................................
    def __dir__(self):

        return ['data', 'mask', 'labels', 'units', 'uncertainty',
                'meta', 'name', 'title', 'is_complex']

    # .........................................................................
    def __eq__(self, other, attrs=None):

        if not isinstance(other, NDArray):

            # try to make some assumption to make usefull comparison.
            if isinstance(other, Quantity):
                otherdata = other.magnitude
                otherunits = other.units
            elif isinstance(other, (float, int, np.ndarray)):
                otherdata = other
                otherunits = False
            else:
                raise ValueError("I do not know how to compare "
                                "{} object with objets of type {} ".format(
                        type(self).__name__,
                        type(other).__name__))

            if not self.has_units and not otherunits:
                eq = np.all(self._data == otherdata)
            elif self.has_units and otherunits:
                eq = np.all(self._data * self._units == otherdata * otherunits)
            else:
                return False
            return eq

        eq = True
        if attrs is None:
            attrs = self.__dir__()
            attrs.remove('name')
            attrs.remove('title')  # name and title will
                                          # not be used for comparison
        for attr in attrs:
            if hasattr(other, "_%s" % attr):
                eq &= np.all(
                        getattr(self, "_%s" % attr) == getattr(other,
                                                               "_%s" % attr))
                if not eq:
                    log.debug("attributes '{}' are not equals "
                        "or one is missing: {}, {}" .format(attr,
                                            getattr(self,  "_%s" % attr),
                                            getattr(other, "_%s" % attr)))
                    return False
        return eq

    # .........................................................................
    def __getitem__(self, items):

        # to avoid slicing error when there is only one element
        #if items == slice(None, None, None) and self.size == 1:
        #    return self.copy()

        new = self.copy()

        # The actual index depends on the complexity of the dimension
        keys, internkeys = self._make_index(items)

        # slicing by index of all internal array
        #new._data = np.array(self._data[internkeys])
        udata = new._uncert_data(force=True)[internkeys]

        #if isinstance(udata, Quantity):
        new._data = unp.nominal_values(np.asarray(udata))

        #else:
        #    pass

        if self.is_labeled:
            # case only of 1D dataset such as Coord
            # we add Ellipsis as labels can be multidimensional
            # (multilabels)
            newkeys = tuple((Ellipsis, keys[-1]))
            new._labels = np.array(self._labels[newkeys])

        if new._data.size == 0:
            if not new.is_labeled or new._labels.size ==0:
                raise IndexError("Empty array of shape {}".format(
                                str(new._data.shape)) + \
                             "resulted from slicing.\n"
                             "Check the indexes and make "
                             "sure to use floats for "
                             "location slicing")

        new._is_complex = self._is_complex

        #if isinstance(udata, Quantity):
        new._mask = udata.mask # np.array(self._mask[keys])

        #if isinstance(udata, Quantity):
        new._uncertainty = unp.std_devs(np.asarray(udata)) # np.array(self._uncertainty[keys])

        if self._coordset is not None:
            new_coordset = self.coordset.copy()
            for i, coord in enumerate(new_coordset):
                new_coordset[i] = coord[keys[i]]
            new._coordset = new_coordset

        new._name = '*' + self._name.lstrip('*')

        return new

    # .........................................................................
    def __setitem__(self, items, value):

        keys, internkeys = self._make_index(items)

        self._data[internkeys] = value

    # .........................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared

        return type(self).__name__ + "1234567890"

    # .........................................................................
    def __iter__(self):

        if self._data.ndim == 0:
            raise ValueError('iteration over a 0-d array')
        for n in range(len(self)):
            yield self[n]

    # .........................................................................
    def __len__(self):

        return self._data.shape[0]

    # .........................................................................
    def __ne__(self, other, attrs=None):

        return not self.__eq__(other, attrs)

    # .........................................................................
    def __repr__(self):

        prefix = type(self).__name__ + ': '
        data = self.uncert_data
        if isinstance(data, Quantity):
            data = data.magnitude
        body = np.array2string( \
                data.squeeze(), separator=', ', \
                prefix=prefix)  # this allow indentation of len of the prefix

        units = ' {:~K}'.format(self.units) if self.has_units \
            else ' unitless'
        return ''.join([prefix, body, units])

    # .........................................................................
    def __str__(self):
        return self._str()


    # -------------------------------------------------------------------------
    # Properties / validators
    # -------------------------------------------------------------------------

    # .........................................................................
    @property
    def coordset(self):
        return None

    # .........................................................................
    @coordset.setter
    def coordset(self, values):
        raise NotImplementedError('Should be implemented in a subclass')

    # .........................................................................
    @validate('_data')
    def _data_validate(self, proposal):
        pv = proposal['value']
        data, complex = interleave(pv)
        # if we have a 1D vector, make a 1 row matrix internally
        if data.ndim == 1:
            data = data.reshape((1,-1))
        # handle the complexity
        if not self.has_complex_dims or len(self._is_complex) != data.ndim:
            # init the _is_complex list
            self._is_complex = [False] * data.ndim
        if data.ndim > 0:
            self._is_complex[-1] |= complex
        if self._copy:
            return data.copy()
        else:
            return data

    # .........................................................................
    @property
    def data(self):
        """:class:`~numpy.ndarray`-like object - The array data
        contained in this object.

        """
        if self.size == 1:
            if self.has_complex_dims:
                # only a single complex
                return interleaved2complex(self._data).squeeze()[()]
            else:
                return self._data.squeeze()[()]

        return self._data.squeeze()

    # .........................................................................
    @data.setter
    def data(self, data):
        # property.setter for data

        if data is None:
            self._data = np.array([[]]).astype(float)  # reinit data

        elif isinstance(data, NDArray):
            log.debug(
                    "init data with data from another NDArray or NDArray subclass")
            # No need to check the validity of the data
            # because the data must have been already
            # successfully initialized for the passed NDArray.data

            for attr in self.__dir__():
                val = getattr(data, "_%s" % attr)
                if self._copy:
                    val = copy.deepcopy(val)
                setattr(self, "_%s" % attr, val)

            if self._copy:
                self._name = "copy of {}".format(data._name)
                self._date = data._date

        elif isinstance(data, NDFrame):  # pandas object
            log.debug("init data with data from pandas NDFrame object")
            self._data = data.values
            self.coordset = data.axes

        elif isinstance(data, Index):  # pandas index object
            log.debug("init data with data from a pandas Index")
            self._data = data.values
            self._title = data.name

        elif isinstance(data, Quantity):
            log.debug("init data with data from a Quantity object")
            self._data_passed_is_quantity = True
            self._data = np.array(data.magnitude, subok=True,
                                  copy=self._copy)
            self._units = data.units

        elif hasattr(data, 'mask'):  # an object with data and mask attributes
            log.debug("init mask from the passed data")
            self._data = np.array(data.data, subok=True,
                                  copy=self._copy)
            if isinstance(data.mask, np.ndarray) and \
                            data.mask.shape == data.data.shape:
                self.mask = np.array(data.mask, dtype=np.bool_, copy=False)

        elif (not hasattr(data, 'shape') or
                  not hasattr(data, '__getitem__') or
                  not hasattr(data, '__array_struct__')):
            log.debug("init data with a non numpy-like array object")
            # Data doesn't look like a numpy array, try converting it to
            # one. Non-numerical input are converted to an array of objects.
            self._data = np.array(data, subok=True, copy=False)

        else:
            log.debug("init data with a numpy array")
            self._data = np.array(data, subok=True, copy=self._copy)

    # .........................................................................
    @default('_date')
    def _date_default(self):
        return datetime(1970, 1, 1, 0, 0)

    # .........................................................................
    @property
    def date(self):
        """A datetime object containing date information
        about the ndarray object, for example a creation date
        or a modification date"""

        return self._date

    # .........................................................................
    @date.setter
    def date(self, date):

        if isinstance(date, datetime):
            self._date = date
        elif isinstance(date, str):
            try:
                self._date = datetime.strptime(date, "%Y/%m/%d")
            except ValueError:
                self._date = datetime.strptime(date, "%d/%m/%Y")

    # .........................................................................
    @default('_id')
    def _id_default(self):

        return str(uuid.uuid1()).split('-')[0]  # a unique id

    # .........................................................................
    @property
    def id(self):
        return self._id

    # .........................................................................
    @default('_labels')
    def _labels_default(self):
        return None

    # .........................................................................
    @property
    def labels(self):
        """:class:`~numpy.ndarray` - An array of objects of any type (but most
        generally string), with the last dimension size equal
        to that of the dimension of data.
        Note that's labelling is possible only for 1D data

        """
        if not self.is_labeled:
            self._labels = np.zeros_like(self._data).astype(object)
        return self._labels

        #
        # # Get the real labels taking account complexity
        # # internally it has the same dim as the full data shape,
        # # but a view is given here where it appears with only as single label
        # # for a complex value, not two
        #
        # _labels = self._labels.copy()
        # if self.has_complex_dims:
        #     for axis in self.iterdims:
        #         if self._is_complex[axis]:
        #             _labels = _labels.swapaxes(axis,-1)
        #             _labels = _labels[..., ::2]
        #             _labels = _labels.swapaxes(axis, -1)
        # return _labels

    # .........................................................................
    @labels.setter
    def labels(self, labels):
        # Property setter for labels

        if labels is None:
            return

        if self.ndim > 1:
            warnings.warn('We cannot set the labels for '
                                     'multidimentional data - '
                          'Thus, these labels are ignored',
                          SpectroChemPyWarning)
            return None

        # make sure labels array is of type np.ndarray
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, subok=True, copy=True).astype(object)

        if not np.any(labels):
            # no labels
            return

        if self.data is None and labels.shape[-1] != self.shape[-1]:
            raise ValueError(
                    "labels {} and data {} shape mismatch!".format(
                    labels.shape, self.shape))

        if self.has_complex_dims:
            for axis in self.iterdims:
                labels = labels.swapaxes(axis, -1)
                if self._is_complex[axis]:
                    labels = labels.repeat(2, axis=-1)
                labels = labels.swapaxes(axis, -1)

        if np.any(self._labels):
            log.info("{0} is already a labeled array.\n".format(
                    type(self).__name__) +
                     "The explicitly passed-in labels will be appended "
                     "to the current labels")
            self._labels = np.stack((self._labels, labels))
        else:
            if self._copy:
                self._labels = labels.copy()
            else:
                self._labels = labels

    # .........................................................................
    @validate('_mask')
    def _mask_validate(self, proposal):
        pv = proposal['value']
        mask = pv
        # if we have a 1D vector, make a 1 row matrix internally
        if mask.ndim == 1:
            mask = mask.reshape((1, -1))
        if self._copy:
            return mask.copy()
        else:
            return mask

    # .........................................................................
    @default('_mask')
    def _mask_default(self):
        return np.zeros(self._data.shape).astype(bool)

    # .........................................................................
    @property
    def mask(self):
        """ Mask for the data.

        The values in the mask array must be `False` where
        the data is *valid* and `True` when it is not (like Numpy
        masked arrays).

        """

        if not self.is_masked:
            self._mask = np.zeros(self._data.shape).astype(bool)

        if self._mask.size == 1:
            return self._mask.squeeze()[()]

        return self._mask.squeeze()

    # .........................................................................
    @mask.setter
    def mask(self, mask):
        # property.setter for mask

        #if mask is None:
        #    return

        # make sure mask is of type np.ndarray
        if not isinstance(mask, np.ndarray) and not isinstance(mask, bool):
            mask = np.array(mask, dtype=np.bool_)

        if not np.any(mask):
            # no mask
            return

        if not isinstance(mask, bool) and mask.shape != self.shape:
            raise ValueError(
                    "mask {} and data {} shape mismatch!".format(
                    mask.shape, self.shape))

        if self.has_complex_dims:
            for axis in self.iterdims:
                mask = mask.swapaxes(axis, -1)
                if self._is_complex[axis]:
                    mask = mask.repeat(2, axis=-1)
                mask = mask.swapaxes(axis, -1)

        if np.any(self._mask):
            # this should happen when a new mask is added to an existing one
            # mask to be combined to an existing one
            log.info("{0} is already a masked array.\n".format(
                    type(self).__name__) +
                     "The new mask will be combined with the current array's mask.")
            self._mask &= mask  # combine (is a copy!)
        else:
            if self._copy:
                self._mask = mask.copy()
            else:
                self._mask = mask

        # if we have a 1D vector, make a 1 row matrix internally
        if self._data.ndim == 1:
            self._mask = self._mask.reshape((1, -1))

    # .........................................................................
    @default('_meta')
    def _meta_default(self):
        return Meta()

    # .........................................................................
    @property
    def meta(self):
        """:class:`~spectrochempy.core.dataset.ndmeta.Meta` instance object -
        Additional metadata for this object.

        """
        return self._meta

    # .........................................................................
    @meta.setter
    def meta(self, meta):
        # property.setter for meta
        if meta is not None:
            self._meta.update(meta)

    # .........................................................................
    @property
    def name(self):
        """`str` - An user friendly name for this object (often not necessary,
        as the title may be used for the same purpose).

        If the name is not provided, the object
        will automatically create a unique name.
        For most usage, the object name needs to be unique.

        """

        return self._name

    # .........................................................................
    @name.setter
    def name(self, name):
        # property.setter for name
        if name is not None:
            self._name = name

    # .........................................................................
    @default('_title')
    def _get_title_default(self):
        return None

    # .........................................................................
    @property
    def title(self):
        """`str` - An user friendly title for this object.

        Unlike the :attr:`name`, the title doesn't need to be unique.
        When the title is provided, it can be used for labelling the object,
        e.g. axe title in a matplotlib plot.

        """
        if not self._title:
            if self._name != self._id:
                self._title = self._name
            else:
                self._title = 'untitled'
        return self._title


    # .........................................................................
    @title.setter
    def title(self, title):
        # property.setter for title
        if title is not None:
            if self._title:
                log.info("Overwriting current title")
            self._title = title

    # .........................................................................
    @validate('_uncertainty')
    def _uncertainty_validate(self, proposal):
        pv = proposal['value']
        uncertainty = pv
        # if we have a 1D vector, make a 1 row matrix internally
        if uncertainty.ndim == 1:
            uncertainty = uncertainty.reshape((1, -1))
        if self._copy:
            return uncertainty.copy()
        else:
            return uncertainty
    # .........................................................................
    @default('_uncertainty')
    def _get_uncertainty_default(self):
        return np.zeros_like(self._data).astype(float)

    # .........................................................................
    @property
    def uncertainty(self):
        """ Uncertainty (std deviation) on the data.

        """
        if not self.is_uncertain:
            self._uncertainty = np.zeros_like(self._data).astype(float)

        if self._uncertainty.size == 1:
            return self._uncertainty.squeeze()[()]

        return self._uncertainty.squeeze()

    # .........................................................................
    @uncertainty.setter
    def uncertainty(self, uncertainty):
        # property setter for uncertainty

        if uncertainty is None:
            return

        # make sure uncertainty is of type np.ndarray or bool
        if not isinstance(uncertainty, np.ndarray):
            uncertainty = np.array(uncertainty, dtype=np.float_)

        if not gt_eps(uncertainty):
            # no uncertainty
            return

        if uncertainty.shape != self.shape:
            raise ValueError(
                    "uncertainty {} and data {} shape mismatch!".format(
                            uncertainty.shape, self.shape))

        if self.has_complex_dims:
            for axis in self.iterdims:
                uncertainty = uncertainty.swapaxes(axis, -1)
                if self._is_complex[axis]:
                    uncertainty = uncertainty.repeat(2, axis=-1)
                uncertainty = uncertainty.swapaxes(axis, -1)

        if self.is_uncertain and np.any(uncertainty != self._uncertainty):
            log.info("Overwriting {} ".format(type(self).__name__) +
                     "current uncertainty with specified uncertainty")

        if self._copy:
            self._uncertainty = uncertainty.copy()
        else:
            self._uncertainty = uncertainty

        # if we have a 1D vector, make a 1 row matrix internally
        if self._data.ndim == 1:
            self._uncertainty = self._uncertainty.reshape((1,-1))


    # .........................................................................
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

    # .........................................................................
    @units.setter
    def units(self, units):

        if units is None:
            return

        if isinstance(units, str):
            units = ur.Unit(units)
        elif isinstance(units, Quantity):
            raise TypeError("Units or string representation "
                            "of unit is expected, not Quantity")

        if self.has_units and units != self._units:
            # first try to cast
            try:
                self.to(units)
            except:
                raise TypeError(
                        "Provided units does not match data units.\n "
                        "To force a change - use the to() method, "
                        "with force flag set to True")

        self._units = units

    # -------------------------------------------------------------------------
    # read-only properties / attributes
    # -------------------------------------------------------------------------

    # .........................................................................
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

    # .........................................................................
    @property
    def dtype(self):
        """`dtype`, read-only property - data type of the underlying array

        """
        if self._data is None:
            return None
        if self._is_complex and np.sum(self._is_complex) > 0:
            return np.complex  # TODO: create a hypercomplex dtype?
        else:
            return self._data.dtype

    # .........................................................................
    @property
    def has_complex_dims(self):
        """ `bool` indicating if at least one of the dimension is complex

        """
        return np.any(self._is_complex)

    # .........................................................................
    @property
    def has_units(self):
        """ `bool` indicating if the data have units

        """
        if self._units:
            if not str(self.units).strip():
                return False
            return True
        return False

    # .........................................................................
    @property
    def is_empty(self):
        """`bool`, read-only property - Whether the array is empty (size==0)
        or not.

        """
        return self._data.size == 0

    # .........................................................................
    @default('_is_complex')
    def _get_is_complex_default(self):
        return [False]*self._data.ndim

    # .........................................................................
    @property
    def is_complex(self):
        """`tuple` of `bool` - Indicate if which dimension is complex.

        If a dimension is complex, real and imaginary part are interlaced
        in the `data` array.

        """
        # if not np.any(self._is_complex):
        #    return None

        isc = []
        for i, size in enumerate(self._data.shape):
            if size > 1:
                isc.append(self._is_complex[i])

        return isc

    # .........................................................................
    @property
    def is_labeled(self):
        """`bool`, read-only property - Whether the axis has labels or not.

        """
        # label cannot exists (for now for nD dataset - only 1D dataset, such
        # as Coord can be labelled.
        if self.ndim > 1:
            return False

        if self._labels is not None and np.any(self._labels != ''):
            return True
        else:
            return False

    # .........................................................................
    @property
    def is_masked(self):
        """`bool`, read-only property - Whether the array is masked or not.

        """
        if self._mask is not None and np.any(self._mask):
            return True
        else:
            return False

    # .........................................................................
    @property
    def is_uncertain(self):
        """`bool`, read-only property - Whether the array has uncertainty
        or not.

        """
        if self._uncertainty is not None and gt_eps(self._uncertainty):
            return True
        else:
            return False

    # .........................................................................
    @property
    def iterdims(self):
        return list(range(self.ndim))

    # .........................................................................
    @property
    def masked_data(self):
        """:class:`~numpy.ndarray`-like object - The actual masked array of data
        contained in this object.

        """
        #if self.is_masked:
        return self._umasked(self._data, self._mask)
        #else:
        #    return self._data
            # here we use .mask not ._mask to get the correct shape

    # .........................................................................
    @property
    def ndim(self):
        """`int`, read-only property - The number of dimensions of
        the dataset.

        """

        return self._data.squeeze().ndim

    # .........................................................................
    @property
    def shape(self):
        """`tuple`, read-only property - A `tuple` with the size of each axis.

        i.e., the number of data element on each axis (possibly complex).

        """
        if self._data is None:
            return ()
        shape = list(self._data.squeeze().shape)
        if self.has_complex_dims:
            for i in self.iterdims:
                if self.is_complex[i]:
                    shape[i] = shape[i] // 2

        return tuple(shape)

    # .........................................................................
    @property
    def size(self):
        """`int`, read-only property - Size of the underlying `ndarray`.

        i.e., the total number of data element
        (possibly complex or hyper-complex in the array).

        """
        if self._data is None or self._data.size == 0:
            # no data but it be that labels are present
            if np.any(self._labels):
                return self._labels.size
            return 0
        size = self._data.size
        if self.has_complex_dims:
            for complex in self._is_complex:
                if complex:
                    size = size // 2
        return size

    # .........................................................................
    @property
    def uncert_data(self):
        """:class:`~numpy.ndarray`-like object - The actual array with
                uncertainty of the data contained in this object.
        """
        return self._uncert_data()

    def _uncert_data(self, force=False):
        # private function that allow to force the masked and uncertainty
        # representation. Useful for slicing

        return self._uarray(self._umasked(self._data, self._mask, force= force),
                            self._uncertainty, self._units, force = force)

    # .........................................................................
    @property
    def unitless(self):
        """`bool`, read-only property - Whether the array has `units` or not.

        """

        return not self.has_units

    # .........................................................................
    @property
    def real(self):
        """:class:`~numpy.ndarray`-like object - The array with
        real part of the data contained in this object.
        """
        if not self._is_complex[-1]:
            return self.copy()

        new = self.copy()
        new._is_complex[-1] = False
        ma = self.masked_data
        ma = ma[..., ::2]
        if isinstance(ma, np.ma.masked_array):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    # .........................................................................
    @property
    def imag(self):
        """:class:`~numpy.ndarray`-like object - The array with
        imaginary part of the data contained in this object.
        """

        new = self.copy()
        new._is_complex[-1] = False
        ma = self.masked_data
        ma = ma[..., 1::2]
        if isinstance(ma, np.ma.masked_array):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    # .........................................................................
    @property
    def RR(self):
        """:class:`~numpy.ndarray`-like object - The array with
        real part in both dimension of hypercomplex 2D data contained in this object.
                """

        if self.ndim != 2:
            raise ValueError('Not a two dimensional array')
        return self.part('RR')

    # .........................................................................
    @property
    def RI(self):
        """:class:`~numpy.ndarray`-like object - The array with
        real-imaginary part of hypercomplex 2D data contained in this object.
        """

        if self.ndim != 2:
            raise ValueError('Not a two dimensional array')
        return self.part('RI')

    # .........................................................................
    @property
    def IR(self):
        """:class:`~numpy.ndarray`-like object - The array with
        imaginary-real part of hypercomplex 2D data contained in this object.
        """

        if self.ndim != 2:
            raise ValueError('Not a two dimensional array')
        return self.part('IR')

    # .........................................................................
    @property
    def II(self):
        """:class:`~numpy.ndarray`-like object - The array with
        imaginary-imaginary part of hypercomplex 2D data contained in this object.
        """

        if self.ndim != 2:
            raise ValueError('Not a two dimensional array')
        return self.part('II')

    # .........................................................................
    @property
    def values(self):
        """:class:`~numpy.ndarray`-like object - The actual values (data, units,
        + uncertainties) contained in this object.

        """
        return self._uarray(self._data, self._uncertainty, self._units)

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    # .........................................................................
    def conj(self, axis=-1, inplace=False):
        """
        Return the conjugate of the NDDataset in the specified dimension

        Parameters
        ----------
        axis : `int`, Optional, default: -1.
            The axis along which the absolute value should be calculated.

        inplace : `bool`, optional, default=``False``
            should we return a new dataset (default) or not (inplace=True)

        Returns
        -------
        array : same type

            Output array.

        See Also
        --------
        :meth:`real`, :meth:`imag`

        """
        if not inplace:  # default is to return a new array
            new = self.copy()
        else:
            new = self  # work inplace

        if new._is_complex[axis]:
            new.swapaxes(axis, -1, inplace=True)
            new._data[..., 1::2] = - new._data[..., 1::2]
            new.swapaxes(axis, -1, inplace=True)

        return new

    conjugate = conj

    # .........................................................................
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

        Examples
        --------
        >>> nd1 = NDArray([1.+2.j,2.+ 3.j])
        >>> print(nd1)
        R[   1.000    2.000]
        I[   2.000    3.000]
        >>> nd2 = nd1
        >>> nd2 is nd1
        True
        >>> nd3 = nd1.copy()
        >>> nd3 is not nd1
        True

        """
        if deep:
            do_copy = copy.deepcopy
        else:
            do_copy = copy.copy

        new = type(self)()
        # new._data = do_copy(self._data)
        for attr in self.__dir__():
            try:
                setattr(new, "_" + attr, do_copy(getattr(self, "_" + attr)))
            except:
                # ensure that if deepcopy do not work, a shadow copy can be done
                setattr(new, "_" + attr, copy.copy(getattr(self, "_" + attr)))
        new._name = str(uuid.uuid1()).split('-')[0]
        new._date = datetime.now()
        return new

    # .........................................................................
    def is_units_compatible(self, other):
        """
        Check the compatibility of units with another NDArray

        Parameters
        ----------
        other : NDArray

        Returns
        -------
        compat : `bool`

        >>> nd1 = NDArray([1.+2.j,2.+ 3.j], units='meters')
        >>> print(nd1)
        R[   1.000    2.000] m
        I[   2.000    3.000] m
        >>> nd2 = NDArray([1.+2.j,2.+ 3.j], units='seconds')
        >>> nd1.is_units_compatible(nd2)
        False
        >>> nd1.to('minutes', force=True)
        >>> nd1.is_units_compatible(nd2)
        True
        >>> nd2[0].data == nd1[0].data
        True

        """

        try:
            other.to(self.units, inplace=False)
        except:
            return False

        return True

    # .........................................................................
    @deprecated('use ``to`` instead')
    def ito(self, other):
        """Inplace scaling of the current object data to different units.

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

    # .........................................................................
    def part(self, select='ALL'):
        """
        Take selected components of an hypercomplex array (RRR, RIR, ...)

        Parameters
        ----------
        select : `str`, optional, default: 'ALL'
            if 'ALL', only real part in all dimensions will be selected.
            ELse a string must specify wich real (R) or imaginary (I) component
            has to be selected along a specific axis. For instance,
            a string such as 'RRI' for a 2D hypercomplex array indicated
            that we take the real component in each dimension except the last
            one, for wich imaginary component is preferred. A star (*)
            indicates that we keep the complex structure along a given
            axis, e.g. R*R

        Returns
        -------
        array : same type

        """
        new = self.copy()
        if select == 'ALL':
            select = 'R' * self.ndim
        ma = self._uncert_data(force=True) #self.masked_data
        for axis, component in enumerate(select):
            if self._is_complex[axis]:
                data = ma.swapaxes(axis, -1)
                if component == 'R':
                    data = data[..., ::2]
                elif component == 'I':
                    data = data[..., 1::2]
                elif component != '*':
                    raise ValueError(
                            'components must be indicated with R, I or *')
                ma = data.swapaxes(axis, -1)
                new._is_complex[axis] = False
        #if isinstance(ma, np.ma.masked_array):
        #    new._data = ma.data
        #    new._mask = ma.mask
        #else:
        #    new._data = ma
        new._mask = ma.mask

        new._uncertainty = unp.std_devs(ma)
        new._data = unp.nominal_values(ma)

        return new

    # .........................................................................
    def set_complex(self, axis):
        """
        Set the data along the given dimension as complex

        Parameters
        ----------
        axis :  `int`
            The axis along which the data must be considered complex.
            The R and I part of the data are supposed to be interlaced, and so
            the effective size of the array along this axis must be even
            or an error will be raised

        """
        if axis < 0:
            axis = self.ndim + axis

        if self._is_complex and self._is_complex[axis]:
            # not necessary in this case, it is already complex
            return

        if self.shape[axis] % 2 != 0:
            raise ValueError("Size of complex indirect dimension "
                             "must be even, not %d" % self.shape[axis])
        self._is_complex[axis] = True

    # .........................................................................
    def set_mask(self, items, axis=-1):
        """
        Mask a given region

        Parameters
        ----------

        start : `ìnt`, `float` `slice` , `str` or any keys allowing slicing a ndarray

            Specify the region or the start of the region
            of the ndarray to be masked

        stop : `ìnt`, `float`, `str` or any keys allowing slicing a ndarray

            Specify the end region of the ndarray to be masked

        """

        try:
            s = self[start:stop:step]
        except IndexError:
            warnings.warn('Specified indexes raised an error. Mask not set',
                          SpectroChemPyWarning)

        pass




    # .........................................................................
    def set_real(self, axis):
        """
        Set the data along the given dimension as real

        Parameters
        ----------
        axis :  `int`
            The axis along which the data must be considered real.
        """
        if axis < 0:
            axis = self.ndim + axis

        self._is_complex[axis] = False

    # # .........................................................................
    # def _old_squeeze(self, axis=None, inplace=False):
    #     """
    #     Remove single-dimensional entries from the shape of an array.
    #
    #     Parameters
    #     ----------
    #     axis :   `None` or `int` or `tuple` of ints, optional
    #
    #         Selects a subset of the single-dimensional entries in the shape.
    #         If an axis is selected with shape entry greater than one,
    #         an error is raised.
    #
    #     inplace : `bool`, optional, default = False
    #
    #         if False a new object is returned
    #
    #     Returns
    #     -------
    #     squeezed_dataset : same type
    #
    #         The input array, but with all or a subset of the dimensions
    #         of length 1 removed.
    #
    #     """
    #
    #     if axis is not None:
    #         if not is_sequence(axis):
    #             axis = [axis]
    #         squeeze_axis = list(axis)
    #
    #         for axis in squeeze_axis:
    #             if axis < 0:
    #                 axis = self._data.ndim + axis
    #
    #             if self._data.shape[axis] > 1:
    #                 raise ValueError(
    #                         '%d is of length greater than one: '
    #                         'cannot be squeezed' % axis)
    #     else:
    #         squeeze_axis = []
    #         for axis, dim in enumerate(self._data.shape):
    #             # we need the real shape here
    #             if dim == 1:
    #                 squeeze_axis.append(axis)
    #
    #     if not inplace:
    #         new = self.copy()
    #     else:
    #         new = self
    #
    #     new._data = self._data.squeeze(tuple(squeeze_axis))
    #
    #     if self.is_masked:
    #         new._mask = self._mask.squeeze(tuple(squeeze_axis))
    #     if self.is_uncertain:
    #         new._uncertainty = self._uncertainty.squeeze(tuple(squeeze_axis))
    #     if self.is_labeled:
    #         # labels is a special case, as several row of label can be presents.
    #         squeeze_labels_axis = []
    #         for axis, dim in enumerate(self._labels.shape):
    #             if dim == 1:
    #                 squeeze_labels_axis.append(axis)
    #         new._labels = self._labels.squeeze(tuple(squeeze_labels_axis))
    #
    #     cplx = []
    #     coordset = []
    #     for axis in range(self._data.ndim):
    #         if axis not in squeeze_axis:
    #             cplx.append(self._is_complex[axis])
    #             if self.coordset:
    #                 coordset.append(self.coordset[axis])
    #
    #     new._is_complex = cplx
    #     if coordset:
    #         new._coordset = CoordSet(coordset)
    #
    #     #if new.ndim ==1 and new.size > 1 :
    #     #    new.data = new.data.reshape((1,self._data.size))
    #
    #     return new

    # .........................................................................
    def swapaxes(self, axis1, axis2, inplace=False):
        """
        Interchange two dims of a NDArray.

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
        swapped : same type

            The object or a new object (inplace=False) is returned with all
            components swapped

        See Also
        --------
        :meth:`transpose`

        """

        if not inplace:
            new = self.copy()
        else:
            new = self

        if self.ndim < 2:  # cannot swap axe for 1D data
            return new

        if axis1 == -1:
            axis1 = self.ndim - 1

        if axis2 == -1:
            axis2 = self.ndim - 1

        new._data = np.swapaxes(new._data, axis1, axis2)

        # all other arrays, except labels have also to be swapped to reflect
        # changes of data ordering.
        # labels are presents only for 1D array, so we do not have to swap them

        if self.is_masked:
            new._mask = np.swapaxes(new._mask, axis1, axis2)

        if new.is_uncertain:
            new._uncertainty = np.swapaxes(new._uncertainty, axis1, axis2)

        # we need also to swap the is_complex list, as well has the metadata

        if self.has_complex_dims:
            new._is_complex[axis1], new._is_complex[axis2] = \
                self._is_complex[axis2], self._is_complex[axis1]

        new._meta = new._meta.swapaxes(axis1, axis2, inplace=False)

        return new

    # .........................................................................
    def to(self, other, inplace=True, force=False):
        """Return the object with data rescaled to different units.

        Parameters
        ----------

        other : :class:`Quantity` or `str`.

            destination units.

        inplace : `bool`, optional, default = `True`.

            if inplace is True, the object itself is returned with
            the new units. If `False` a copy is created.

        force: `bool`, optional, default: False

            If True the change of units is forced, even for imcompatible units

        Returns
        -------

        object : same type

            same object or a copy depending on `ìnplace` with new units.

        Examples
        --------

        >>> np.random.seed(12345)
        >>> ndd = NDArray( data = np.random.random((3, 3)),
        ...                mask = [[True, False, False],
        ...                        [False, True, False],
        ...                        [False, False, True]],
        ...                units = 'meters')
        >>> print(ndd)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[  --    0.316    0.184]
        [   0.205   --    0.596]
        [   0.965    0.653   --]] m

        We want to change the units to seconds for instance
        but there is no relation with meters,
        so an error is generated during the change

        >>> ndd.to('second')
        Traceback (most recent call last):
        ...
        Exception: Cannot set this unit

        However, we can force the change
        >>> ndd.to('second', force=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        >>> print(ndd)
        [[  --    0.316    0.184]
        [   0.205   --    0.596]
        [   0.965    0.653   --]] s

        """
        if inplace:
            new = self
        else:
            new = self.copy()

        if other is None:
            units = None
            if self.units is None:
                return new
        elif isinstance(other, str):
            units = ur.Unit(other)
        elif hasattr(other, 'units'):
            units = other.units
        else:
            units = ur.Unit(other)

        if self.has_units:
            try:
                q = Quantity(1., self._units).to(units)
                scale = q.magnitude
                new._data = new._data * scale  # new * scale #
                if new._uncertainty is not None:
                    new._uncertainty = new._uncertainty * scale
                new._units = q.units

            except DimensionalityError as exc:
                if force:
                    new._units = units
                    log.info('units forced to change')
                else:
                    raise exc

        else:
            if force:
                new._units = units
            else:
                warnings.warn("There is no units for this NDArray!",
                              SpectroChemPyWarning)

        # if not inplace:
        return new

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    # .........................................................................
    def _str(self, sep='\n', ufmt=' {:~K}'):

        prefix = ['']
        if self.has_complex_dims:
            for axis in self.iterdims:
                for item in prefix[:]:  # work on a copy of prefix as it will
                    # change during execution of this loop
                    prefix.remove(item)
                    prefix.append(item + 'R')
                    if self.is_complex[axis]:
                        prefix.append(item + 'I')

        units = ufmt.format(self.units) if self.has_units else ''

        def mkbody(d, pref, units):
            body = np.array2string(
                    d.squeeze(), separator=' ',
                    prefix=pref)
            body = body.replace('\n',sep)
            text = ''.join([pref, body, units])
            text += sep
            return text

        if 'I' not in ''.join(
                prefix):  # case of pure real data (not hypercomplex)
            data = self.uncert_data
            if isinstance(data, Quantity):
                data = data.magnitude
            text = mkbody(data, '', units)
        else:
            text = ''
            for pref in prefix:
                data = self.part(pref).uncert_data
                if isinstance(data, Quantity):
                    data = data.magnitude
                text += mkbody(data, pref, units)

        text = text[:-1]  # remove the trailing '\n'
        return text

    # .........................................................................
    def _repr_html_(self):

        prefix = ['']
        sep = '<br/>'
        ufmt = ' {:~H}'

        return self._str(sep=sep, ufmt=ufmt)

    # .........................................................................
    def _argsort(self, by='value', pos=None, descend=False):
        # found the indices sorted by values or labels

        if by == 'value':
            args = np.argsort(self.data)

        elif 'label' in by and not self.is_labeled:
            by = 'value'
            pos = None
            warnings.warn('no label to sort, use ``value`` by default',
                              SpectroChemPyWarning)
            args = np.argsort(self.data)

        elif 'label' in by and self.is_labeled:
            labels = self._labels
            if len(self._labels.shape) > 1:
                # multidimentional labels
                if not pos:
                    pos = 0
                    #try to find a pos in the by string
                    pattern = re.compile("label\[(\d)\]")
                    p = pattern.search(by)
                    if p is not None:
                        pos = int(p[1])
                labels = self._labels[pos]  # TODO: this must be checked
            args = np.argsort(labels)

        else:
            by = 'value'
            warnings.warn(
                    'parameter `by` should be set to `value` or `label`, '
                    'use ``value`` by default',
                              SpectroChemPyWarning)
            args = np.argsort(self.data)

        if descend:
            args = args[::-1]

        return args

    # .........................................................................
    def _loc2index(self, loc, axis):
        # Return the index of a location along the axis
        # Not implemented in this base class

        raise NotImplementedError('not implemented for {} objects'.format(
                type(self).__name__))

    # .........................................................................
    def _get_slice(self, key, axis, iscomplex=False):

        if not isinstance(key, slice):
            start = key
            if not isinstance(key, (int, np.int)):
                start = self._loc2index(key, axis)
            else:
                if key < 0:  # reverse indexing
                    start = self.shape[axis] + key
            stop = start + 1
            step = 1
        else:
            start, stop, step = key.start, key.stop, key.step

            if start is not None and not isinstance(start, (int, np.int_)):
                start = self._loc2index(start, axis)

            if stop is not None and not isinstance(stop, (int, np.int_)):
                stop = self._loc2index(stop, axis)
                if stop < start: # and self.coordset[axis].is_reversed:
                    start, stop = stop, start
                stop = stop + 1

            if step is not None and not isinstance(step, (int, np.int_)):
                raise NotImplementedError('step in location slicing is not yet possible.')
                # TODO: we have may be a special case with datetime
                step = 1

        if step is None:
            step = 1

        keys = slice(start, stop, step)

        if iscomplex:
            if start is not None:
                start = start * 2
            if stop is not None:
                stop = stop * 2

        internkeys = slice(start, stop, step)

        return keys, internkeys

    # .........................................................................
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

        #
        if self._data.ndim != self.ndim:
            # case or 1D spectra or of array with complex dimensions
            # this need some attention to have a correct slicing
            # because, the user is not aware of the internal representation
            newkeys=[]
            i=0
            for size in self._data.shape:
                # loop on the real shape, and make the keys correspondind to
                # dimension which are not of lenght one.
                # The other have to be replaced by a slice(None) or 0
                if size == 1:
                    newkeys.append(slice(None))
                else:
                    if not keys:
                        #list of keys already completely used
                        newkeys.append(slice(None))
                    else:
                        newkeys.append(keys.pop(0))
            # TODO: check for complex data !!!!
            keys = newkeys[:]
        else:
            # pad the list with additional dimensions
            for i in range(len(keys), self.ndim):
                keys.append(slice(None))

        # replace all keys by index slices (and get internal slice index for
        # complex array)

        internkeys = keys[:]

        for axis, key in enumerate(keys):
            complex = self._is_complex[axis]

            keys[axis], internkeys[axis] = self._get_slice(key, axis, complex)

        return tuple(keys), tuple(internkeys)

    # .........................................................................
    def _sort(self, by='value', pos=None, descend=False, inplace=False):
        # sort an ndarray in place using data or label values

        if not inplace:
            new = self.copy()
        else:
            new = self

        args = self._argsort(by, pos, descend)

        return self._take(args)

    # .........................................................................
    def _take(self, indices):
        # get a ndarray with passed indices

        new = self.copy()
        if new._data.size > 0:
            new._data = new._data[..., indices]
        if new.is_labeled:
            new._labels = new._labels[..., indices]
        if new.is_masked:
            new._mask = new._mask[..., indices]

        return new

    # .........................................................................
    @staticmethod
    def _umasked(data, mask, force=False):
        # This ensures that a masked array is returned.

        if np.any(mask) or force:
            if not np.any(mask):
                mask = np.zeros_like(data).astype(bool)
            data = np.ma.masked_array(data, mask)

        return data

    # .........................................................................
    @staticmethod
    def _uarray(data, uncertainty, units=None, force=False):
        # return the array with uncertainty and units if any

        if (uncertainty is None or not gt_eps(uncertainty)) and not force:
            uar = data
        else:
            try:
                if not np.any(uncertainty):
                    uncertainty = np.zeros_like(data).astype(float)
                uar = unp.uarray(data, uncertainty)
            except TypeError:
                uar = data


        if units:
            return Quantity(uar, units)
        else:
            return uar


# =============================================================================
# CoordSet
# =============================================================================

class CoordSet(HasTraits):
    """A collection of Coord objects for a NDArray object
     with a validation method.

    Parameters
    ----------
    coords : NDarray or NDArray subclass objects.

       Any instance of a NDArray can be accepted as coordinates for a
       given dimension.

       If an instance of CoordSet is found, instead, this means that all
       coordinates in this set describe the same axis

    is_same_dim : bool, optional, default=False

        if true, all elements of coords describes a single dimension.
        By default, this is false, which means that each item describes
        a different dimension.

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
        _copy = kwargs.pop('copy', False)

        super(CoordSet, self).__init__(**kwargs)

        self._coords = []

        if all([isinstance(coords[i], (NDArray, CoordSet))
                for i in range(len(coords))]):
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

                if not isinstance(item, (NDArray, CoordSet)):
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
                    raise ValueError(
                                        'Coordinates must be of the same size '
                                        'for a dimension with multiple '
                                        'coordinates')

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    # .........................................................................
    @property
    def name(self):
        return self._name

    # .........................................................................
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

    # .........................................................................
    @property
    def titles(self):
        """`list` - Get/Set a list of axis titles.

        """
        _titles = []
        for item in self._coords:
            if isinstance(item, NDArray):
                _titles.append(item.title if item.title else item.name)
            elif isinstance(item, CoordSet):
                _titles.append([el.title if el.title else el.name
                                for el in item])
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
        """`list` - Get/Set a list of axis labels.

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
        """`list` - Get/Set a list of axis units.

        """
        return [item.units for item in self._coords]

    # .........................................................................
    @units.setter
    def units(self, value):
        if is_sequence(value):
            for i, item in enumerate(value):
                self._coords[i].units = item

    # .........................................................................
    @property
    def isempty(self):
        """`bool`, read-only property - `True` if there is no coords defined.

        """
        return len(self._coords) == 0

    # .........................................................................
    @property
    def is_same_dim(self):
        """`bool`, read-only property -
        `True` if the coords define a single dimension.

        """
        return self._is_same_dim

    # .........................................................................
    @property
    def sizes(self):
        """`int`, read-only property -
        gives the size of the axis or coords for each dimention"""
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
        """:class:`~numpy.ndarray`-like object - A list of the Coord object
        present in this coordset

        """
        return self._coords

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------

    # .........................................................................
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
        out = ("CoordSet object <" + ', '.join(['<object {}>']
                                               * len(self._coords)) + ">")
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
        return not self.__eq__(
                other)
