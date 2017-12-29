# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module implements the base |NDArray| class.

"""

__all__ = ['NDArray']

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
from numpy.ma.core import MaskedConstant, masked, nomask
from traitlets import (List, Unicode, Instance, Bool, Union, Any, Float,
                       HasTraits, default, validate)
from pandas.core.generic import NDFrame, Index

# =============================================================================
# local imports
# =============================================================================

from ..utils.meta import Meta
from ..units import Unit, ur, Quantity
from ..application import log
from ..utils import (EPSILON, INPLACE, StdDev, is_sequence,
                                 numpyprintoptions, interleaved2complex,
                                 interleave, SpectroChemPyWarning, docstrings,
                                 make_func_from)
from ..extern.traittypes import Array
from ..extern.pint.errors import (DimensionalityError)
from ..extern.uncertainties import unumpy as unp


# =============================================================================
# Some initializations
# =============================================================================

# set options for printing data (numpy arrays) contained in a NDArray.
numpyprintoptions()  # set up the numpy print format


# ============================================================================
# Functions
# ============================================================================

gt_eps = lambda arr: np.any(arr > EPSILON)
"""lambda function to check that an array has at least some values
 greater than epsilon """


# =============================================================================
# The basic NDArray class
# =============================================================================

class NDArray(HasTraits):
    """
    The basic |NDArray| object.

    The |NDArray| class is an array
    (numpy |ndarray|-like) container, usually not intended to be used directly,
    as its basic functionalities may be quite limited, but to be subclassed.

    Indeed, both the class |Coord| which handles the coordinates in a given
    dimension and the class |NDDataset| which implements a full dataset (with
    coordinates) are derived from |NDArray| in |scpy|.


    The key distinction from raw numpy \|ndarrays| is the presence of
    optional properties such as labels, masks, uncertainties,
    units and/or extensible metadata dictionary.

    This is a base class in |scpy| on which for instance, the |Coord|  and
    |NDDataset| are builded. This class is normaly intended to be subclassed,
    but it can also be used at such.

    """

    _data = Array(Float(), allow_none=True)
    _coordset = Instance(List, allow_none=True)
    _mask = Union((Array(Bool()), Instance(MaskedConstant)))
    _uncertainty = Array(Float(), allow_none=True)
    _labels = Array(Any(), allow_none=True)
    _units = Instance(Unit, allow_none=True)
    _is_complex = List(Bool(), allow_none=True)

    _date = Instance(datetime)

    _meta = Instance(Meta, allow_none=True)
    _title = Unicode(allow_none=True)

    _id = Unicode()
    _name = Unicode(allow_none=True)
    _copy = Bool()

    _labels_allowed = Bool(True)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    # .........................................................................
    @docstrings.get_sectionsf('NDArray')
    @docstrings.dedent
    def __init__(self, data=None, **kwargs):
        """
        Parameters
        ----------
        data : array of floats.
            Data array contained in the object.
            The data can be a list, a tuple,
            a |ndarray|, a |ndarray|-like,
            a |NDArray| or any subclass of |NDArray|. Any size or shape of data
            is accepted. If not given, an empty |NDArray| will be inited.
            At the initialisation the provided data will casted to a
            numpy-ndarray. If a subclass of |NDArray| is passed that contains
            already mask, labels,units or uncertainties, these elements will be
            used to accordingly set those of the created object.
            If possible, the provided data  will not be copied for `data`
            input,
            but will be passed by reference, so you should make a copy of the
            `data` before passing them if that's the desired behavior or
            set the `copy` argument to True.
        labels : array of objects, optional
            Labels for the `data`. labels can be used only for 1D-datasets.
            The labels array may have an
            additional dimension, meaning several series of labels for the same
            data. The given array can be a list,
            a tuple, a `ndarray`, a
            `ndarray`-like, a `NDArray`
            or any subclass of `NDArray`.
        mask : array of bool or `nomask`, optional
            Mask for the data. The mask array must have the same shape as
            the data. The given array can be a list, a tuple,
            a |ndarray|. Each values in the array must be `False`
            where the data is *valid* and True when it is not (like in numpy
            masked arrays). If `data` is already a
            :class:`~numpy.ma.MaskedArray`, or any array object
            (such as a |NDArray| or subclass of it), providing a `mask` here
            will causes the mask from the masked array to be ignored.
        units : |Unit| instance or str, optional
            Units of the data. If data is a |Quantity|
            then `units` is set to the unit of the `data`; if a unit is also
            explicitly provided an error is raised. Handling of units use
            a fork of the `pint <https://pint.readthedocs.org/en/0.6>`_
            package (BSD Licence) which is embedded in |scpy|)
        uncertainty : array of floats, optional
            Standard deviation on the `data`. An array giving the uncertainty
            on each values of the data array.
            The array must have the same shape as the data. The given array
            can be a list, a tuple, a |ndarray|.
            Handling of uncertainty use a fork of the `uncertainties
            <http://pythonhosted.org/uncertainties/>`_ package (BSD Licence)
            which is embedded in |scpy|.
        title : str, optional
            The title of the axis. It will later be used for instance
            for labelling plots of the data. It is optional but recommended to
            give a title to each ndarray.
        name : str, optional
            The name of the ndarray. Default is set automatically.
            It must be unique.
        meta : dict-like object, optional.
            Additional metadata for this object. Must be dict-like but no
            further restriction is placed on meta.
        copy : bool, optional
            Perform a copy of the passed object.


        Examples
        --------
        Empty initialization

        >>> import spectrochempy as scp # doctest: +ELLIPSIS
        SpectroChemPy's API ...
        >>> ndd = scp.NDArray()

        Initialization with a ndarray

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

        NDArray can be also created using keywords arguments.
        Here is a masked array, with units:

        >>> ndd = NDArray( data = np.random.random((3, 3)),
        ...                mask = [[True, False, True],
        ...                        [False, True, False],
        ...                        [False, False, True]],
        ...                units = 'absorbance')
        >>> print(ndd)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[  --    0.295   --]
        [   0.086   --    0.516]
        [   0.689    0.857   --]] a.u.


        """
        self._copy = kwargs.pop('copy', False)  # by default
        # we try to keep a reference to the data, not copy them

        self._is_complex = kwargs.pop('is_complex', None)

        self.data = data

        self.title = kwargs.pop('title', None)

        # a unique id / name
        self.name = kwargs.pop('name', "{}_{}".format(type(self).__name__,
                                                      self.id.split('-')[0]))

        self.mask = kwargs.pop('mask', nomask)

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


    def implements(self, name=None):
        # For compatibility with pyqtgraph
        # Rather than isinstance(obj, NDDataset) use object.implements(
        # 'NDDataset')
        # This is useful to check type without importing the module
        if name is None:
            return ['NDArray']
        else:
            return name == 'NDArray'


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
                    # log.debug("attributes '{}' are not equals "
                    #           "or one is missing: "
                    #           "{}, {}".format(attr,
                    #                           getattr(self,
                    #                                   "_%s" % attr),
                    #                           getattr(other,
                    #                                   "_%s" % attr)))
                    return False
        return eq

    # .........................................................................
    def __getitem__(self, items):

        inplace = False
        if isinstance(items,tuple) and items[-1]==INPLACE:
            items=list(items)[:-1]
            inplace = True

        if inplace:
            new = self
        else:
            new = self.copy()


        # The actual index depends on the complexity of the dimension
        keys, internkeys = self._make_index(items)

        # slicing by index of all internal array
        udata = new._uncert_data[internkeys]
        if new.is_uncertain:
            new._data = unp.nominal_values(np.asarray(udata))
        else:
            new._data = np.asarray(udata)

        if self.is_labeled:
            # case only of 1D dataset such as Coord
            # we add Ellipsis as labels can be multidimensional
            # (multilabels)
            newkeys = tuple((Ellipsis, keys[-1]))
            new._labels = np.array(self._labels[newkeys])

        if new._data.size == 0:
            if not new.is_labeled or new._labels.size == 0:
                raise IndexError("Empty array of shape {}".format(
                    str(new._data.shape)) + \
                                 "resulted from slicing.\n"
                                 "Check the indexes and make "
                                 "sure to use floats for "
                                 "location slicing")

        new._is_complex = self._is_complex

        if hasattr(udata, 'mask'):
            new._mask = udata.mask
        else:
            new._mask = nomask

        if new.is_uncertain:
            new._uncertainty = unp.std_devs(np.asarray(udata))
        else:
            new._uncertainty = None

        if self._coordset is not None:
            new_coordset = self.coordset.copy()
            for i, coord in enumerate(new_coordset):
                new_coordset[i] = coord[keys[i]]
            new._coordset = new_coordset

        # this is a modified dataset
        new._name = '*' + self._name.lstrip('*')

        return new

    # .........................................................................
    def __setitem__(self, items, value):

        # TODO: this may not work for complex data in other dimensions than the
        # last
        keys, internkeys = self._make_index(items)
        if self.ndim == 1:
            keys = keys[-1]
        if isinstance(value, (bool, np.bool_, MaskedConstant)):
            # the mask is modified, not the data
            if value is masked:
                value = True
            if not np.any(self._mask):
                self._mask = np.zeros_like(self._data).astype(np.bool_)
            self._mask[internkeys] = value
        elif isinstance(value, StdDev):
            # the uncertainties are modified
            self._uncertainty[internkeys] = value.data
        else:
            if self.ndim > 1 and np.any(self.is_complex[:-1]):
                raise NotImplementedError("Sorry but setting values for"
                                          "hypercomplex array "
                                          "is not yet possible")
            self.data[keys] = value

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
        print_unit=True
        if not (self.is_empty and self.is_labeled):
            data = self.uncert_data
        else:
            data = self.get_labels()
            print_unit=False

        if isinstance(data, Quantity):
            data = data.magnitude

        body = np.array2string( data.squeeze(), separator=', ', prefix=prefix)
        # this allow indentation of len of the prefix

        units = ''
        if print_unit:
            units = ' {:~K}'.format(self.units) \
                                              if self.has_units else ' unitless'

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
        # not implemented in NDArray but in the NDDataset subclass
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
            data = data.reshape((1, -1))
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
        """
        |ndarray|, dtype:float - The `data` array.

        A special storage is used for complex or hypercomplex data.
        If a dimension is complex, real and imaginary part are interleaved
        in the `data` array.

        .. note::
            See the :ref:`userguide` for more information


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
            #log.debug(
            #    "init data with data from another NDArray or NDArray
            # subclass")
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
            #log.debug("init data with data from pandas NDFrame object")
            self._data = data.values
            self.coordset = data.axes

        elif isinstance(data, Index):  # pandas index object
            #log.debug("init data with data from a pandas Index")
            self._data = data.values
            self._title = data.name

        elif isinstance(data, Quantity):
            #log.debug("init data with data from a Quantity object")
            self._data_passed_is_quantity = True
            self._data = np.array(data.magnitude, subok=True,
                                  copy=self._copy)
            self._units = data.units

        elif hasattr(data, 'mask'):  # an object with data and mask attributes
            #log.debug("init mask from the passed data")
            self._data = np.array(data.data, subok=True,
                                  copy=self._copy)
            if isinstance(data.mask, np.ndarray) and \
                            data.mask.shape == data.data.shape:
                self.mask = np.array(data.mask, dtype=np.bool_, copy=False)

        elif (not hasattr(data, 'shape') or
                  not hasattr(data, '__getitem__') or
                  not hasattr(data, '__array_struct__')):
            #log.debug("init data with a non numpy-like array object")
            # Data doesn't look like a numpy array, try converting it to
            # one. Non-numerical input are converted to an array of objects.
            self._data = np.array(data, subok=True, copy=False)

        else:
            #log.debug("init data with a numpy array")
            self._data = np.array(data, subok=True, copy=self._copy)

    # .........................................................................
    @default('_date')
    def _date_default(self):
        return datetime(1970, 1, 1, 0, 0)

    # .........................................................................
    @property
    def date(self):
        """
        `Datetime` object - creation date

        """

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
    @default('_labels')
    def _labels_default(self):
        return None

    # .........................................................................
    @property
    def labels(self):
        """
        |ndarray|, dtype:object - Labels for `data`.

        An array of objects of any type (but most
        generally string), with the last dimension size equal
        to that of the dimension of data.
        Note that's labelling is possible only for 1D data. One classical
        application is the labelling of coordinates to display informative
        strings instead of numerical values.

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
                     "The explicitly provided labels will be appended "
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
        """
        |ndarray|, dtype:bool - Mask for the data

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

        if mask is nomask or mask is masked:
            pass

        elif isinstance(mask, (np.bool_, bool)):
            if not mask:
                mask = nomask
            else:
                mask = masked

        else:

            # from now, make sure mask is of type np.ndarray if it provided

            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=np.bool_)

            if not np.any(mask):
                # all element of the mask are false
                mask = nomask

            elif mask.shape != self.shape:
                raise ValueError(
                    "mask {} and data {} shape mismatch!".format(
                        mask.shape, self.shape))

            # mask need to be replicated on the imaginary part if data are complex
            if self.has_complex_dims and mask is not nomask:
                for axis in self.iterdims:
                    mask = mask.swapaxes(axis, -1)
                    if self._is_complex[axis]:
                        mask = mask.repeat(2, axis=-1)
                    mask = mask.swapaxes(axis, -1)

        # finally set the mask of the object

        if isinstance(mask, MaskedConstant):
            self._mask = mask
            return

        elif np.any(self._mask):
            # this should happen when a new mask is added to an existing one
            # mask to be combined to an existing one
            log.info("{0} is already a masked array.\n".format(
                type(self).__name__) +
                     "The new mask will be combined with the current array's mask.")
            self._mask |= mask  # combine (is a copy!)

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
        """
        |Meta| instance object - Additional metadata.

        """
        return self._meta

    # .........................................................................
    @meta.setter
    def meta(self, meta):
        # property.setter for meta
        if meta is not None:
            self._meta.update(meta)

    # ..................................................
    @property
    def name(self):
        """
        str - An user friendly name.

        It is really optional as the title may be used for the same purpose).
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
    def _title_default(self):
        return None

    # .........................................................................
    @property
    def title(self):
        """
        str - An user friendly title.

        Unlike the `name`, the title doesn't need to be unique.
        When the title is provided, it can be used for labeling the object,
        e.g., axe title in a matplotlib plot.

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
        """
        |ndarray|, dtype:float - Uncertainty (std deviation) on the data.

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

        if self.ndim == 1:
            uncertainty = uncertainty.squeeze()

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
            self._uncertainty = self._uncertainty.reshape((1, -1))

    # .........................................................................
    @property
    def units(self):
        """
        |Unit| instance object - The units of the data.

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
    @default('_id')
    def _id_default(self):

        return str(uuid.uuid1())  # a unique id

    # .........................................................................
    @property
    def id(self):
        """
        str - Object identifier (Readonly property).

        """
        return self._id

    # .........................................................................
    @property
    def dimensionless(self):
        """
        bool - True if the `data` array is dimensionless (Readonly property).

        Notes
        -----
        `Dimensionless` is different of `unitless` which means no unit.

        See Also
        --------
        unitless, has_units

        """
        if self.unitless:
            return False

        return self._units.dimensionless

    # .........................................................................
    @property
    def dtype(self):
        """
        dtype - Type of the underlying `data` array (Readonly property).

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
        """
        bool - True if at least one of the `data` array dimension is complex
        (Readonly property).

        """
        return np.any(self._is_complex)

    # .........................................................................
    @property
    def has_units(self):
        """
        bool - True if the `data` array have units (Readonly property).

        See Also
        --------
        unitless, dimensionless

        """
        if self._units:
            if not str(self.units).strip():
                return False
            return True
        return False

    # .........................................................................
    @property
    def is_empty(self):
        """
        bool - True if the `data` array is empty (size=0) (Readonly property).


        """
        return self._data.size == 0

    # .........................................................................
    @default('_is_complex')
    def _get_is_complex_default(self):
        return [False] * self._data.ndim

    # .........................................................................
    @property
    def is_complex(self):
        """
        tuple of bool - True if one of the `data` dimensions is complex
        (Readonly property).

        Notes
        -----
        If a dimension is complex, real and imaginary part
        are interleaved in the `data` array.

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
        """
        bool - True if the `data` array have labels (Readonly property).

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
        """
        bool - True if the `data` array has masked values (Readonly property).

        """
        if self._mask is nomask or self._mask is None:
            return False

        if isinstance(self._mask, (np.bool_, bool)):
            return self._mask

        if isinstance(self._mask, np.ndarray) and np.any(self._mask):
            return True
        else:
            return False

    # .........................................................................
    @property
    def is_uncertain(self):
        """
        bool - True if the `data` array has uncertainty (Readonly property).



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
        """
        |ma_ndarray| - The actual masked `data` array (Readonly property).

        """
        return self._umasked(self.data, self.mask)

    # .........................................................................
    @property
    def _masked_data(self):
         return self._umasked(self._data, self._mask)

    # .........................................................................
    @property
    def ndim(self):
        """
        int - The number of dimensions of the `data` array (Readonly property).

        """

        return self._data.squeeze().ndim

    # .........................................................................
    @property
    def shape(self):
        """
        tuple on int - A tuple with the size of each axis (Readonly property).

        The number of `data` element on each dimensions (possibly complex).

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
        """
        int - Size of the underlying `data` array (Readonly property).

        The total number of data element (possibly complex or hypercomplex
        in the array).

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
        """
        |ndarray|, dtype:object - The actual array with
        `uncertainty` of the `data` contained in this object (Readonly property).

        """
        return self._uarray(self.masked_data, self.uncertainty, self._units)

    # .........................................................................
    @property
    def _uncert_data(self):
        # private function that allow to force the masked and uncertainty
        # representation. Useful for slicing

        return self._uarray(self._masked_data, self._uncertainty, self._units)

    # .........................................................................
    @property
    def unitless(self):
        """
        bool - True if the `data` have `units` (Readonly property).

        """

        return not self.has_units

    # .........................................................................
    @property
    def real(self):
        """
        |ndarray|, dtype:float - The array with real part of the `data` (
        Readonly property).

        """
        if not self._is_complex[-1]:
            return self.copy()

        new = self.copy()
        new._is_complex[-1] = False
        ma = self._masked_data
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
        """
        |ndarray|, dtype:float - The array with imaginary part of the `data`
        (Readonly property).

        """
        if not self._is_complex[-1]:
            warnings(
                "This array is not complex, so we can't get imaginary data",
                SpectroChemPyWarning)

        new = self.copy()
        new._is_complex[-1] = False
        ma = self._masked_data
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
        """
        |ndarray|, dtype:float - The array with real part in both dimension of
        hypercomplex 2D `data` (Readonly property).

        """

        if self.ndim != 2:
            raise ValueError('Not a two dimensional array')
        return self.part('RR')

    # .........................................................................
    @property
    def RI(self):
        """
        |ndarray|, dtype:float - The array with real-imaginary part of
        hypercomplex 2D `data` (Readonly property).

        """

        if self.ndim != 2:
            raise ValueError('Not a two dimensional array')
        return self.part('RI')

    # .........................................................................
    @property
    def IR(self):
        """
        |ndarray|, dtype:float - The array with imaginary-real part of
        hypercomplex 2D `data` (Readonly property).

        """

        if self.ndim != 2:
            raise ValueError('Not a two dimensional array')
        return self.part('IR')

    # .........................................................................
    @property
    def II(self):
        """
        |ndarray|, dtype:float - The array with imaginary-imaginary part of
        hypercomplex 2D data (Readonly property).

        """

        if self.ndim != 2:
            raise ValueError('Not a two dimensional array')
        return self.part('II')

    # .........................................................................
    @property
    def values(self):
        """
        |ndarray|, dtype:object - The actual values (data, units,
        uncertainties) contained in this object (Readonly property).

        """
        return self._uarray(self._data, self._uncertainty, self._units)

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    # .........................................................................
    @docstrings.dedent
    def conjugate(self, axis=-1, inplace=False):
        """
        Conjugate of the NDDataset in the specified dimension

        Parameters
        ----------
        %(generic_method.parameters.axis|inplace)s

        Returns
        -------
        %(generic_method.returns.object)s

        See Also
        --------
        conj, real, imag, RR, RI, IR, II, part, set_complex, is_complex

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

    conj = make_func_from(conjugate)
    conj.__doc__ = "Short alias of `conjugate`"

    # .........................................................................
    def copy(self, deep=True, memo=None):
        """
        Make a disconnected copy of the current object.

        Parameters
        ----------
        deep : bool, optional
            If True a deepcopy is performed which is the default behavior
        memo : Not used
            This parameter ensure compatibility with deepcopy() from the copy
            package.

        Returns
        -------
        object
            An exact copy of the current object.

        Examples
        --------
        >>> nd1 = NDArray([1.+2.j,2.+ 3.j])
        >>> nd1
        NDArray: [   1.000,    2.000,    2.000,    3.000] unitless
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
        Check the compatibility of units with another object

        Parameters
        ----------
        other : |ndarray|
            The ndarray object for which we want to compare units compatibility

        Returns
        -------
        result
            True if units are compatible

        Examples
        --------
        >>> nd1 = NDArray([1.+2.j,2.+ 3.j], units='meters')
        >>> print(nd1)
        R[   1.000    2.000] m
        I[   2.000    3.000] m
        >>> nd2 = NDArray([1.+2.j,2.+ 3.j], units='seconds')
        >>> nd1.is_units_compatible(nd2)
        False
        >>> nd1.ito('minutes', force=True)
        NDArray: [   1.000,    2.000,    2.000,    3.000] min
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
    def ito(self, other, force=False):
        """
        Inplace scaling of the current object data to different units.

        (same as `to` with inplace= True).

        Parameters
        ----------
        other : |Unit|, |Quantity| or str
            Destination units.
        force : bool, optional, default= `False`
            If True the change of units is forced, even for incompatible units

        Returns
        -------
        object
            same object with new units.

        See Also
        --------
        to

        """
        return self.to(other, inplace=True, force=force)

    # .........................................................................
    def part(self, select='ALL'):
        """
        Take selected components of an hypercomplex array (RRR, RIR, ...)

        Parameters
        ----------
        select : str, optional, default='ALL'
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
        %(generic_method.returns.object)s

        """
        new = self.copy()
        if select == 'ALL':
            select = 'R' * self.ndim
        if self.ndim == 1:
            select = '*' + select
        ma = self._uncert_data
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
        # if isinstance(ma, np.ma.masked_array):
        #    new._data = ma.data
        #    new._mask = ma.mask
        # else:
        #    new._data = ma
        if hasattr(ma, 'mask'):
            new._mask = ma.mask
        else:
            new._mask = nomask

        new._uncertainty = unp.std_devs(ma)
        new._data = unp.nominal_values(ma)

        return new

    # .........................................................................
    def remove_masks(self):
        """
        Remove all masks previously set on this array

        """
        self._mask = nomask

    # .........................................................................
    @docstrings.dedent
    def set_complex(self, axis):
        """
        Set the data along the given dimension as complex

        Parameters
        ----------
        %(generic_method.parameters.axis)s

        Notes
        -----
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
    @docstrings.dedent
    def set_real(self, axis):
        """
        Set the data along the given dimension as real

        Parameters
        ----------
        %(generic_method.parameters.axis)s

        """
        if axis < 0:
            axis = self.ndim + axis

        self._is_complex[axis] = False

    # .........................................................................
    @docstrings.dedent
    def swapaxes(self, axis1, axis2, inplace=False):
        """
        Interchange two dims of a NDArray.

        Parameters
        ----------
        axis1 : int
            First dimension index
        axis2 : int
            Second dimension index
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        See Also
        --------
        transpose

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
    @docstrings.dedent
    def to(self, other, inplace=False, force=False):
        """
        Return the object with data rescaled to different units.

        Parameters
        ----------
        other : |Quantity| or str.
            Destination units.
        %(generic_method.parameters.inplace)s
        force: bool, optional, default: False
            If True the change of units is forced, even for imcompatible units

        Returns
        -------
        %(generic_method.returns.object)s

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
        spectrochempy.extern.pint.errors.DimensionalityError: Cannot convert from 'meter' ([length]) to 'second' ([time])

        However, we can force the change

        >>> ndd.to('second', force=True)
        NDArray: [[  --,    0.316,    0.184],
                  [   0.205,   --,    0.596],
                  [   0.965,    0.653,   --]] s

        By default the conversion is not done inplace, so the original is not
        modified:

        >>> print(ndd) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[  --    0.316    0.184]
         [   0.205   --    0.596]
         [   0.965    0.653   --]] m


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

        if self.data is None:
            return ''

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
            body = body.replace('\n', sep)
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
        # probably not useful
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
                    # try to find a pos in the by string
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
                if key < 0:  # reverse indexing (on the real shape!)
                    start = self._data.shape[axis] + key
            stop = start + 1
            step = 1
        else:
            start, stop, step = key.start, key.stop, key.step

            if start is not None and not isinstance(start, (int, np.int_)):
                start = self._loc2index(start, axis)

            if stop is not None and not isinstance(stop, (int, np.int_)):
                stop = self._loc2index(stop, axis)
                if stop < start:  # and self.coordset[axis].is_reversed:
                    start, stop = stop, start
                stop = stop + 1

            if step is not None and not isinstance(step, (int, np.int_)):
                raise NotImplementedError(
                    'step in location slicing is not yet possible.')
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
            return key, key  # TODO: jsut check with complex!!!

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

        if self._data.ndim != self.ndim:
            # case or 1D spectra or of array with complex dimensions
            # this need some attention to have a correct slicing
            # because, the user should not be aware of the internal
            # representation
            newkeys = []
            i = 0
            for size in self._data.shape:
                # loop on the real shape, and make the keys corresponding to
                # dimensions which are not of length 1.
                # The other have to be replaced by a slice(None) or 0
                if size == 1:
                    newkeys.append(slice(None))
                else:
                    if not keys:
                        # list of keys already completely used
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
    def _umasked(data, mask):
        # This ensures that a masked array is returned.

        if np.any(mask):
            if not np.any(mask):
                mask = np.zeros_like(data).astype(bool)
            data = np.ma.masked_array(data, mask)

        return data

    # .........................................................................
    @staticmethod
    def _uarray(data, uncertainty, units=None):
        # return the array with uncertainty and units if any

        # the handling of uncertainties have a great price in performance.
        # see discussion: https://github.com/lebigot/uncertainties/issues/57
        # Let's avoid it if not necessary
        #

        if (uncertainty is None or not gt_eps(uncertainty)):
            uar = data
        else:
            uar = unp.uarray(data, uncertainty)

        if units:
            return Quantity(uar, units)
        else:
            return uar


# ============================================================================
if __name__ == '__main__':

    pass
