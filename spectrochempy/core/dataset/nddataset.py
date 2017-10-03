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
from six import string_types
from traitlets import (List, Unicode, Instance, default,
                       Bool, observe, All)

from spectrochempy.core.units import Quantity
from spectrochempy.utils import SpectroChemPyWarning
from spectrochempy.utils import is_sequence, is_number
from spectrochempy.utils import (numpyprintoptions,
                                 get_user_and_node)
from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndaxes import Axis, Axes, AxisError
from spectrochempy.core.dataset.ndmath import NDMath, set_operators
#from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.dataset.ndio import NDIO

from spectrochempy.application import log

# =============================================================================
# Local imports
# =============================================================================

# =============================================================================
# Constants
# =============================================================================

__all__ = ['NDDataset',
           'NDDatasetError',
           'NDDatasetWarning',
           #dataset
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

    axes : An instance of :class:`~spectrochempy.core.dataset.ndaxes.Axes`,
    optional

        It contains the `axis` coordinates and `labels` for the different
        dimensions of the `data`. if `axes` is provided, it must specified
        the `axis` and `labels` for all dimensions of the `data`.
        Multiple axis can be specified in an Axes instance for each dimension.

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

    iscopy :  `bool`, optional, default = `False`.

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
    >>> x.data
    array([       1,        2,        3])

    """
    author = Unicode(get_user_and_node(),
                     desc='Name of the author of this dataset',
                     config=True)

    # private metadata in addition to those of the base NDArray class
    _modified = Instance(datetime)
    _description = Unicode
    _history = List

    _axes = Instance(Axes, allow_none=True)

    _iscopy = Bool(False)

    @default('_iscopy')
    def _get_iscopy_default(self):
        return False

    def __init__(self,
                 data=None,
                 axes=None,
                 mask=None,
                 uncertainty=None,
                 units=None,
                 meta=None,
                 name=None,
                 title=None,
                 axesunits=None,
                 axestitles=None,
                 is_complex=None,
                 **kwargs):

        self._iscopy = False  # kwargs.pop('iscopy', False)

        if is_complex is not None:
            self._data_is_complex = is_complex

        super(NDDataset, self).__init__(**kwargs)

        self._modified = self._date

        self._description = ''
        self._history = []

        # If we want a deepcopy of the passed data
        if self._iscopy:
            data = copy.deepcopy(data)
            axes = copy.deepcopy(axes)
            mask = copy.deepcopy(mask)
            uncertainty = copy.deepcopy(uncertainty)
            units = copy.copy(units)  # FIX:? deepcopy not working
            meta = copy.deepcopy(meta)
            axesunits = copy.deepcopy(axesunits)
            axestitles = copy.deepcopy(axestitles)

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

        self.axes = axes

        self.axestitles = axestitles

        self.axesunits = axesunits

        # This must come after self's units has been set so that the units
        # of the uncertainty, if any, can be converted to the units of the
        # units of self.
        self.uncertainty = uncertainty

    def _check_for_complex_data(self, data):

        if data.dtype != np.complex:
            # given data in the last dimension are not complex!
            self._data = data
            if self._data_is_complex is None:
                self._is_complex = [False] * data.ndim
            else:
                self._is_complex = self._data_is_complex
            return

        # input complex values are only accepted for the last dimension
        # They will be interlaced in the final dataset

        if self._data_is_complex is None:
            # make the data in the last dimension (or the dimension
            # specified by is complex)
            # compatible with the structure of
            # NDDataset which does not accept complex data (must be interlaced)
            self._is_complex = [False] * data.ndim
        else:
            self._is_complex = self._data_is_complex

        self.set_complex(axis=-1)

        newshape = list(data.shape)
        newshape[-1] = newshape[-1] * 2
        newdata = np.zeros(newshape)
        newdata[..., ::2] = data.real
        newdata[..., 1::2] = data.imag

        self._data = newdata[:]

    @property
    def data(self):
        """
        :class:`~numpy.ndarray`-like object - The actual array data
        contained in this object.

        """
        return self._data

    @data.setter
    def data(self, data):
        # property.setter for data
        if data is None:
            self._data = np.array([]).astype(float)  # reinit data
            log.debug("init data with an empty array of type float")

        elif isinstance(data, (NDDataset, NDArray)):
            log.debug("init data with data from another NDArray")
            # No need to check the data because data must have successfully
            # initialized.
            self._name = "copy of {}".format(data._name) if self._iscopy \
                else data._name
            self._check_for_complex_data(data._data)
            self._mask = data._mask
            self._axes = data._axes
            self._units = data._units
            self._meta = data._meta
            self._title = data._title
            self.uncertainty = data.uncertainty

        elif isinstance(data, NDFrame):  # pandas object
            log.debug("init data with data from pandas NDFrame object")
            self._check_for_complex_data(data.values)
            self.axes = data.axes

        elif isinstance(data, pd.Index):  # pandas index object
            log.debug("init data with data from a pandas Index")
            self._check_for_complex_data(np.array(data.values, subok=True,
                                                  copy=self._iscopy))

        elif isinstance(data, Quantity):
            log.debug("init data with data from a Quantity object")
            self._data_passed_is_quantity = True
            self._check_for_complex_data(np.array(data.magnitude, subok=True,
                                                  copy=self._iscopy))
            self._units = data.units

        elif hasattr(data, 'mask'):  # an object with data and mask attributes
            log.debug("init mask from the passed data")
            self._data_passed_with_mask = True
            self._check_for_complex_data(np.array(data.data, subok=True,
                                                  copy=self._iscopy))
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
            self._check_for_complex_data(np.array(data, subok=True, copy=False))

        else:
            log.debug("init data with a numpy array")
            self._check_for_complex_data(np.array(data, subok=True,
                                                  copy=self._iscopy))

    @property
    def mask(self):
        """
        :class:`~numpy.ndarray`-like - Mask for the data.

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
                log.info("Overwriting NDDataset's current "
                         "mask with specified mask")

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
    def uncertainty(self):
        """
        :class:`~numpy.ndarray` -  Uncertainty (std deviation) on the data.

        """
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        # property setter for uncertainty
        if value is not None:
            if self.uncertainty is not None and self.uncertainty.size > 0:
                log.info("Overwriting NDDataset's current uncertainty being"
                         " overwritten with specified uncertainty")
            if not isinstance(value, np.ndarray):
                raise ValueError('Uncertainty must be specified as a ndarray')
                # TODO: make this a little less strict
                # so it accept other list structure

            if value.shape != self._data.shape:

                if not self.has_complex_dims:
                    raise ValueError(
                            'uncertainty shape does not match array data shape')
                else:  # complex data
                    pass

            self._uncertainty = value

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

        Describes the history of actions made on this dataset

        """
        return self._history

    @history.setter
    def history(self, value):
        self._history.append(value)

    @property
    def axes(self):
        """
        :class:`~spectrochempy.core.dataset.ndaxes.Axes` instance

        Contain the axes of the dataset

        """
        return self._axes

    @axes.setter
    def axes(self, value):
        if value is not None:
            if self._axes is not None:
                log.info("Overwriting NDDataset's current "
                         "axes with specified axes")

            for i, axis in enumerate(value):
                if isinstance(axis, Axes):
                    size = axis.sizes[i]
                else:
                    size = axis.size
                if size != self.shape[i]:
                    raise AxisError(
                            'the size of each axis coordinates must '
                            'be equal to that of the respective data dimension')

            if not isinstance(value, Axes):
                self._axes = Axes(value)
            else:
                self._axes = value

    @default('_axes')
    def _get_axes_default(self):
        return None  # Axes([None for dim in self.shape])

    @property
    def axestitles(self):
        """
        `List` - A list of the :class:`~spectrochempy.core.dataset.ndaxes.Axis`
        titles.

        """
        if self.axes is not None:
            return self.axes.titles

    @axestitles.setter
    def axestitles(self, value):
        if self.axes is not None:
            self.axes.titles = value

    @property
    def axesunits(self):
        """
        `List`- A list of the :class:`~spectrochempy.core.dataset.ndaxes.Axis`
        units
        """
        if self.axes is not None:
            return self.axes.units

    @axesunits.setter
    def axesunits(self, value):

        if self.axes is not None:
            self.axes.units = value

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

        Return the x axis, i.e. coords(-1)

        """
        return self.axes[-1]

    @x.setter
    def x(self, value):
        self.axes[-1] = value

    @property
    def y(self):
        """
        Read-pnly properties

        Return the y axis, i.e. coords(-2) for 2D dataset.

        """
        if self.ndim > 1:
            return self.axes[-2]

    @y.setter
    def y(self, value):
        self.axes[-2] = value

    @property
    def z(self):
        """
        Read-pnly properties

        Return the z axis, i.e. coords(-3) fpr 3D dataset

        """
        if self.ndim > 2:
            return self.axes[-3]

    @z.setter
    def z(self, value):
        self.axes[-3] = value

    @property
    def date(self):
        """
        Date of the dataset creation
        """
        return self._date

    @property
    def modified(self):
        """
        Date of modification

        """
        return self._modified

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------
    def squeeze(self, axis=None, inplace=False):
        """
        Remove single-dimensional entries from the shape of an array.

        Parameters
        ----------
        axis :   `None` or `int` or `tuple` of ints, optional

            Selects a subset of the single-dimensional entries in the shape.
            If an axis is selected with shape entry greater than one,
            an error is raised.

        inplace : `bool`, optional, default = False

            if False a new object is returned

        Returns
        -------
        squeezed_dataset : same type

            The input array, but with all or a subset of the dimensions
            of length 1 removed.

        """

        if axis is not None:
            if not is_sequence(axis):
                axis = [axis]
            squeeze_axis = list(axis)

            for axis in squeeze_axis:
                if axis < 0:
                    axis = self.ndim + axis

                if self.shape[axis] > 1:
                    raise IndexError(
                            '%d is of length greater than one: '
                            'cannot be squeezed' % axis)
        else:
            squeeze_axis = []
            for axis, dim in enumerate(self.shape):
                if dim == 1:
                    squeeze_axis.append(axis)

        if not inplace:
            new = self.copy()
        else:
            new = self

        new._data = self._data.squeeze(tuple(squeeze_axis))
        new._mask = self._mask.squeeze(tuple(squeeze_axis))
        new._uncertainty = self._uncertainty.squeeze(tuple(squeeze_axis))

        axes = []
        cplx = []
        for axis in range(self.ndim):
            if axis not in squeeze_axis:
                axes.append(self.axes[axis])
                cplx.append(self._is_complex[axis])

        new._is_complex = cplx
        new._axes = Axes(axes)

        return new

    def coords(self, axis=-1):
        """
        This method return the the coordinates along the given axis

        Parameters
        ----------
        axis : `int` or `unicode`

            An axis index or name, default=-1 for the last axis

        Returns
        -------
        coords : :class:`~numpy.ndarray`

        """
        return self.axes[axis]  # .coords

    def transpose(self, axes=None, inplace=False):
        """
        Permute the dimensions of a NDDataset.

        Parameters
        ----------
        axes : list of ints, optional

            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        inplace : `bool`, optional, default = `False`.

            By default a new dataset is returned.
            Change to `True` to chnage data inplace.

        Returns
        -------
        transposed_dataset : same type

            The nd-dataset or a new nd-dataset (inplace=False)
            is returned with axes
            transposed

        See Also
        --------
        :meth:`swapaxes`

        """
        if not inplace:
            new = self.copy()
        else:
            new = self

        if axes is None:
            axes = range(self.ndim - 1, -1, -1)

        new._data = np.transpose(new._data, axes)
        new._axes._transpose(axes)
        new._is_complex = [new._is_complex[axis] for axis in axes]
        return new

    def swapaxes(self, axis1, axis2, inplace=False):
        """
        Interchange two axes of the NDDataset.

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

            The object or a new object (inplace=False) is returned with axes
            swapped

        See Also
        --------
        :meth:`transpose`

        """
        if not inplace:
            new = self.copy()
        else:
            new = self

        if axis1 == -1:
            axis1 = self.ndim - 1

        if axis2 == -1:
            axis2 = self.ndim - 1

        new._data = np.swapaxes(new._data, axis1, axis2)
        if new._axes:
            new._axes[axis1], new._axes[axis2] = \
                new._axes[axis2], new._axes[axis1]
        new._is_complex[axis1], new._is_complex[axis2] = \
            new._is_complex[axis2], new._is_complex[axis1]

        return new

    def sort(self, axis=0, pos=None, by='axis', descend=False, inplace=False):
        """
        Returns a copy of the dataset sorted along a given axis
        using the numeric or label values.

        Parameters
        ----------
        axis : `int` , optional, default = 0.

            Axis id along which to sort.

        pos: `int` , optional

            If labels are multidimentional  - allow to sort on a define
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

            The object or a new object (inplace=False) is returned with axes
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
                args = self.axes[axis]._argsort(by=by, pos=pos, descend=descend)
                new.axes[axis] = self.axes[axis]._take(args)
                indexes.append(args)
            else:
                indexes.append(slice(None))

        new._data = new._data[indexes]
        new._mask = new._mask[indexes]
        new._uncertainty = new._uncertainty[indexes]

        return new

    def real(self, axis=-1):
        """
        Compute the real part of the elements of the NDDataset.

        Parameters
        ----------
        axis : `int` , optional, default = -1

            The axis along which the angle should be calculated

        Returns
        -------
        real_dataset : same type

            Output array.

        See Also
        --------
        :meth:`imag`, :meth:`conj`, :meth:`abs`

        """
        new = self.copy()
        if not new._is_complex[axis]:
            return new
        new.swapaxes(-1, axis, inplace=True)
        new._data = new._data[..., ::2]
        new._is_complex[axis] = False
        new.swapaxes(-1, axis, inplace=True)
        return new

    def imag(self, axis=-1):
        """
        Imaginary part

        Compute the imaginary part of the elements of the NDDataset.

        Parameters
        ----------
        axis : `int` , optional

            The axis along which the angle should be calculated.

        Returns
        -------
        imag_dataset : same type

            Output array.

        See Also
        --------
        :meth:`real`,:meth:`conj`, :meth:`abs`

        """
        new = self.copy()
        if not new._is_complex[axis]:
            logging.error(
                    'The current dataset is not complex. '
                    'Imag = None is returned.')
            return None

        new.swapaxes(-1, axis, inplace=True)
        new._data = new._data[..., 1::2]
        new._is_complex[axis] = False
        new.swapaxes(-1, axis, inplace=True)
        return new

    def conj(self, axis=-1):
        """
        Return the conjugate of the NDDataset.

        Parameters
        ----------
        axis : `int` , optional, default = -1

            The axis along which the conjugate value should be calculated

        Returns
        -------
        conj_dataset : same type

            Output array.

        See Also
        --------
        :meth:`real`, :meth:`imag`, :meth:`abs`

        """
        new = self.copy()
        if not new._is_complex[axis]:
            return new  # not a complex, return inchanged

        new.swapaxes(-1, axis)
        new._data[..., 1::2] = -new._data[..., 1::2]
        new.swapaxes(-1, axis)

        return new

    conjugate = conj

    def abs(self, axis= -1):
        """
        Returns the absolute value of a complex NDDataset.

        Parameters
        ----------
        axis : int

            Optional, default: 1.

            The axis along which the absolute value should be calculated.



        Returns
        -------
        nddataset : same type,

            Output array.

        See Also
        --------
        :meth:`real`, :meth:`imag`, :meth:`conj`


        """
        new = self.copy()
        if not new.has_complex_dims or not new.is_complex[axis]:
            return np.fabs(new)  # not a complex, return fabs should be faster

        new = new.real(axis) ** 2 + new.imag(axis) ** 2
        new._is_complex[axis] = False
        new._data = np.sqrt(new)._data

        return new
    absolute = abs

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

        if self.axes:
            new_axis = self.axes[axis][::2]
            self.axes[axis] = new_axis

    # Create the returned values of functions should be same class as input.
    # The units should have been handled by __array_wrap__ already

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    def __dir__(self):
        return ['data', 'mask', 'units', 'uncertainty',
                'meta', 'name', 'title', 'is_complex',
                'axes', 'description', 'history', 'date', 'modified'
                ]

    def __repr__(self):
        prefix = type(self).__name__ + '('
        body = np.array2string(self._data, separator=', ', prefix=prefix)
        return ''.join([prefix, body, ')'])

    def __str__(self):
        # Display the metadata of the object and partially the data

        # print field names/values (class/sizes)
        # data.name, .author, .date,
        out = '\n' + '-' * 80 + '\n'
        # out += '   name or id: %s \n' % self.name
        out += '       author: {}\n'.format(self.author)
        out += '      create  d: {}\n'.format(self._date)
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

        if self._history:
            pars = self.history
            out += '      history: '
            if pars:
                out += '{}\n'.format(wrapper1.fill(pars[0]))
            for par in pars[1:]:
                out += '{}'.format(textwrap.indent(par, ' ' * 15))

            if not out.endswith('\n'):
                out += '\n'

        uncertainty = "(+/-%s)" % self.uncertainty if \
            self.uncertainty is not None else ""
        units = '{:~K}'.format(
                self.units) if self.units is not None else 'unitless'
        sh = ' size' if self.ndim < 2 else 'shape'
        shapecplx = (x for x in
                     itertools.chain.from_iterable(
                             zip(self.shape, self.is_complex)))
        shape = (' x '.join(['{}{}'] * len(self.shape))).format(
                *shapecplx).replace(
                'False', '').replace('True', '(complex)')
        size = self.size
        sizecplx = '' if not self.has_complex_dims else " (complex)"

        out += '   data title: {}\n'.format(self.title)
        out += '    data size: {}{}\n'.format(size, sizecplx) if self.ndim < 2 \
            else '   data shape: {}\n'.format(shape)

        out += '   data units: {}\n'.format(units)
        data_str = str(
                self._uarray(self._data, self._uncertainty)).replace('\n\n',
                                                                     '\n')
        out += '  data values:\n'
        out += '{}\n'.format(textwrap.indent(str(data_str), ' ' * 9))
        if self.axes is not None:
            for i, axis in enumerate(self.axes):
                axis_str = str(axis).replace('\n\n', '\n')
                out += '       axis {}:\n'.format(i)
                out += textwrap.indent(axis_str, ' ' * 9)
        out += '-' * 80

        return out

    def __getattr__(self, item):
        # when the attribute was not found

        if item in [ "__numpy_ufunc__"] or '_validate' in item or \
                        '_changed' in item:
            # raise an error so that masked array will be handled correctly
            # with arithmetic operators and more
            raise AttributeError

            # look from the plugins
            # attr = super(NDDataset, self)._getattr(item)

            # if attr is not None:
            #    return attr

            # log.warning('not found attribute: %s' % item)

    def __getitem__(self, item):
        # we need axes (but they might be not present...
        # in this case axes are simply the indexes
        if self.axes is None:
            # create temp axes
            axes = Axes([Axis(np.arange(l)) for l in self._data.shape])
        else:
            axes = self.axes

        # transform the passed index (if necessary) to integer indexes
        key = self._make_index(axes, item)

        # normal integer based slicing
        new_data = self._data[key].squeeze()
        new_mask = self._mask[key].squeeze()
        new_uncertainty = self._uncertainty[key].squeeze()

        # perform the axes slicing (and unsqueeze the data!)
        new_axes = axes.copy()
        for i, ax in enumerate(new_axes):
            if self.is_complex[i]:
                # the slice has been multiplied by 2 in _get_slice
                # (see below)
                # so we have to get back to a nowmal slice for the axis
                start, stop, step = key[i].start, key[i].stop, key[i].step
                if start is not None:
                    start = start // 2
                if stop is not None:
                    stop = stop // 2
                if step is not None:
                    step = step // 2
                _key = slice(start, stop, step)
            else:
                _key = key[i]
            new_axes[i] = ax[_key]

        sh = list(new_data.shape)
        for i, ax in enumerate(new_axes):
            cplx = self.is_complex[i]
            if (ax.size == 1) and not cplx:  # and len(new_axes)>1:
                # new_axes.remove(ax)
                # we don't want to squeeze the extraction by default
                # we will possibly squeeze them later
                sh.insert(i, 1)
        new_data = new_data.reshape(tuple(sh))
        if new_mask is not None:
            new_mask = new_mask.reshape(tuple(sh))
        if new_uncertainty is not None:
            new_uncertainty = new_uncertainty.reshape(tuple(sh))

        if new_data.size == 0:
            raise IndexError("Empty array of shape {} resulted from slicing.\n"
                             "Check the indexes and make "
                             "sure to use floats for "
                             "location slicing".format(str(new_data.shape)))

        new = self.copy()
        new._name = '*' + self._name
        new._data = new_data
        new._mask = new_mask
        new._axes = new_axes
        new._uncertainty = new_uncertainty

        return new

    def __eq__(self, other):
        eq = super(NDDataset, self).__eq__(other)
        eq &= (np.all(self._uncertainty == other._uncertainty))
        return eq

    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return str(type(self)) + "1234567890"

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
                             zip(self.shape, self.is_complex)))

        shape = (' x '.join(['{}{}'] * len(self.shape))).format(
                *shapecplx).replace('False', '').replace('True', '(complex)')

        size = self.size
        sizecplx = '' if not self.has_complex_dims else " (complex)"
        size = '{}{}'.format(size, sizecplx) \
                                        if self.ndim < 2 else '{}'.format(shape)

        data = '<table>\n'
        data += tr.format("Title", self.title)
        data += tr.format("Size",size)
        data += tr.format("Units", units)
        data_str = str(self._uarray(self._data, self._uncertainty))
        data_str = data_str.replace('\n\n', '\n')
        data += tr.format("Values", textwrap.indent(str(data_str), ' ' * 9))
        data += '</table>\n'  # end of row data

        out += tr.format('data', data)

        if self.axes is not None:
            for i, axis in enumerate(self.axes):
                axis_str = axis._repr_html_().replace('\n\n', '\n')
                out += tr.format("axis %i"%i,
                                 textwrap.indent(axis_str, ' ' * 9))

        out += '</table><br/>\n'

        return out

    def _loc2index(self, loc, axis):
        # Return the index of a location (label or coordinates) along the axis

        # underlying axis array and labels
        coords = axis._data
        labels = axis._labels

        if isinstance(loc, string_types) and labels is not None:
            # it's probably a label
            indexes = np.argwhere(labels == loc).flatten()
            if indexes.size > 0:
                return indexes[0]
            else:
                raise ValueError(
                        'Could not find this label: {}'.format(loc))

        elif isinstance(loc, datetime):
            # not implemented yet
            return None  # TODO: date!

        elif is_number(loc):
            index = (np.abs(coords - loc)).argmin()
            if loc > coords.max() or loc < coords.min():
                warn('This coordinate ({}) is outside the axis limits.\n'
                     'The closest limit index is returned'.format(loc),
                     NDDatasetWarning)
            return index

        else:
            raise ValueError('Could not find this location: {}'.format(loc))

    def _get_slice(self, key, axis, iscomplex=False):

        if not isinstance(key, slice):
            start = key
            if not isinstance(key, (int, np.int)):
                start = self._loc2index(key, axis)
            else:
                if key < 0:  # reverse indexing
                    start = axis.size + key
            stop = start + 1
            step = None
        else:
            start, stop, step = key.start, key.stop, key.step

            if not (start is None or isinstance(start, (int, np.int))):
                start = self._loc2index(start, axis)

            if not (stop is None or isinstance(stop, (int, np.int))):
                stop = self._loc2index(stop, axis) + 1

            if step is not None and not isinstance(step, (int, np.int)):
                warn('step in location slicing is not yet possible. Set to 1')
                # TODO: we have may be a special case with datetime
                step = None

        if iscomplex:
            if start is not None:
                start = start * 2
            if stop is not None:
                stop = stop * 2
            if step is not None:
                step = step * 2

        newkey = slice(start, stop, step)

        return newkey

    def _make_index(self, axes, key):

        if isinstance(key, np.ndarray) and key.dtype == np.bool:
            # this is a boolean selection
            # we can proceed directly
            return key

        # we need to have a list of slice for each argument or a single slice
        #  acting on the axis=0
        # the given key can be a single argument or a single slice:

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

        # replace the non index slice or non slide by index slices
        for i_axis, key in enumerate(keys):
            axis = axes[i_axis]
            keys[i_axis] = self._get_slice(key, axis,
                                           iscomplex=self.is_complex[i_axis])

        return tuple(keys)

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
        if change['name'] == '_data' and self._date == datetime(1, 1, 1, 0, 0):
            self._date = datetime.now()
            self._modified = datetime.now()

        # change to complex
        # change type of data to complex
        # require modification of the axes, if any
        if change['name'] == '_is_complex':
            pass

        # all the time -> update modified date
        self._modified = datetime.now()

        return

# make some function also accesiibles from the module
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
