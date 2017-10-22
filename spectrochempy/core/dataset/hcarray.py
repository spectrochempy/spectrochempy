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

from spectrochempy.utils import is_sequence, EPSILON

import numpy as np
import copy

_classes = ['HCArray', 'hcarray']

nomask = np.ma.nomask

# =============================================================================
# HCArray object
# =============================================================================
# subclass of the numpy MaskedArray type

class HCArray(np.ma.MaskedArray):
    """
    This is a subclass of the ndarray with attributes
    allowing the handling of hypercomplex data

    :class:`hcarray` is an alias of :class:`HCArray`

    Parameters
    ----------
    data : array_like
        Input data.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    dtype : dtype, optional
        Data type of the output.
        If `dtype` is None, the type of the data argument (``data.dtype``)
        is used. If `dtype` is not None and different from ``data.dtype``,
        a copy is performed.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead.
        Default is False.
    subok : bool, optional
        Whether to return a subclass of `MaskedArray` if possible (True) or a
        plain `MaskedArray`. Default is True.
    ndmin : int, optional
        Minimum number of dimensions. Default is 0.
    fill_value : scalar, optional
        Value used to fill in the masked values when necessary.
        If None, a default based on the data-type is used.
    keep_mask : bool, optional
        Whether to combine `mask` with the mask of the input data, if any
        (True), or to use only `mask` for the output (False). Default is True.
    hard_mask : bool, optional
        Whether to use a hard mask or not. With a hard mask, masked values
        cannot be unmasked. Default is False.
    shrink : bool, optional
        Whether to force compression of an empty mask. Default is True.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C', then the array
        will be in C-contiguous order (last-index varies the fastest).
        If order is 'F', then the returned array will be in
        Fortran-contiguous order (first-index varies the fastest).
        If order is 'A' (default), then the returned array may be
        in any order (either C-, Fortran-contiguous, or even discontiguous),
        unless a copy is required, in which case it will be C-contiguous.
    is_complex : list, optional,
        A list specifying which dimension is complex or not

    Examples
    --------

    Initialize with real data
    >>> x = np.array(range(4), dtype=float)
    >>> y = hcarray(x)
    >>> y  # repr()
    hcarray([ 0.,  1.,  2.,  3.])
    >>> print(y) # str()
    [ 0.  1.  2.  3.]

    >>> y.is_complex
    [False]

    >>> y.make_complex(-1)
    >>> y.is_complex
    [True]

    >>> x = np.array([[0, 1.+1.5j],[1-2.j, 2.+1j]], dtype=np.complex128)
    >>> y = hcarray(x)
    >>> y
    hcarray([[ 0. ,  0. ,  1. ,  1.5],
             [ 1. , -2. ,  2. ,  1. ]])
    >>> print(y)
    [[ 0.+0.j   1.+1.5j]
     [ 1.-2.j   2.+1.j ]]

    >>> y.is_complex
    [False, True]

    >>> x = np.array([[0, 1.+1.5j],[1-2.j, 2.+1j]], dtype=np.complex128)
    >>> y = hcarray(x, is_complex=[True, True])
    >>> y.is_complex
    [True, True]
    >>> print(y)
    [[ 0.+0.j   1.+1.5j]
     [ 1.-2.j   2.+1.j ]]
    >>> print(y.real)
    [[ 0.  1.]
     [ 1.  2.]]
    >>> print(y.RR)
    [ 0.  1.]
    >>> print(y.RI)
    [ 0.   1.5]
    >>> print(y.IR)
    [ 1.  2.]
    >>> print(y.II)
    [-2.  1.]

    Now make dimension 0 real.  Note that the dimension are not change,
    we just force all item to be real components
    >>> y.make_real(0)
    >>> y.is_complex
    [False, True]
    >>> print(y.real)
    [[ 0.  1.]
     [ 1.  2.]]
    >>> print(y.RR)
    [[ 0.  1.]
     [ 1.  2.]]
    >>> print(y.RI)
    [[ 0.   1.5]
     [-2.   1. ]]

    >>> # np.sqrt(y) This can work with this hcarray #TODO: work with the ufunc

    """
    _mask = nomask
    _fill_value = 1e+20
    is_complex = []

    def __new__(cls, data=None, mask=nomask, dtype=None, copy=False,
                subok=True, ndmin=0, fill_value=None, keep_mask=True,
                hard_mask=None, shrink=True, order=None, is_complex=None,
                **options):

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if isinstance(data, cls):
            _data = data.view(type(data))
        else:
            _data = np.asanyarray(data).view(cls)

        if mask is nomask:
            # Case 1. : no mask in input.
            # Erase the current mask ?
            if not keep_mask:
                # With a reduced version
                if shrink:
                    _data._mask = nomask
                # With full version
                else:
                    _data._mask = np.zeros(_data.shape, dtype=bool)

        else:
            # Case 2. : With a mask in input.
            # If mask is boolean, create an array of True or False
            if mask is True:
                mask = np.ones(_data.shape, dtype=bool)
            elif mask is False:
                mask = np.zeros(_data.shape, dtype=bool)
            else:
                # Read the mask with the current mdtype
                mask = np.array(mask, copy=copy, dtype=bool)

            # Make sure the mask and the data have the same shape
            if mask.shape != _data.shape:
                (nd, nm) = (_data.size, mask.size)
                if nm == 1:
                    mask = np.resize(mask, _data.shape)
                elif nm == nd:
                    mask = np.reshape(mask, _data.shape)
                else:
                    msg = "Mask and data not compatible: data size is %i, " + \
                          "mask size is %i."
                    raise Exception(msg % (nd, nm))
                copy = True
            # Set the mask to the new value
            if _data._mask is nomask:
                _data._mask = mask
            else:
                if not keep_mask:
                    _data._mask = mask
                else:
                    _data._mask = np.logical_or(mask, _data._mask)

        # Update fill_value.
        if fill_value is None:
            fill_value = getattr(data, '_fill_value', None)
        # But don't run the check unless we have something to check.
        if fill_value is not None:
            _data._fill_value = super(HCArray,cls)._check_fill_value(fill_value, _data.dtype)
        # Process extra options ..
        if hard_mask is None:
            _data._hardmask = getattr(data, '_hardmask', False)
        else:
            _data._hardmask = hard_mask
        _data._baseclass = np.ndarray


        if is_complex is None:
            is_complex = [False] * _data.ndim  # by default real in all dimnension

        if is_sequence(is_complex) and len(is_complex)!=_data.ndim:
            raise ValueError("size of the `is_complex` list argument "
                             "doesn't match the nb of dims")

        if is_sequence(is_complex) and (data.dtype == complex):
            is_complex[-1] = True

        if _data.dtype != complex:
            _data = _data.astype(np.float64)  # be sure all element are float or complex

        # reshape so there is no more complex data... They will be interleaved
        if _data.ndim>1:
            _data = _data.view(dtype=np.float64).reshape(data.shape[0],-1)
        else:
            _data = _data.view(dtype=np.float64).reshape(-1)

        for i in range(_data.ndim):
            if is_complex[i] and _data.shape[i] % 2 != 0:
                raise ValueError("Size of complex indirect dimension "
                                 "must be even, not %d"%_data.shape[i])

        # add the new attribute to the created instance
        _data.is_complex=is_complex[:]

        # Finally, we must return the newly created object:
        return _data

    def make_complex(self, axis):
        if self.shape[axis] % 2 != 0:
            raise ValueError("Size of complex indirect dimension "
                             "must be even, not %d"%self.shape[axis])
        self.is_complex[axis] = True

    def make_real(self, axis):
        self.is_complex[axis] = False
        pass

    @property
    def real(self):
        r = self[..., ::2]
        r.is_complex[-1]=False
        return r.squeeze()

    @property
    def imag(self):
        r = self[..., 1::2]
        r.is_complex[-1]=False
        return r.squeeze()

    def part(self, select='ALL'):
        ar = self.copy()
        if select=='ALL':
            select = 'R'*self.ndim
        for axis, component in enumerate(select):
            if self.is_complex[axis]:
                ar = ar.swapaxes(axis,-1)
                if component == 'R':
                    ar = ar[..., ::2]
                elif component == 'I':
                    ar = ar[..., 1::2]
                ar = ar.swapaxes(axis, -1)
                ar.is_complex[axis] = False
        return ar.squeeze()

    @property
    def RR(self):
        if self.ndim != 2: raise TypeError('Not a two dimensional array')
        return self.part('RR')

    @property
    def RI(self):
        if self.ndim != 2: raise TypeError('Not a two dimensional array')
        return self.part('RI')

    @property
    def IR(self):
        if self.ndim != 2: raise TypeError('Not a two dimensional array')
        return self.part('IR')

    @property
    def II(self):
        if self.ndim != 2: raise TypeError('Not a two dimensional array')
        return self.part('II')

    @property
    def trueshape(self):
        return self.part('ALL').shape

    @property
    def truesize(self):
        return self.part('ALL').size

    def __array_finalize__(self, obj):
        super(HCArray, self).__array_finalize__(obj)
        self.is_complex = getattr(obj, 'is_complex',None).copy()

    def __getitem__(self, item):
        is_complex = self.is_complex
        # TODO: complexity handling!
        data  = super(HCArray, self).__getitem__(item)
        data.is_complex = is_complex
        return data

    def __str__(self):
         if self.is_complex[-1]:
             ar = self.real + self.imag*1j
         else:
             ar = self
         return ar.__str__()

hcarray = HCArray