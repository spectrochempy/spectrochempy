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
import functools
import logging
import operator

# =============================================================================
# third-party imports
# =============================================================================
from six import PY3
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from uncertainties import unumpy as unp
from spectrochempy.core.units import Quantity
from spectrochempy.core.dataset.ndarray import NDArray

# =============================================================================
# Constants
# =============================================================================

__all__ =['NDMath', 'set_operators']

from spectrochempy.application import log



class NDMath(object):
    """
    Examples
    --------

    >>> from spectrochempy.api import *
    >>> source = NDDataset.load('mydataset.scp')
    >>> source             #doctest: +ELLIPSIS
    NDDataset([[    2.06,...,     1.24]])
    >>> np.negative(source) #doctest: +ELLIPSIS
    NDDataset([[   -2.06,...,    -1.24]])

    """

    @property
    def __array_struct__(self):
        return self._data.__array_struct__

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------

    def pipe(self, func, *args, **kwargs):
        """Apply func(self, \*args, \*\*kwargs)

        Parameters
        ----------
        func : function

            function to apply to the NDDataset.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the NADArray object.

        args : positional arguments passed into ``func``.

        kwargs : a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Notes
        -----

        Use ``.pipe`` when chaining together functions that expect
        on `NDDataset`.

        """
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError('%s is both the pipe target and a keyword '
                                 'argument' % target)
            kwargs[target] = self
            return func(*args, **kwargs)

            return func(self, *args, **kwargs)

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    # the following methods are to give NDArray based class
    # a behavior similar to np.ndarray regarding the ufuncs

    # def __array_prepare(self, *args, **kwargs):
    #    pass

    def __array_wrap__(self, *args):
        # called when element-wise ufuncs are applied to the array

        f, objs, huh = args[1]

        # case of complex data
        if self.is_complex is not None:
            if self.is_complex[-1] and \
                            f.__name__ in ['real', 'imag', 'conjugate', 'absolute']:
                return getattr(objs[0], f.__name__)()
            if self.is_complex[-1] and f.__name__ in ["fabs", ]:
                # fonction not available for complex data
                raise ValueError("{} does not accept complex data ".format(f))

        # not a complex data or a function not in the ['real', 'imag', 'conjugate']
        if f.__name__ in ['absolute']:
            f = np.fabs

        data, uncertainty, units, mask = self._op(f, objs, ufunc=True)
        history = 'ufunc ' + f.__name__
        return self._op_result(data, uncertainty, units, mask, history)

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    def _op_result(self, data, uncertainty=None, units=None, mask=None, history=None):

        new = self.copy()
        new._data = copy.deepcopy(data)

        if uncertainty is not None:
            new._uncertainty = copy.deepcopy(uncertainty)
        if units is not None:
            new._units = copy.copy(units)
        if mask is not None:
            new._mask = copy.copy(mask)
        if history is not None and hasattr(new, 'history'):
            new._history.append(history.strip())
        return new

    @staticmethod
    def _op(f, objs, ufunc=False):

        sameunits = False

        fname = f.__name__

        if fname in ['lt', 'le', 'ge', 'gt', 'add', 'sub']:
            sameunits = True

        objs = list(objs)
        obj = copy.deepcopy(objs.pop(0))

        isdataset = True
        isaxe = False
        iscomplex = False

        if isinstance(obj, NDArray):
            d = obj.data

            # units?
            if not obj.unitless:
                q = Quantity(1., obj.units)  # create a Quantity from the units
            else:
                q = 1.

            if hasattr(obj, 'axes'):
                # uncertainties?
                # if any create an UFloat type if any uncertainty
                d = obj._uarray(d, obj._uncertainty)

                # complex
                iscomplex = obj.is_complex[-1]

            else:
                isdataset = False
                isaxe = True

            # mask?
            d = obj._umasked(d, obj._mask)

        else:
            # assume an array or a scalar (possibly a Quantity)
            isdataset = False

            if hasattr(obj, 'units'):
                if not obj.dimensionless:
                    q = Quantity(1., obj.units)  # create a Quantity from the units
                else:
                    q = 1.
                d = d = obj.magnitude
            else:
                q = 1.
                d = obj

        # other operand
        args = []
        argunits = []
        argcomplex = []

        # TODO: check the units with respect to some ufuncs or ops
        for o in objs:
            other = copy.deepcopy(o)
            if isinstance(other, NDArray):

                # if isaxe:
                #     raise TypeError('when the first argument is an Axis, '
                #                      'second argument cannot be an Axis or '
                #                      'NDDataset instance')
                #
                # if isdataset and not hasattr(other, '_axes'):
                #     raise TypeError('when the first argument is a NDDataset, '
                #                     'second argument cannot be an Axis'
                #                     ' instance')

                if isdataset and other._axes != obj._axes:
                    raise ValueError("axes properties do not match")

                # rescale according to units
                if not other.unitless:
                    if hasattr(obj, 'units'):  # obj is a Quantity
                        if sameunits:
                            other.to(obj._units, inplace=True)  # must also rescale uncertainty
                        argunits.append(Quantity(1., other._units))
                    else:
                        argunits.append(1.)
                else:
                    argunits.append(1.)

                arg = other.data
                arg = other._uarray(arg, other._uncertainty)

                # mask?
                arg = other._umasked(arg, other._mask)

            else:
                if isinstance(other, Quantity):
                    arg = other.magnitude
                    argunits.append(Quantity(1., other._units))
                else:
                    arg = other
                    argunits.append(1.)
            args.append(arg)

            # complex?
            if hasattr(other, '_is_complex') and other.is_complex is not None:
                argcomplex.append(other.is_complex[-1])
            else:
                argcomplex.append(False)

        if ufunc:
            # with use of the numpy functions of uncertainty package
            # some transformation to handle missing function in the unumpy module
            if fname == 'exp2':
                data = unp.exp(d * np.log(2.))
            elif fname == 'log2':
                data = unp.log(d) / np.log(2.)
            elif fname == 'negative':
                data = -d
            elif fname == 'reciprocal':
                data = 1. / d
            elif fname == 'square':
                data = d ** 2
            elif fname == 'rint':
                data = obj._uarray(np.rint(unp.nominal_values(d)),
                                   unp.std_devs(d))
            elif fname == 'sign':
                data = np.sign(unp.nominal_values(d))
            else:
                data = getattr(unp, fname)(d, *args)
        else:
            if iscomplex:
                # pack to complex
                d = d[..., ::2] + 1j * d[..., 1::2]
            data = f(d, *args)
            if iscomplex:
                # unpack
                new = np.empty_like(obj.data)
                new[..., ::2] = data.real
                new[..., 1::2] = data.imag
                data = new

        # unpack the data
        uncertainty = unp.std_devs(data)
        data = unp.nominal_values(data).astype(float)

        # get possible mask
        if isinstance(data, np.ma.MaskedArray):
            mask = data.mask
            data = data.data
        else:
            mask = np.zeros_like(data, dtype=bool)

        # redo the calculation with the units to found the final one
        q = f(q, *argunits)
        if hasattr(q, 'units'):
            units = q.units
        else:
            units = None

        return data, uncertainty, units, mask

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self):
            data, uncertainty, units, mask = self._op(f, [self])
            history = 'unary op : ' + f.__name__
            return self._op_result(data, uncertainty, units, mask, history)

        return func

    @staticmethod
    def _binary_op(f, reflexive=False, **kwargs):
        @functools.wraps(f)
        def func(self, other):
            if not reflexive:
                objs = [self, other]
            else:
                objs = [other, self]
            data, uncertainty, units, mask = self._op(f, objs)
            history = 'binary op : ' + f.__name__ + ' with %s ' % str(other)
            return self._op_result(data, uncertainty, units, mask, history)

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            objs = [self, other]
            data, uncertainty, units, mask = self._op(f, objs)
            self._data = data
            self._uncertainty = uncertainty
            self._units = units
            self._mask = mask
            if hasattr(self, '_history'):
                self._history.append('inplace binary op : ' + f.__name__ + ' with %s ' % str(other))
            return self

        return func


# =============================================================================
# ARITHMETIC ON NDDATASET
# =============================================================================

# unary operators
UNARY_OPS = ['neg', 'pos', 'abs']

# binary operators
CMP_BINARY_OPS = ['lt', 'le', 'ge', 'gt']

NUM_BINARY_OPS = ['add', 'sub', 'and', 'xor', 'or',
                  'mul', 'truediv', 'floordiv', 'pow']
if not PY3:
    NUM_BINARY_OPS.append('div')


def _op_str(name):
    return '__%s__' % name


def _get_op(name):
    return getattr(operator, _op_str(name))


def set_operators(cls, priority=50):
    # adapted from Xarray

    cls.__array_priority__ = priority

    # unary ops
    for name in UNARY_OPS:
        setattr(cls, _op_str(name), cls._unary_op(_get_op(name)))

    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, _op_str(name), cls._binary_op(_get_op(name)))

    for name in NUM_BINARY_OPS:
        # only numeric operations have in-place and reflexive variants
        setattr(cls, _op_str('r' + name),
                cls._binary_op(_get_op(name), reflexive=True))

        setattr(cls, _op_str('i' + name),
                cls._inplace_binary_op(_get_op('i' + name)))
