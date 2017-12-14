# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version 2.1)
# See full LICENSE agreement in the root directory
# =============================================================================



__all__ = ['interleave', 'interleaved2complex', 'StdDev', 'set_operators']

import operator
import numpy as np
from traitlets import HasTraits, Float

def interleave(data):
    """
    This function make an array where real and imaginary part are interleaved

    Parameters
    ==========
    data : complex ndarray
        If the array is not complex, then data are
        returned inchanged

    Returns
    =======
    data : ndarray with interleaved complex data

    iscomplex : is the data are really complex it is set to true

    """
    if np.any(np.iscomplex(data)) or data.dtype == np.complex:
        # unpack (we must double the last dimension)
        newshape = list(data.shape)
        newshape[-1] *= 2
        new = np.empty(newshape)
        new[..., ::2] = data.real
        new[..., 1::2] = data.imag
        return new, True
    else:
        return data, False


def interleaved2complex(data):
    """
    Make a complex array from interleaved data

    """
    return data[..., ::2] + 1j * data[..., 1::2]


# =============================================================================
# helper class for uncertainties
# =============================================================================

class StdDev(HasTraits):
    """
    A helper class used to set the uncertainty on array values

    Examples
    --------


    """
    data = Float(0.)

# =============================================================================
# ARITHMETIC ON NDDATASET
# =============================================================================

# unary operators
UNARY_OPS = ['neg', 'pos', 'abs']

# binary operators
CMP_BINARY_OPS = ['lt', 'le', 'ge', 'gt']

NUM_BINARY_OPS = ['add', 'sub', 'and', 'xor', 'or',
                  'mul', 'truediv', 'floordiv', 'pow']


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