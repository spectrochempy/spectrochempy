# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================


__all__ = ['interleaved2complex', 'set_operators']

import operator
import numpy as np
from traitlets import HasTraits, Float


def interleaved2complex(data):
    """
    Make a complex array from interleaved data

    """
    return data[..., ::2] + 1j * data[..., 1::2]


# ======================================================================================================================
# ARITHMETIC ON NDDATASET
# ======================================================================================================================

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
