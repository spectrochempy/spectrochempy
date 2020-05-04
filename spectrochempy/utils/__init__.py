# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
Package containing various utilities classes and functions.

"""
# some useful constants
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

EPSILON = epsilon = np.finfo(float).eps
"Minimum value before considering it as zero value"

INPLACE = "INPLACE"
"Flag used to specify inplace slicing"

# masked arrays
# ----------------------------------------------------------------------------------------------------------------------
from numpy.ma.core import (masked as MASKED, nomask as NOMASK, MaskedArray,
                           MaskedConstant)

# import util files content
# ----------------------------------------------------------------------------------------------------------------------
from .exceptions import *
from .fake import *
from .file import *
from .misc import *
from .print import *
from .system import *
from .excel import *
from .matplolib_utils import *
from .arrays import *
from .docstring import *
from .meta import *
from .configurable import MetaConfigurable

# internal utilities
# ----------------------------------------------------------------------------------------------------------------------
class _TempBool(object):
    """Wrapper around a boolean defining an __enter__ and __exit__ method

    Notes
    -----
    If you want to use this class as an instance property, rather use the
    :func:`_temp_bool_prop` because this class as a descriptor is ment to be a
    class descriptor"""

    #: default boolean value for the :attr:`value` attribute
    default=False

    #: boolean value indicating whether there shall be a validation or not
    value = False

    def __init__(self, default=False):
        """
        Parameters
        ----------
        default=bool
            value of the object"""
        self.default=default
        self.value = default
        self._entered = []

    def __enter__(self):
        self.value = not self.default
        self._entered.append(1)

    def __exit__(self, type, value, tb):
        self._entered.pop(-1)
        if not self._entered:
            self.value = self.default

    def __bool__(self):
        return self.value

    def __repr__(self):
        return repr(bool(self))

    def __str__(self):
        return str(bool(self))

    def __call__(self, value=None):
        """
        Parameters
        ----------
        value : bool or None
            If None, the current value will be negated. Otherwise the current
            value of this instance is set to the given `value`"""
        if value is None:
            self.value = not self.value
        else:
            self.value = value

    def __get__(self, instance, owner):
        return self

    def __set__(self, instance, value):
        self.value = value


def _temp_bool_prop(propname, doc="", default=False):
    """Creates a property that uses the :class:`_TempBool` class

    Parameters
    ----------
    propname : str
        The attribute name to use. The _TempBool instance will be stored in the
        ``'_' + propname`` attribute of the corresponding instance
    doc : str
        The documentation of the property
    default=bool
        The default value of the _TempBool class"""

    def getx(self):
        if getattr(self, '_' + propname, None) is not None:
            return getattr(self, '_' + propname)
        else:
            setattr(self, '_' + propname, _TempBool(default))
        return getattr(self, '_' + propname)

    def setx(self, value):
        getattr(self, propname).value = bool(value)

    def delx(self):
        getattr(self, propname).value = default

    return property(getx, setx, delx, doc)
