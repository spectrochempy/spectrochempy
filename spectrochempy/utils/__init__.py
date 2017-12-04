# -*- coding: utf-8 -*-
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

"""
This package provides utilities classes and functions.

"""
import numpy as np

EPSILON = epsilon = np.finfo(float).eps

from .decorators import *
from .exceptions import *
from .file import *
from .misc import *
from .system import *
from .utilities import *
from .introspect import *
from .matplolib_utils import *
from .arrayutils import *
from .version import *
from .docstring import *
from .traittypes import *
from .meta import *

# internal utilities

class _TempBool(object):
    """Wrapper around a boolean defining an __enter__ and __exit__ method

    Notes
    -----
    If you want to use this class as an instance property, rather use the
    :func:`_temp_bool_prop` because this class as a descriptor is ment to be a
    class descriptor"""

    #: default boolean value for the :attr:`value` attribute
    default = False

    #: boolean value indicating whether there shall be a validation or not
    value = False

    def __init__(self, default=False):
        """
        Parameters
        ----------
        default: bool
            value of the object"""
        self.default = default
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
        value: bool or None
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
    propname: str
        The attribute name to use. The _TempBool instance will be stored in the
        ``'_' + propname`` attribute of the corresponding instance
    doc: str
        The documentation of the property
    default: bool
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
