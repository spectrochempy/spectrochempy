# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import warnings
from .print import pstr

__all__ = ['SpectroChemPyWarning',
           'SpectroChemPyDeprecationWarning',
           'SpectroChemPyException',
           'UnitsCompatibilityError',
           'deprecated',
          ]


# ======================================================================================================================
# Exception and warning  subclass
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
class SpectroChemPyWarning(Warning):
    """
    The base warning class for SpectroChemPy warnings.
    
    """


# ----------------------------------------------------------------------------------------------------------------------
class SpectroChemPyDeprecationWarning(SpectroChemPyWarning):
    """
    A warning class to indicate that something is deprecated.
    
    """


# ----------------------------------------------------------------------------------------------------------------------
class SpectroChemPyException(Exception):
    """
    The base exception class for SpectroChemPy
    
    """

# ----------------------------------------------------------------------------------------------------------------------
class UnitsCompatibilityError(SpectroChemPyException):
    """
    Exception raised when units are not compatible, preventing some mathematical operations
    
    """
    


# ----------------------------------------------------------------------------------------------------------------------
def deprecated(message):
    """
    Deprecation decorator

    Parameters
    ----------
    message : str,
        the deprecation message

    """

    def deprecation_decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn("The function `{} is deprecated : {}".format(
                func.__name__, message),
                SpectroChemPyDeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator

# ======================================================================================================================
# EOF
