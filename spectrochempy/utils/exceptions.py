# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import warnings

__all__ = ['SpectroChemPyWarning', 'SpectroChemPyException', 'UnitsCompatibilityError', 'DimensionsCompatibilityError',
           'CoordinateMismatchError', 'ProtocolError', 'deprecated', ]


# ======================================================================================================================
# Exception and warning  subclass
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
class SpectroChemPyWarning(Warning):
    """
    The base warning class for SpectroChemPy warnings.
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
class DimensionsCompatibilityError(SpectroChemPyException):
    """
    Exception raised when dimensions are not compatible for concatenation for instance
    """


# ----------------------------------------------------------------------------------------------------------------------
class CoordinateMismatchError(SpectroChemPyException):
    """
    Exception raised when object coordinates differ
    """


class ProtocolError(SpectroChemPyException):

    def __init__(self, protocol, available_protocols):
        print(f'The `{protocol}` protocol is unknown or not yet implemented:\n'
              f'it is expected to be one of {tuple(available_protocols)}')


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
            warnings.warn("The function `{} is deprecated : {}".format(func.__name__, message), DeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator

# ======================================================================================================================
# EOF
