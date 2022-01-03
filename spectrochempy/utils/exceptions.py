# -*- coding: utf-8 -*-

# ============================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ============================================================================================

import warnings
from contextlib import contextmanager

__all__ = [
    "SpectroChemPyWarning",
    "SpectroChemPyException",
    "UnitsCompatibilityError",
    "DimensionsCompatibilityError",
    "CoordinateMismatchError",
    "ProtocolError",
    "deprecated",
    "ignored",
]


# ==============================================================================
# Exception and Warning Subclass
# ==============================================================================

# ------------------------------------------------------------------------------
class SpectroChemPyWarning(Warning):
    """
    The base warning class for SpectroChemPy warnings.
    """


# ------------------------------------------------------------------------------
class SpectroChemPyException(Exception):
    """
    The base exception class for SpectroChemPy.
    """


# ------------------------------------------------------------------------------
class UnitsCompatibilityError(SpectroChemPyException):
    """
    Exception raised when units are not compatible,
    preventing some mathematical operations.
    """


# ------------------------------------------------------------------------------
class DimensionsCompatibilityError(SpectroChemPyException):
    """
    Exception raised when dimensions are not compatible
    for concatenation for instance.
    """


# ------------------------------------------------------------------------------
class CoordinateMismatchError(SpectroChemPyException):
    """
    Exception raised when object coordinates differ.
    """


class ProtocolError(SpectroChemPyException):
    def __init__(self, protocol, available_protocols):
        """
        This exception is issued when a wrong protocol is secified to the spectrochempy importer.

        Parameters
        ==========
        protocol: str
            The protocol string that was at the origin of the exception.
        available_protocols: list of str
            The available (implemented) protocols.
        """
        self.message = (
            f"IO - The `{protocol}` protocol is unknown or not yet implemented.\n"
        )
        f"It is expected to be one of {tuple(available_protocols)}"

        super().__init__(self.message)


# ------------------------------------------------------------------------------
def deprecated(message):
    """
    Deprecation decorator.

    Parameters
    ----------
    message : str,
        The deprecation message.
    """

    def deprecation_decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                "The function `{} is deprecated : {}".format(func.__name__, message),
                DeprecationWarning,
            )
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator


# ..............................................................................
try:
    from contextlib import ignored
except ImportError:

    @contextmanager
    def ignored(*exceptions):
        """
        A context manager for ignoring exceptions.

        This is equivalent to::

            try :
                <body>
            except exceptions :
                pass

        parameters
        ----------
        *exceptions : Exception
            One or several exceptions to ignore.

        Examples
        --------

        >>> import os
        >>> from spectrochempy.utils import ignored
        >>>
        >>> with ignored(OSError):
        ...     os.remove('file-that-does-not-exist')
        """

        try:
            yield
        except exceptions:
            pass


# ==============================================================================
# EOF
