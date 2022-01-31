# -*- coding: utf-8 -*-

# ============================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ============================================================================================

import warnings
from contextlib import contextmanager
from .misc import DEFAULT_DIM_NAME

__all__ = [
    "SpectroChemPyWarning",
    "SpectroChemPyException",
    "UnitsCompatibilityError",
    "IncompatibleShapeError",
    "InvalidDimensionNameError",
    "InvalidCoordinatesTypeError",
    "InvalidCoordinatesSizeError",
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


class IncompatibleShapeError(SpectroChemPyException):
    """
    Exception raised when shapes of the elements are incompatibles for math operations.
    """

    def __init__(self, obj1, obj2, extra_msg=""):

        shape1 = obj1.shape if hasattr(obj1, "shape") else "<no shape>"
        shape2 = obj2.shape if hasattr(obj2, "shape") else "<no shape>"
        self.message = (
            f"The shape of obj1:[{obj1.__class__.__name__}, shape={shape1}] and that of"
            f" obj2:[{obj2.__class__.__name__}, shape={shape2}] are different!. {extra_msg}"
        )


# ------------------------------------------------------------------------------
class InvalidDimensionNameError(SpectroChemPyException):
    """
    Exception raised when dimension name are invalid.
    """

    def __init__(self, name, available_names=DEFAULT_DIM_NAME):

        self.message = (
            f"dim name must be one of {tuple(available_names)} "
            f"with an optional subdir indication (e.g., 'x_2'. dim=`{name}` was given!"
        )

        super().__init__(self.message)


class InvalidCoordinatesTypeError(SpectroChemPyException):
    """
    Exception raised when coordinates type is invalid.
    """


class InvalidCoordinatesSizeError(SpectroChemPyException):
    """
    Exception raised when size of coordinates does not match what is expected.
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
    """
    This exception is issued when a wrong protocol is secified to the
    spectrochempy importer.

    Parameters
    ----------
    protocol : str
        The protocol string that was at the origin of the exception.
    available_protocols : list of str
        The available (implemented) protocols.
    """

    def __init__(self, protocol, available_protocols):

        self.message = (
            f"IO - The `{protocol}` protocol is unknown or not yet implemented.\n"
        )
        f"It is expected to be one of {tuple(available_protocols)}"

        super().__init__(self.message)


# ------------------------------------------------------------------------------
def deprecated(type="method", replace="", extra_msg=""):
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
                f" `{func.__name__}` {type} is now deprecated and could be completely removed in version 0.5.*."
                + f" Use `{replace}`."
                if replace
                else "" + f" {extra_msg}."
                if extra_msg
                else "",
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

        Parameters
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
