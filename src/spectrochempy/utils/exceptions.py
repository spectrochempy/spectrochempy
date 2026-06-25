# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""SpectroChemPy specific exceptions."""

import inspect
from contextlib import contextmanager
from contextlib import suppress

import pint

# import pytz


# ======================================================================================
# Warning subclasses
# ======================================================================================
class KeyErrorWarning(UserWarning):
    """Warning raised when an issue arise regarding units."""


class UnitErrorWarning(UserWarning):
    """Warning raised when an issue arise regarding units."""


class LabelErrorWarning(UserWarning):
    """Warning raised when an issue arise regarding labels."""


class ValueErrorWarning(UserWarning):
    """Warning raised when an issue arise arguments or attributes."""


class NeedsUpdateWarning(UserWarning):
    """Warning raised when an issue arise arguments or attributes."""


# ======================================================================================
# Exception Subclasses
# ======================================================================================
class SpectroChemPyError(Exception):
    """The base exception class for SpectroChemPy."""

    def __init__(self, message):
        self.message = message

        super().__init__(message)


# ======================================================================================
# Exceptions for configurable models
# ======================================================================================
class NotTransformedError(SpectroChemPyError):
    """
    Exception raised when a model has not yet been applied to a dataset, but one use one of its method.

    Parameters
    ----------
    attr : method, optional
        The method from which the error was issued. In general, it is determined
        automatically.

    """

    def __init__(self, attr=None, message=None):
        if message is not None:
            super().__init__(message)
            return
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if attr is None else attr
        model = frame.f_locals["self"].name
        message = (
            f"To use `{caller}` ,  the method `apply` of model `{model}`"
            f" should be executed first"
        )
        super().__init__(message)


class NotFittedError(NotTransformedError):
    """
    Exception raised when an analysis estimator is not fitted but one use one of its method.

    Parameters
    ----------
    attr : method, optional
        The method from which the error was issued. In general, it is determined
        automatically.

    """

    def __init__(self, attr=None, message=None):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if attr is None else attr
        model = frame.f_locals["self"].name
        message = (
            f"To use `{caller}` ,  the method `fit` of model `{model}`"
            f" should be executed first"
        )
        super().__init__(message=message)


class CastingError(SpectroChemPyError):
    """Exception raised when an array cannot be cast to the required data type."""

    def __init__(self, dtype, message):
        message = f" Assigned value has type {dtype} but {message}"
        super().__init__(message)


class InvalidNameError(SpectroChemPyError):
    """Exception when a object name is not valid."""


class ShapeError(SpectroChemPyError):
    """Exception raised when an array cannot be set due to a wrong shape."""

    def __init__(self, shape, message):
        message = f" Assigned value has shape {shape} but {message}"
        super().__init__(message)


class MissingDataError(SpectroChemPyError):
    """Exception raised when no data is present in an object."""


class NDDatasetAttributeError(SpectroChemPyError):
    """Exception raised when a dataset attribute was not found."""

    def __init__(self, attr):
        message = f" NDDataset attribute `{attr}` was not found."
        super().__init__(message)


class CoordinatesAttributeError(SpectroChemPyError):
    """Exception raised when a dataset attribute was not found."""

    def __init__(self, attr):
        message = f" Coord attribute `{attr}` was not found."
        super().__init__(message)


class MissingCoordinatesError(SpectroChemPyError):
    """Exception raised when no coordinates in present in an object."""


class LabelsError(SpectroChemPyError):
    """
    Exception raised when an array cannot be labeled.

    For instance, if the array is multidimensional.
    """


# class UnknownTimeZoneError(pytz.UnknownTimeZoneError):
#    """
#    Exception raised when Timezone code is not recognized.
#    """


class UnitsCompatibilityError(SpectroChemPyError):
    """Exception raised when units are not compatible, preventing some mathematical operations."""


class InvalidUnitsError(SpectroChemPyError):
    """Exception raised when units is not valid."""


def format_incompatible_units_message(operation, left_units, right_units, dim=None):
    """
    Build a clear error message for incompatible units in coordinate-aware operations.

    Parameters
    ----------
    operation : str
        Short operation description, e.g. ``"align datasets"``.
    left_units, right_units : any
        Units involved in the failed operation.
    dim : str, optional
        Dimension name, when available.
    """
    location = f" along dimension '{dim}'" if dim is not None else ""
    return (
        f"Cannot {operation}{location}: incompatible coordinate units "
        f"({left_units} and {right_units}). "
        "Convert the coordinates to compatible units before retrying."
    )


class InvalidReferenceError(SpectroChemPyError):
    """Exception raised when a reference to another coordinate is not valid."""


class DimensionalityError(pint.DimensionalityError):
    """Exception raised when units have a dimensionality problem."""


class CoordinatesMismatchError(SpectroChemPyError):
    """Exception raised when object coordinates differ."""

    def __init__(self, obj1, obj2, extra_msg=""):
        self.message = f"Coordinates [{obj1}] and [{obj2}] mismatch. {extra_msg}"
        super().__init__(self.message)


class DimensionsCompatibilityError(SpectroChemPyError):
    """Exception raised when dimensions are not compatible for concatenation for instance."""


class IncompatibleShapeError(SpectroChemPyError):
    """Exception raised when shapes of the elements are incompatibles for math operations."""

    def __init__(self, obj1, obj2, extra_msg=""):
        self.message = f"Shapes of [{obj1}] and [{obj2}] mismatch. {extra_msg}"
        super().__init__(self.message)


class NonWritableCoordSetError(SpectroChemPyError):
    """Exception raised when the CoordSEt is readonly, but an attempt to write it has been done."""


class InvalidDimensionNameError(SpectroChemPyError):
    """Exception raised when dimension name are invalid."""

    from spectrochempy.utils.constants import DEFAULT_DIM_NAME

    def __init__(self, name, available_names=DEFAULT_DIM_NAME):
        self.message = (
            f"dim name must be one of {tuple(available_names)} "
            f"with an optional subdir indication (e.g., 'x_2') but axis=`"
            f"{name}` was given!"
        )
        super().__init__(self.message)


class InvalidCoordinatesSizeError(SpectroChemPyError):
    """Exception raised when size of coordinates does not match what is expected."""


class InvalidCoordinatesTypeError(SpectroChemPyError):
    """Exception raised when coordinates type is invalid."""


class InvalidCoordSetSizeError(SpectroChemPyError):
    """Exception raised when size of coordset does not match what is expected."""


class ProtocolError(SpectroChemPyError):
    """
    Exception raised when an unsupported protocol is specified to the importer.

    Parameters
    ----------
    protocol : str
        The protocol string that was at the origin of the exception.
    available_protocols : list of str
        The available (implemented) protocols.
    filename : str or path-like, optional
        The file that could not be read.
    detected_protocol : str, optional
        The protocol inferred from the filename when it conflicts with ``protocol``.
    """

    def __init__(
        self,
        protocol,
        available_protocols,
        filename=None,
        detected_protocol=None,
    ):
        supported = ", ".join(f"'{item}'" for item in sorted(set(available_protocols)))
        location = (
            f"Cannot read '{filename}'" if filename is not None else "Cannot read"
        )

        if detected_protocol is not None:
            self.message = (
                f"{location} with protocol='{protocol}'.\n"
                f"The filename indicates protocol='{detected_protocol}'.\n"
                f"Choose protocol='{detected_protocol}' or omit the protocol argument."
            )
        else:
            self.message = (
                f"{location} with protocol='{protocol}'.\n"
                "The requested protocol is unknown or not implemented.\n"
                f"Supported protocols are: {supported}."
            )

        super().__init__(self.message)


class UnsupportedOriginError(SpectroChemPyError, NotImplementedError):
    """Exception raised when a reader does not support a requested origin."""

    def __init__(self, filename, protocol, origin, supported_origins):
        supported = ", ".join(f"'{item}'" for item in supported_origins)
        protocol_name = protocol.upper()
        message = (
            f"Cannot read {protocol_name} file '{filename}' with origin='{origin}'.\n"
            f"Supported {protocol_name} origins are: {supported}.\n"
            "Remove the origin argument or choose a supported origin."
        )
        super().__init__(message)


class WrongFileFormatError(SpectroChemPyError, TypeError):
    """Exception raised when the file format is incorrect or not supported."""


@contextmanager
def ignored(*exc):
    """
    Ignore exceptions within a context.

    This is equivalent to::

        try :
            <body>
        except exc :
            pass

    Parameters
    ----------
    *exc : Exception
        One or several exceptions to ignore.

    Examples
    --------
    >>> import os
    >>> from spectrochempy.utils.exceptions import ignored
    >>>
    >>> with ignored(OSError):
    ...     os.remove('file-that-does-not-exist')

    """
    with suppress(exc):
        yield
