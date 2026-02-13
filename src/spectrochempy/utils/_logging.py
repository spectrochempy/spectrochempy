# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Logging utilities for SpectroChemPy.

This module provides logging functions that do NOT depend on the application
module, ensuring that core modules can use logging without triggering
matplotlib imports or other application-level dependencies.

These functions are designed to be compatible with the logging functions
previously provided by spectrochempy.application.application.
"""

import logging
import warnings

from spectrochempy.utils.exceptions import SpectroChemPyError

# Get or create the SpectroChemPy logger
_logger = logging.getLogger("spectrochempy")


def _ensure_logger_configured():
    """Ensure the logger has a basic handler configured."""
    if not _logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)


def info_(msg, *args, **kwargs):
    """
    Log an info message.

    Parameters
    ----------
    msg : str
        The message to log.
    *args : tuple
        Additional arguments passed to logging.info.
    **kwargs : dict
        Additional keyword arguments.
    """
    _ensure_logger_configured()
    _logger.info(msg, *args, **kwargs)


def debug_(msg, *args, **kwargs):
    """
    Log a debug message.

    Parameters
    ----------
    msg : str
        The message to log.
    *args : tuple
        Additional arguments passed to logging.debug.
    **kwargs : dict
        Additional keyword arguments.
    """
    _ensure_logger_configured()
    _logger.debug("DEBUG | " + msg, *args, **kwargs)


def error_(*args, **kwargs):
    """
    Report an error.

    This function handles several call signatures:
    - error_(exception): Report an exception (logged, not raised)
    - error_("message"): Report a SpectroChemPyError with the message
    - error_(ExceptionType, "message"): Report a specific exception type with message

    Parameters
    ----------
    *args : tuple
        Arguments specifying the error.
        - If first arg is an Exception, that exception is reported (logged)
        - If single string arg, a SpectroChemPyError is created
        - If two args (type, message), the specified exception type is raised
    **kwargs : dict
        Additional keyword arguments.
    """
    _ensure_logger_configured()

    if not args:
        raise KeyError("wrong arguments have been passed to error_")

    if isinstance(args[0], Exception):
        e = args[0]
        etype = type(e)
        emessage = str(e)
    elif len(args) == 1 and isinstance(args[0], str):
        etype = SpectroChemPyError
        emessage = str(args[0])
    elif len(args) == 2:
        etype = args[0]
        emessage = str(args[1])
    else:
        raise KeyError("wrong arguments have been passed to error_")

    # Log the error (not raise) - consistent with original application.error_ behavior
    _logger.error(f"{etype.__name__}: {emessage}")


def warning_(msg, *args, **kwargs):
    """
    Issue a warning message.

    Parameters
    ----------
    msg : str
        The warning message.
    *args : tuple
        Additional arguments passed to warnings.warn.
    **kwargs : dict
        Additional keyword arguments (including stacklevel).
    """
    # Default stacklevel is 2 to point to the caller's location
    stacklevel = kwargs.pop("stacklevel", 2)
    warnings.warn(msg, *args, stacklevel=stacklevel, **kwargs)
