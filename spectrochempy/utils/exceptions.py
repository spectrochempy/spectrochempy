# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import warnings
from .print import pstr

__all__ = ['SpectroChemPyWarning',
           'SpectroChemPyDeprecationWarning',
           'SpectroChemPyException',
           'deprecated',
           'info_', 'debug_', 'error_', 'warning_'
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
def deprecated(message):
    """
    Deprecation decorator

    Parameters
    ----------
    message: str,
        the deprecation message

    """

    def deprecation_decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn("The function `{} is deprecated: {}".format(
                func.__name__, message),
                SpectroChemPyDeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator


# ======================================================================================================================
# logging functions
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def info_(*args, **kwargs):
    from spectrochempy.core import log
    s = ""
    for a in args:
        s += pstr(a, **kwargs)
        s = s.replace('\0', '')
    log.info(s)


# ----------------------------------------------------------------------------------------------------------------------
def debug_(*args):
    from spectrochempy.core import log
    s = ""
    for a in args:
        s += pstr(a)
        s = s.replace('\0', '')
    log.debug(s)


# ----------------------------------------------------------------------------------------------------------------------
def error_(*args):
    from spectrochempy.core import log
    s = ""
    for a in args:
        s += pstr(a)
        s = s.replace('\0', '')
    log.error(s)


# ----------------------------------------------------------------------------------------------------------------------
def warning_(*args):
    s = ""
    for a in args:
        s += pstr(a)
        s = s.replace('\0', '')
    warnings.warn(s, SpectroChemPyWarning)

# ======================================================================================================================
# EOF
