# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import functools
import inspect
from warnings import warn


def preserve_signature(f):
    """
    A decorator for decorators, which preserves the signature of the function
    being wrapped. This preservation allows IDE function parameter hints to work
    on the wrapped function. To do this, the ``__signature__`` dunder is defined, or
    inherited, from the function being wrapped to the resulting wrapped function.

    Copied from
    https://github.com/PlasmaPy/PlasmaPy/blob/main/plasmapy/utils/decorators/helpers.py
    (PlasmaPy, LICENSE BSD-3)

    Parameters
    ----------
    f: callable
        The function being wrapped.

    Returns
    -------
    callable
        Wrapped version of the function.

    Examples
    --------
    >>> def a_decorator(f):
    ...     @preserve_signature
    ...     @functools.wraps(f)
    ...     def wrapper(*args, **kwargs):
    ...         return wrapper(*args, **kwargs)
    ...
    ...     return wrapper
    """
    # add '__signature__' if it does not exist
    # - this will preserve parameter hints in IDE's
    if not hasattr(f, "__signature__"):
        f.__signature__ = inspect.signature(f)

    return f


# noinspection PyDeprecation
@preserve_signature
def deprecated(name=None, *, kind="method", replace="", removed=None, extra_msg=""):
    """
    Deprecation decorator.

    Parameters
    ----------
    name : str
        If name is specified, kind is mandatory set to attribute
        and the deprecated function is no more acting as a decorator.
    kind : str
        By default, it is method.
    replace : str, optional, default:None
        Name of the method that replace the deprecated one or None
    extra_msg : str
        Additional message.
    removed : str, optional
        Version string when this method will be removed
    """

    def output_warning_message(name, kind, replace, removed, extra_msg):
        sreplace = f"Use `{replace}` instead. " if replace is not None else ""
        msg = f" The `{name}` {kind} is now deprecated. {sreplace}"
        sremoved = f"version {removed}" if removed else "future version"
        msg += f"`{name}` {kind} will be removed in {sremoved}. "
        msg += extra_msg
        warn(
            msg,
            category=DeprecationWarning,
        )

    if name is not None:
        kind = "attribute"
        output_warning_message(name, kind, replace, removed, extra_msg)
        return

    def deprecation_decorator(func):
        @preserve_signature
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__qualname__
            if name.endswith("__init__"):
                name = name.split(".", maxsplit=1)[0]
            output_warning_message(name, kind, replace, removed, extra_msg)
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator
