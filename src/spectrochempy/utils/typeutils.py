# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
from numbers import Number

import numpy as np


def is_iterable(obj):
    """
    Determine if an object is iterable.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool

    """
    from spectrochempy.core.units import Quantity

    # I don't know why Quantity has a method iter,
    # but we don't want to use it as an iterable if it is not.
    if isinstance(obj, Quantity) and not hasattr(obj, "shape"):
        return False

    return hasattr(obj, "__iter__")


def is_number(obj):
    """
    Check if an object is a number.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool

    """
    # try:
    #     if is_numpy_array(obj) or is_duck_array(obj):
    #         return False
    #     obj + 1
    #     return True
    # except TypeError:
    #     return False
    return isinstance(obj, Number)


def is_sequence(obj):
    """
    Determine if an object is iterable but is not a string.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool

    """
    return not is_string(obj) and is_iterable(obj)


def is_string(obj):
    """
    Check if an object is a string.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool

    """
    return isinstance(obj, str)


def is_numpy_array(obj):
    """
    Check if the array represents a numpy ndarray instance.

    Parameters
    ----------
    obj : object
     Object to check.

    Returns
    -------
    bool

    """
    if isinstance(obj, np.ndarray):
        return True
    return None


def is_duck_array(obj):
    """
    Check if the object represents a duck array instance.

    Parameters
    ----------
    obj : object
     Object to check.

    Returns
    -------
    bool

    """
    if is_numpy_array(obj):
        return False

    return (
        hasattr(obj, "__array_function__")
        and hasattr(obj, "shape")
        and hasattr(obj, "ndim")
        and hasattr(obj, "dtype")
    )


def is_quantity(obj):
    """
    Check if the object represent a quantity instance.

    Parameters
    ----------
    obj : object
     Object to check.

    Returns
    -------
    bool

    """
    return hasattr(obj, "_units") and hasattr(obj, "_magnitude")


def is_sequence_with_quantity_elements(obj):
    """
    Check if at least one element of a sequence is a quantity.

    Parameters
    ----------
    obj : object
     Object to check.

    Returns
    -------
    bool

    """
    return is_sequence(obj) and any(is_quantity(item) for item in obj)


def is_array_like(obj):
    """
    Check if an obj is an array-like object.

    An object, but not a numpy array, from which an array can be extracted.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool

    """
    return (
        is_quantity(obj)
        or is_sequence_with_quantity_elements(obj)
        or is_duck_array(obj)
        or hasattr(obj, "__xarray__")
    ) and not is_numpy_array(obj)


def is_datetime64(obj):
    """
    Check if obj or obj.data is a np.datetime64 object.

    Parameters
    ----------
    obj : datetime or np.ndarray like obj
        If obj implement NDDataset or Coord obj, then the data attribute is checked.

    Returns
    -------
    bool
        Whether the obj or obj.data has a np.datetime64 dtype or not

    """
    dt64 = np.datetime64
    if hasattr(obj, "_data"):
        data = obj._data[0] if obj._data is not None else None
    else:
        data = obj
    return isinstance(data, dt64)
