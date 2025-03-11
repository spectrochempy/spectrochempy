# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Various methods and classes used in other part of the program."""

import warnings

import numpy as np
from quaternion import as_float_array
from quaternion import as_quat_array

from spectrochempy.utils.constants import TYPE_COMPLEX

typequaternion = np.dtype(np.quaternion)


# ======================================================================================
# Public methods
# ======================================================================================
def as_quaternion(*args):
    """
    Recombine the arguments to produce a numpy array with quaternion dtype.

    Parameters
    ----------
    *args : `ndarray` with dtype: `float` or `complex`
        The quaternion array components: If there is 4 components, then we assume it is
        the four components of the
        quaternion array: w, x, y, z. If there is only two, they are casted to complex
        and correspond respectively
        to w + i.x and y + j.z.

    """
    if len(args) == 4:
        # we assume here that the for components have been provided w, x, y, z
        w, x, y, z = args

    if len(args) == 2:
        r, i = args
        w, x, y, z = r.real, r.imag, i.real, i.imag

    data = as_quat_array(
        list(zip(w.flatten(), x.flatten(), y.flatten(), z.flatten(), strict=False)),
    )
    return data.reshape(w.shape)


def quat_as_complex_array(arr):
    """
    Recombine the component of a quaternion array into a tuple of two complex array.

    Parameters
    ----------
    arr : quaternion ndarray
        The arr will be separated into (w + i.x) and (y + i.z).

    Returns
    -------
    tuple
        Tuple of two complex array.

    """
    if arr.dtype != np.quaternion:
        # no change
        return arr

    wt, xt, yt, zt = as_float_array(arr).T
    w, x, y, z = wt.T, xt.T, yt.T, zt.T

    return (w + 1j * x), (y + 1j * z)


def get_component(data, select="REAL"):
    """
    Take selected components of an hypercomplex array (RRR, RIR, ...).

    Parameters
    ----------
    data : `ndarray`
        Input data array.
    select : `str`, optional, default: 'REAL'
        If 'REAL', only real component in all dimensions will be selected.
        Else a string must specify which real (R) or imaginary (I) component
        has to be selected along a specific dimension. For instance,
        a string such as 'RRI' for a 2D hypercomplex array indicated
        that we take the real component in each dimension except the last
        one, for which imaginary component is preferred.

    Returns
    -------
    component
        A component of the complex or hypercomplex array.

    .. warning::
        The definition is somewhat different from Bruker, as we order the component in
        the order of the dimensions in
        dataset:
        e.g., for dims = ['y','x'], 'IR' means that the `y` component is imaginary
        while the `x` is real.

    """
    if not select:
        return data

    new = data.copy()

    if select == "REAL":
        select = "R" * new.ndim

    w = x = y = z = None

    if new.dtype == typequaternion:
        w, x, y, z = as_float_array(new).T
        w, x, y, z = w.T, x.T, y.T, z.T
        if select == "R":
            new = w + x * 1j
        elif select == "I":
            new = y + z * 1j
        elif select == "RR":
            new = w
        elif select == "RI":
            new = x
        elif select == "IR":
            new = y
        elif select == "II":
            new = z
        else:
            raise ValueError(
                f"something wrong: cannot interpret `{select}` for hypercomplex (quaternion) data!",
            )

    elif new.dtype in TYPE_COMPLEX:
        w, x = new.real, new.imag
        if (select == "R") or (select == "RR"):
            new = w
        elif (select == "I") or (select == "RI"):
            new = x
        else:
            raise ValueError(
                f"something wrong: cannot interpret `{select}` for complex data!",
            )
    else:
        warnings.warn(
            f"No selection was performed because datasets with complex data have no "
            f"`{select}` component. ",
            stacklevel=2,
        )

    return new


def interleaved2complex(data):
    """Make a complex array from interleaved data."""
    return data[..., ::2] + 1j * data[..., 1::2]


def interleaved2quaternion(data):
    """Make a complex array from interleaved data."""
    return data[..., ::2] + 1j * data[..., 1::2]
