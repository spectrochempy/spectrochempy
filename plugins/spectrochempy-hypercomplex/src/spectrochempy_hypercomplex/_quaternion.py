# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Optional quaternion support — internal to the hypercomplex plugin.

The public API is exposed through :class:`HyperAccessor`.
"""

import warnings

import numpy as np

from spectrochempy.utils.constants import TYPE_COMPLEX

# Try to import the quaternion library (optional dependency)
_HAS_QUATERNION = False
typequaternion = None
as_float_array = None
as_quat_array = None

try:
    from quaternion import as_float_array  # noqa: F811
    from quaternion import as_quat_array  # noqa: F811

    _HAS_QUATERNION = True
    typequaternion = np.dtype(np.quaternion)
except ImportError:
    pass


def _check():
    if not _HAS_QUATERNION:
        raise ImportError(
            "Missing optional dependency 'numpy-quaternion'. "
            "Use pip or conda to install numpy-quaternion."
        )


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
    _check()
    if len(args) == 4:
        w, x, y, z = args
    elif len(args) == 2:
        r, i = args
        w, x, y, z = r.real, r.imag, i.real, i.imag
    else:
        raise ValueError("as_quaternion requires 2 or 4 arguments")

    stacked = np.stack([w, x, y, z], axis=-1)
    return as_quat_array(stacked)


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
    _check()
    if typequaternion is None or arr.dtype != typequaternion:
        return arr

    float_arr = as_float_array(arr)
    w = float_arr[..., 0]
    x = float_arr[..., 1]
    y = float_arr[..., 2]
    z = float_arr[..., 3]

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

    if typequaternion is not None and new.dtype == typequaternion:
        float_arr = as_float_array(new)
        w = float_arr[..., 0]
        x = float_arr[..., 1]
        y = float_arr[..., 2]
        z = float_arr[..., 3]
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
