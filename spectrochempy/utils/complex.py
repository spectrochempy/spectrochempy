#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

import numpy as np
from quaternion import as_float_array, as_quat_array


def as_quaternion(*args):
    """
    Recombine the arguments to produce a numpy array with quaternion dtype.

    Parameters
    ----------
    *args : ndarray with dtype:float or complex
        The quaternion array components: If there is 4 components,
        then we assume it is the four components of the
        quaternion array: w, x, y, z. If there is only two, they are cast to
        complex and correspond respectively to w + i.x and y + j.z.
    """
    if len(args) == 4:
        # we assume here that the for components have been provided w, x, y, z
        w, x, y, z = args

    if len(args) == 2:
        r, i = args
        w, x, y, z = r.real, r.imag, i.real, i.imag

    data = as_quat_array(list(zip(w.flatten(), x.flatten(), y.flatten(), z.flatten())))
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
    if not arr.dtype == np.quaternion:
        # no change
        return arr

    wt, xt, yt, zt = as_float_array(arr).T
    w, x, y, z = wt.T, xt.T, yt.T, zt.T

    return (w + 1j * x), (y + 1j * z)
