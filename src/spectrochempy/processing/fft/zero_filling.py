# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["zf_auto", "zf_double", "zf_size", "zf"]

__dataset_methods__ = __all__

import functools

import numpy as np

from spectrochempy.application.application import error_
from spectrochempy.utils.numutils import largest_power_of_2


# ======================================================================================
# Decorators
# ======================================================================================
def _zf_method(method):
    @functools.wraps(method)
    def wrapper(dataset, **kwargs):
        # On which axis do we want to shift (get axis from arguments)
        axis, dim = dataset.get_axis(**kwargs, negative_axis=True)

        # output dataset inplace (by default) or not
        new = dataset.copy() if not kwargs.pop("inplace", False) else dataset

        swapped = False
        if axis != -1:
            new.swapdims(axis, -1, inplace=True)  # must be done in  place
            swapped = True

        x = new.coordset[dim]

        if not x.linear:
            # this method is not valid for non-linear coordinates
            error_(
                "zero-filling apply only to linear coordinates\n"
                "The processing was thus cancelled",
            )
            return dataset  # return the original dataset

        if hasattr(x, "_use_time_axis"):
            x._use_time_axis = True  # we need to have dimensionless or time units

        # get the lastcoord
        if x.unitless or x.dimensionless or x.units.dimensionality == "[time]":
            # we can apply the method
            data = method(new.data, **kwargs)
            new._data = data

            # we need to increase the x coordinates array to match the new data size
            offset = x.data[0]
            size = x.size
            inc = np.ptp(x._data) / (size - 1)
            x._data = np.arange(offset, offset + new._data.shape[-1] * inc, inc)
            # update with the new td
            new.meta.td[-1] = x.size
            new.history = f"`{method.__name__}` shift performed on dimension `{dim}` with parameters: {kwargs}"

        else:
            error_(
                "zero-filling apply only to dimensions with [time] dimensionality or dimensionless coords\n"
                "The processing was thus cancelled",
            )
            return dataset  # return the original dataset

        # restore original data order if it was swapped
        if swapped:
            new.swapdims(axis, -1, inplace=True)  # must be done inplace

        return new

    return wrapper


# ======================================================================================
# Private methods
# ======================================================================================
def _zf_pad(data, pad=0, mid=False, **kwargs):
    """
    Zero fill by padding with zeros.

    Parameters
    ----------
    dataset : ndarray
        Array of NMR data.
    pad : int
        Number of zeros to pad data with.
    mid : bool
        True to zero fill in middle of data.

    Returns
    -------
    ndata : ndarray
        Array of NMR data to which `pad` zeros have been appended to the end or
        middle of the data.

    """
    size = list(data.shape)
    size[-1] = int(pad)
    z = np.zeros(size, dtype=data.dtype)

    if mid:
        h = int(data.shape[-1] / 2.0)
        return np.concatenate((data[..., :h], z, data[..., h:]), axis=-1)
    return np.concatenate((data, z), axis=-1)


# ======================================================================================
# Public methods
# ======================================================================================
@_zf_method
def zf_double(dataset, n, mid=False, **kwargs):
    """
    Zero fill by doubling original data size once or multiple times.

    Parameters
    ----------
    dataset : ndataset
        Array of NMR data.
    n : int
        Number of times to double the size of the data.
    mid : bool
        True to zero fill in the middle of data.

    Returns
    -------
    ndata : ndarray
        Zero filled array of NMR data.

    """
    return _zf_pad(dataset, int((dataset.shape[-1] * 2**n) - dataset.shape[-1]), mid)


@_zf_method
def zf_size(dataset, size=None, mid=False, **kwargs):
    """
    Zero fill to given size.

    Parameters
    ----------
    dataset : `NDDataset`
        Input dataset.
    size : int
        Size of data after zero filling.
    mid : bool
        True to zero fill in the middle of data.

    Returns
    -------
    `NDDataset`
        Modified dataset.

    """
    if size is None:
        size = dataset.shape[-1]
    return _zf_pad(dataset, pad=int(size - dataset.shape[-1]), mid=mid)


def zf_auto(dataset, mid=False):
    """
    Zero fill to next largest power of two.

    Parameters
    ----------
    dataset : ndarray
        Array of NMR data.
    mid : bool
        True to zero fill in the middle of data.

    Returns
    -------
    ndata : ndarray
        Zero filled array of NMR data.

    """
    return zf_size(dataset, size=largest_power_of_2(dataset.shape[-1]), mid=mid)


zf = zf_size
