# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["zf_auto", "zf_double", "zf_size", "zf"]

__dataset_methods__ = __all__

import functools

import numpy as np

from spectrochempy.application.application import error_
from spectrochempy.utils.decorators import _processing_history_parameters
from spectrochempy.utils.decorators import _processing_requested_dimension
from spectrochempy.utils.decorators import _processing_scientific_parameters
from spectrochempy.utils.numutils import largest_power_of_2


# ======================================================================================
# Decorators
# ======================================================================================
def _zf_method(method):
    @functools.wraps(method)
    def wrapper(dataset, **kwargs):
        requested_dim = _processing_requested_dimension(kwargs)
        axis, dim = dataset.get_axis(**kwargs, negative_axis=True)
        resolved_axis = axis % dataset.ndim
        source_size = dataset.shape[resolved_axis]
        source_coord = dataset.coordset[dim]
        validation_coord = source_coord.copy()
        if hasattr(validation_coord, "_use_time_axis"):
            validation_coord._use_time_axis = True

        # Refused calls return the original dataset. Validate a coordinate copy
        # before copying or temporarily permuting an in-place caller.
        if not validation_coord.linear:
            error_(
                "zero-filling apply only to linear coordinates\n"
                "The processing was thus cancelled",
            )
            return dataset
        if not (
            validation_coord.unitless
            or validation_coord.dimensionless
            or validation_coord.units.dimensionality == "[time]"
        ):
            error_(
                "zero-filling apply only to dimensions with [time] dimensionality or dimensionless coords\n"
                "The processing was thus cancelled",
            )
            return dataset

        inplace = kwargs.pop("inplace", False)
        new = dataset.copy() if not inplace else dataset

        swapped = False
        if axis != -1:
            new._swapdims_without_history(axis, -1, inplace=True)
            swapped = True

        x = new.coordset[dim]
        if hasattr(x, "_use_time_axis"):
            x._use_time_axis = True

        requested_parameters = _processing_scientific_parameters(method, kwargs)
        data = method(new.data, **kwargs)
        new._data = data

        # Increase the selected coordinate to match the new data size.
        offset = x.data[0]
        size = x.size
        inc = np.ptp(x._data) / (size - 1)
        x._data = np.arange(offset, offset + new._data.shape[-1] * inc, inc)
        new.meta.td[-1] = x.size
        result_size = new._data.shape[-1]

        if swapped:
            new._swapdims_without_history(axis, -1, inplace=True)

        scientific_parameters = dict(requested_parameters)
        if method.__name__ == "zf_size":
            scientific_parameters["size"] = result_size
        parameters = _processing_history_parameters(
            method,
            kwargs,
            requested_dim=requested_dim,
            resolved_dim=dim,
            resolved_axis=resolved_axis,
            inplace=inplace,
            scientific_parameters=scientific_parameters,
        )
        parameters.update(
            {
                "source_size": source_size,
                "result_size": result_size,
            }
        )
        if requested_parameters != scientific_parameters:
            parameters["requested_parameters"] = requested_parameters

        new._append_history_entry(
            operation=method.__name__,
            parameters=parameters,
            message=(
                f"Applied {method.__name__} zero filling on dimension {dim} "
                f"with parameters: {kwargs}"
            ),
        )
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
