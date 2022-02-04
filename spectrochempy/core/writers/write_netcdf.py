# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
Plugin module to extend NDDataset with a netCDF export method.
"""
import numpy as np

from spectrochempy.optional import import_optional_dependency


# from spectrochempy.core.dataset.ndplot import PreferencesSet
from spectrochempy.core import debug_

from spectrochempy.core.writers.exporter import Exporter, exportermethod
from spectrochempy.utils.datetimeutils import encode_datetime64

__all__ = ["write_netcdf"]
__dataset_methods__ = __all__


# ...............................................................................
def write_netcdf(dataset, filename, **kwargs):
    """
    Write a dataset or a list of datasets in netCDF format.

    See https://www.unidata.ucar.edu/software/netcdf/ for information about
    this format

    Parameters
    ----------
    filename: str or pathlib object, optional
        If not provided, a dialog is opened to select a file for writing
    directory : str, optional
        Where to write the specified `filename`. If not specified, write in the current directory.
    description: str, optional
        A Custom description.

    Returns
    -------
    out : `pathlib` object
        path of the saved file.

    Examples
    --------

    The extension `nc` will be added automatically
    >>> X.write_netcdf('myfile')
    """
    exporter = Exporter()
    kwargs["filetypes"] = ["NetCDF - Network Common Data Form (*.nc)"]
    kwargs["suffix"] = ".nc"
    return exporter(dataset, filename, **kwargs)


write_nc = write_netcdf
write_nc.__doc__ = "This method is an alias of `write_netcdf`."


def _encode_cf_variable(var):

    # unpack for decoding
    dims, data, attrs, encoding = (
        var.dims,
        var.data,
        var.attrs.copy(),
        var.encoding.copy(),
    )

    # apply to datetime64
    if np.issubdtype(data.dtype, np.datetime64):
        new, attrs = encode_datetime64(data, attrs)
        data = new.reshape(data.shape)

        # case of ROI attrs
        newROI, att = encode_datetime64(attrs["roi"])
        attrs["roi"] = newROI
        attrs["roi_units"] = att["units"]

    return type(var)(dims, data, attrs, encoding)


@exportermethod
def _write_netcdf(*args, **kwargs):

    dataset, filename = args
    dataset.filename = filename

    xarr = dataset.to_xarray()

    netcdf_file = import_optional_dependency("netCDF4.Dataset", errors="ignore")
    if netcdf_file is None:
        debug_(
            "netCDF4 package was not found. Using Scipy.io instead but "
            "then writing is limited to netCDF3 format as scipy does not "
            "support netCDF4 format."
        )

    # For the moment we limit this writing to single dataset

    # here we run cf-encoder to adapt the variable to the CF conventions
    xarr._variable = _encode_cf_variable(xarr.variable)
    xarr._coords = {k: _encode_cf_variable(v) for k, v in xarr._coords.items()}

    xarr.to_netcdf(filename)
