# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Plugin module to extend NDDataset with a netCDF export method.
"""
from spectrochempy.optional import import_optional_dependency
from spectrochempy.core import warning_

HAS_NETCDF4 = True
netcdf_file = import_optional_dependency("netCDF4.Dataset", errors="ignore")
if netcdf_file is None:
    HAS_NETCDF4 = False
    warning_(
        "netCDF4 package was not found. Using Scipy.io instead but then "
        "writing is limited to netCDF3 format as scipy does not support "
        "netCDF4 format."
    )
    from scipy.io import netcdf_file

from spectrochempy.core.writers.exporter import Exporter, exportermethod

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


@exportermethod
def _write_netcdf(*args, **kwargs):

    raise NotImplementedError("WIP")
