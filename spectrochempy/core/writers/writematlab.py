# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Plugin module to extend NDDataset with a JCAMP-DX export method.
"""

from spectrochempy.core.writers.exporter import Exporter, exportermethod

__all__ = ["write_matlab", "write_mat"]
__dataset_methods__ = __all__


# ...............................................................................
def write_matlab(*args, **kwargs):
    """
    Write a dataset in CSV format.

    Parameters
    ----------
    *args
    **kwargs

    Returns
    -------
    out : `pathlib` object
        Path of the saved file.

    Examples
    --------

    The extension will be added automatically
    >>> X.write_matlab('myfile')
    """
    exporter = Exporter()
    kwargs["filetypes"] = ["MATLAB files (*.mat)"]
    kwargs["suffix"] = ".mat"
    return exporter(*args, **kwargs)


write_mat = write_matlab
write_mat.__doc__ = "This method is an alias of `write_matlab`."


@exportermethod
def _write_matlab(*args, **kwargs):
    raise NotImplementedError
