# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
Plugin module to extend NDDataset with a JCAMP-DX export method.
"""

from spectrochempy.core.writers.exporter import Exporter, exportermethod

__all__ = ["write_matlab", "write_mat"]
__dataset_methods__ = __all__


# ...............................................................................
def write_matlab(dataset, filename, **kwargs):
    """
    Write a dataset in CSV format.

    Parameters
    ----------
    dataset : |NDDataset|
        Dataset to write.
    filename : str or pathlib object, optional
        If not provided, a dialog is opened to select a file for writing.
    **kwargs : dict
        See other parameters.

    Other Parameters
    ----------------
    directory : str, optional
        Where to write the specified `filename`. If not specified, write in the current directory.
    comment : str, optional
        A Custom comment.

    Returns
    -------
    out : `pathlib` object
        path of the saved file.

    Examples
    --------

    The extension will be added automatically
    >>> X.write_matlab('myfile')
    """
    exporter = Exporter()
    kwargs["filetypes"] = ["MATLAB files (*.mat)"]
    kwargs["suffix"] = ".mat"
    return exporter(dataset, filename, **kwargs)


write_mat = write_matlab
write_mat.__doc__ = "This method is an alias of `write_matlab`."


@exportermethod
def _write_matlab(*args, **kwargs):
    raise NotImplementedError
