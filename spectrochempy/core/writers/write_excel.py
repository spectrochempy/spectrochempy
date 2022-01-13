# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Plugin module to extend NDDataset with a JCAMP-DX export method.
"""
# import os as os

from spectrochempy.core.writers.exporter import Exporter, exportermethod

__all__ = ["write_excel"]
__dataset_methods__ = __all__


# ...............................................................................
def write_excel(*args, **kwargs):
    """
    Write a dataset in XLS format.

    Parameters
    ----------
    filename: str or pathlib objet, optional
        If not provided, a dialog is opened to select a file for writing
    protocol : {'scp', 'matlab', 'jcamp', 'csv', 'excel'}, optional
        Protocol used for writing. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
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

    The extension will be added automatically
    >>> X.write_xls('myfile')
    """
    exporter = Exporter()
    kwargs["filetypes"] = ["Microsoft Excel files (*.xls)"]
    kwargs["suffix"] = ".xls"
    return exporter(*args, **kwargs)


write_xls = write_excel
write_xls.__doc__ = "This method is an alias of `write_excel`."


@exportermethod
def _write_excel(*args, **kwargs):
    raise NotImplementedError
