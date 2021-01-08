# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
"""
Plugin module to extend NDDataset with a JCAMP-DX export method.
"""
# import os as os

from spectrochempy.core.writers.exporter import Exporter, exportermethod

__all__ = ['write_csv']
__dataset_methods__ = __all__


# .......................................................................................................................
def write_csv(*args, **kwargs):
    """
    Writes a dataset in CSV format

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
    csv_delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is the one set in SpectroChemPy `Preferences`.

    Returns
    -------
    out : `pathlib` object
        path of the saved file

    Examples
    --------

    The extension will be added automatically
    # >>> import spectrochempy as scp
    # >>> ds = scp.NDDataset([1,2,3])
    # >>> f = ds.write_csv('myfile')
    # f.name = myfile.csv
    """
    exporter = Exporter()
    kwargs['filetypes'] = ['CSV files (*.csv)']
    kwargs['suffix'] = '.csv'
    return exporter(*args, **kwargs)


@exportermethod
def _write_csv(*args, **kwargs):
    raise NotImplementedError
