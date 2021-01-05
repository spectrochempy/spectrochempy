# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================


"""Plugin module to extend NDDataset with a JCAMP-DX export method.

"""
# import os as os

from spectrochempy.core.writers.exporter import docstrings, Exporter, exportermethod

__all__ = ['write_csv']
__dataset_methods__ = __all__


# .......................................................................................................................
@docstrings.dedent
def write_csv(*args, **kwargs):
    """
    Writes a dataset in CSV format


    Parameters
    ----------


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
