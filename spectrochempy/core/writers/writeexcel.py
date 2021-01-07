# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================


"""Plugin module to extend NDDataset with a JCAMP-DX export method.

"""
# import os as os

from spectrochempy.core.writers.exporter import docstrings, Exporter, exportermethod

__all__ = ['write_excel']
__dataset_methods__ = __all__


# .......................................................................................................................
@docstrings.dedent
def write_excel(*args, **kwargs):
    """
    Writes a dataset in XLS format


    Parameters
    ----------


    Returns
    -------
    out : `pathlib` object
        path of the saved file

    Examples
    --------

    The extension will be added automatically
    >>> X.write_xls('myfile')

    """
    exporter = Exporter()
    kwargs['filetypes'] = ['Microsoft Excel files (*.xls)']
    kwargs['suffix'] = '.xls'
    return exporter(*args, **kwargs)


write_xls = write_excel
write_xls.__doc__ = 'This method is an alias of `write_excel` '


@exportermethod
def _write_excel(*args, **kwargs):
    raise NotImplementedError
