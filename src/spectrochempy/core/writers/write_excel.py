# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Internal unsupported Excel-writer compatibility stub."""
# import os as os

from spectrochempy.core.writers.exporter import Exporter
from spectrochempy.core.writers.exporter import exportermethod

__all__ = ["write_excel", "write_xls"]
__dataset_methods__ = __all__


def write_excel(*args, **kwargs):
    """
    Unsupported internal compatibility stub for XLS export.

    Parameters
    ----------
    filename: str or pathlib object, optional
        If not provided, a dialog is opened to select a file for writing
    **kwargs
        Ignored compatibility arguments.

    Returns
    -------
    object
        This function always raises when called through the exporter path.

    Notes
    -----
    Excel export is not part of the supported SpectroChemPy core API.
    Public `write_excel()` / `write_xls()` entry points were removed.

    """
    exporter = Exporter()
    kwargs["filetypes"] = ["Microsoft Excel files (*.xls)"]
    kwargs["suffix"] = ".xls"
    return exporter(*args, **kwargs)


write_xls = write_excel
write_xls.__doc__ = "This method is an alias of `write_excel` ."


@exportermethod
def _write_excel(*args, **kwargs):
    raise NotImplementedError("Excel export is not yet implemented")
