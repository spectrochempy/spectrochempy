# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Plugin module to extend NDDataset with a minimal MATLAB export method."""

import numpy as np
from scipy.io import savemat

from spectrochempy.core.writers.exporter import Exporter
from spectrochempy.core.writers.exporter import exportermethod

__all__ = ["write_matlab", "write_mat"]
__dataset_methods__ = __all__


def write_matlab(*args, **kwargs):
    r"""
    Write a dataset to a MATLAB `.mat` exchange file.

    This writer targets simple MATLAB / Octave interoperability through
    `scipy.io.savemat`. It is not SpectroChemPy native persistence and does
    not claim full round-trip fidelity.

    Parameters
    ----------
    filename : str or pathlib object, optional
        If not provided, a dialog is opened to select a file for writing.
    **kwargs
        Additional keyword arguments accepted by the generic writer API.
        This specialized writer always exports MATLAB `.mat` files.

    Returns
    -------
    out : `pathlib` object
        Path of the saved file.

    Examples
    --------
    The extension will be added automatically
    >>> X.write_matlab('myfile')

    Using the explicit namespace API
    >>> scp.matlab.write(X, 'myfile')

    """
    exporter = Exporter()
    kwargs["filetypes"] = ["MATLAB files (*.mat)"]
    kwargs["suffix"] = ".mat"
    return exporter(*args, **kwargs)


write_mat = write_matlab
write_mat.__doc__ = "This method is an alias of `write_matlab` ."


@exportermethod
def _write_matlab(*args, **kwargs):
    dataset, filename = args

    values = _matlab_export_data(dataset)
    payload = {
        "data": values,
        "dims": np.asarray([str(dim) for dim in dataset.dims], dtype=object),
        "coords": {},
        "coord_units": {},
        "coord_titles": {},
        "name": str(dataset.name or ""),
        "title": str(dataset.title or ""),
    }

    if dataset.units is not None:
        payload["units"] = _matlab_stringify_units(dataset.units)
    if dataset.description:
        payload["description"] = str(dataset.description)

    for dim in dataset.dims:
        coord = dataset.coord(dim)
        if coord is None or coord.is_empty:
            continue
        payload["coords"][str(dim)] = np.asarray(coord.data)
        if coord.units is not None:
            payload["coord_units"][str(dim)] = _matlab_stringify_units(coord.units)
        if coord.title and coord.title != "<untitled>":
            payload["coord_titles"][str(dim)] = str(coord.title)

    savemat(filename, payload, do_compression=False)
    return filename


def _matlab_export_data(dataset):
    """Return a MATLAB-compatible numeric array for minimal exchange export."""
    if dataset.ndim not in (1, 2):
        raise NotImplementedError(
            "MATLAB export is only implemented for 1D and 2D NDDataset objects."
        )

    values = np.asarray(dataset.data)
    if values.dtype.kind == "c":
        raise TypeError("MATLAB export does not support complex NDDataset data.")
    if values.dtype.kind not in "biuf":
        raise TypeError(
            "MATLAB export only supports numeric boolean/integer/float "
            "NDDataset data."
        )

    if dataset.is_masked:
        values = np.asarray(np.ma.filled(dataset.masked_data, np.nan), dtype=float)

    return values


def _matlab_stringify_units(units):
    """Return a simple ASCII unit string for MATLAB exchange payloads."""
    return f"{units:~D}"
