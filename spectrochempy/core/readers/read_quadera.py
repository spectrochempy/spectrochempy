# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Plugin module to extend NDDataset with the import methods method.
"""

__all__ = ["read_quadera"]
__dataset_methods__ = __all__

import re
from datetime import datetime
from warnings import warn

import numpy as np

from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.core.readers.importer import Importer, _importer_method, _openfid
from spectrochempy.utils.docstrings import _docstring

# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_quadera")


@_docstring.dedent
def read_quadera(*paths, **kwargs):
    """
    Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file with extension :file:`.asc`\ .

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    --------
    %(Importer.returns)s

    Other Parameters
    ----------------
    timestamp: `bool`\ , optional, default: `True`
        Returns the acquisition timestamp as `Coord`.
        If set to `False`\ , returns the time relative to the acquisition time of the
        data
    %(Importer.other_parameters)s

    See Also
    ---------
    %(Importer.see_also.no_read_quadera)s

    Notes
    ------
    Currently the acquisition time is that of the first channel as the timeshift of
    other channels are typically
    within few seconds, and the data of other channels are NOT interpolated
    Todo: check with users whether data interpolation should be made

    Examples
    ---------

    >>> scp.read_quadera('msdata/ion_currents.asc')
    NDDataset: [float64] A (shape: (y:16975, x:10))
    """
    kwargs["filetypes"] = ["Quadera files (*.asc)"]
    kwargs["protocol"] = ["asc"]
    importer = Importer()
    return importer(*paths, **kwargs)


# --------------------------------------------------------------------------------------
# Private methods
# --------------------------------------------------------------------------------------
@_importer_method
def _read_asc(*args, **kwargs):
    _, filename = args

    fid, kwargs = _openfid(filename, mode="r", **kwargs)

    lines = fid.readlines()
    fid.close()

    timestamp = kwargs.get("timestamp", True)

    # the list of channels is 2 lines after the line starting with "End Time"
    i = 0
    while not lines[i].startswith("End Time"):
        i += 1

    i += 2
    # reads channel names
    channels = re.split(r"\t+", lines[i].rstrip("\t\n"))[1:]
    nchannels = len(channels)

    # the next line contains the columns titles, repeated for each channels
    # this routine assumes that for each channel are  Time / Time relative [s] / Ion Current [A]
    # check it:
    i += 1
    colnames = re.split(r"\t+", lines[i].rstrip("\t"))

    if (
        colnames[0] == "Time"
        or colnames[1] != "Time Relative [s]"
        or colnames[2] != "Ion Current [A]"
    ):
        warn(
            "Columns names are  not those expected: the reading of your .asc file  could yield "
            "please notify this to the developers of scpectrochempy"
        )
    if nchannels > 1 and colnames[3] != "Time":  # pragma: no cover
        warn(
            "The number of columms per channel is not that expected: the reading of your .asc file  could yield "
            "please notify this to the developers of spectrochempy"
        )

    # the remaining lines contain data and time coords
    ntimes = len(lines) - i - 1

    times = np.empty((ntimes, nchannels), dtype=object)
    reltimes = np.empty((ntimes, nchannels))
    ioncurrent = np.empty((ntimes, nchannels))
    i += 1

    prev_timestamp = 0
    for j, line in enumerate(lines[i:]):
        data = re.split(r"[\t+]", line.rstrip("\t"))
        for k in range(nchannels):
            datetime_ = datetime.strptime(
                data[3 * k].strip(" "), "%m/%d/%Y %H:%M:%S.%f"
            )
            times[j][k] = datetime_.timestamp()
            # hours are given in 12h clock format, so we need to add 12h when hour is in the afternoon
            if times[j][k] < prev_timestamp:
                times[j][k] += 3600 * 12
            reltimes[j][k] = data[1 + 3 * k].replace(",", ".")
            ioncurrent[j][k] = data[2 + 3 * k].replace(",", ".")
        prev_timestamp = times[j][k]

    dataset = NDDataset(ioncurrent)
    dataset.name = filename.stem
    dataset.title = "ion current"
    dataset.units = "amp"

    if timestamp:
        _y = Coord(times[:, 0], title="acquisition timestamp (UTC)", units="s")
    else:
        _y = Coord(times[:, 0] - times[0, 0], title="Time", units="s")

    _x = Coord(labels=channels)
    dataset.set_coordset(y=_y, x=_x)

    # Set origin, description and history
    dataset.history = f"Imported from Quadera asc file {filename}"

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset
