# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

"""
Plugin module to extend NDDataset with the import methods method.
"""

__all__ = ["read_quadera"]
__dataset_methods__ = __all__

import io
from warnings import warn
from datetime import datetime, timezone
import re

import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset, Coord
from spectrochempy.core.readers.importer import Importer, importermethod


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_quadera(*paths, **kwargs):
    """
    Read a Pfeiffer Vacuum's QUADERA® mass spectrometer software file with extension ``.asc``.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_quadera
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
    timestamp: bool, optional
        returns the acquisition timestamp as Coord (Default=True).
        If set to false, returns the time relative to the acquisition time of the data
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel', 'asc'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False)
    description: str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the ``tests/tests_readers`` directory
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True)
    recursive : bool, optional
        Read also in subfolders. (default=False)

    Notes
    ------
    Currently the acquisition time is that of the first channel as the timeshift of other channels are typically
    within few seconds, and the data of other channels are NOT interpolated
    Todo: check with users wether data interpolation should be made

    See Also
    --------
    read : Read generic files.
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.

    Examples
    ---------

    >>> scp.read_quadera('msdata/ion_currents.asc')
    NDDataset: [float64] A (shape: (y:16975, x:10))
    """
    kwargs["filetypes"] = ["Quadera files (*.asc)"]
    kwargs["protocol"] = ["asc"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ------------------------------------------------------------------
# Private methods
# ------------------------------------------------------------------


@importermethod
def _read_asc(*args, **kwargs):
    _, filename = args
    content = kwargs.get("content", False)

    if content:  # pragma: no cover
        fid = io.BytesIO(content)
    else:
        fid = open(filename, "r")

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

    # Set the NDDataset date
    dataset._date = datetime.now(timezone.utc)
    dataset._modified = dataset.date

    # Set origin, description and history
    dataset.history = f"{dataset.date}:imported from Quadera asc file {filename}"

    return dataset


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
