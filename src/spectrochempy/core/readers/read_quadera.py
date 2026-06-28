# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Plugin module to extend NDDataset with the import methods method."""

__all__ = ["read_quadera"]

import re
from datetime import UTC
from datetime import datetime
from warnings import warn

import numpy as np

from spectrochempy.core.dataset.nddataset import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid


# ======================================================================================
# Public functions
# ======================================================================================
def read_quadera(*paths, **kwargs):
    r"""
    Open a Pfeiffer Vacuum's QUADERA mass spectrometer software file.

    This is the explicit QUADERA reader in the public import API. Use
    :func:`spectrochempy.read` for generic format autodetection and
    ``scp.quadera.read(...)`` or :func:`spectrochempy.read_quadera` when the
    QUADERA format is already known.

    Non-merged multi-file reads may return a list-like `ScpObjectList`
    exposing helper methods for dataset selection. See
    :func:`spectrochempy.read` for the complete description of the generic
    import convention and multi-object return behavior.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ..., kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to ``False``.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    object : `NDDataset` or `ScpObjectList` of `NDDataset`
        The returned dataset(s). When several datasets are returned, the
        result is a list-like `ScpObjectList` with helper attributes such as
        ``.names``, ``.select_largest()``, ``.select_by_name()``,
        ``.filter_by_ndim()``, and ``.filter_by_shape()``.

    Other Parameters
    ----------------
    content : `bytes` object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly
        useful for a GUI Dash application to handle drag and drop of files into a
        Browser.
    csv_delimiter : `str`, optional, default: `~spectrochempy.preferences.csv_delimiter`
        Set the column delimiter in CSV file.
    description : `str`, optional
        A custom description.
    directory : `~pathlib.Path` object objects or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `True` and several filenames or a ``directory`` have been provided as
        arguments, then a single `NDDataset` with merged dataset (stacked along the first
        dimension) is returned. In the case not all datasets have compatible dimensions or types/origins,
        then several NDDatasets can be returned for different groups of compatible datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used, for instance, when reading a directory with different types of
        files and merging compatible datasets into separate groups by origin.

        It is also used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading, for example ``'scp'``, ``'omnic'``,
        ``'opus'``, ``'matlab'``, ``'jcamp'``, ``'csv'``, or ``'excel'``.
        If not provided, the correct protocol is inferred whenever possible
        from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic reader inferring protocol from the filename extension.
    :func:`spectrochempy.read_zip` : Read Zip archives (containing spectrochempy readable files)
    :func:`spectrochempy.read_dir` : Read an entire directory.
    :func:`spectrochempy.read_opus` : Read OPUS spectra.
    :func:`spectrochempy.read_labspec` : Read Raman LABSPEC spectra (:file:`.txt`).
    :func:`spectrochempy.read_omnic` : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    :func:`spectrochempy.read_soc` : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_jcamp` : Read Infrared JCAMP-DX files (:file:`.jdx`, :file:`.dx`).
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).

    Examples
    --------
    Reading a single QUADERA file

    >>> scp.read_quadera('irdata/quadera.QD')
    NDDataset: [float64] a.u. (shape: (y:1, x:16384))

    Using the explicit namespace API

    >>> scp.quadera.read('irdata/quadera.QD')
    NDDataset: [float64] a.u. (shape: (y:1, x:16384))

    """
    kwargs["filetypes"] = ["QUADERA files (*.QD)"]
    kwargs["protocol"] = ["quadera"]
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
            "please notify this to the developers of scpectrochempy",
            stacklevel=2,
        )
    if nchannels > 1 and colnames[3] != "Time":  # pragma: no cover
        warn(
            "The number of columms per channel is not that expected: the reading of your .asc file  could yield "
            "please notify this to the developers of spectrochempy",
            stacklevel=2,
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
                data[3 * k].strip(" "),
                "%m/%d/%Y %H:%M:%S.%f",
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
    dataset.filename = filename
    dataset.title = "ion current"
    dataset.units = "amp"

    if timestamp:
        _y = Coord(times[:, 0], title="acquisition timestamp (UTC)", units="s")
    else:
        _y = Coord(times[:, 0] - times[0, 0], title="Time", units="s")

    _x = Coord(labels=channels)
    dataset.set_coordset(y=_y, x=_x)

    # Set origin, acquisition date, description and history
    dataset.origin = "quadera"
    dataset.acquisition_date = datetime.fromtimestamp(min(times[:, 0]), tz=UTC)
    dataset.history = f"Imported from Quadera asc file {filename}"

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset
