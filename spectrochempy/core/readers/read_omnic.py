#  -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =====================================================================================================================
#
"""
This module extend NDDataset with the import method for OMNIC generated data
files.
"""
__all__ = ["read_omnic", "read_spg", "read_spa", "read_srs"]
__dataset_methods__ = __all__

from datetime import datetime, timezone, timedelta
import io
import struct

import numpy as np

from spectrochempy.core import info_
from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import importermethod, Importer
from spectrochempy.core.units import Quantity


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_omnic(*paths, **kwargs):
    """
    Open a Thermo Nicolet OMNIC file.

    Open Omnic file or a list of files with extension ``.spg`` , ``.spa`` or
    ``.srs`` and set data/metadata in the current dataset.

    The collected metadata are:
    - names of spectra
    - acquisition dates (UTC)
    - units of spectra (absorbance, transmittance, reflectance, Log(1/R),
    Kubelka-Munk, Raman intensity, photoacoustics, volts)
    - units of xaxis (wavenumbers in cm^-1, wavelengths in nm or micrometer,
    Raman shift in cm^-1)
    - spectra history (but only incorporated in the NDDataset if a single
    spa is read)

    An error is generated if attempt is made to inconsistent datasets: units
    of spectra and
    xaxis, limits and number of points of the xaxis.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name
        for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e.
        no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    out
        The dataset or a list of dataset corresponding to a (set of) .spg,
        .spa or .srs file(s).

    Other Parameters
    -----------------
    directory : str, optional
        From where to read the specified `filename`. If not specified,
        read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been
        provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description : str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content
        can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is
        particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For examples on how to use this feature, one can look in the
        ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided
        `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current
        reading protocol (default=True).
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read : Generic read method.
    read_dir : Read a set of data from a directory.
    read_spg : Read Omnic files *.spg.
    read_spa : Read Omnic files *.spa.
    read_srs : Read Omnic files *.srs.
    read_opus : Read Bruker OPUS files.
    read_topspin : Read TopSpin NMR files.
    read_csv : Read *.csv.
    read_matlab : Read MATLAB files *.mat.
    read_zip : Read zipped group of files.

    Examples
    ---------
    Reading a single OMNIC file  (providing a windows type filename relative
    to the default ``datadir``)

    >>> scp.read_omnic('irdata\\\\nh4y-activation.spg')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Reading a single OMNIC file  (providing a unix/python type filename
    relative to the default ``datadir``)
    Note that here read_omnic is called as a classmethod of the NDDataset class

    >>> scp.NDDataset.read_omnic('irdata/nh4y-activation.spg')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata')
    >>> p = folder / 'nh4y-activation.spg'
    >>> scp.read_omnic(p)
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    The directory can also be specified independently, either as a string or
    a pathlib object

    >>> scp.read_omnic('nh4y-activation.spg', directory=folder)
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Multiple files not merged (return a list of datasets)

    >>> le = scp.read_omnic('irdata/nh4y-activation.spg', 'wodger.spg')
    >>> len(le)
    2
    >>> le[1]
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read_omnic('irdata/nh4y-activation.spg', 'wodger.spg', merge=True)
    NDDataset: [float64] a.u. (shape: (y:57, x:5549))

    Multiple files to merge : they are passed as a list (note the brakets)
    instead of using the keyword `merge`

    >>> scp.read_omnic(['irdata/nh4y-activation.spg', 'wodger.spg'])
    NDDataset: [float64] a.u. (shape: (y:57, x:5549))

    Multiple files not merged : they are passed as a list but `merge` is set
    to false

    >>> l2 = scp.read_omnic(['irdata/nh4y-activation.spg', 'wodger.spg'], merge=False)
    >>> len(l2)
    2

    Read without a filename. This has the effect of opening a dialog for
    file(s) selection

    >>> nd = scp.read_omnic()

    Read in a directory (assume that only OPUS files are present in the
    directory
    (else we must use the generic `read` function instead)

    >>> l3 = scp.read_omnic(directory='irdata/subdir/1-20')
    >>> len(l3)
    3

    Again we can use merge to stack all 4 spectra if thet have compatible
    dimensions.

    >>> scp.read_omnic(directory='irdata/subdir', merge=True)
    NDDataset: [float64] a.u. (shape: (y:4, x:5549))

    An example, where bytes contents are passed directly to the read_omnic
    method.

    >>> datadir = scp.preferences.datadir
    >>> filename1 = datadir / 'irdata' / 'subdir' / '7_CZ0-100 Pd_101.SPA'
    >>> content1 = filename1.read_bytes()
    >>> filename2 = datadir / 'wodger.spg'
    >>> content2 = filename2.read_bytes()
    >>> listnd = scp.read_omnic({filename1.name: content1, filename2.name: content2})
    >>> len(listnd)
    2
    >>> scp.read_omnic({filename1.name: content1, filename2.name: content2}, merge=True)
    NDDataset: [float64] a.u. (shape: (y:3, x:5549))
    """

    kwargs["filetypes"] = ["OMNIC files (*.spa *.spg)", "OMNIC series (*.srs)"]
    kwargs["protocol"] = ["omnic", "spa", "spg", "srs"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ..............................................................................
def read_spg(*paths, **kwargs):
    """
    Open a Thermo Nicolet file or a list of files with extension ``.spg``.

    Open Omnic file or a list of files with extension ``.spg`` and set
    data/metadata in the current dataset.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name
        for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e.
        no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    read_spg
        The dataset or a list of dataset corresponding to a (set of) .spg
        file(s).

    Other Parameters
    -----------------
    directory : str, optional
        From where to read the specified `filename`. If not specified,
        read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been
        provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description : str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content
        can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is
        particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For examples on how to use this feature, one can look in the
        ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided
        `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current
        reading protocol (default=True).
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read : Generic read method.
    read_dir : Read a set of data from a directory.
    read_omnnic : Read Omnic files.
    read_spa : Read Omnic files *.spa.
    read_srs : Read Omnic files *.srs.
    read_opus : Read Bruker OPUS files.
    read_topspin : Read TopSpin NMR files.
    read_csv : Read *.csv.
    read_matlab : Read MATLAB files *.mat.
    read_zip : Read zipped group of files.

    Notes
    -----
    This method is an alias of ``read_omnic``, except that the type of file
    is contrain to *.spg.

    Examples
    ---------

    >>> scp.read_spg('irdata/nh4y-activation.spg')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))
    """

    kwargs["filetypes"] = ["OMNIC files (*.spg)"]
    kwargs["protocol"] = ["spg", "spa"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ..............................................................................
def read_spa(*paths, **kwargs):
    """
    Open a Thermo Nicolet file or a list of files with extension ``.spa``.

    Open Omnic file or a list of files with extension ``.spa`` and set
    data/metadata in the current dataset.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name
        for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e.
        no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    read_spa
        The dataset or a list of dataset corresponding to the (set of) .spa
        file(s).

    Other Parameters
    -----------------
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'backgroung' returns
        the backgroung interferogram of the spa file if present or None if absent.
    directory : str, optional
        From where to read the specified `filename`. If not specified,
        read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been
        provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description: str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content
        can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is
        particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For examples on how to use this feature, one can look in the
        ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided
        `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current
        reading protocol (default=True).
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read : Generic read method.
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Notes
    -----
    This method is an alias of ``read_omnic``, except that the type of file
    is contrain to *.spa.

    Examples
    ---------

    >>> scp.read_spa('irdata/subdir/20-50/7_CZ0-100 Pd_21.SPA')
    NDDataset: [float64] a.u. (shape: (y:1, x:5549))
    >>> scp.read_spa(directory='irdata/subdir', merge=True)
    NDDataset: [float64] a.u. (shape: (y:4, x:5549))
    """

    kwargs["filetypes"] = ["OMNIC files (*.spa)"]
    kwargs["protocol"] = ["spa"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ..............................................................................
def read_srs(*paths, **kwargs):
    """
    Open a Thermo Nicolet file or a list of files with extension ``.srs``.

    Open Omnic file or a list of files with extension ``.srs`` and set
    data/metadata in the current dataset.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name
        for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e.
        no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    NDDataset
        The dataset or a list of dataset corresponding to a (set of) series
        or backgroun files.

    Other Parameters
    -----------------
    return_bg : bool, optional
        Default value is False. When set to 'True' returns the series background
    directory : str, optional
        From where to read the specified `filename`. If not specified,
        read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been
        provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description: str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content
        can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is
        particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For examples on how to use this feature, one can look in the
        ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided
        `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current
        reading protocol (default=True)
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read : Generic read method.
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Notes
    -----
    This method is an alias of ``read_omnic``, except that the type of file
    is constrained to *.srs.

    Examples
    ---------
    >>> scp.read_srs('irdata/omnic series/rapid_scan_reprocessed.srs')
    NDDataset: [float64] a.u. (shape: (y:643, x:3734))
    """

    kwargs["filetypes"] = ["OMNIC series (*.srs)"]
    kwargs["protocol"] = ["srs"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================

# ..............................................................................
@importermethod
def _read_spg(*args, **kwargs):
    # read spg file

    dataset, filename = args
    sortbydate = kwargs.pop("sortbydate", True)
    content = kwargs.get("content", False)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, "rb")

    # Read name:
    # The name starts at position hex 1e = decimal 30. Its max length
    # is 256 bytes. It is the original filename under which the group has been saved: it
    # won't match with the actual filename if a subsequent renaming has been done in the OS.

    spg_title = _readbtext(fid, 30)

    # Count the number of spectra
    # From hex 120 = decimal 304, individual spectra are described
    # by blocks of lines starting with "key values",
    # for instance hex[02 6a 6b 69 1b 03 82] -> dec[02 106  107 105 27 03 130]
    # Each of these lines provides positions of data and metadata in the file:
    #
    #     key: hex 02, dec  02: position of spectral header (=> nx, firstx,
    #     lastx, nscans, nbkgscans)
    #     key: hex 03, dec  03: intensity position
    #     key: hex 04, dec  04: user text position
    #     key: hex 1B, dec  27: position of History text
    #     key: hex 64, dec 100: ?
    #     key: hex 66  dec 102: sample interferogram
    #     key: hex 67  dec 103: background interferogram
    #     key: hex 69, dec 105: ?
    #     key: hex 6a, dec 106: ?
    #     key: hex 6b, dec 107: position of spectrum title, the acquisition
    #     date follows at +256(dec)
    #     key: hex 80, dec 128: ?
    #     key: hex 82, dec 130: rotation angle ?
    #
    # the number of line per block may change from file to file but the total
    # number of lines is given at hex 294, hence allowing counting the
    # number of spectra:

    # read total number of lines
    fid.seek(294)
    nlines = _fromfile(fid, "uint16", count=1)

    # read "key values"
    pos = 304
    keys = np.zeros(nlines)
    for i in range(nlines):
        fid.seek(pos)
        keys[i] = _fromfile(fid, dtype="uint8", count=1)
        pos = pos + 16

    # the number of occurrences of the key '02' is number of spectra
    nspec = np.count_nonzero((keys == 2))

    if nspec == 0:  # pragma: no cover
        raise IOError(
            "Error : File format not recognized" " - information markers not found"
        )

    # container to hold values
    nx, firstx, lastx = (
        np.zeros(nspec, "int"),
        np.zeros(nspec, "float"),
        np.zeros(nspec, "float"),
    )
    xunits = []
    xtitles = []
    units = []
    titles = []

    # Extracts positions of '02' keys
    key_is_02 = keys == 2  # ex: [T F F F F T F (...) F T ....]'
    indices02 = np.nonzero(key_is_02)  # ex: [1 9 ...]
    position02 = (
        304 * np.ones(len(indices02[0]), dtype="int") + 16 * indices02[0]
    )  # ex: [304 432 ...]

    for i in range(nspec):
        # read the position of the header
        fid.seek(position02[i] + 2)
        pos_header = _fromfile(fid, dtype="uint32", count=1)
        # get infos
        info = _read_header(fid, pos_header)
        nx[i] = info["nx"]
        firstx[i] = info["firstx"]
        lastx[i] = info["lastx"]
        xunits.append(info["xunits"])
        xtitles.append(info["xtitle"])
        units.append(info["units"])
        titles.append(info["title"])

    # check the consistency of xaxis and data units
    if np.ptp(nx) != 0:  # pragma: no cover
        raise ValueError(
            "Error : Inconsistent data set -"
            " number of wavenumber per spectrum should be "
            "identical"
        )
    elif np.ptp(firstx) != 0:  # pragma: no cover
        raise ValueError(
            "Error : Inconsistent data set - " "the x axis should start at same value"
        )
    elif np.ptp(lastx) != 0:  # pragma: no cover
        raise ValueError(
            "Error : Inconsistent data set -" " the x axis should end at same value"
        )
    elif len(set(xunits)) != 1:  # pragma: no cover
        raise ValueError(
            "Error : Inconsistent data set - " "data units should be identical"
        )
    elif len(set(units)) != 1:  # pragma: no cover
        raise ValueError(
            "Error : Inconsistent data set - " "x axis units should be identical"
        )
    data = np.ndarray((nspec, nx[0]), dtype="float32")

    # Now the intensity data

    # Extracts positions of '03' keys
    key_is_03 = keys == 3
    indices03 = np.nonzero(key_is_03)
    position03 = 304 * np.ones(len(indices03[0]), dtype="int") + 16 * indices03[0]

    # Read number of spectral intensities
    for i in range(nspec):
        data[i, :] = _getintensities(fid, position03[i])

    # Get spectra titles & acquisition dates:
    # container to hold values
    spectitles, acquisitiondates, timestamps = [], [], []

    # Extract positions of '6B' keys (spectra titles & acquisition dates)
    key_is_6B = keys == 107
    indices6B = np.nonzero(key_is_6B)
    position6B = 304 * np.ones(len(indices6B[0]), dtype="int") + 16 * indices6B[0]

    # Read spectra titles and acquisition date
    for i in range(nspec):
        # determines the position of informatioon
        fid.seek(position6B[i] + 2)  # go to line and skip 2 bytes
        spa_title_pos = _fromfile(fid, "uint32", 1)

        # read filename
        spa_title = _readbtext(fid, spa_title_pos)
        spectitles.append(spa_title)

        # and the acquisition date
        fid.seek(spa_title_pos + 256)
        timestamp = _fromfile(fid, dtype="uint32", count=1)
        # since 31/12/1899, 00:00
        acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(
            seconds=int(timestamp)
        )
        acquisitiondates.append(acqdate)
        timestamp = acqdate.timestamp()
        # Transform back to timestamp for storage in the Coord object
        # use datetime.fromtimestamp(d, timezone.utc))
        # to transform back to datetime object

        timestamps.append(timestamp)

        # Not used at present
        # -------------------
        # extract positions of '1B' codes (history text), sometimes absent,
        # e.g. peakresolve)
        #  key_is_1B = (keys == 27)
        #  indices1B =  # np.nonzero(key_is_1B)
        #  position1B = 304 * np.ones(len(indices1B[0]), dtype='int') + 16 * indices6B[0]
        #  if len(position1B) != 0:  # read history texts
        #     for j in range(nspec):  determine the position of information
        #        f.seek(position1B[j] + 2)  #
        #        history_pos = _fromfile(f,  'uint32', 1)
        #        history =  _readbtext(f, history_pos[0])
        #        allhistories.append(history)

    fid.close()

    # Create Dataset Object of spectral content
    dataset.data = data
    dataset.units = units[0]
    dataset.title = titles[0]
    dataset.name = filename.stem
    dataset.filename = filename

    # now add coordinates
    # _x = Coord(np.around(np.linspace(firstx[0], lastx[0], nx[0]), 3),
    #           title=xtitles[0], units=xunits[0])
    spacing = (lastx[0] - firstx[0]) / int(nx[0] - 1)
    _x = LinearCoord(
        offset=firstx[0],
        increment=spacing,
        size=int(nx[0]),
        title=xtitles[0],
        units=xunits[0],
    )

    _y = Coord(
        timestamps,
        title="acquisition timestamp (GMT)",
        units="s",
        labels=(acquisitiondates, spectitles),
    )

    dataset.set_coordset(y=_y, x=_x)

    # Set description, date and history
    # Omnic spg file don't have specific "origin" field stating the oirigin of the data
    dataset.description = kwargs.get(
        "description", f"Omnic title: {spg_title}\nOmnic " f"filename: {filename}"
    )

    dataset._date = datetime.now(timezone.utc)

    dataset.history = str(dataset.date) + ":imported from spg file {} ; ".format(
        filename
    )
    if sortbydate:
        dataset.sort(dim="y", inplace=True)
        dataset.history = str(dataset.date) + ":sorted by date"

    # debug_("end of reading")

    return dataset


# ..............................................................................
@importermethod
def _read_spa(*args, **kwargs):
    dataset, filename = args
    content = kwargs.get("content", False)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, "rb")

    return_ifg = kwargs.get("return_ifg", None)

    # Read name:
    # The name  starts at position hex 1e = decimal 30. Its max length
    # is 256 bytes. It is the original filename under which the spectrum has
    # been saved: it won't match with the actual filename if a subsequent
    # renaming has been done in the OS.
    spa_name = _readbtext(fid, 30)

    # The acquisition date (GMT) is at hex 128 = decimal 296.
    # Second since 31/12/1899, 00:00
    fid.seek(296)
    timestamp = _fromfile(fid, dtype="uint32", count=1)
    acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=int(timestamp)
    )
    acquisitiondate = acqdate

    # Transform back to timestamp for storage in the Coord object
    # use datetime.fromtimestamp(d, timezone.utc)) to transform back to datetime object
    timestamp = acqdate.timestamp()

    # From hex 120 = decimal 304, the spectrum is described
    # by a block of lines starting with "key values",
    # for instance hex[02 6a 6b 69 1b 03 82] -> dec[02 106  107 105 27 03 130]
    # Each of these lines provides positions of data and metadata in the file:
    #
    #     key: hex 02, dec  02: position of spectral header (=> nx,
    #                                 firstx, lastx, nscans, nbkgscans)
    #     key: hex 03, dec  03: intensity position
    #     key: hex 04, dec  04: user text position
    #     key: hex 1B, dec  27: position of History text
    #     key: hex 64, dec 100: ?
    #     key: hex 66  dec 102: sample interferogram
    #     key: hex 67  dec 103: background interferogram
    #     key: hex 69, dec 105: ?
    #     key: hex 6a, dec 106: ?
    #     key: hex 80, dec 128: ?
    #     key: hex 82, dec 130: rotation angle
    #
    # The line preceding the block start with '01'
    # The lines after the block generally start with '00', except in few cases where
    # they start by '01'. In such cases, the '53' key is also present
    # (before the '1B').

    # scan "key values"
    pos = 304
    while "continue":
        fid.seek(pos)
        key = _fromfile(fid, dtype="uint8", count=1)

        if key == 2:
            # read the position of the header
            fid.seek(pos + 2)
            pos_header = _fromfile(fid, dtype="uint32", count=1)
            info = _read_header(fid, pos_header)

        elif key == 3 and return_ifg is None:
            intensities = _getintensities(fid, pos)

        elif key == 27:
            fid.seek(pos + 2)
            history_pos = _fromfile(fid, "uint32", 1)
            spa_history = _readbtext(fid, history_pos)

        elif key == 102 and return_ifg == "sample":
            s_ifg_intensities = _getintensities(fid, pos)

        elif key == 103 and return_ifg == "background":
            b_ifg_intensities = _getintensities(fid, pos)

        elif key == 00 or key == 1:
            break

        pos += 16

    fid.close()

    if (return_ifg == "sample" and "s_ifg_intensities" not in locals()) or (
        return_ifg == "background" and "b_ifg_intensities" not in locals()
    ):
        info_("No interferogram found, read_spa returns None")
        return None
    elif return_ifg == "sample":
        intensities = s_ifg_intensities
    elif return_ifg == "background":
        intensities = b_ifg_intensities
    # load intensity into the  NDDataset
    dataset.data = np.array(intensities[np.newaxis], dtype="float32")

    if return_ifg == "background":
        title = "sample acquisition timestamp (GMT)"  # bckg acquisition date is not known for the moment...
    else:
        title = "acquisition timestamp (GMT)"  # no ambiguity here

    _y = Coord(
        [timestamp],
        title=title,
        units="s",
        labels=([acquisitiondate], [filename]),
    )

    # useful when a part of the spectrum/ifg has been blanked:
    dataset.mask = np.isnan(dataset.data)

    if return_ifg is None:
        default_description = f"Omnic name: {spa_name}\nOmnic filename: {filename.name}"
        dataset.units = info["units"]
        dataset.title = info["title"]

        # now add coordinates
        nx = info["nx"]
        firstx = info["firstx"]
        lastx = info["lastx"]
        xunit = info["xunits"]
        xtitle = info["xtitle"]

        spacing = (lastx - firstx) / (nx - 1)

        _x = LinearCoord(
            offset=firstx, increment=spacing, size=nx, title=xtitle, units=xunit
        )

    else:  # interferogram
        if return_ifg == "sample":
            default_description = (
                f"Omnic name: {spa_name} : sample IFG\nOmnic filename: {filename.name}"
            )
        else:
            default_description = f"Omnic name: {spa_name} : background IFG\nOmnic filename: {filename.name}"
        spa_name += ": Sample IFG"
        dataset.units = "V"
        dataset.title = "detector signal"
        _x = LinearCoord(
            offset=0,
            increment=1,
            size=len(intensities),
            title="data points",
            units=None,
        )

    dataset.set_coordset(y=_y, x=_x)
    dataset.name = spa_name  # to be consistent with omnic behaviour
    dataset.filename = str(filename)

    # Set origin, description, history, date
    # Omnic spg file don't have specific "origin" field stating the oirigin of the data

    dataset.description = kwargs.get("description", default_description)
    if "spa_history" in locals():
        dataset.history = (
            "Omnic 'DATA PROCESSING HISTORY' : \n----------------------------------\n"
            + spa_history
        )
    dataset.history = str(datetime.now(timezone.utc)) + ":imported from spa file(s)"

    dataset._date = datetime.now(timezone.utc)

    if dataset.x.units is None and dataset.x.title == "data points":
        # interferogram
        dataset.meta.interferogram = True
        dataset.meta.td = list(dataset.shape)
        dataset.x._zpd = int(np.argmax(dataset)[-1])
        dataset.meta.laser_frequency = Quantity("15798.26 cm^-1")
        dataset.x.set_laser_frequency()
        dataset.x._use_time_axis = (
            False  # True to have time, else it will be optical path difference
        )

    return dataset


# ..............................................................................
@importermethod
def _read_srs(*args, **kwargs):
    dataset, filename = args
    frombytes = kwargs.get("frombytes", False)

    return_bg = kwargs.get("return_bg", False)

    if frombytes:
        # in this case, filename is actually a byte content
        fid = io.BytesIO(filename)  # pragma: no cover
    else:
        fid = open(filename, "rb")

    # determine whether the srs is reprocessed. At pos=292 (hex:124) appears a difference between
    # and reprocessed series
    fid.seek(292)
    key = _fromfile(fid, dtype="uint8", count=16)[0]
    if key == 39:  # (hex: 27)
        is_reprocessed = False
    elif key == 15:  # (hex = 0F)
        is_reprocessed = True
    # if key == 72 (hex:48), could be TGA

    """ At pos=304 (hex:130) is the position of the '02' key for series. Herte we don't use it.
    Instead, we use the following sequence :
    b'\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\x50\x43\x47'
    which appears 3 times in rapid-scan srs. They are used to assert the srs file is rapid_scan
    and to locate headers and data:
    - The 1st one is located 152 bytes after the series header position
    - The 2nd one is located 152 bytes before the background header position and
       56 bytes before either the background data / or the background title and infos
       followed by the background data
    - The 3rd one is located 64 bytes before the series data (spectre/ifg names and
    intensities"""

    sub = b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\x50\x43\x47"

    # find the 3 starting indexes of sub.we will use the 1st (-> series info),
    # the 2nd (-> background) and the 3rd (-> data)
    fid.seek(0)
    bytestring = fid.read()
    start = 0
    index = []
    while start != -1:
        i = bytestring.find(sub, start + 1)
        index.append(i)
        start = i
    index = np.array(index[:-1])

    if len(index) != 3:
        raise NotImplementedError("Only implemented for rapidscan")

    index += [-152, -152, 60]

    # read series data, except if the user asks for the background
    if not return_bg:
        info = _read_header(fid, index[0])
        # container for names and data
        names = []
        data = np.zeros((info["ny"], info["nx"]))

        # now read the spectra/interferogram names and data
        # the first one....
        pos = index[2]
        names.append(_readbtext(fid, pos))
        pos += 84
        fid.seek(pos)
        data[0, :] = _fromfile(fid, dtype="float32", count=info["nx"])[:]
        pos += info["nx"] * 4
        # ... and the remaining ones:
        for i in np.arange(info["ny"])[1:]:
            pos += 16
            names.append(_readbtext(fid, pos))
            pos += 84
            fid.seek(pos)
            data[i, :] = _fromfile(fid, dtype="float32", count=info["nx"])[:]
            pos += info["nx"] * 4

        # now get series history
        if not is_reprocessed:
            history = info["history"]
        else:
            # In reprocessed series the updated "DATA PROCESSING HISTORY" is located right after
            # the following 16 byte sequence:
            sub = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF"
            pos = bytestring.find(sub) + 16
            history = _readbtext(fid, pos)

    # read the background if the user asked for it.
    if return_bg:

        # First get background info
        info = _read_header(fid, index[1])

        if "background_name" not in info.keys():
            # it is a short header
            fid.seek(index[1] + 208)
            data = _fromfile(fid, dtype="float32", count=info["nx"])
        else:
            # longer header, in such case the header indicates a spectrum
            # but the data are those of an ifg... For now need more examples
            return None

    # Create NDDataset Object for the series
    if not return_bg:
        dataset = NDDataset(data)
    else:
        dataset = NDDataset(np.expand_dims(data, axis=0))

    dataset.units = info["units"]
    dataset.title = info["title"]
    dataset.origin = "omnic"

    # now add coordinates
    spacing = (info["lastx"] - info["firstx"]) / (info["nx"] - 1)
    _x = LinearCoord(
        offset=info["firstx"],
        increment=spacing,
        size=info["nx"],
        title=info["xtitle"],
        units=info["xunits"],
    )

    # specific infos for series data
    if not return_bg:
        dataset.name = info["name"]
        _y = Coord(
            np.around(np.linspace(info["firsty"], info["lasty"], info["ny"]), 3),
            title="Time",
            units="minute",
            labels=names,
        )

    else:
        _y = Coord()

    dataset.set_coordset(y=_y, x=_x)

    # Set origin, description and history
    dataset.origin = "omnic"
    dataset.description = kwargs.get("description", "Dataset from omnic srs file.")

    if "history" in locals():
        dataset.history.append(
            "Omnic 'DATA PROCESSING HISTORY' :\n"
            "--------------------------------\n" + history
        )
    dataset.history.append(
        str(datetime.now(timezone.utc)) + ": imported from srs file " + str(filename)
    )

    if dataset.x.units is None and dataset.x.title == "data points":
        # interferogram
        dataset.meta.interferogram = True
        dataset.meta.td = list(dataset.shape)
        dataset.x._zpd = int(np.argmax(dataset)[-1])  # zero path difference
        dataset.meta.laser_frequency = Quantity("15798.26 cm^-1")
        dataset.x.set_laser_frequency()
        dataset.x._use_time_axis = (
            False  # True to have time, else it will  be optical path difference
        )

        # uncomment below to load the last datafield has the same dimension as the time axis
        # its function is not known. related to Grams-schmidt ?

        # pos = _nextline(pos)
        # found = False
        # while not found:
        #     pos += 16
        #     f.seek(pos)
        #     key = _fromfile(f, dtype='uint8', count=1)
        #     if key == 1:
        #         pos += 4
        #         f.seek(pos)
        #         X = _fromfile(f, dtype='float32', count=info['ny'])
        #         found = True
        #
        # X = NDDataset(X)
        # _x = Coord(np.around(np.linspace(0, info['ny']-1, info['ny']), 0),
        #            title='time',
        #            units='minutes')
        # X.set_coordset(x=_x)
        # X.name = '?'
        # X.title = '?'
        # X.description = 'unknown'
        # X.history = str(datetime.now(timezone.utc)) + ':imported from srs

    fid.close()

    return dataset


# ..............................................................................
def _fromfile(fid, dtype, count):
    # to replace np.fromfile in case of io.BytesIO object instead of byte
    # object
    t = {
        "uint8": "B",
        "int8": "b",
        "uint16": "H",
        "int16": "h",
        "uint32": "I",
        "int32": "i",
        "float32": "f",
        "char8": "c",
    }
    typ = t[dtype] * count
    if dtype.endswith("16"):
        count = count * 2
    elif dtype.endswith("32"):
        count = count * 4

    out = struct.unpack(typ, fid.read(count))
    if len(out) == 1:
        return out[0]
    return np.array(out)


# ..............................................................................
def _readbtext(fid, pos):
    # Read some text in binary file, until b\0\ is encountered.
    # Returns utf-8 string
    fid.seek(pos)
    btext = fid.read(1)
    while not (btext[len(btext) - 1] == 0):  # while the last byte is not zero
        btext = btext + fid.read(1)  # append 1 byte

    btext = btext[0 : len(btext) - 1]  # cuts the last byte
    try:
        text = btext.decode(encoding="utf-8")  # decode btext to string
    except UnicodeDecodeError:
        try:
            text = btext.decode(encoding="latin_1")
        except UnicodeDecodeError:  # pragma: no cover
            text = btext.decode(encoding="utf-8", errors="ignore")
    return text


# ..............................................................................
def _nextline(pos):
    # reset current position to the beginning of next line (16 bytes length)
    return 16 * (1 + pos // 16)


# ..............................................................................
def _read_header(fid, pos):
    """
    read spectrum/ifg/series header

    Parameters
    ----------
    fid : BufferedReader
        The buffered binary stream.

    pos : int
        The position of the header (see Notes).

    Returns
    -------
        dict, int
        Dictionary and current position in file

    Notes
    -----
        So far, the header structure is as follows:
        - starts with b'\x01' , b'\x02', b'\x03' ... maybe indicating the header "type"
        - nx (UInt32): 4 bytes behind
        - xunits (UInt8): 8 bytes behind. So far, we have the following correspondence:
            `x\01`: wavenumbers, cm-1
            `x\02`: datapoints (interferogram)
            `x\03`: wavelength, nm
            `x\04': wavelength, um
            `x\20': Raman shift, cm-1
        - data units (UInt8): 12 bytes behind. So far, we have the following correspondence:
            `x\11`: absorbance
            `x\10`: transmittance (%)
            `x\0B`: reflectance (%)
            `x\0C`: Kubelka_Munk
            `x\16`:  Volts (interferogram)
            `x\1A`:  photoacoustic
            `x\1F`: Raman intensity
        - first x value (float32), 16 bytes behind
        - last x value (float32), 20 bytes behind
        - scan points (UInt32), 28 bytes behind
        - zpd (UInt32),  32 bytes behind
        - number of scans (UInt32), 36 bytes behind
        ... infos from 40 to 51 bytes behind are not none yet
        - number of background scans (UInt32), 52 bytes behind

        For spa and spg infos between 56 and 207 bytes behind are not none yet
        - spectrum history (text), 208 bytes behind

        For "rapid-scan" srs files:
        - series name (text), 938 bytes behind
        - collection length (float32), 1002 bytes behind
        - last y (float 32), 1006 bytes behind
        - first y (float 32), 1010 bytes behind
        - ny (UInt32), 1026
        ... y unit could be at pos+1030 with 01 = minutes ?
        - history (text), 1200 bytes behind (only initila hgistopry.
           When reprocessed, updated histopry is at the end of the file after the
           b`\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF` sequance
    """

    out = {}
    # determine the type of file
    fid.seek(0)
    bytes = fid.read(18)
    if bytes == b"Spectral Data File":
        filetype = "spa, spg"
    elif bytes == b"Spectral Exte File":
        filetype = "srs"

    # nx
    fid.seek(pos + 4)
    out["nx"] = _fromfile(fid, "uint32", count=1)

    # xunits
    fid.seek(pos + 8)
    key = _fromfile(fid, dtype="uint8", count=1)
    if key == 1:
        out["xunits"] = "cm^-1"
        out["xtitle"] = "wavenumbers"
    elif key == 2:
        out["xunits"] = None
        out["xtitle"] = "data points"
    elif key == 3:  # pragma: no cover
        out["xunits"] = "nm"
        out["xtitle"] = "wavelengths"
    elif key == 4:  # pragma: no cover
        out["xunits"] = "um"
        out["xtitle"] = "wavelengths"
    elif key == 32:  # pragma: no cover
        out["xunits"] = "cm^-1"
        out["xtitle"] = "raman shift"
    else:  # pragma: no cover
        out["xunits"] = None
        out["xtitle"] = "xaxis"
        info_("The nature of x data is not recognized, xtitle is set to 'xaxis'")

    # data units
    fid.seek(pos + 12)
    key = _fromfile(fid, dtype="uint8", count=1)
    if key == 17:
        out["units"] = "absorbance"
        out["title"] = "absorbance"
    elif key == 16:  # pragma: no cover
        out["units"] = "percent"
        out["title"] = "transmittance"
    elif key == 11:  # pragma: no cover
        out["units"] = "percent"
        out["title"] = "reflectance"
    elif key == 12:  # pragma: no cover
        out["units"] = None
        out["title"] = "log(1/R)"
    elif key == 20:  # pragma: no cover
        out["units"] = "Kubelka_Munk"
        out["title"] = "Kubelka-Munk"
    elif key == 22:
        out["units"] = "V"
        out["title"] = "detector signal"
    elif key == 26:  # pragma: no cover
        out["units"] = None
        out["title"] = "photoacoustic"
    elif key == 31:  # pragma: no cover
        out["units"] = None
        out["title"] = "Raman intensity"
    else:  # pragma: no cover
        out["units"] = None
        out["title"] = "intensity"
        info_("The nature of data is not recognized, title set to 'Intensity'")

    # firstx, lastx
    fid.seek(pos + 16)
    out["firstx"] = _fromfile(fid, "float32", 1)
    fid.seek(pos + 20)
    out["lastx"] = _fromfile(fid, "float32", 1)
    fid.seek(pos + 28)

    out["scan_pts"] = _fromfile(fid, "uint32", 1)
    fid.seek(pos + 32)
    out["zpd"] = _fromfile(fid, "uint32", 1)
    fid.seek(pos + 36)
    out["nscan"] = _fromfile(fid, "uint32", 1)
    fid.seek(pos + 52)
    out["nbkgscan"] = _fromfile(fid, "uint32", 1)

    if filetype == "spa, spg":
        out["history"] = _readbtext(fid, pos + 208)

    if filetype == "srs":
        if out["nbkgscan"] == 0:
            # an interferogram in rapid scan mode
            if out["firstx"] > out["lastx"]:
                out["firstx"], out["lastx"] = out["lastx"], out["firstx"]

        out["name"] = _readbtext(fid, pos + 938)
        fid.seek(pos + 1002)
        out["coll_length"] = _fromfile(fid, "float32", 1) * 60
        fid.seek(pos + 1006)
        out["lasty"] = _fromfile(fid, "float32", 1)
        fid.seek(pos + 1010)
        out["firsty"] = _fromfile(fid, "float32", 1)
        fid.seek(pos + 1026)
        out["ny"] = _fromfile(fid, "uint32", 1)
        #  y unit could be at pos+1030 with 01 = minutes ?
        out["history"] = _readbtext(fid, pos + 1200)

        if _readbtext(fid, pos + 208)[:10] == "Background":
            # it is the header of a background
            out["background_name"] = _readbtext(fid, pos + 208)[10:]

    return out


# ..............................................................................
def _getintensities(fid, pos):
    # get intensities from the 03 (spectrum)
    # or 66 (sample ifg) or 67 (bg ifg) key,
    # returns a ndarray

    fid.seek(pos + 2)  # skip 2 bytes
    intensity_pos = _fromfile(fid, "uint32", 1)
    fid.seek(pos + 6)
    intensity_size = _fromfile(fid, "uint32", 1)
    nintensities = int(intensity_size / 4)

    # Read and return spectral intensities
    fid.seek(intensity_pos)
    return _fromfile(fid, "float32", int(nintensities))


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
