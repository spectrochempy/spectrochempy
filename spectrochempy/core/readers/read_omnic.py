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

from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import importermethod, Importer
from spectrochempy.units import Quantity


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_omnic(*paths, **kwargs):
    """
    Open a Thermo Nicolet OMNIC file.

    Open Omnic file or a list of files with extension ``.spg`` , ``.spa`` or
    ``.srs``
    and set data/metadata in the current dataset.

    The collected metatdata are:
    - names of spectra
    - acquisition dates (UTC)
    - units of spectra (absorbance, transmittance, reflectance, Log(1/R),
    Kubelka-Munk, Raman intensity,
    photoacoustics, volts)
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
    **kwargs : dict
        See other parameters.

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
        For exemples on how to use this feature, one can look in the
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

    The diretory can also be specified independantly, either as a string or
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
    **kwargs : dict
        See other parameters.

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
        For exemples on how to use this feature, one can look in the
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
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_spa
        The dataset or a list of dataset corresponding to a (set of) .spg
        file(s).

    Other Parameters
    -----------------
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv',
    'excel'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
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
        For exemples on how to use this feature, one can look in the
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
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_srs
        The dataset or a list of dataset corresponding to a (set of) .spg
        file(s).

    Other Parameters
    -----------------
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv',
    'excel'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
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
        For exemples on how to use this feature, one can look in the
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
    is contrain to *.srs.

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

    # Read title:
    # The file title starts at position hex 1e = decimal 30. Its max length
    # is 256 bytes.
    #  It is the original filename under which the group has been saved: it
    #  won't match with
    #  the actual filename if a subsequent renaming has been done in the OS.

    spg_title = _readbtext(fid, 30)

    # Check if it is really a spg file (in this case title his the filename
    # with extension spg)
    # if spg_title[-4:].lower() != ".spg":  # pragma: no cover
    #     # probably not a spg content
    #     # try .spa
    #     fid.close()
    #     return _read_spa(*args, **kwargs)

    # Count the number of spectra
    # From hex 120 = decimal 304, individual spectra are described
    # by blocks of lines starting with "key values",
    # for instance hex[02 6a 6b 69 1b 03 82] -> dec[02 106  107 105 27 03 130]
    # Each of theses lines provides positions of data and metadata in the file:
    #
    #     key: hex 02, dec  02: position of spectral header (=> nx, firstx,
    #     lastx, nscans, nbkgscans)
    #     key: hex 03, dec  03: intensity position
    #     key: hex 04, dec  04: user text position
    #     key: hex 1B, dec  27: position of History text
    #     key: hex 69, dec 105: ?
    #     key: hex 6a, dec 106: ?
    #     key: hex 6b, dec 107: position of spectrum title, the acquisition
    #     date follows at +256(dec)
    #     key: hex 80, dec 128: ?
    #     key: hex 82, dec 130: ?
    #
    # the number of line per block may change from one omnic version to
    # another,
    # but the total number of lines is given at hex 294, hence allowing
    # counting
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

    # the number of occurences of the key '02' is number of spectra
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
        info02 = _readheader02(fid, position02[i])
        nx[i] = info02["nx"]
        firstx[i] = info02["firstx"]
        lastx[i] = info02["lastx"]
        xunits.append(info02["xunits"])
        xtitles.append(info02["xtitle"])
        units.append(info02["units"])
        titles.append(info02["title"])

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
        spa_title = _readbtext(fid, spa_title_pos)  # [0])
        spectitles.append(spa_title)

        # and the acquisition date
        fid.seek(spa_title_pos + 256)
        timestamp = _fromfile(fid, dtype="uint32", count=1)  #
        # since 31/12/1899, 00:00
        acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(
            seconds=int(timestamp)
        )
        acquisitiondates.append(acqdate)
        timestamp = acqdate.timestamp()
        # Transform back to timestamp for storage in the Coord object
        # use datetime.fromtimestamp(d, timezone.utc))
        # to transform back to datetime obkct

        timestamps.append(timestamp)

        # Not used at present  # -------------------  # extract positions of
        # '1B' codes (history text  #  --  #  #  #  # sometimes absent,
        # e.g. peakresolve)  # key_is_1B = (keys == 27)  # indices1B =  # np.nonzero(
        #  key_is_1B)  #  #  # position1B = 304 * np.ones(len(
        # indices1B[0]), dtype='int') + 16 * indices6B[0]  #
        #  if len(  # position1B) != 0:  #    # read history texts
        #    for j in range(  # nspec):  #        #  #  determine the position of information
        #  #  # f.seek(position1B[j] + 2)  #        history_pos =  #  _fromfile(f,  # 'uint32', 1)
        #        # read history  #  #    history =  # _readbtext(f, history_pos[0])
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

    # Set origin and description
    dataset.origin = "omnic"
    dataset.description = kwargs.get(
        "description", f"Omnic title: {spg_title}\nOmnic " f"filename: {filename}"
    )

    # Set the NDDataset date
    dataset._date = datetime.now(timezone.utc)

    # Set origin, description and history
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

    # Read title:
    # The file title  starts at position hex 1e = decimal 30. Its max length
    # is 256 bytes. It is the original
    # filename under which the group has  been saved: it won't match with
    # the actual filename if a subsequent
    # renaming has been done in the OS.

    spa_title = _readbtext(fid, 30)

    # The acquisition date (GMT) is at hex 128 = decimal 296.
    # The format is HFS+ 32 bit hex value, little endian

    fid.seek(296)

    # days since 31/12/1899, 00:00
    timestamp = _fromfile(fid, dtype="uint32", count=1)
    acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=int(timestamp)
    )
    acquisitiondate = acqdate

    # Transform back to timestamp for storage in the Coord object
    # use datetime.fromtimestamp(d, timezone.utc))
    # to transform back to datetime object
    timestamp = acqdate.timestamp()

    # From hex 120 = decimal 304, the spectrum is described
    # by blocks of lines starting with "key values",
    # for instance hex[02 6a 6b 69 1b 03 82] -> dec[02 106  107 105 27 03 130]
    # Each of theses lines provides positions of data and metadata in the file:
    #
    #     key: hex 02, dec  02: position of spectral header (=> nx,
    #                                 firstx, lastx, nscans, nbkgscans)
    #     key: hex 03, dec  03: intensity position
    #     key: hex 04, dec  04: user text position
    #     key: hex 1B, dec  27: position of History text
    #     key: hex 66  dec 102: sample inferogram
    #     key: hex 67  dec 103: background inferogram
    #     key: hex 69, dec 105: ?
    #     key: hex 6a, dec 106: ?
    #     key: hex 80, dec 128: ?
    #     key: hex 82, dec 130: rotation angle

    gotinfos = [False, False, False]  # spectral header, intensity, history
    # scan "key values"
    pos = 304
    while not (all(gotinfos)):
        fid.seek(pos)
        key = _fromfile(fid, dtype="uint8", count=1)
        if key == 2:
            info02 = _readheader02(fid, pos)
            nx = info02["nx"]
            firstx = info02["firstx"]
            lastx = info02["lastx"]
            xunit = info02["xunits"]
            xtitle = info02["xtitle"]
            units = info02["units"]
            title = info02["title"]
            gotinfos[0] = True

        elif key == 3:
            intensities = _getintensities(fid, pos)
            gotinfos[1] = True

        elif key == 27:
            fid.seek(pos + 2)
            history_pos = _fromfile(fid, "uint32", 1)
            # read history
            history = _readbtext(fid, history_pos)
            gotinfos[2] = True

        elif not key:  # pragma: no cover
            break

        pos += 16

    fid.close()

    # load spectral content into the  NDDataset
    dataset.data = np.array(intensities[np.newaxis], dtype="float32")
    dataset.units = units
    dataset.title = title
    dataset.name = filename.stem
    dataset.filename = str(filename)

    # now add coordinates
    # _x = Coord(np.around(np.linspace(firstx, lastx, nx, 3)), title=xtitle,
    # units=xunit)
    spacing = (lastx - firstx) / (nx - 1)
    _x = LinearCoord(
        offset=firstx, increment=spacing, size=nx, title=xtitle, units=xunit
    )

    _y = Coord(
        [timestamp],
        title="acquisition timestamp (GMT)",
        units="s",
        labels=([acquisitiondate], [filename]),
    )
    dataset.set_coordset(y=_y, x=_x)

    # Set origin, description, history, date
    dataset.origin = "omnic"
    dataset.description = kwargs.get(
        "description", f"Omnic title: {spa_title}\nOmnic " f"filename: {filename.name}"
    )
    dataset.history = str(datetime.now(timezone.utc)) + ":imported from spa files"
    dataset.history = history
    dataset._date = datetime.now(timezone.utc)

    if dataset.x.units is None and dataset.x.title == "data points":
        # interferogram
        dataset.meta.interferogram = True
        dataset.meta.td = list(dataset.shape)
        dataset.x._zpd = int(np.argmax(dataset)[-1])  # zero path difference
        dataset.meta.laser_frequency = Quantity("15798.26 cm^-1")
        dataset.x.set_laser_frequency()
        dataset.x._use_time_axis = (
            False  # True to have time, else it will  # be optical path difference
        )

    return dataset


# ..............................................................................
@importermethod
def _read_srs(*args, **kwargs):
    dataset, filename = args
    frombytes = kwargs.get("frombytes", False)

    if frombytes:
        # in this case, filename is actualy a byte content
        fid = io.BytesIO(filename)  # pragma: no cover
    else:
        fid = open(filename, "rb")
    # at pos=306 (hex:132) is the position of the xheader
    fid.seek(306)
    pos_xheader = _fromfile(fid, dtype="int32", count=1)
    info, pos = _read_xheader(fid, pos_xheader)

    # reset current position at the start of next line
    pos = _nextline(pos)

    if info["mode"] != "rapidscan":
        raise NotImplementedError("Only implemented for rapidscan")

    # read the data part of series files
    found = False
    background = None
    names = []
    data = np.zeros((info["ny"], info["nx"]))

    # find the position of the background and of the first interferogram
    # based on
    # empirical "fingerprints".

    while not found:
        pos += 16
        fid.seek(pos)
        line = _fromfile(fid, dtype="uint8", count=16)
        if np.all(line == [15, 0, 0, 0, 2, 0, 0, 0, 24, 0, 0, 0, 0, 0, 72, 67]):
            # hex 0F 00 00 00 02 00 00 00 18 00 00 00 00 00 48 43
            # this is a fingerprint of header of data fields for
            # non-processed series
            # the first one is the background
            if background is None:
                pos += 52
                fid.seek(pos)
                key = _fromfile(fid, dtype="uint16", count=1)

                if key > 0:  # pragma: no cover
                    # a background file was selected; it is present as a
                    # single sided interferogram
                    #  key could be the zpd of the double sided interferogram
                    background_size = key - 2
                    pos += 8
                    background_name = _readbtext(fid, pos)
                    pos += 256  # max length of text
                    pos += 8  # unknown info ?
                    fid.seek(pos)
                    background = _fromfile(fid, dtype="float32", count=background_size)
                    pos += background_size * 4
                    pos = _nextline(pos)

                elif key == 0:
                    # no background file was selected; the background is the
                    # one that was recorded with the series
                    background_size = info["nx"]
                    pos += 8
                    fid.seek(pos)
                    background = _fromfile(fid, dtype="float32", count=background_size)
                    pos += background_size * 4
                    background_name = _readbtext(fid, pos)
                    # uncomment below to read unused data (noise measurement ?)
                    # pos += 268
                    # f.seek(pos)
                    # noisy_data = _fromfile(f, dtype='float32', count=499)
                    pos = _nextline(pos)

                # Create a NDDataset for the background

                background = NDDataset(background)
                _x = Coord(
                    np.around(np.linspace(0, background_size - 1, background_size), 0),
                    title="data points",
                    units="dimensionless",
                )
                background.set_coordset(x=_x)
                background.name = background_name
                background.units = "V"
                background.title = "volts"
                background.origin = "omnic"
                background.description = "background from omnic srs file."
                background.history = (
                    str(datetime.now(timezone.utc)) + ":imported from srs file"
                )

            else:  # this is likely the first interferogram of the series
                found = True
                names.append(_readbtext(fid, pos + 64))
                pos += 148

        elif np.all(
            line == [2, 0, 0, 0, 24, 0, 0, 0, 0, 0, 72, 67, 0, 80, 67, 71]
        ) or np.all(line == [30, 0, 0, 0, 2, 0, 0, 0, 24, 0, 0, 0, 0, 0, 72, 67]):
            # hex 02 00 00 00 18 00 00 00 00 00 48 43 00 50 43 47
            # this is likely header of data field of reprocessed series
            # the first one is skipped TODO: check the nature of these data
            if background is None:  # pragma: no cover
                # skip
                background = NDDataset()
            else:  # this is likely the first spectrum of the series
                found = True
                names.append(_readbtext(fid, pos + 64))
                pos += 148

    # read first data
    fid.seek(pos)
    data[0, :] = _fromfile(fid, dtype="float32", count=info["nx"])[:]
    pos += info["nx"] * 4
    # and the remaining part:
    for i in np.arange(info["ny"])[1:]:
        pos += 16
        names.append(_readbtext(fid, pos))
        pos += 84
        fid.seek(pos)
        data[i, :] = _fromfile(fid, dtype="float32", count=info["nx"])[:]
        pos += info["nx"] * 4

    # Create NDDataset Object for the series
    dataset = NDDataset(data)
    dataset.name = info["name"]
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

    _y = Coord(
        np.around(np.linspace(info["firsty"], info["lasty"], info["ny"]), 3),
        title="Time",
        units="minute",
        labels=names,
    )

    dataset.set_coordset(y=_y, x=_x)

    # Set origin, description and history
    dataset.origin = "omnic"
    dataset.description = kwargs.get("description", "Dataset from omnic srs file.")

    dataset.history = str(
        datetime.now(timezone.utc)
    ) + ":imported from srs file {} ; ".format(filename)

    if dataset.x.units is None and dataset.x.title == "data points":
        # interferogram
        dataset.meta.interferogram = True
        dataset.meta.td = list(dataset.shape)
        dataset.x._zpd = int(np.argmax(dataset)[-1])  # zero path difference
        dataset.meta.laser_frequency = Quantity("15798.26 cm^-1")
        dataset.x.set_laser_frequency()
        dataset.x._use_time_axis = (
            False  # True to have time, else it will  # be optical path difference
        )

    # uncomment below to load the last datafield
    # has the same dimension as the time axis
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
    # X.origin = 'omnic'
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
    fid.seek(pos)  # read first byte, ensure entering the while loop
    btext = fid.read(1)
    while not (btext[len(btext) - 1] == 0):  # while the last byte of btext differs from
        # zero
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
    # reset current position to the begining of next line (16 bytes length)
    return 16 * (1 + pos // 16)


# ..............................................................................
def _readheader02(fid, pos):
    # read spectrum header, pos is the position of the 02 key
    # returns a dict
    fid.seek(pos + 2)  # go to line and skip 2 bytes
    info_pos = _fromfile(fid, dtype="uint32", count=1)

    # other positions:
    #   nx_pos = info_pos + 4
    #   xaxis unit code = info_pos + 8
    #   data unit code = info_pos + 12
    #   fistx_pos = info_pos + 16
    #   lastx_pos = info_pos + 20
    #   nscan_pos = info_pos + 36;
    #   nbkgscan_pos = info_pos + 52;

    fid.seek(info_pos + 4)
    out = {"nx": _fromfile(fid, "uint32", 1)}

    # read xaxis unit
    fid.seek(info_pos + 8)
    key = _fromfile(fid, dtype="uint8", count=1)
    if key == 1:
        out["xunits"] = "cm ^ -1"
        out["xtitle"] = "wavenumbers"
    elif key == 2:  # pragma: no cover
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
        out[
            "xtitle"
        ] = "xaxis"  # warning: 'The nature of data is not  # recognized, xtitle set to \'xaxis\')

    # read data unit
    fid.seek(info_pos + 12)
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
        out["title"] = "Log(1/R)"
    elif key == 20:  # pragma: no cover
        out["units"] = "Kubelka_Munk"
        out["title"] = "Kubelka-Munk"
    elif key == 22:  # pragma: no cover
        out["units"] = "V"
        out["title"] = "detector signal"
    elif key == 26:  # pragma: no cover
        out["units"] = None
        out["title"] = "photoacoustic"
    elif key == 31:  # pragma: no cover
        out["units"] = None
        out["title"] = "raman intensity"
    else:  # pragma: no cover
        out["units"] = None
        out[
            "title"
        ] = "intensity"  # warning: 'The nature of data is not  # recognized, title set to \'Intensity\')

    fid.seek(info_pos + 16)
    out["firstx"] = _fromfile(fid, "float32", 1)
    fid.seek(info_pos + 20)
    out["lastx"] = _fromfile(fid, "float32", 1)
    fid.seek(info_pos + 36)
    out["nscan"] = _fromfile(fid, "uint32", 1)
    fid.seek(info_pos + 52)
    out["nbkgscan"] = _fromfile(fid, "uint32", 1)

    return out


# ..............................................................................
def _read_xheader(fid, pos):
    # read spectrum header, pos is the position of the 03 or 01 key
    # for series files
    # return a dict and updated position in the file
    # Todo: merge with _readheader02

    fid.seek(pos)
    key = _fromfile(fid, dtype="uint8", count=1)

    if key not in (1, 3):
        raise ValueError(  # pragma: no cover
            "xheader key={} not recognized yet.".format(key)
            + " Please report this error (and the corresponding srs "
            "file) to the developers"
            "They will do their best to fix the issue"
        )
    else:
        out = {"xheader": key}

    #   positions
    #   nx_pos = info_pos + 4
    #   xaxis unit code = info_pos + 8
    #   data unit code = info_pos + 12
    #   fistx_pos = info_pos + 16
    #   lastx_pos = info_pos + 20
    #   scan_pts_pos = info_pos + 29,
    #   nscan_pos = info_pos + 36;
    #   nbkgscan_pos = info_pos + 52;

    fid.seek(pos + 4)
    out["nx"] = _fromfile(fid, "uint32", count=1)

    # read xaxis unit
    fid.seek(pos + 8)
    key = _fromfile(fid, dtype="uint8", count=1)
    if key == 1:  # pragma: no cover
        out["xunits"] = "cm ^ -1"
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
        out[
            "xtitle"
        ] = "xaxis"  # warning: 'The nature of data is not  # recognized, xtitle set to \'xaxis\')
    # read data unit
    fid.seek(pos + 12)
    key = _fromfile(fid, dtype="uint8", count=1)
    if key == 17:  # pragma: no cover
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
        out["title"] = "raman intensity"
    else:  # pragma: no cover
        out["title"] = None
        out["title"] = "intensity"
        # warning: 'The nature of data is not
        # recognized, title set to \'Intensity\')

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
    if out["nbkgscan"] == 0:  # then probably interferogram in rapid scan mode
        #     out['units'] = 'V'
        #     out['title'] = 'Volts'
        #     out['xunits'] = 'dimensionless'
        #     out['xtitle'] = 'Data points'
        if out["firstx"] > out["lastx"]:  # pragma: no cover
            out["firstx"], out["lastx"] = out["lastx"], out["firstx"]
        out["mode"] = "rapidscan"
    else:  # pragma: no cover
        out["mode"] = "GC-IR or TGA-IR"

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
    return out, pos + 1026


# ..............................................................................
def _getintensities(fid, pos):
    # get intensities from the 03 key
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
