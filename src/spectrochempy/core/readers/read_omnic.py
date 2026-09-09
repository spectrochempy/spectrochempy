# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Extend NDDataset with the import method for OMNIC generated data files.

This module provides functions to read OMNIC generated data files.
"""

__all__ = ["read_omnic", "read_spg", "read_spa", "read_srs"]

import io
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import info_
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.datetimeutils import UTC
from spectrochempy.utils.datetimeutils import utcnow
from spectrochempy.utils.decorators import warn_deprecated
from spectrochempy.utils.file import fromfile


# ======================================================================================
# Public functions
# ======================================================================================
def read_omnic(*paths, **kwargs):
    r"""
    Open a Thermo Nicolet OMNIC file.

    This is the explicit OMNIC reader in the public import API. Use
    :func:`spectrochempy.read` for generic format autodetection and
    ``scp.omnic.read(...)`` or :func:`spectrochempy.read_omnic` when the OMNIC
    format is already known.

    Open Omnic file or a list of :file:`.spg`, :file:`.spa` or
    :file:`.srs` files and set data/metadata in the current dataset.

    The collected metadata are:
    - names of spectra
    - acquisition dates (UTC)
    - units of spectra (absorbance, transmittance, reflectance, Log(1/R),
    Kubelka-Munk, Raman intensity, photoacoustics, volts)
    - units of xaxis (wavenumbers in :math:`cm^{-1}`, wavelengths in nm or micrometer,
    Raman shift in :math:`cm^{-1}`)
    - spectra history (but only incorporated in the NDDataset if a single
    spa is read)

    An error is generated when an SPG file contains spectra with inconsistent
    x-axis definitions, unless ``allow_inconsistent_x=True`` is specified. In
    that case, a list containing one `NDDataset` per spectrum is returned.

    Non-merged multi-file reads may also return a list-like `ScpObjectList`
    exposing helper methods for dataset selection. See
    :func:`spectrochempy.read` for the complete description of the generic
    import convention and multi-object return behavior.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ...,  kwargs )

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
        useful for a web application to handle drag and drop of files into a
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
    allow_inconsistent_x : `bool`, optional, default: `False`
        Allow SPG files whose spectra have different x-axis definitions. When
        enabled, return one `NDDataset` per spectrum instead of merging the
        spectra. This option has no effect on SPA or SRS files.

    See Also
    --------
    read : Generic reader inferring protocol from the filename extension.
    :func:`spectrochempy.read_zip` : Read Zip archives (containing spectrochempy readable files)
    :func:`spectrochempy.read_dir` : Read an entire directory.
    :func:`spectrochempy.read_opus` : Read OPUS spectra.
    :func:`spectrochempy.read_labspec` : Read Raman LABSPEC spectra (:file:`.txt`).
    :func:`spectrochempy.read_omnic` : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    :func:`spectrochempy.read_soc` : Read Surface Optics Corp. files (:file:`.ddr`, :file:`.hdr`, or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).
    :func:`spectrochempy.read_spg` : Alias of `read_omnic`.
    :func:`spectrochempy.read_spa` : Alias of `read_omnic`.
    :func:`spectrochempy.read_srs` : Alias of `read_omnic`.

    Examples
    --------
    Reading a single OMNIC file  (providing a windows type filename relative
    to the default ``datadir`` )

    >>> scp.read_omnic('irdata\\nh4y-activation.spg')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Reading a single OMNIC file  (providing a unix/python type filename relative
    to the default ``datadir`` )

    Note that here read_omnic is called as a classmethod of the NDDataset class

    >>> scp.read_omnic('irdata/nh4y-activation.spg')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Using the explicit namespace API

    >>> scp.omnic.read('irdata/nh4y-activation.spg')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata')
    >>> p = folder / 'nh4y-activation.spg'
    >>> scp.read_omnic(p)
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Multiple files not merged (return a list-like multi-dataset result).
    Note that a directory is specified

    >>> le = scp.read_omnic('irdata/nh4y-activation.spg', 'wodger.spg')
    >>> len(le)
    2
    >>> le[1]
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read_omnic('irdata/nh4y-activation.spg', 'wodger.spg', merge=True)
    NDDataset: [float64] a.u. (shape: (y:57, x:5549))

    Multiple files to merge : they are passed as a list instead of using the keyword
    `merge`

    >>> scp.read_omnic(['irdata/nh4y-activation.spg', 'wodger.spg'])
    NDDataset: [float64] a.u. (shape: (y:57, x:5549))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> l2 = scp.read_omnic(['irdata/nh4y-activation.spg', 'wodger.spg'], merge=False)
    >>> len(l2)
    2
    >>> names = l2.names
    >>> len(names)
    2
    >>> largest = l2.select_largest()
    >>> largest.ndim
    2

    Read without a filename. This has the effect of opening a dialog for file(s)
    selection

    >>> nd = scp.read_omnic()

    Read in a directory (assume that only OMNIC files are present in the directory
    (else we must use the generic `read` function instead)

    >>> l3 = scp.read_omnic(directory='irdata/OMNIC/1-20')
    >>> len(l3)
    3

    Again we can use merge to stack all 4 spectra if they have compatible dimensions.

    >>> scp.read_omnic(directory='irdata/OMNIC/1-20', merge=True)
    [NDDataset: [float64] a.u. (shape: (y:1, x:5549)), NDDataset: [float64] a.u. (shape: (y:4, x:5549))]

    """
    kwargs["filetypes"] = ["OMNIC files (*.spa *.spg)", "OMNIC series (*.srs)"]
    kwargs["protocol"] = ["omnic", "spa", "spg", "srs"]
    importer = Importer()
    return importer(*paths, **kwargs)


def read_spg(*paths, **kwargs):
    r"""
    Open a Thermo Nicolet file or a list of files with extension ``.spg``.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
    The data source(s) can be specified by the name or a list of name for the
    file(s) to be loaded:

        - e.g., ( filename1, filename2, ...,  kwargs )

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
    result is a list-like `ScpObjectList`.

    Other Parameters
    ----------------
    content : `bytes` object, optional
    Instead of passing a filename for further reading, a bytes content can be
    directly provided as bytes objects.
    The most convenient way is to use a dictionary. This feature is particularly
    useful for a web application to handle drag and drop of files into a
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
    allow_inconsistent_x : `bool`, optional, default: `False`
    Allow spectra with different x-axis definitions and return one `NDDataset`
    per spectrum instead of a merged dataset.

    See Also
    --------
    read_spg : Read grouped Omnic spectra.
    read_spa : Read single Omnic spectra.
    read_srs : Read series Omnic spectra.
    read : Generic reader inferring protocol from the filename extension.
    :func:`spectrochempy.read_zip` : Read Zip archives (containing spectrochempy readable files)
    :func:`spectrochempy.read_dir` : Read an entire directory.
    :func:`spectrochempy.read_opus` : Read OPUS spectra.
    :func:`spectrochempy.read_labspec` : Read Raman LABSPEC spectra (:file:`.txt`).
    :func:`spectrochempy.read_omnic` : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    :func:`spectrochempy.read_soc` : Read Surface Optics Corp. files (:file:`.ddr`, :file:`.hdr`, or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).

    Notes
    -----
    This method is an alias of `read_omnic`, except that the type of file
    is constrained to ``.spg``.

    Examples
    --------
    >>> scp.read_spg('irdata/nh4y-activation.spg')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))

    """
    kwargs["filetypes"] = ["OMNIC files (*.spg)"]
    kwargs["protocol"] = ["spg"]
    importer = Importer()
    return importer(*paths, **kwargs)


def read_spa(*paths, **kwargs):
    r"""
    Open a Thermo Nicolet file or a list of files with extension ``.spa``.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
    The data source(s) can be specified by the name or a list of name for the
    file(s) to be loaded:

        - e.g., ( filename1, filename2, ...,  kwargs )

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
    result is a list-like `ScpObjectList`.

    Other Parameters
    ----------------
    content : `bytes` object, optional
    Instead of passing a filename for further reading, a bytes content can be
    directly provided as bytes objects.
    The most convenient way is to use a dictionary. This feature is particularly
    useful for a web application to handle drag and drop of files into a
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
    :func:`spectrochempy.read_soc` : Read Surface Optics Corp. files (:file:`.ddr`, :file:`.hdr`, or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).

    Notes
    -----
    This method is an alias of `read_omnic`, except that the type of file
    is constrained to ``.spa``.

    Examples
    --------
    >>> scp.read_spa('irdata/subdir/20-50/7_CZ0-100 Pd_21.SPA')
    NDDataset: [float64] a.u. (shape: (y:1, x:5549))
    >>> scp.read_spa(directory='irdata/subdir', merge=True)
    NDDataset: [float64] a.u. (shape: (y:4, x:5549))

    """
    kwargs["filetypes"] = ["OMNIC files (*.spa)"]
    kwargs["protocol"] = ["spa"]
    importer = Importer()
    return importer(*paths, **kwargs)


def read_srs(*paths, **kwargs):
    r"""
    Open a Thermo Nicolet file or a list of files with extension ``.srs``.

    .. note::
       The reverse-engineered binary layout of the ``.srs`` format is
       documented in :doc:`/devguide/file_formats/omnic/srs`.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
    The data source(s) can be specified by the name or a list of name for the
    file(s) to be loaded:

        - e.g., ( filename1, filename2, ...,  kwargs )

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
    result is a list-like `ScpObjectList`.
        When return_bg is set to 'True', the series background is returned.

    Other Parameters
    ----------------
    return_bg : bool, optional
        Default value is False. When set to 'True' returns the series background

    .. note::
       SRS **spectral** series are now read normalized to the public
       SpectroChemPy convention used by :func:`read_spa`: the wavenumber X
       coordinate is exposed descending (high to low wavenumber) with the
       intensity data matched to it. This normalization is applied
       automatically, per spectrum/background record, from each record's own
       endpoints. Rapid-scan interferograms keep their ascending
       data-points coordinate and are not treated as spectral data.

    reverse_x : bool, optional
        .. deprecated::
            No longer needed. Spectral orientation is handled automatically as
            described above; this historical workaround (introduced for issue
            #858) is now a no-op and will be removed according to the
            SpectroChemPy deprecation policy. Supplying the ``reverse_x``
            keyword (with any value, ``True`` or ``False``) emits a
            ``DeprecationWarning`` and is ignored.

    content : `bytes` object, optional
    Instead of passing a filename for further reading, a bytes content can be
    directly provided as bytes objects.
    The most convenient way is to use a dictionary. This feature is particularly
    useful for a web application to handle drag and drop of files into a
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
    :func:`spectrochempy.read_soc` : Read Surface Optics Corp. files (:file:`.ddr`, :file:`.hdr`, or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).

    Notes
    -----
    This method is an alias of `read_omnic`, except that the type of file
    is constrained to ``.srs``.

    Examples
    --------
    >>> scp.read_srs('irdata/omnic_series/rapid_scan_reprocessed.srs')
    NDDataset: [float64] a.u. (shape: (y:643, x:3734))

    """
    kwargs["filetypes"] = ["OMNIC series (*.srs)"]
    kwargs["protocol"] = ["srs"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_spg(*args, **kwargs):
    # read spg file

    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    # Read name:
    # The name starts at position hex 1e = decimal 30. Its max length
    # is 256 bytes. It is the original filename under which the group has been saved: it
    # won't match with the actual filename if a subsequent renaming has been done in the
    # OS.

    spg_title = _readbtext(fid, 30, 256)

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
    #     key: hex 66  dec 102: sample interferogram
    #     key: hex 67  dec 103: background interferogram
    #     key: hex 6a, dec 106: canonical acquisition-parameter block
    #     key: hex 6b, dec 107: position of spectrum title, the acquisition
    #     date follows at +256(dec)
    #
    # the number of line per block may change from file to file but the total
    # number of lines is given at hex 294, hence allowing counting the
    # number of spectra:

    # read total number of lines
    fid.seek(294)
    nlines = fromfile(fid, "uint16", count=1)

    # read "key values"
    pos = 304
    keys = np.zeros(nlines)
    for i in range(nlines):
        fid.seek(pos)
        keys[i] = fromfile(fid, dtype="uint8", count=1)
        pos += 16

    # Extract experiment info blocks (key 0x82, subtype 0x79).
    experiment_infos = []
    key_is_82 = keys == 130
    for idx in np.nonzero(key_is_82)[0]:
        entry_pos = 304 + 16 * idx
        fid.seek(entry_pos + 2)
        blk_pos = fromfile(fid, "uint32", 1)
        fid.seek(entry_pos + 6)
        blk_len = fromfile(fid, "uint32", 1)
        if blk_len >= 50:
            cur = fid.tell()
            fid.seek(blk_pos)
            blk_data = fid.read(blk_len)
            fid.seek(cur)
            exp = _decode_experiment_info_block(blk_data)
            if exp:
                experiment_infos.append(exp)

    # the number of occurrences of the key '02' is number of spectra
    nspec = np.count_nonzero(keys == 2)

    if nspec == 0:  # pragma: no cover
        raise OSError(
            "Error : File format not recognized - information markers not found",
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
        pos_header = fromfile(fid, dtype="uint32", count=1)
        # get infos
        info = _read_header(fid, pos_header, is_first_spectrum=(i == 0))
        nx[i] = info["nx"]
        firstx[i] = info["firstx"]
        lastx[i] = info["lastx"]
        xunits.append(info["xunits"])
        xtitles.append(info["xtitle"])
        units.append(info["units"])
        titles.append(info["title"])

    # Extract positions of intensity and spectrum metadata blocks before
    # checking consistency because they are needed by both return paths.
    key_is_03 = keys == 3
    indices03 = np.nonzero(key_is_03)
    position03 = 304 * np.ones(len(indices03[0]), dtype="int") + 16 * indices03[0]

    key_is_6B = keys == 107
    indices6B = np.nonzero(key_is_6B)
    position6B = 304 * np.ones(len(indices6B[0]), dtype="int") + 16 * indices6B[0]

    spectitles, acquisitiondates, timestamps = [], [], []
    for i in range(nspec):
        fid.seek(position6B[i] + 2)  # go to line and skip 2 bytes
        spa_title_pos = fromfile(fid, "uint32", 1)

        # read omnic filename
        spa_title = _readbtext(fid, spa_title_pos, 256)
        spectitles.append(spa_title)

        # and the acquisition date
        fid.seek(spa_title_pos + 256)
        timestamp = fromfile(fid, dtype="uint32", count=1)
        # since 31/12/1899, 00:00
        acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=UTC) + timedelta(
            seconds=int(timestamp),
        )
        acquisitiondates.append(acqdate)
        timestamps.append(acqdate.timestamp())

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
        #        history_pos = fromfile(f,  'uint32', 1)
        #        history =  _readbtext(f, history_pos[0])
        #        allhistories.append(history)

    xaxis_consistent = (
        np.ptp(nx) == 0
        and np.ptp(firstx) == 0
        and np.ptp(lastx) == 0
        and len(set(xunits)) == 1
        and len(set(units)) == 1
    )

    if not xaxis_consistent:
        if kwargs.get("allow_inconsistent_x", False):
            datasets = []
            for i in range(nspec):
                single = dataset.__class__(
                    np.expand_dims(_getintensities(fid, position03[i]), axis=0)
                )
                single.units = units[i]
                single.title = titles[i]
                single.name = f"{filename.stem}_spectrum_{i}"
                single.filename = filename
                single.set_coordset(
                    y=Coord(
                        [timestamps[i]],
                        title="acquisition timestamp (GMT)",
                        units="s",
                        labels=([acquisitiondates[i]], [spectitles[i]]),
                    ),
                    x=Coord.linspace(
                        firstx[i],
                        lastx[i],
                        nx[i],
                        title=xtitles[i],
                        units=xunits[i],
                    ),
                )
                single.acquisition_date = acquisitiondates[i]
                single.origin = "omnic"
                single.description = kwargs.get(
                    "description",
                    f"Omnic title: {spg_title}\nOmnic filename: {filename}",
                )
                single._date = utcnow()
                single.history = f"Imported from spg file {filename} (spectrum {i})."
                datasets.append(single)

            fid.close()
            return datasets

        inconsistencies = []
        if np.ptp(nx) != 0:
            inconsistencies.append(
                f"number of wavenumbers per spectrum varies: {nx.tolist()}"
            )
        if np.ptp(firstx) != 0:
            inconsistencies.append(f"x-axis start values differ: {firstx.tolist()}")
        if np.ptp(lastx) != 0:
            inconsistencies.append(f"x-axis end values differ: {lastx.tolist()}")
        if len(set(xunits)) != 1:
            inconsistencies.append(f"x-axis units differ: {list(set(xunits))}")
        if len(set(units)) != 1:
            inconsistencies.append(f"spectra units differ: {list(set(units))}")
        fid.close()
        raise ValueError(
            "Error: Inconsistent data set - "
            f"{', '.join(inconsistencies)}. "
            "Use allow_inconsistent_x=True to return one NDDataset per spectrum."
        )

    data = np.ndarray((nspec, nx[0]), dtype="float32")
    for i in range(nspec):
        data[i, :] = _getintensities(fid, position03[i])

    fid.close()

    # Create Dataset Object of spectral content
    dataset.data = data
    dataset.units = units[0]
    dataset.title = titles[0]
    dataset.name = filename.stem
    dataset.filename = filename

    # now add coordinates
    _x = Coord.linspace(
        firstx[0],
        lastx[0],
        nx[0],
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
    if acquisitiondates:
        dataset.acquisition_date = min(acquisitiondates)
    dataset.origin = "omnic"

    # Set description, date and history
    # Omnic spg file don't have specific "origin" field stating the oirigin of the data
    dataset.description = kwargs.get(
        "description",
        f"Omnic title: {spg_title}\nOmnic filename: {filename}",
    )

    dataset._date = utcnow()

    dataset.history = f"Imported from spg file {filename}."

    # Attach acquisition metadata from the header.
    # Acquisition parameters (collection_length, reference_frequency,
    # optical_velocity) are global to the SPG file — the header at
    # position02 stores them once, not per spectrum.  Reusing the last
    # parsed header is therefore intentional.
    dataset.meta.collection_length = info["collection_length"] / 100 * ur("s")
    dataset.meta.optical_velocity = info["optical_velocity"]
    dataset.meta.laser_frequency = info["reference_frequency"] * ur("cm^-1")

    # Attach experiment info if available.
    # If all decoded 0x79 blocks are identical, attach once; if different,
    # attach the first set only (conservative).
    if experiment_infos:
        first = experiment_infos[0]
        for meta_key, val in first.items():
            setattr(dataset.meta, f"omnic_{meta_key}", val)
        if not dataset.description.strip() and first.get("experiment_title"):
            dataset.description = first["experiment_title"]

    if kwargs.pop("sortbydate", True):
        dataset.sort(dim="y", inplace=True)
        dataset.history = "Sorted by date"

    # debug_("end of reading")

    return dataset


@_importer_method
def _read_spa(*args, **kwargs):
    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    return_ifg = kwargs.get("return_ifg", None)
    if return_ifg not in (None, "sample", "background"):
        raise ValueError(
            f"Invalid return_ifg value: {return_ifg!r}. "
            "Expected None, 'sample', or 'background'."
        )

    # Read name:
    # The name  starts at position hex 1e = decimal 30. Its max length
    # is 256 bytes. It is the original filename under which the spectrum has
    # been saved: it won't match with the actual filename if a subsequent
    # renaming has been done in the OS.
    spa_name = _readbtext(fid, 30, 256)

    # The raw acquisition-time value is at file offset 296.
    fid.seek(296)
    raw_timestamp = int(fromfile(fid, dtype="uint32", count=1))

    # The active key table is counted at +294 and consists of 16-byte records
    # beginning at +304. Terminators, padding, and post-table variant data are
    # intentionally outside the parsed record list.
    # Current dispatch handles the primary header/payload, comments, history,
    # associated IFGs, optical velocity, and Experiment Information blocks.
    # Detailed key semantics and variant scope are documented in spa.rst.
    #

    records = _read_spa_key_table(fid)
    has_library_text = any(record.key == 0x53 for record in records)
    fid.seek(304 + 16 * len(records))
    post_table_key = fid.read(1)
    # This is the discriminator for the observed validated library/retrieved
    # layout: 0x53 together with the post-table 0x01 grid. It is not a
    # universal semantic interpretation of either structure.
    is_library_variant = has_library_text and post_table_key == b"\x01"
    if is_library_variant:
        acquisitiondate = None
        timestamp = 0.0
    else:
        acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=UTC) + timedelta(
            seconds=raw_timestamp,
        )
        acquisitiondate = acqdate
        timestamp = acqdate.timestamp()
    spa_comments = []  # several custom comments can be present
    _exp_info = None
    acquisition_parameters = {}
    for record in records:
        key = record.key

        if key == 2:
            info = _read_header(fid, record.position)

        elif key == 3 and return_ifg is None:
            intensities = _read_spa_float32_payload(fid, record)

        elif key == 4:
            fid.seek(record.position)
            spa_comments.append(fid.read(record.length).decode("latin-1", "replace"))

        elif key == 27:
            spa_history = _readbtext(fid, record.position, record.length)

        elif key == 102 and return_ifg == "sample":
            s_ifg_intensities = _read_spa_float32_payload(fid, record)

        elif key == 103 and return_ifg == "background":
            b_ifg_intensities = _read_spa_float32_payload(fid, record)

        elif key == 106:
            acquisition_parameters.update(_read_spa_acquisition_parameters(fid, record))

        elif key == 130 and _exp_info is None and record.length >= 50:
            fid.seek(record.position)
            blk_data = fid.read(record.length)
            _exp_info = _decode_experiment_info_block(blk_data)

    fid.close()

    if (return_ifg == "sample" and "s_ifg_intensities" not in locals()) or (
        return_ifg == "background" and "b_ifg_intensities" not in locals()
    ):
        info_("No interferogram found, read_spa returns None")
        return None
    if return_ifg == "sample":
        intensities = s_ifg_intensities
    elif return_ifg == "background":
        intensities = b_ifg_intensities
    # load intensity into the  NDDataset
    dataset.data = np.array(intensities[np.newaxis], dtype="float32")

    if return_ifg == "background":
        title = "sample acquisition timestamp (GMT)"  # bckg acquisition date is not known for the moment...
    else:
        title = "acquisition timestamp (GMT)"  # no ambiguity here

    if acquisitiondate is None:
        _y = Coord([0], title="spectrum", units=None, labels=([None], [filename]))
    else:
        _y = Coord(
            [timestamp],
            title=title,
            units="s",
            labels=([acquisitiondate], [filename]),
        )

    # useful when a part of the spectrum/ifg has been blanked:
    dataset.mask = np.isnan(dataset.data)

    if return_ifg is None:
        default_description = f"# Omnic name: {spa_name}\n# Filename: {filename.name}"
        dataset.units = info["units"]
        dataset.title = info["title"]

        # now add coordinates
        nx = info["nx"]
        native_last_x = info["native_last_x"]
        native_first_x = info["native_first_x"]
        xunit = info["xunits"]
        xtitle = info["xtitle"]

        _x = Coord.linspace(
            native_last_x,
            native_first_x,
            int(nx),
            title=xtitle,
            units=xunit,
        )

    else:  # interferogram
        if return_ifg == "sample":
            default_description = (
                f"# Omnic name: {spa_name} : sample IFG\n # Filename: {filename.name}"
            )
        else:
            default_description = f"# Omnic name: {spa_name} : background IFG\n # Filename: {filename.name}"
        spa_name += ": Sample IFG"
        dataset.units = "V"
        dataset.title = "detector signal"

        _x = Coord.arange(
            len(intensities),
            title="data points",
            units=None,
        )

    dataset.set_coordset(y=_y, x=_x)
    dataset.name = spa_name  # to be consistent with omnic behaviour
    dataset.filename = filename
    if return_ifg != "background" and acquisitiondate is not None:
        dataset.acquisition_date = acquisitiondate
    dataset.origin = "omnic"

    # Set origin, description, history, date
    # Omnic spg file don't have specific "origin" field stating the oirigin of the data

    dataset.description = kwargs.get("description", default_description) + "\n"
    if spa_comments:
        dataset.description += "# Comments from Omnic:\n"
        for comment in spa_comments:
            dataset.description += comment + "\n---------------------\n"

    dataset.history = "Imported from spa file(s)"

    if "spa_history" in locals() and len(spa_history.strip()) > 0:
        dataset.history = (
            "Data processing history from Omnic :\n------------------------------------\n"
            + spa_history
        )

    dataset._date = utcnow()

    dataset.meta.collection_length = info["collection_length"] / 100 * ur("s")
    optical_velocity = acquisition_parameters.get("optical_velocity")
    if optical_velocity is None:
        # Retain compatibility with supported files that do not carry 0x6a.
        optical_velocity = info["optical_velocity"]
    dataset.meta.optical_velocity = optical_velocity
    if info["xtitle"] == "raman shift":
        dataset.meta.laser_frequency = info["raman_excitation_frequency"] * ur("cm^-1")
        dataset.meta.omnic_reference_frequency = info["reference_frequency"] * ur(
            "cm^-1"
        )
    else:
        dataset.meta.laser_frequency = info["reference_frequency"] * ur("cm^-1")
    dataset.meta.sample_spacing = info["sample_spacing"]

    if not is_library_variant:
        dataset.meta.omnic_scan_points = int(info["scan_points"])
        dataset.meta.omnic_interferogram_peak_position = int(info["peak_position"])
        dataset.meta.omnic_sample_scans = int(info["sample_scans"])
        dataset.meta.omnic_background_scans = int(info["background_scans"])
        dataset.meta.omnic_fft_points = int(info["fft_points"])
        dataset.meta.omnic_background_gain = float(info["background_gain"])
        dataset.meta.omnic_aperture = float(info["aperture"])
        for name in ("digitizer_bits", "sample_gain"):
            if name in acquisition_parameters:
                setattr(
                    dataset.meta, f"omnic_{name}", int(acquisition_parameters[name])
                )
        for name in ("high_pass", "low_pass"):
            if name in acquisition_parameters:
                setattr(
                    dataset.meta,
                    f"omnic_{name}",
                    acquisition_parameters[name] * ur("cm^-1"),
                )

    if _exp_info is not None:
        for meta_key, val in _exp_info.items():
            setattr(dataset.meta, f"omnic_{meta_key}", val)
        if not dataset.description.strip() and _exp_info.get("experiment_title"):
            dataset.description = _exp_info["experiment_title"]

    if dataset.x.units is None and dataset.x.title == "data points":
        # interferogram: build the OPD axis from the reference frequency and
        # the native sample spacing (spacing = sample_spacing / (2 * nu)).
        dataset.meta.interferogram = True
        dataset.meta.td = list(dataset.shape)
        # This is the data-derived peak index, not formal physical ZPD.
        dataset.x._zpd = int(np.argmax(dataset)[-1])
        dataset.x.set_laser_frequency(
            frequency=info["reference_frequency"], sample_spacing=info["sample_spacing"]
        )
        dataset.x._use_time_axis = (
            False  # True to have time, else it will be optical path difference
        )

    return dataset


@_importer_method
def _read_srs(*args, **kwargs):
    dataset, filename = args
    frombytes = kwargs.get("frombytes", False)

    return_bg = kwargs.get("return_bg", False)
    if "reverse_x" in kwargs:
        warn_deprecated(
            "reverse_x",
            subject="The `reverse_x` keyword argument of `read_srs`",
            kind="keyword argument",
            action="is deprecated",
            replace=None,
            policy=True,
            extra_msg=(
                "SRS spectral orientation is now handled automatically "
                "(descending wavenumber, data matched to it). This option is a "
                "no-op and will be removed according to SpectroChemPy policy."
            ),
            stacklevel=4,
        )

    # in this case, filename is actually a byte content
    fid = io.BytesIO(filename) if frombytes else open(filename, "rb")  # noqa: SIM115

    # read the file and determine whether it is a rapidscan or a high speed real time
    is_rapidscan, is_highspeed, is_tg = False, False, False

    """ At pos=304 (hex:130) is the position of the '02' key for series. Here we don't use it.
    Instead, we use one of the following sequence :

    RapidScan series:
    ----------------
    the following sequence appears 3 times in the file
    b'\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\x50\x43\x47

    They are used to assert the srs file is rapid_scan and to locate headers and data:
    - The 1st one is located 152 bytes after the series header position
    - The 2nd one is located 152 bytes before the background header position and
       56 bytes before either the background data / or the background title and infos
       followed by the background data
    - The 3rd one is located 60 bytes before the series data (spectre/ifg names and
    intensities


    High Speed Real time series:
    ---------------------------
    the following sequence appears 4 times in the file:
    b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\xc8\xaf\x47"

    They are used to assert the srs file is high speed and to locate headers and data:
    - The 1st one is located 152 bytes after the series header position
    - The 2nd one is located 152 bytes before the background header position and
       56 bytes before either the background data / or the background title and infos
       followed by the background data
    - The 3rd one is located 60 bytes before some data (don't know yet what it is)
    - The 4th one is located 60 bytes before the series data (spectra)

    TGA/IR or GC series:
    ---------------------------
    the following sequence appears 3 times in the file:
    b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00", the next bytes can differ from
    one file to another.

    As it is common to other types of TG/IR they will be used to assert if the srs file
    is TGA/IR or GC  *after the other formats*. They also allows locating headers and
    data:
    - The 1st one is located 152 bytes after the series header position
    - The 2nd one is located 152 bytes before the background header position and
       56 bytes before either the background data / or the background title and infos
       followed by the background data
    - The 3rd one is located 60 bytes before the series data (spectre/ifg names and
    intensities ?
    """

    sub_rs = b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\x50\x43\x47"
    sub_hs = b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\xc8\xaf\x47"
    sub_tg = b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00"

    # find the first occurrence and determine whether the srs is rapidscan or high
    # speed real time
    fid.seek(0)
    bytestring = fid.read()

    # try rapidscan first:
    pos = bytestring.find(sub_rs, 1)
    if pos > 0:
        is_rapidscan = True
    else:
        # not rapidscan, try high speed real time
        pos = bytestring.find(sub_hs, 1)
        if pos > 0:
            is_highspeed = True
        else:
            # neith rapid scan nor high speed real time, try TGA/IR
            pos = bytestring.find(sub_tg, 1)
            if pos > 0:
                is_tg = True

            else:
                raise NotImplementedError(
                    "The reader is only implemented for Rapid Scan, "
                    "High Speed Real Time, GC or TGA srs files. If you think "
                    "your file belongs to one of these types, or if "
                    "you'd like an update of the reader to read your "
                    "file type, please report the issue on "
                    "https://github.com/spectrochempy/spectrochempy"
                    "/issues ",
                )

    if is_rapidscan:
        # determine whether the srs is reprocessed. At pos=292 (hex:124) appears a
        # difference between pristine and reprocessed series
        fid.seek(292)
        key = fromfile(fid, dtype="uint8", count=16)[0]
        if key == 39:  # (hex: 27)
            is_reprocessed = False
        elif key == 15:  # (hex = 0F)
            is_reprocessed = True
        else:
            raise NotImplementedError(
                "The file is not recognized as a Rapid Scan "
                "srs file. Please report the issue on "
                "https://github.com/spectrochempy/spectrochempy"
                "/issues ",
            )

        # find the 2 following starting indexes of sub_rs.
        # we will use the 1st (-> series info), the 2nd (-> background) and
        # the 3rd  (-> data)

        fid.seek(0)
        bytestring = fid.read()
        index = [pos]
        while pos != -1:
            pos = bytestring.find(sub_rs, pos + 1)
            index.append(pos)

        index = np.array(index[:-1]) + [-152, -152, 60]

        if len(index) != 3:
            raise NotImplementedError(
                "The file is not recognized as a Rapid Scan "
                "srs file. Please report the issue on "
                "https://github.com/spectrochempy/spectrochempy"
                "/issues ",
            )

        pos_info_data = index[0]
        pos_info_bg = index[1]
        pos_data = index[2]

        # read series data, except if the user asks for the background
        if not return_bg:
            info = _read_header(fid, pos_info_data)
            names, data = _read_srs_spectra(fid, pos_data, info["ny"], info["nx"])

            # now get series history
            if not is_reprocessed:
                history = info["history"]
            else:
                # In reprocessed series the updated "DATA PROCESSING HISTORY" is located right after
                # the following 16 byte sequence:
                sub = (
                    b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
                )
                pos = bytestring.find(sub) + 16
                history = _readbtext(fid, pos, None)

        # read the background if the user asked for it.
        if return_bg:
            # First get background info
            info = _read_header(fid, pos_info_bg)

            if "background_name" not in info:
                # it is a short header
                fid.seek(index[1] + 208)
                data = fromfile(fid, dtype="float32", count=info["nx"])
            else:
                # longer header, in such case the header indicates a spectrum
                # but the data are those of an ifg... For now need more examples
                return None

            # uncomment below to load the last datafield has the same dimension as the time axis
            # its function is not known. related to Grams-schmidt ?

            # pos = _nextline(pos)
            # found = False
            # while not found:
            #     pos += 16
            #     f.seek(pos)
            #     key = fromfile(f, dtype='uint8', count=1)
            #     if key == 1:
            #         pos += 4
            #         f.seek(pos)
            #         X = fromfile(f, dtype='float32', count=info['ny'])
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

    if is_highspeed:
        # find the 3 following starting indexes of sub.
        # 1st -> series info),
        # 2nd -> background ?
        # 3rd -> data ?
        # 4th  -> ?
        fid.seek(0)
        bytestring = fid.read()
        index = [pos]
        while pos != -1:
            pos = bytestring.find(sub_hs, pos + 1)
            index.append(pos)

        index = np.array(index[:-1]) + [-152, -152, 0, 60]

        pos_info_data = index[0]
        pos_bg = index[1]
        pos_x = index[2]
        pos_data = index[3]

        if len(index) != 4:
            raise NotImplementedError(
                "The file is not recognized as a High Speed Real "
                "Time srs file. Please report the issue on "
                "https://github.com/spectrochempy/spectrochempy"
                "/issues ",
            )

        if not return_bg:
            info = _read_header(fid, pos_info_data)
            # container for names and data

            names, data = _read_srs_spectra(fid, pos_data, info["ny"], info["nx"])

            # Get series history. on the sample file, the history seems overwritten by
            # some post-processing, so info["history"] returns a corrupted string.
            # The "DATA PROCESSING HISTORY" (as indicated by omnic) is located right
            # after the following 16 byte sequence:
            sub = b"\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff"
            pos = bytestring.find(sub) + 16
            history = _readbtext(fid, pos, None)

            # read the background if the user asked for it.

        elif return_bg:
            # First get background info
            info = _read_header(fid, pos_bg)

            if "background_name" not in info:
                # it is a short header
                fid.seek(index[1] + 208)
                data = fromfile(fid, dtype="float32", count=info["nx"])
            else:
                # longer header, in such case the header indicates a spectrum
                # but the data are those of an ifg... For now need more examples
                return None

    if is_tg:
        fid.seek(0)
        bytestring = fid.read()
        index = [pos]
        while pos != -1:
            pos = bytestring.find(sub_tg, pos + 1)
            index.append(pos)

        index = np.array(index[:-1]) + [-152, -152, 60]

        if len(index) != 3:
            raise NotImplementedError(
                "The file is not recognized as a TG IR or GC "
                "srs file. Please report the issue on "
                "https://github.com/spectrochempy/spectrochempy"
                "/issues ",
            )

        pos_info_data = index[0]
        pos_info_bg = index[1]
        pos_data = index[2]

        # read series data, except if the user asks for the background
        if not return_bg:
            info = _read_header(fid, pos_info_data)
            names, data = _read_srs_spectra(fid, pos_data, info["ny"], info["nx"])
            # Note: info["history"] is empty in TG IR or GC series
            # the position of the history is indicated at pos 856 or 878 depending on the
            # file.

        # read the background if the user asked for it.
        if return_bg:
            # First get background info
            info = _read_header(fid, pos_info_bg)

            if "background_name" not in info:
                # it is a short header
                fid.seek(index[1] + 208)
                data = fromfile(fid, dtype="float32", count=info["nx"])
            else:
                # longer header, in such case the header indicates a spectrum
                # but the data are those of an ifg... For now need more examples
                return None

    # Create NDDataset object for the series / background.
    #
    # The raw SRS spectral intensity array is stored ascending-wavenumber, but
    # the public SpectroChemPy / `read_spa` convention presents OMNIC spectra
    # with descending wavenumber and data matched to it. Spectral records are
    # normalized here using each record's own firstx/lastx endpoints (which may
    # differ between the series header and the background header). Interferogram
    # records keep the raw ascending data-points coordinate and are never
    # reversed. `_read_header` returns raw firstx/lastx without reordering them.
    #
    # The X axis is classified into one of three cases:
    #
    #   * spectral record: known physical spectral coordinate (xunit codes
    #     1/3/4/32); normalize to the public descending-wavenumber convention.
    #   * interferogram: explicit data-points axis (xunit code 2); keep the raw
    #     ascending coordinate and leave data/order unchanged.
    #   * unknown X-axis type: `xunits` is None but the axis is not a
    #     data-points axis (the `_read_header` fallback for an unrecognized
    #     x-unit code). Never silently classify this as an interferogram, and
    #     do not apply spectral normalization to an axis whose meaning is
    #     unknown. Leave the record in its raw storage orientation and warn.
    if info["xunits"] is not None:
        # spectral record
        data_out = data[::-1] if return_bg else data[:, ::-1]
        x0, x1 = max(info["firstx"], info["lastx"]), min(info["firstx"], info["lastx"])
    elif info["xtitle"] == "data points":
        # interferogram
        data_out = data
        x0, x1 = info["firstx"], info["lastx"]
    else:
        # unknown X-axis type
        warning_(
            "The nature of the SRS X axis is not recognized: "
            "xunits is None and xtitle is "
            f"{info['xtitle']!r}. The record is left in its raw storage "
            "orientation and is not treated as an interferogram."
        )
        data_out = data
        x0, x1 = info["firstx"], info["lastx"]

    if return_bg:
        dataset = NDDataset(np.expand_dims(data_out, axis=0))
    else:
        dataset = NDDataset(data_out)

    # in case part of the spectra/ifg has been blanked:
    dataset.mask = np.isnan(dataset.data)

    dataset.units = info["units"]
    dataset.title = info["title"]
    dataset.origin = "omnic"
    dataset.filename = filename

    # now add coordinates

    _x = Coord.linspace(
        x0,
        x1,
        int(info["nx"]),
        title=info["xtitle"],
        units=info["xunits"],
    )

    # specific infos for series data
    if not return_bg:
        dataset.name = info["name"]
        _y = Coord(
            np.around(np.linspace(info["time_min"], info["lasty"], info["ny"]), 3),
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
            "--------------------------------\n" + history,
        )
    dataset.history.append(str(utcnow()) + ": imported from srs file " + str(filename))

    dataset.meta.laser_frequency = info["reference_frequency"] * ur("cm^-1")
    dataset.meta.collection_length = info["collection_length"] * ur("s")
    dataset.meta.optical_velocity = info["optical_velocity"]

    if dataset.x.units is None and dataset.x.title == "data points":
        # interferogram
        dataset.meta.interferogram = True
        dataset.meta.td = list(dataset.shape)
        dataset.x._zpd = int(np.argmax(dataset)[-1])  # zero path difference
        dataset.x.set_laser_frequency()
        dataset.x._use_time_axis = (
            False  # True to have time, else it will  be optical path difference
        )

    fid.close()

    return dataset


def _readbtext(fid, pos, size):
    # Read some text in binary file of given size. If size is None, the text is read
    # until a null byte (b"\x00") is encountered.
    # Returns utf-8 string
    fid.seek(pos)
    if size is None:
        btext = b""
        while fid.read(1) != b"\x00":
            btext += fid.read(1)
    else:
        btext = fid.read(size)
    btext = re.sub(b"\x00+", b"\n", btext)

    if btext[:1] == b"\n":
        btext = btext[1:]

    if btext[-1:] == b"\n":
        btext = btext[:-1]

    try:
        text = btext.decode(encoding="utf-8")  # decode btext to string
    except UnicodeDecodeError:
        try:
            text = btext.decode(encoding="latin_1")
        except UnicodeDecodeError:  # pragma: no cover
            text = btext.decode(encoding="utf-8", errors="ignore")
    return text


def _nextline(pos):
    # reset current position to the beginning of next line (16 bytes length)
    return 16 * (1 + pos // 16)


def _read_header(fid, pos, is_first_spectrum=True):
    r"""
    Read spectrum/ifg/series header.

    Parameters
    ----------
    fid : BufferedReader
        The buffered binary stream.

    pos : int
        The position of the header (see Notes).

    is_first_spectrum : bool, optional
            Indicates if this is the first spectrum being read. Default is True.

    Returns
    -------
        dict, int
        Dictionary and current position in file

    Notes
    -----
        So far, the header structure is as follows (offsets are relative to the
        header base position; the positioning of the srs series header and the
        full field map are documented in the public format reference
        :doc:`/devguide/file_formats/omnic/srs`):

        - starts with b'\x01' , b'\x02', b'\x03' ... maybe indicating the header "type"
        - nx (UInt32): 4 bytes behind
        - xunits (UInt8): 8 bytes behind. So far, we have the following
          correspondence (the byte value of the unit code):

            * \x01 : wavenumbers, cm-1
            * \x02 : datapoints (interferogram)
            * \x03 : wavelength, nm
            * \x04 : wavelength, um
            * \x20 : Raman shift, cm-1

        - data units (UInt8): 12 bytes behind. So far, we have the following
          correspondence:

            * \x0B : reflectance (%)
            * \x0C : Kubelka_Munk
            * \x0F : single beam
            * \x11 : absorbance
            * \x10 : transmittance (%)
            * \x16 : Volts (interferogram)
            * \x1A : photoacoustic
            * \x1F : Raman intensity

        - first x value (float32), 16 bytes behind
        - last x value (float32), 20 bytes behind
        - ... unknown
        - scan points (UInt32), 28 bytes behind
        - interferogram peak position (UInt32), 32 bytes behind
        - number of scans (UInt32), 36 bytes behind
        - ... unknown
        - number of background scans (UInt32), 52 bytes behind
        - ... unknown
        - general-header collection length in 1/100th of sec (UInt32), 68
          bytes behind. This is the shared acquisition-time field; for srs
          series the public "collection length" is derived from the series
          minimum time at +1002 (see below), not from this field.
        - ... unknown
        - reference frequency (float32), 80 bytes behind
        - ...
        - optical velocity (float32), 188 bytes behind
        - ...
        - spectrum history (text), 208 bytes behind

        For "rapid-scan" srs files:

        - series name (text), 938 bytes behind
        - series minimum / first time in minutes (float32), 1002 bytes
          behind. Historically described as "collection length": the public
          `collection_length` is this value converted to seconds (x60), while
          `time_min` keeps it in minutes and is used as the time-axis anchor.
          Do not conflate it with the general-header collection length at +68.
        - series maximum / last time in minutes (float32), 1006 bytes behind
        - regular time step in minutes (float32), 1010 bytes behind. This
          field was historically misnamed "first y": it is the series step,
          not the minimum (the minimum is the +1002 field above).
        - ny (UInt32), 1026
        - ... y unit could be at pos+1030 with 01 = minutes ?
        - history (text), 1200 bytes behind (only initial history.
           When reprocessed, updated history is at the end of the file after the
           b'\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff' sequence

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
    out["nx"] = fromfile(fid, "uint32", count=1)

    # xunits
    fid.seek(pos + 8)
    key = fromfile(fid, dtype="uint8", count=1)
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
    key = fromfile(fid, dtype="uint8", count=1)
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
    elif key == 15:  # pragma: no cover
        out["units"] = None
        out["title"] = "single beam"
    elif key == 20:  # pragma: no cover
        out["units"] = "Kubelka_Munk"
        out["title"] = "Kubelka-Munk"
    elif key == 21:
        out["units"] = None
        out["title"] = "reflectance"
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
        if is_first_spectrum:
            info_(
                f"The nature of data is not recognized (key == {key}), title set to 'Intensity'"
            )

    # Native endpoint names follow OMNIC terminology: +16 is Last X and +20
    # is First X. Historical aliases are retained for the shared readers.
    fid.seek(pos + 16)
    out["native_last_x"] = fromfile(fid, "float32", 1)
    fid.seek(pos + 20)
    out["native_first_x"] = fromfile(fid, "float32", 1)
    out["firstx"] = out["native_last_x"]
    out["lastx"] = out["native_first_x"]
    fid.seek(pos + 28)

    out["scan_points"] = fromfile(fid, "uint32", 1)
    out["scan_pts"] = out["scan_points"]
    fid.seek(pos + 32)
    out["peak_position"] = fromfile(fid, "uint32", 1)
    out["zpd"] = out["peak_position"]
    fid.seek(pos + 36)
    out["sample_scans"] = fromfile(fid, "uint32", 1)
    out["nscan"] = out["sample_scans"]
    fid.seek(pos + 44)
    out["fft_points"] = fromfile(fid, "uint32", 1)
    fid.seek(pos + 48)
    out["trailing_geometry"] = fromfile(fid, "uint32", 1)
    fid.seek(pos + 52)
    out["background_scans"] = fromfile(fid, "uint32", 1)
    out["nbkgscan"] = out["background_scans"]
    fid.seek(pos + 56)
    out["background_gain"] = fromfile(fid, "float32", 1)
    fid.seek(pos + 68)
    out["collection_length"] = fromfile(fid, "uint32", 1)
    fid.seek(pos + 80)
    out["reference_frequency"] = fromfile(fid, "float32", 1)
    fid.seek(pos + 84)
    out["sample_spacing"] = fromfile(fid, "float32", 1)
    fid.seek(pos + 92)
    out["aperture"] = fromfile(fid, "float32", 1)
    fid.seek(pos + 96)
    out["raman_excitation_frequency"] = fromfile(fid, "float32", 1)
    fid.seek(pos + 188)
    out["optical_velocity"] = fromfile(fid, "float32", 1)

    if filetype == "spa, spg":
        out["history"] = _readbtext(fid, pos + 208, None)

    if filetype == "srs":
        out["name"] = _readbtext(fid, pos + 938, 256)
        # Hack because name seems not to be well read for srs
        out["name"] = out["name"].split("\n")[0]
        fid.seek(pos + 1002)
        # The stored float32 at +1002 is the OMNIC series *minimum / first time*
        # (in minutes). `collection_length` keeps the historical public meaning
        # (that value converted to seconds); `time_min` is the same field kept in
        # minutes, used as the time-axis start. Do not conflate the two.
        collection_length = fromfile(fid, "float32", 1)
        out["collection_length"] = collection_length * 60
        out["time_min"] = collection_length
        fid.seek(pos + 1006)
        out["lasty"] = fromfile(fid, "float32", 1)
        fid.seek(pos + 1010)
        out["firsty"] = fromfile(fid, "float32", 1)
        fid.seek(pos + 1026)
        out["ny"] = fromfile(fid, "uint32", 1)
        #  y unit could be at pos+1030 with 01 = minutes ?
        out["history"] = _readbtext(fid, pos + 1200, None)

        if _readbtext(fid, pos + 208, 256)[:10] == "Background":
            # it is the header of a background
            out["background_name"] = _readbtext(fid, pos + 208, 256)[10:]

    return out


def _read_srs_spectra_name(fid, pos):
    """
    Read the 84-byte per-spectrum SRS name record and return its label.

    The SRS spectrum record is exactly 84 bytes: a null-terminated human-readable
    name followed by binary metadata and, after the record, the spectral bytes.
    Reading beyond the record (as the historical 256-byte read did) leaks binary
    metadata and spectrum data into the label, so this SRS-specific helper stops
    at the first null and never inspects more than the 84-byte record.

    `_readbtext` is intentionally left untouched to avoid regressions in the
    shared SPA/SPG readers.
    """
    fid.seek(pos)
    record = fid.read(84)
    name = record.split(b"\x00", 1)[0]
    try:
        return name.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return name.decode("latin_1")
        except UnicodeDecodeError:  # pragma: no cover
            return name.decode("utf-8", errors="ignore")


def _read_srs_spectra(fid, pos_data, n_spectra, n_points):
    """
    Read the spectra/interferogram names and data of a series.

    fid: BufferedReader
    pos_data: int
    n_spectra: int
    n_points: int

    returns: names (list), spectral data (ndarray)
    """
    # container for names and data
    names = []
    data = np.zeros((n_spectra, n_points))

    # read the spectra/interferogram names and data
    # the first one....
    pos = pos_data
    names.append(_read_srs_spectra_name(fid, pos))
    pos += 84
    fid.seek(pos)
    data[0, :] = fromfile(fid, dtype="float32", count=n_points)[:]
    pos += n_points * 4
    # ... and the remaining ones:
    for i in np.arange(n_spectra)[1:]:
        pos += 16
        names.append(_read_srs_spectra_name(fid, pos))
        pos += 84
        fid.seek(pos)
        data[i, :] = fromfile(fid, dtype="float32", count=n_points)[:]
        pos += n_points * 4

    return names, data


@dataclass(frozen=True)
class _SpaKeyRecord:
    """One counted 16-byte record from an SPA key table."""

    key: int
    position: int
    length: int


def _read_spa_key_table(fid):
    """
    Read the counted active records from an SPA key table.

    The active table starts at file offset 304 and contains ``nlines`` records,
    where ``nlines`` is the little-endian uint16 at offset 294. Any
    variant-dependent terminator, padding, or post-table grid is deliberately
    outside this representation; see the public SPA format reference.
    """
    fid.seek(294)
    count_bytes = fid.read(2)
    if len(count_bytes) != 2:
        raise ValueError("Invalid SPA key table: missing record count")
    nlines = int.from_bytes(count_bytes, byteorder="little")
    table_end = 304 + 16 * nlines

    fid.seek(0, io.SEEK_END)
    if table_end > fid.tell():
        raise ValueError("Invalid SPA key table: truncated record table")

    fid.seek(304)
    records = []
    for _ in range(nlines):
        raw_record = fid.read(16)
        if len(raw_record) != 16:  # pragma: no cover - table_end guards this
            raise ValueError("Invalid SPA key table: truncated record")
        records.append(
            _SpaKeyRecord(
                key=raw_record[0],
                position=int.from_bytes(raw_record[2:6], byteorder="little"),
                length=int.from_bytes(raw_record[6:10], byteorder="little"),
            )
        )
    return records


def _read_spa_float32_payload(fid, record):
    """Read a float32 payload referenced by an SPA key record."""
    fid.seek(record.position)
    return fromfile(fid, "float32", int(record.length / 4))


def _read_spa_acquisition_parameters(fid, record):
    """Read mature acquisition fields from the canonical 0x6a block."""
    if record.length < 20:
        return {}
    fid.seek(record.position)
    data = fid.read(record.length)
    parameters = {}

    if len(data) >= 20:
        parameters["digitizer_bits"] = struct.unpack_from("<I", data, 16)[0]
    if len(data) >= 28:
        parameters["high_pass"] = struct.unpack_from("<f", data, 20)[0]
        parameters["low_pass"] = struct.unpack_from("<f", data, 24)[0]
    if len(data) >= 48:
        parameters["sample_gain"] = struct.unpack_from("<f", data, 44)[0]
    if len(data) >= 52:
        parameters["optical_velocity"] = struct.unpack_from("<f", data, 48)[0]
    return parameters


def _getintensities(fid, pos):
    # get intensities from the 03 (spectrum)
    # or 66 (sample ifg) or 67 (bg ifg) key,
    # returns a ndarray

    fid.seek(pos + 2)  # skip 2 bytes
    intensity_pos = fromfile(fid, "uint32", 1)
    fid.seek(pos + 6)
    intensity_size = fromfile(fid, "uint32", 1)
    nintensities = int(intensity_size / 4)

    # Read and return spectral intensities
    fid.seek(intensity_pos)
    return fromfile(fid, "float32", int(nintensities))


def _decode_experiment_info_block(data: bytes) -> dict | None:
    """
    Decode an OMNIC Experiment Information block (key 0x82, subtype 0x79).

    Parameters
    ----------
    data : bytes
        Raw block data read from the file.

    Returns
    -------
    dict or None
        Dictionary with fixed-slot subtype-0x79 fields, or None if the block
        is not a supported subtype.
    """
    if not data or data[0] != 0x79:
        return None

    def _read_slot(start, end):
        if start >= len(data):
            return None
        slot = data[start : min(end, len(data))]
        value = slot.split(b"\x00", 1)[0]
        if not value:
            return None
        return value.decode("utf-8", errors="replace").strip()

    result: dict[str, str] = {}
    for name, start, end in (
        ("experiment_path", 10, 90),
        ("experiment_title", 90, 154),
        ("experiment_description", 154, 413),
        ("accessory_name", 413, 670),
    ):
        value = _read_slot(start, end)
        if value:
            result[name] = value

    return result or None
