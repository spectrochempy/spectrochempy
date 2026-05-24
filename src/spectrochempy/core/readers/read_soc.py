# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Extend NDDataset with the import method for Surface Optics Corp. (soc) data files."""

__all__ = ["read_soc", "read_ddr", "read_sdr", "read_hdr"]


from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.read_omnic import _read_spa


# ======================================================================================
# Public functions
# ======================================================================================
def read_soc(*paths, **kwargs):
    r"""
    Read a Surface Optics Corp. file or a list of files with extension :file:`.ddr`, :file:`.hdr` or :file:`.sdr`.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ..., kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to `False`.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    `NDDataset` or list of `NDDataset`
        The returned dataset(s).

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
        A Custom description.
    directory : `~pathlib.Path` object or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `False` (default), individual datasets are preserved and returned as a list.
        If `True` and several filenames have been provided with compatible dimensions,
        they are merged into a single `NDDataset` (stacked along the first dimension).
        When datasets have incompatible dimensions or origins, they may be grouped into
        multiple merged datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading. It can be one of {``'scp'``, ``'omnic'``,
        ``'opus'``, ````, ``'matlab'``, ``'jcamp'``, ``'csv'``,
        ``'excel'``}. If not provided, the correct protocol
        is inferred (whenever it is possible) from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
        the background interferogram of the spa file if present or None if absent.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic read function.

    """
    kwargs["filetypes"] = ["Surface Optics Corp. (*.ddr *.hdr *.sdr)"]
    kwargs["protocol"] = ["soc", "ddr", "hdr", "sdr"]
    # For SOC files with multiple subfiles, don't merge them by default
    # Each subfile should be a separate dataset
    if "merge" not in kwargs:
        kwargs["merge"] = False
    importer = Importer()
    return importer(*paths, **kwargs)


def read_ddr(*paths, **kwargs):
    r"""
    Open a Surface Optics Corp. file or a list of files with extension :file:`.ddr`.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ..., kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to `False`.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    `NDDataset` or list of `NDDataset`
        The returned dataset(s).

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
        A Custom description.
    directory : `~pathlib.Path` object or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `False` (default), individual datasets are preserved and returned as a list.
        If `True` and several filenames have been provided with compatible dimensions,
        they are merged into a single `NDDataset` (stacked along the first dimension).
        When datasets have incompatible dimensions or origins, they may be grouped into
        multiple merged datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading. It can be one of {``'scp'``, ``'omnic'``,
        ``'opus'``, ````, ``'matlab'``, ``'jcamp'``, ``'csv'``,
        ``'excel'``}. If not provided, the correct protocol
        is inferred (whenever it is possible) from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
        the background interferogram of the spa file if present or None if absent.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic read function.
    read_soc : Read Surface Optics Corp. files.

    """
    kwargs["filetypes"] = ["Surface Optics Corp. (*.ddr)"]
    kwargs["protocol"] = ["ddr"]
    # Don't merge files by default - preserve individual datasets
    if "merge" not in kwargs:
        kwargs["merge"] = False
    importer = Importer()
    return importer(*paths, **kwargs)


def read_hdr(*paths, **kwargs):
    r"""
    Open a Surface Optics Corp. file or a list of files with extension :file:`.hdr`.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ..., kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to `False`.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    `NDDataset` or list of `NDDataset`
        The returned dataset(s).

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
        A Custom description.
    directory : `~pathlib.Path` object or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `False` (default), individual datasets are preserved and returned as a list.
        If `True` and several filenames have been provided with compatible dimensions,
        they are merged into a single `NDDataset` (stacked along the first dimension).
        When datasets have incompatible dimensions or origins, they may be grouped into
        multiple merged datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading. It can be one of {``'scp'``, ``'omnic'``,
        ``'opus'``, ````, ``'matlab'``, ``'jcamp'``, ``'csv'``,
        ``'excel'``}. If not provided, the correct protocol
        is inferred (whenever it is possible) from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
        the background interferogram of the spa file if present or None if absent.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic read function.
    read_soc : Read Surface Optics Corp. files.

    """
    kwargs["filetypes"] = ["Surface Optics Corp. (*.hdr)"]
    kwargs["protocol"] = ["hdr"]
    # Don't merge files by default - preserve individual datasets
    if "merge" not in kwargs:
        kwargs["merge"] = False
    importer = Importer()
    return importer(*paths, **kwargs)


def read_sdr(*paths, **kwargs):
    r"""
    Open a Surface Optics Corp. file or a list of files with extension :file:`.sdr`.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ..., kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to `False`.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    `NDDataset` or list of `NDDataset`
        The returned dataset(s).

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
        A Custom description.
    directory : `~pathlib.Path` object or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `False` (default), individual datasets are preserved and returned as a list.
        If `True` and several filenames have been provided with compatible dimensions,
        they are merged into a single `NDDataset` (stacked along the first dimension).
        When datasets have incompatible dimensions or origins, they may be grouped into
        multiple merged datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading. It can be one of {``'scp'``, ``'omnic'``,
        ``'opus'``, ````, ``'matlab'``, ``'jcamp'``, ``'csv'``,
        ``'excel'``}. If not provided, the correct protocol
        is inferred (whenever it is possible) from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
        the background interferogram of the spa file if present or None if absent.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic read function.
    read_soc : Read Surface Optics Corp. files.

    """
    kwargs["filetypes"] = ["Surface Optics Corp. (*.sdr)"]
    kwargs["protocol"] = ["sdr"]
    # Don't merge files by default - preserve individual datasets
    if "merge" not in kwargs:
        kwargs["merge"] = False
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_ddr(*args, **kwargs):
    ds = _read_spa(*args, **kwargs)
    ds.history[-1] = "Imported from ddr file(s)"
    return ds


@_importer_method
def _read_hdr(*args, **kwargs):
    ds = _read_spa(*args, **kwargs)
    ds.history[-1] = "Imported from hdr file(s)"
    return ds


@_importer_method
def _read_sdr(*args, **kwargs):
    ds = _read_spa(*args, **kwargs)
    ds.history[-1] = "Imported from sdr file(s)"
    return ds
