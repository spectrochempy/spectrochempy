# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module extend NDDataset with the import method for Thermo galactic (spc) data files.
"""
__all__ = ["read_soc", "read_ddr", "read_sdr", "read_hdr"]
__dataset_methods__ = __all__


from spectrochempy.core.readers.importer import Importer, _importer_method
from spectrochempy.core.readers.read_omnic import _read_spa

# ======================================================================================================================
# Public function
# ======================================================================================================================


def read_soc(*paths, **kwargs):
    """
    Open a Surface Optics Corps. file or a list of files with extension ``.ddr``, ``.hdr`` or ``.sdr``.

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
    read_soc
        The dataset or a list of dataset corresponding to a (set of) file(s).

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
    read_spa : Read Omnic *.spa spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Examples
    ---------

    """

    kwargs["filetypes"] = ["Surface Optics Corp. (*.ddr *.hdr *.sdr)"]
    kwargs["protocol"] = ["soc", "ddr", "hdr", "sdr"]
    importer = Importer()
    return importer(*paths, **kwargs)


def read_ddr(*paths, **kwargs):
    """
    Open a Surface Optics Corps. file or a list of files with extension ``.ddr``.

    Open Surface Optics Corps. file or a list of files with extension ``.ddr`` and set
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
    read_ddr
        The dataset or a list of dataset corresponding to the (set of)
        file(s).

    Other Parameters
    -----------------
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
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

    Examples
    ---------

    """

    kwargs["filetypes"] = ["Surface Optics Corp. (*.ddr)"]
    kwargs["protocol"] = ["ddr"]
    importer = Importer()
    return importer(*paths, **kwargs)


def read_hdr(*paths, **kwargs):
    """
    Open a Surface Optics Corps. file or a list of files with extension ``.hdr``.

    Open Surface Optics Corps. file or a list of files with extension ``.hdr`` and set
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
    read_ddr
        The dataset or a list of dataset corresponding to the (set of)
        file(s).

    Other Parameters
    -----------------
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
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

    Examples
    ---------

    """

    kwargs["filetypes"] = ["Surface Optics Corp. (*.hdr)"]
    kwargs["protocol"] = ["hdr"]
    importer = Importer()
    return importer(*paths, **kwargs)


def read_sdr(*paths, **kwargs):
    """
    Open a Surface Optics Corps. file or a list of files with extension ``.sdr``.

    Open Surface Optics Corps. file or a list of files with extension ``.sdr`` and set
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
    read_ddr
        The dataset or a list of dataset corresponding to the (set of)
        file(s).

    Other Parameters
    -----------------
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
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

    Examples
    ---------

    """

    kwargs["filetypes"] = ["Surface Optics Corp. (*.sdr)"]
    kwargs["protocol"] = ["sdr"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================


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


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
