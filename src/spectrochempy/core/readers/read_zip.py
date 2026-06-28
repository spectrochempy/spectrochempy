# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["read_zip"]

import zipfile

from spectrochempy.core.readers.filetypes import registry
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid
from spectrochempy.core.readers.importer import read
from spectrochempy.utils._logging import warning_


# ======================================================================================
# Public functions
# ======================================================================================
def read_zip(*paths, **kwargs):
    r"""
    Read Zip archives (containing spectrochempy readable files).

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
        result is a list-like `ScpObjectList`.

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
    only : `int`, optional
        Limit ZIP import to the first readable files discovered in the archive.
        This is mainly useful for sampling large archives or for tests. The
        limit is applied according to archive discovery order after filtering
        ignored entries such as ``__MACOSX`` and ``.DS_Store``.
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
    :func:`spectrochempy.read_dir` : Read an entire directory.
    :func:`spectrochempy.read_opus` : Read OPUS spectra.
    :func:`spectrochempy.read_labspec` : Read Raman LABSPEC spectra (:file:`.txt`).
    :func:`spectrochempy.read_omnic` : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    :func:`spectrochempy.read_soc` : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_jcamp` : Read Infrared JCAMP-DX files (:file:`.jdx`, :file:`.dx`).
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).

    Examples
    --------
    Reading a single Zip file

    >>> scp.read_zip('irdata/zip/zipfile.zip')
    NDDataset: [float64] a.u. (shape: (y:2, x:5549))

    Keeping separate datasets from an archive

    >>> datasets = scp.read_zip('irdata/zip/agirdata.zip', merge=False)
    >>> count = len(datasets)
    >>> names = datasets.names
    >>> _ = len(names)
    >>> largest = datasets.select_largest()
    >>> _ = largest.ndim

    """
    kwargs["filetypes"] = ["Zip archives (*.zip)"]
    kwargs["protocol"] = ["zip"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_zip(*args, **kwargs):
    # read zip file
    _, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    with zipfile.ZipFile(fid) as zf:
        filelist = zf.filelist
        only = kwargs.pop("only", len(filelist))

        datasets = []
        ignored_files = []
        supported_extensions = set(
            list(zip(*(registry.aliases + registry.filetypes), strict=False))[0]
        )

        def extract(children, **kwargs):
            extension = children.name.split(".")[-1]
            if extension.lower() not in supported_extensions:
                return None
            origin = kwargs.get("origin", "")
            return read(
                children.name, content=children.read_bytes(), origin=origin, merge=False
            )

        count = 0
        for zipinfo in filelist:
            if count == only:
                break
            if "__MACOSX" in zipinfo.filename or ".DS_Store" in zipinfo.filename:
                continue
            file = zipfile.Path(zf, at=zipinfo.filename)
            if file.is_dir():
                continue
            d = extract(file, **kwargs)
            if d is not None:
                datasets.append(d)
                count += 1
            else:
                ignored_files.append(zipinfo.filename)

        if ignored_files:
            ignored_list = "\n".join(f"- {name}" for name in ignored_files)
            warning_(
                f"Some files in {filename} were ignored because no reader is available:\n"
                f"{ignored_list}"
            )

        if len(datasets) == 1:
            return datasets[0]
        return datasets
