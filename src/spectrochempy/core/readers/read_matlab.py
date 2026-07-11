# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Plugin module to extend NDDataset with the import methods method."""

__all__ = ["read_matlab", "read_mat"]

import contextlib
from datetime import datetime

import numpy as np
import scipy.io as sio

from spectrochempy.core.dataset.nddataset import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid
from spectrochempy.utils._logging import info_
from spectrochempy.utils._logging import warning_


# ======================================================================================
# Public functions
# ======================================================================================
def read_matlab(*paths, **kwargs):
    r"""
    Open Matlab files.

    This is the explicit Matlab reader in the public import API. Use
    :func:`spectrochempy.read` for generic format autodetection and
    ``scp.matlab.read(...)`` or :func:`spectrochempy.read_matlab` when the
    Matlab format is already known.

    A Matlab file may contain one or several numeric variables. Compatible
    variables can be grouped into a single dataset, while non-merged multi-file
    or multi-variable reads may return a list-like `ScpObjectList` with helper
    methods for dataset selection. See :func:`spectrochempy.read` for the
    complete description of the generic import convention and multi-object
    return behavior.

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
    :func:`spectrochempy.read_soc` : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_jcamp` : Read Infrared JCAMP-DX files (:file:`.jdx`, :file:`.dx`).
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).

    Notes
    -----
    A single Matlab file may hold several variables.  Each numeric variable is
    converted to an `NDDataset` named after the Matlab variable; the importer
    then groups them by compatible shape, stacking same-shape arrays into a
    single `NDDataset` and returning incompatible ones as separate datasets.
    Non-numeric variables (e.g. character arrays) and Matlab-internal entries
    (``__header__``, ``__version__``, ``__globals__``) are skipped.  When a file
    holds a single numeric variable a lone `NDDataset` is returned.

    When several datasets are returned, the result keeps the Matlab variable
    names and can be queried directly, for example with ``.names``,
    ``.select_largest()``, ``.select_by_name()``, ``.filter_by_ndim()``, or
    ``.filter_by_shape()``.

    Examples
    --------
    Reading a single Matlab file

    >>> scp.read_matlab('irdata/matlab/matlabdata.mat')
    NDDataset: [float64] a.u. (shape: (y:1, x:3))

    Using the explicit namespace API

    >>> scp.matlab.read('irdata/matlab/matlabdata.mat')
    NDDataset: [float64] a.u. (shape: (y:1, x:3))

    Selecting datasets from a multi-variable Matlab file

    >>> datasets = scp.read_matlab('irdata/matlab/matlabdata.mat', merge=False)
    >>> names = datasets.names
    >>> len(names)
    1
    >>> largest = datasets.select_largest()
    >>> largest.ndim
    2
    >>> filtered = datasets.filter_by_ndim(2)
    >>> len(filtered)
    1
    >>> datasets.select_by_name('data').name
    'data'

    """
    kwargs["filetypes"] = ["Matlab files (*.mat *.dso)"]
    kwargs["protocol"] = ["matlab"]
    importer = Importer()
    return importer(*paths, **kwargs)


read_mat = read_matlab


# --------------------------------------------------------------------------------------
# Private methods
# --------------------------------------------------------------------------------------
@_importer_method
def _read_mat(*args, **kwargs):
    _, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    dic = sio.loadmat(fid)

    datasets = []
    for name, data in dic.items():
        dataset = NDDataset()
        if name == "__header__":
            dataset.description = str(data, "utf-8", "ignore")
            continue
        if name.startswith("__"):
            continue

        if data.dtype in [
            np.dtype("float64"),
            np.dtype("float32"),
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
            np.dtype("int64"),
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("uint64"),
        ]:
            # this is an array of numbers
            dataset.data = data
            dataset.name = name
            dataset.filename = filename
            dataset.origin = "matlab"
            dataset.history = "Imported from .mat file"
            # TODO: reshape from fortran/Matlab order to C opder
            # for 3D or higher datasets ?
            datasets.append(dataset)

        elif data.dtype.char == "U":
            # this is an array of string
            info_(
                f"The mat file contains an array of strings named '{name}' which will not be converted to NDDataset",
            )
            continue

        elif all(
            name_ in data.dtype.names for name_ in ["moddate", "axisscale", "imagesize"]
        ):
            # this is probably a DSO object
            dataset = _read_dso(dataset, name, data)
            datasets.append(dataset)

        else:
            warning_(f"unsupported data type : {data.dtype}")
            # TODO: implement DSO reader
            datasets.append([name, data])

    return datasets


@_importer_method
def _read_dso(dataset, name, data):
    name_mat = data["name"][0][0]
    name = "" if len(name_mat) == 0 else name_mat[0]

    typedata_mat = data["type"][0][0]
    typedata = "" if len(typedata_mat) == 0 else typedata_mat[0]

    if typedata != "data":
        return (name, data)

    author_mat = data["author"][0][0]
    author = "*unknown*" if len(author_mat) == 0 else author_mat[0]

    date_mat = data["date"][0][0]
    if len(date_mat) == 0:
        date = datetime(1, 1, 1, 0, 0)
    else:
        date = datetime(
            int(date_mat[0][0]),
            int(date_mat[0][1]),
            int(date_mat[0][2]),
            int(date_mat[0][3]),
            int(date_mat[0][4]),
            int(date_mat[0][5]),
        )

    dat = data["data"][0][0]

    # look at coords and labels
    # only the first label and axisscale are taken into account
    # the axisscale title is used as the coordinate title

    coords = []
    for i in range(len(dat.shape)):
        coord = datac = None  # labels = title = None
        labelsarray = data["label"][0][0][i][0]
        if len(labelsarray):  # some labels might be present
            if isinstance(labelsarray[0], np.ndarray):
                labels = data["label"][0][0][i][0][0]
            else:
                labels = data["label"][0][0][i][0]
            if len(labels):
                coord = Coord(labels=[str(label) for label in labels])
            if len(data["label"][0][0][i][1]):
                if isinstance(data["label"][0][0][i][1][0], np.ndarray):
                    if len(data["label"][0][0][i][1][0]):
                        coord.name = data["label"][0][0][i][1][0][0]
                elif isinstance(data["label"][0][0][i][1][0], str):
                    coord.name = data["label"][0][0][i][1][0]

        axisdataarray = data["axisscale"][0][0][i][0]
        if len(axisdataarray):  # some axiscale might be present
            if isinstance(axisdataarray[0], np.ndarray):
                if len(axisdataarray[0]) == dat.shape[i]:
                    datac = axisdataarray[0]  # take the first axiscale data
                elif axisdataarray[0].size == dat.shape[i]:
                    datac = axisdataarray[0][0]

            if datac is not None:
                if isinstance(coord, Coord):
                    coord.data = datac
                else:
                    coord = Coord(data=datac)

            if len(data["axisscale"][0][0][i][1]):  # some titles might be present
                try:
                    coord.title = data["axisscale"][0][0][i][1][0]
                except Exception:
                    with contextlib.suppress(Exception):
                        coord.title = data["axisscale"][0][0][i][1][0][0]

        if not isinstance(coord, Coord):
            coord = Coord(data=list(range(dat.shape[i])), title="index")

        coords.append(coord)

    dataset.data = dat
    dataset.set_coordset(*list(coords))
    dataset.author = author
    dataset.origin = "dso"
    dataset.name = name
    dataset.acquisition_date = date

    # TODO: reshape from fortran/Matlab order to C order
    #  for 3D or higher datasets ?

    for i in data["description"][0][0]:
        dataset.description += i

    for entry in data["history"][0, 0].ravel():
        dataset.history = entry.item()

    dataset.history = "Imported by spectrochempy."
    return dataset
