# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

"""
Plugin module to extend NDDataset with the import methods method.
"""

__all__ = ["read_matlab", "read_mat"]
__dataset_methods__ = __all__

import io
from warnings import warn
from datetime import datetime, timezone

import numpy as np
import scipy.io as sio

from spectrochempy.core.dataset.nddataset import NDDataset, Coord
from spectrochempy.core.readers.importer import Importer, importermethod


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_matlab(*paths, **kwargs):
    """
    Read a matlab file with extension ``.mat`` and return its content as a list.

    The array of numbers (i.e. matlab matrices) and Eigenvector's DataSet Object (DSO, see
    `DSO <https://www.eigenvector.com/software/dataset.htm>`_ ) are returned as NDDatasets.  The
    content not recognized by SpectroChemPy is returned as a tuple (name, object).

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
    read_matlab
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel'}, optional
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

    Examples
    ---------

    >>> scp.read_matlab('matlabdata/dso.mat')
    NDDataset: [float64] unitless (shape: (y:20, x:426))

    See ``read_omnic`` for more examples of use
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
    """
    kwargs["filetypes"] = ["MATLAB files (*.mat *.dso)"]
    kwargs["protocol"] = ["matlab", "mat", "dso"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ..............................................................................
read_mat = read_matlab


# ------------------------------------------------------------------
# Private methods
# ------------------------------------------------------------------


@importermethod
def _read_mat(*args, **kwargs):
    _, filename = args
    content = kwargs.get("content", False)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, "rb")

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
            dataset.history = (
                str(datetime.now(timezone.utc)) + ":imported from .mat file \n"
            )
            # TODO: reshape from fortran/Matlab order to C opder
            # for 3D or higher datasets ?
            datasets.append(dataset)

        elif all(
            name_ in data.dtype.names
            for name_ in ["moddate", "axisscale", "imageaxisscale"]
        ):
            # this is probably a DSO object
            dataset = _read_dso(dataset, name, data)
            datasets.append(dataset)

        else:
            warn(f"unsupported data type : {data.dtype}")
            # TODO: implement DSO reader
            datasets.append([name, data])

    return datasets


@importermethod
def _read_dso(dataset, name, data):
    name_mat = data["name"][0][0]
    if len(name_mat) == 0:
        name = ""
    else:
        name = name_mat[0]

    typedata_mat = data["type"][0][0]
    if len(typedata_mat) == 0:
        typedata = ""
    else:
        typedata = typedata_mat[0]

    if typedata != "data":
        return (name, data)

    else:
        author_mat = data["author"][0][0]
        if len(author_mat) == 0:
            author = "*unknown*"
        else:
            author = author_mat[0]

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
                        try:
                            coord.title = data["axisscale"][0][0][i][1][0][0]
                        except Exception:
                            pass

            if not isinstance(coord, Coord):
                coord = Coord(data=[j for j in range(dat.shape[i])], title="index")

            coords.append(coord)

        dataset.data = dat
        dataset.set_coordset(*[coord for coord in coords])
        dataset.author = author
        dataset.name = name
        dataset.date = date

        # TODO: reshape from fortran/Matlab order to C order
        #  for 3D or higher datasets ?

        for i in data["description"][0][0]:
            dataset.description += i

        for i in data["history"][0][0][0][0]:
            dataset.history.append(i)

        dataset.history = (
            str(datetime.now(timezone.utc)) + ": Imported by spectrochempy "
        )
    return dataset


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
