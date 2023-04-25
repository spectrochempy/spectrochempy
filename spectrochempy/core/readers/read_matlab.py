# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Plugin module to extend NDDataset with the import methods method.
"""

__all__ = ["read_matlab", "read_mat"]
__dataset_methods__ = __all__

from datetime import datetime

import numpy as np
import scipy.io as sio

from spectrochempy.application import info_, warning_
from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.core.readers.importer import Importer, _importer_method, _openfid
from spectrochempy.utils.docstrings import _docstring

# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_matlab")


@_docstring.dedent
def read_matlab(*paths, **kwargs):
    """
    Read a matlab file with extension :file:`.mat` and return its content as a list.

    The array of numbers (*i.e.,* matlab matrices) and Eigenvector's DataSet Object
    (``DSO``, see `DSO <https://www.eigenvector.com/software/dataset.htm>`__ ) are
    returned as NDDatasets.  The content not recognized by SpectroChemPy is returned
    as a tuple (name, object).

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    --------
    %(Importer.returns)s

    Other Parameters
    ----------------
    %(Importer.other_parameters)s

    See Also
    ---------
    %(Importer.see_also.no_read_matlab)s

    Examples
    ---------

    >>> scp.read_matlab('matlabdata/dso.mat')
    NDDataset: [float64] unitless (shape: (y:20, x:426))
    """
    kwargs["filetypes"] = ["MATLAB files (*.mat *.dso)"]
    kwargs["protocol"] = ["matlab", "mat", "dso"]
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
            dataset.history = "Imported from .mat file"
            # TODO: reshape from fortran/Matlab order to C opder
            # for 3D or higher datasets ?
            datasets.append(dataset)

        elif data.dtype.char == "U":
            # this is an array of string
            info_(
                f"The mat file contains an array of strings named '{name}' which will not be converted to NDDataset"
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
            dataset.history = i

        dataset.history = "Imported by spectrochempy."
    return dataset
