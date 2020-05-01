# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================


"""Plugin module to extend NDDataset with the import methods method.

"""

from datetime import datetime

import numpy as np
import scipy.io as sio

__all__ = ['read_matlab']

__dataset_methods__ = __all__

from spectrochempy.core.dataset.nddataset import NDDataset, Coord
from spectrochempy.utils import readfilename
from ...core import debug_


def read_matlab(dataset=None, **kwargs):
    """
    Open a matlab file with extension ``.mat`` and returns its content as a list
    
    The array of numbers (i.e. matlab matrices) and Eigenvector's DataSet Object (DSO, see
    `DSO <https://www.eigenvector.com/software/dataset.htm>`_ ) are returned as NDDatasets.  The
    content not recognized by SCpy  is returned as a tuple (name, object)
    
    Parameters
    ----------
    dataset : |NDDataset|
        The dataset (or list of datasets) to store the data and metadata read from the file(s).
        If None, a |NDDataset| is created.
    filename : None, str, or list of str
        Filename of the file(s) to load. If `None`: opens a dialog box to select
        ``.mat`` files. If str : a single filename. It list of str:
        a list of filenames.
    directory : str, optional, default="".
        From where to read the specified filename. If not specified, read in
        the defaults datadir.
    
    Returns
    -------
    dataset : list or |NDDataset|
        A dataset or a list of datasets or tuples (name, object) if some content
        is not recognized in the .mat file.

    """
    debug_("reading .mat file")

    # filename will be given by a keyword parameter except the first parameters
    # is already the filename
    filename = kwargs.get('filename', None)

    # check if the first parameter is a dataset
    # because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(dataset, (str, list)) and dataset != '':
            filename = dataset

        dataset = NDDataset()  # create an instance of NDDataset

    # check if directory was specified
    directory = kwargs.get("directory", None)

    # returns a list of files to read
    files = readfilename(filename,
                         directory=directory,
                         filetypes=['MAT files (*.mat)'])

    if not files:
        # there is no files, return nothing
        return None

    files = files['.mat']

    datasets = []

    for file in files:
        content = sio.whosmat(file)
        f = sio.loadmat(file)

        for x in content:
            if x[2] in ['double', 'single', 'int8', 'int16',
                        'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
                # this is an array of numbers
                name = x[0]
                data = f[name]

                ds = NDDataset()
                ds.data = data
                ds.name = name
                ds.history.append(str(datetime.now()) + ':imported from .mat file \n')
                datasets.append(ds)
                # TODO: reshape from fortran/Matlab order to C opder
                # for 3D or higher datasets ?

            elif x[2] == 'object':
                # this is probably a DSO object

                ds = _read_DSO(f, x)
                datasets.append(ds)
            else:
                debug_('unsupported data type')
                # TODO: implement DSO reader
                datasets.append((x[0], f[x[0]]))

    if len(datasets) == 1:
        return datasets[0]
    else:
        return datasets


def _read_DSO(f, x):
    dso = x[0]

    name_mat = f[dso]['name'][0][0]
    if len(name_mat) == 0:
        name = ''
    else:
        name = name_mat[0]

    typedata_mat = f[dso]['type'][0][0]
    if len(typedata_mat) == 0:
        typedata = ''
    else:
        typedata = typedata_mat[0]

    if typedata != 'data':
        return ((x[0], f[x[0]]))

    else:
        author_mat = f[dso]['author'][0][0]
        if len(author_mat) == 0:
            author = '*unknown*'
        else:
            author = author_mat[0]

        date_mat = f[dso]['date'][0][0]
        if len(date_mat) == 0:
            date = datetime(1, 1, 1, 0, 0)
        else:
            date = datetime(int(date_mat[0][0]), int(date_mat[0][1]),
                            int(date_mat[0][2]), int(date_mat[0][3]),
                            int(date_mat[0][4]),
                            int(date_mat[0][5]))

        data = f[dso]['data'][0][0]

        # look at coords and labels
        # only the first label and axisscale are taken into account
        # the axisscale title is used as the coordinate title

        coords = []
        for i in range(len(data.shape)):
            coord = datac = labels = title = None
            labelsarray = f[dso]['label'][0][0][i][0]
            if len(labelsarray):  # some labels might be present
                if isinstance(labelsarray[0], np.ndarray):
                    labels = f[dso]['label'][0][0][i][0][0]
                else:
                    labels = f[dso]['label'][0][0][i][0]
                if len(labels):
                    coord = (Coord(labels=[str(label) for label in labels]))
                if len(f[dso]['label'][0][0][i][1]):
                    if isinstance(f[dso]['label'][0][0][i][1][0], np.ndarray):
                        if len(f[dso]['label'][0][0][i][1][0]):
                            coord.name = f[dso]['label'][0][0][i][1][0][0]
                    elif isinstance(f[dso]['label'][0][0][i][1][0], str):
                        coord.name = f[dso]['label'][0][0][i][1][0]

            axisdataarray = f[dso]['axisscale'][0][0][i][0]
            if len(axisdataarray):  # some axiscale might be present
                if isinstance(axisdataarray[0], np.ndarray):
                    if len(axisdataarray[0]) == data.shape[i]:
                        datac = axisdataarray[0]  # take the first axiscale data
                    elif axisdataarray[0].size == data.shape[i]:
                        datac = axisdataarray[0][0]

                if datac is not None:
                    if isinstance(coord, Coord):
                        coord.data = datac
                    else:
                        coord = Coord(data=datac)

                if len(f[dso]['axisscale'][0][0][i][1]):  # some titles might be present
                    try:
                        coord.title = f[dso]['axisscale'][0][0][i][1][0]
                    except:
                        try:
                            coord.title = f[dso]['axisscale'][0][0][i][1][0][0]
                        except:
                            pass

            if not isinstance(coord, Coord):
                coord = Coord(data=[j for j in range(data.shape[i])], title='index')

            coords.append(coord)

        ds = NDDataset(data,
                       author=author,
                       coords=coords,
                       name=name,
                       date=date)

        ds.name = name
        ds.date = date

        # TODO: reshape from fortran/Matlab order to C order
        #  for 3D or higher datasets ?

        for i in f[dso]['description'][0][0]:
            ds.description += i

        for i in f[dso]['history'][0][0][0][0]:
            ds.history.append(i)

        ds.history = (str(datetime.now()) + ': Imported by spectrochempy ')
    return ds
