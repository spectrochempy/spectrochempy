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
import scipy.io as sio

__all__ = ['read_matlab']

__dataset_methods__ = __all__

from spectrochempy.core.dataset.ndio import NDIO
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import log, general_preferences as prefs
from spectrochempy.utils import readfilename, SpectroChemPyWarning


def read_matlab(dataset=None, **kwargs):
    """Open a matlab file with extension ``.mat`` and set data/metadata in the current dataset

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset (or list od datasets) to store the data and metadata read from the file(s).
        If None, a |NDDataset| is created.
    filename : `None`, `str`, or list of `str`
        Filename of the file(s) to load. If `None`: opens a dialog box to select
        ``.mat`` files. If `str`: a single filename. It list of str:
        a list of filenames.
    directory: str, optional, default="".
        From where to read the specified filename. If not specified, read in
        the defaults datadir.


    Returns
    -------
    dataset : |NDDataset|
        A dataset or a list of datasets,  It some content is not recognized as an array or a dataset
        it is returned aa a tuple (name, object)

    Examples
    --------

    """
    log.debug("reading .mat file")

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
        if len(content) > 1:
            log.debug("several elements")

        for x in content:

            if x[2] in ['double', 'single', 'int8', 'int16',
                        'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
                # this is an array of numbers
                name = x[0]
                data = f[name]

                ds = NDDataset()
                ds.data = data
                ds.name = name
                datasets.append(ds)

            else:
                log.debug('unsupported data type')
                # TODO: implement DSO reader
                datasets.append((x[0], f[x[0]]))

    if len(datasets) == 1:
        return (datasets[0])
    else:
        return datasets
