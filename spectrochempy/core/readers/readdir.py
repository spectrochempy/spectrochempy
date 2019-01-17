# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""This module extend NDDataset with some import methods.

"""
__all__ = ['read_dir']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------

import os
import warnings
import glob

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------

from spectrochempy.dataset.ndio import NDIO
from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.application import log, general_preferences as prefs
from spectrochempy.utils import readfilename, readdirname
from spectrochempy.gui.dialogs import opendialog
from spectrochempy.core.readers.readomnic import read_omnic
from spectrochempy.core.readers.readcsv import read_csv


# function for reading data in a directory
# --------------------------------------
def read_dir(dataset=None, directory=None, **kwargs):
    """
    Open readable files in a directory and store data/metadata in a dataset or
    a list of datasets according to the following rules:

    * 2D spectroscopic data (e.g. valid \*.spg files) from distinct files are
      stored in distinct NDdatasets.
    * 1D spectroscopic data (e.g., \*.spa files) in a given directory are grouped
      into single NDDataset, providing their unique dimension are compatible. If not,
      an error is generated.

    Notes
    ------
    Only implemented for OMNIC files (\*.spa, \*.spg), \*.csv, and the
    native format for spectrochempy : \*.scp).

    Parameters
    ----------
    dataset : `NDDataset`
        The dataset to store the data and metadata.
        If None, a NDDataset is created
    directory: str, optional.
        If not specified, opens a dialog box.
    parent_dir: str, optional.
        The parent directory where to look at
    sortbydate: bool, optional,  default:True.
        Sort spectra by acquisition date
    recursive: bool, optional,  default = True.
        Read also subfolders

    Returns
    --------
    nddataset : |NDDataset| or list of |NDDataset|

    Examples
    --------
    >>> A = NDDataset.read_dir('irdata')
    >>> print(A)
    [NDDataset: [  ...

    >>> B = NDDataset.read_dir()

    """

    log.debug("starting reading in a folder")

    # check if the first parameter is a dataset
    # because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameter must be the directory
        if isinstance(dataset, str) and dataset != '':
            directory = dataset
            dataset = None

    # # check directory
    # if directory is None:
    #     directory = opendialog( single=False,
    #                             directory=directory,
    #                             caption='Select folder to read',
    #                             filters = 'directory')
    #
    #     if not directory:
    #         # if the dialog has been cancelled or return nothing
    #         return None
    #
    # if directory is not None and not isinstance(directory, str):
    #     raise TypeError('directory should be of str type, not ' + type(directory))
    #
    # if directory and not os.path.exists(directory):
    #     # the directory may be located in our default datadir
    #     # or be the default datadir
    #     if directory == os.path.basename(prefs.datadir):
    #         return prefs.datadir
    #
    #     _directory = os.path.join(prefs.datadir, directory)
    #
    # if not os.path.isdir(directory):
    #     raise ValueError('\"' + directory + '\" is not a valid directory')

    parent_dir = kwargs.get('parent_dir', None)
    directory = readdirname(directory, parent_dir=parent_dir)

    if not directory:
        # probably cancel has been chosen in the open dialog
        log.info("No directory was selected.")
        return

    datasets = []

    recursive = kwargs.get('recursive', True)
    if recursive:
        for i, root in enumerate(os.walk(directory)):
            if i == 0:
                log.debug("reading root directory")
            else:
                log.debug("reading subdirectory")
            datasets += _read_single_dir(root[0])
    else:
        log.debug("reading root directory only")
        datasets += _read_single_dir(directory)

    if len(datasets) == 1:
        log.debug("finished read_dir()")
        return datasets[0]  # a single dataset is returned

    log.debug("finished read_dir()")
    return datasets  # several datasets returned


def _read_single_dir(directory):
    # lists all filenames of readable files in directory:
    filenames = [os.path.join(directory, f) for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f))]

    datasets = []

    if not filenames:
        log.debug("empty directory")
        return datasets

    files = readfilename(filenames, directory=directory)

    for extension in files.keys():
        if extension == '.spg':
            for filename in files[extension]:
                datasets.append(NDDataset.read_omnic(filename,
                                                     sortbydate=True))

        elif extension == '.spa':
            datasets.append(NDDataset.read_omnic(files[extension],
                                                 sortbydate=True))

        elif extension == '.csv':
            datasets.append(NDDataset.read_csv(filename=files[extension],
                                               sortbydate=True))

        elif extension == '.scp':
            # does not work. see test_load
            # datasets.append(NDDataset.read(files[extension], protocol=extension[1:]))
            pass
        # else the files are not readable
        else:
            pass

    # TODO: extend to other implemented readers (NMR !)
    return datasets


if __name__ == '__main__':
    pass
