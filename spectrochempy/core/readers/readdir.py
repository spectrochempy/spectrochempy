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
from spectrochempy.utils import readfilename, SpectroChemPyWarning
#from spectrochempy.core.readers.readomnic import read_omnic
#from spectrochempy.core.readers.readcsv import read_csv


# function for reading data in a directory
# --------------------------------------
def read_dir(dataset=None, **kwargs):
    """
    Open readable files in a directory and store data/metadata in a dataset or
    a list of datasets according to the following rules:

    * 2D spectroscopic data (e.g. valid \*.spg files) from distinct files are
      stored in distinct NDdatasets.
    * 1D spectroscopic data (e.g., \*.spa files) in a given directory are grouped
      into single NDDataset, providing their unique dimension are compatible.

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
    [NDDataset: [[   2.057,    2.061, ...,    2.013,    2.012],
                [   2.033,    2.037, ...,    1.913,    1.911],
                ...,
                [   1.794,    1.791, ...,    1.198,    1.198],
                [   1.816,    1.815, ...,    1.240,    1.238]] a.u., NDDataset: [  -0.091,    3.547, ...,    4.317,   -0.091] unitless, NDDataset: [[   2.057,    2.061, ...,    2.013,    2.012],
                [   2.033,    2.037, ...,    1.913,    1.911],
                ...,
                [   1.794,    1.791, ...,    1.198,    1.198],
                [   1.816,    1.815, ...,    1.240,    1.238]] a.u.]

    >>> B = NDDataset.read_dir()

    """

    # TODO add recursive: bool [optional default = True ]. read also subfolders
    log.debug("starting reading in a folder")

    # check if the first parameter is a dataset
    # because we allow not to pass it
    directory = None
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameter must be the directory
        if isinstance(dataset, str) and dataset != '':
            directory = dataset

        dataset = None # NDDataset()  # we don't need to create the NDDataset now

    # get the directory name possibly passed in the parameters
    if not directory:
        directory = kwargs.get('directory', None)

    # check is the directory name was passed
    #
    directory = readfilename(directory=directory,
                             filetypes='directory')

    if directory is None:
        return

    if not isinstance(directory, str):
        raise TypeError('Error: directory should be of str type, not %s'%type(directory))

    datasets = []

    recursive = kwargs.get('recursive', False)
    if recursive:
        files = glob.glob(os.path.join(directory,'**','*.*'), recursive=recursive)
    else:
        files = glob.glob(os.path.join(directory, '*.*'))

    files = readfilename(files)

    for extension in files.keys():

        extension = extension.lower()

        # only possible if the data are compatible
        # unlikely true if recursive is True. So by default it is better
        # to set it to False (except if we are sure the concatenation is possible)

        if extension == '.spg':
            for filename in files[extension]:
                datasets.append(NDDataset.read_omnic(filename))

        elif extension == '.spa':
            datasets.append(NDDataset.read_omnic(files[extension],
                                                 sortbydate=True))

        elif extension == '.csv':
            datasets.append(NDDataset.read_csv(filename=files[extension],
                                               sortbydate=True))

        elif extension == '.scp':
            datasets.append(NDDataset.read(files[extension], protocol=extension[1:],
                                           sortbydate=True, **kwargs))

        # else the files are not readable
        else:
            pass
            #TODO: to extend we must implement some other readers
            # case of NMR bruker spectra, for which there is no extension!

    return datasets

if __name__ == '__main__':

    pass

