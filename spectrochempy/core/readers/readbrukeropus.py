# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""This module to extend NDDataset with the import methods method.

"""
__all__ = ['read_bruker_opus']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

import os
from brukeropusreader import read_file
from datetime import datetime, timezone, timedelta

import numpy as np




# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------------------------------------------------
from spectrochempy.core import debug_
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.utils import readfilename
# ======================================================================================================================
# Public functions
# ======================================================================================================================

# .............................................................................
def read_bruker_opus(dataset=None, **kwargs):
    """Open Bruker Opus file(s) and group them in a single dataset. Only the spectrum is
    extracted ("AB" field). Returns an error if dimensions are incompatibles.

    Parameters
    ----------
    filename : `None`, `str`, or list of `str`
        Filename of the file(s) to load. If `None` : opens a dialog box to select
         files. If `str` : a single filename. It list of str :
        a list of filenames.
    directory : str, optional, default="".
        From where to read the specified filename. If not specified, read in
        the defaults datadir.

    Returns
    -------
    dataset : |NDDataset|
        A dataset corresponding to the (set of) bruker file(s).

    Examples
    --------
    >>> A = NDDataset.read_bruker_opus('irdata\\spectrum.0001')
    >>> print(A)

    """
    debug_("reading bruker opus files")

    # filename will be given by a keyword parameter except if the first parameters is already
    # the filename
    filename = kwargs.get('filename', None)

    # check if the first parameter is a dataset because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(dataset, (str, list)) and dataset != '':
            filename = dataset

    # check if directory was specified
    directory = kwargs.get("directory", None)
    sortbydate = kwargs.get("sortbydate", True)

    # returns a list of files to read
    files = readfilename(filename,
                         directory=directory,
                         filetypes=['Bruker files (*.*)',
                                    'all files (*)'],
                         dictionary=False)
    #todo: see how to use regular expression in Qt filters

    if not files:
        # there is no files, return nothing
        return None

    xaxis = None
    intensities = []
    names = []
    for file in files:
        opus_data = read_file(file)
        if not xaxis:
            xaxis = Coord(opus_data.get_range("AB"),
                      title='Wavenumbers', units='cm^-1')
        elif (opus_data.get_range("AB") != xaxis.data).any():
            raise ValueError("spectra have incompatible dimensions (xaxis)")

        intensities.append(opus_data["AB"])
        names.append(opus_data["Sample"]['SNM'])
        acqdate = opus_data["AB Data Parameter"]["DAT"]
        acqtime = opus_data["AB Data Parameter"]["TIM"]
        dtdate = datetime.strptime(acqdate, '%d/%m/%Y')
        dttime = datetime.strptime(acqtime.split()[0], '%H:%M:%S.%f')
        #dttimezone = datetime.strptime(acqtime.split()[1], '(%Z)')
    dataset = NDDataset(intensities)
    dataset.units = 'absorbance'
    dataset.title = 'Absorbance'
    dataset.set_coords(x=xaxis)

    # Set origin, description and history
    dataset.origin = "opus"
    dataset.description = ('Dataset from opus files. \n')

    dataset.history = str(datetime.now()) + ':import from opus files \n'

#    if sortbydate:
#        dataset.sort(dim='y', inplace=True)
#        dataset.history.append(str(datetime.now()) + ':Sorted')

    # Set the NDDataset date
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    # debug_("end of reading")

    return dataset
