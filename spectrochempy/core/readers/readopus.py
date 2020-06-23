# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""This module to extend NDDataset with the import methods method.

"""
__all__ = ['read_opus']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------


from brukeropusreader import read_file
from warnings import warn
from datetime import datetime, timezone, timedelta
from numpy import linspace

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
def read_opus(dataset=None, **kwargs):
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
    >>> A = NDDataset.read_opus('irdata\\spectrum.0001')
    >>> print(A)
    NDDataset: [float64] a.u. (shape: (y:1, x:2568))


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
    # todo: see how to use regular expression in Qt filters

    if not files:
        # there is no files, return nothing
        return None

    xaxis = None
    intensities = []
    names = []
    acquisitiondates = []
    timestamps = []
    for file in files:
        opus_data = read_file(file)
        try:
            opus_data["AB"]
        except KeyError:  # not an absorbance spectrum
            warn("opus file {} could not be read".format(file))
            continue

        npt = opus_data['AB Data Parameter']['NPT']
        fxv = opus_data['AB Data Parameter']['FXV']
        lxv = opus_data['AB Data Parameter']['LXV']
        xdata = linspace(fxv, lxv, npt)

        if not xaxis:
            xaxis = Coord(xdata, title='Wavenumbers', units='cm^-1')

        elif (xdata != xaxis.data).any():
            raise ValueError("spectra have incompatible dimensions (xaxis)")

        intensities.append(opus_data["AB"][:npt])
        names.append(opus_data["Sample"]['SNM'])
        acqdate = opus_data["AB Data Parameter"]["DAT"]
        acqtime = opus_data["AB Data Parameter"]["TIM"]
        GMT_offset_hour = float(acqtime.split('GMT')[1].split(')')[0])
        date_time = datetime.strptime(acqdate + '_' + acqtime.split()[0],
                                      '%d/%m/%Y_%H:%M:%S.%f')
        UTC_date_time = date_time - timedelta(hours=GMT_offset_hour)
        UTC_date_time = UTC_date_time.replace(tzinfo=timezone.utc)
        # Transform to timestamp for storage in the Coord object
        # use datetime.fromtimestamp(d, timezone.utc)) to transform back to datetime
        timestamp = UTC_date_time.timestamp()
        acquisitiondates.append(UTC_date_time)
        timestamps.append(timestamp)

    # return if none of the files could be read:
    if not xaxis:
        return

    yaxis = Coord(timestamps,
                  title='Acquisition timestamp (GMT)',
                  units='s',
                  labels=(acquisitiondates, names))

    dataset = NDDataset(intensities)
    dataset.set_coords(y=yaxis, x=xaxis)
    dataset.units = 'absorbance'
    dataset.title = 'Absorbance'

    # Set origin, description and history
    dataset.origin = "opus"
    dataset.description = 'Dataset from opus files. \n'

    if sortbydate:
        dataset.sort(dim='y', inplace=True)

    dataset.history = str(datetime.now()) + ': import from opus files \n'
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    return dataset
