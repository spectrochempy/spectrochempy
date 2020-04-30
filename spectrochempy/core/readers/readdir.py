# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# =============================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""This module extend NDDataset with some import methods.

"""
__all__ = ['read_dir', 'read_carroucell']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import warnings
import datetime
import scipy.interpolate

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import xlrd

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------------------------------------------------

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.utils import readfilename, readdirname
from ...core import info_, print_

# function for reading data in a directory
# ----------------------------------------------------------------------------------------------------------------------
def read_dir(dataset=None, directory=None, **kwargs):
    """
    Open readable files in a directory and store data/metadata in a dataset or
    a list of datasets according to the following rules :

    * 2D spectroscopic data (e.g. valid \*.spg files or matlab arrays, etc...) from
      distinct files are stored in distinct NDdatasets.
    * 1D spectroscopic data (e.g., \*.spa files) in a given directory are grouped
      into single NDDataset, providing their unique dimension are compatible. If not,
      an error is generated.

    Notes
    ------
    Only implemented for OMNIC files (\*.spa, \*.spg), \*.csv, \*.mat and the
    native format for spectrochempy : \*.scp).

    Parameters
    ----------
    dataset : `NDDataset`
        The dataset to store the data and metadata.
        If None, a NDDataset is created
    directory : str, optional.
        If not specified, opens a dialog box.
    parent_dir : str, optional.
        The parent directory where to look at
    sortbydate : bool, optional,  default:True.
        Sort spectra by acquisition date
    recursive : bool, optional,  default=True.
        Read also subfolders

    Returns
    --------
    nddataset : |NDDataset| or list of |NDDataset|

    Examples
    --------
    >>> A = NDDataset.read_dir('irdata')
    >>> print(A)
    [NDDataset: [[ ...

    >>> B = NDDataset.read_dir()

    """

    #debug_("starting reading in a folder")

    # check if the first parameter is a dataset
    # because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameter must be the directory
        if isinstance(dataset, str) and dataset != '':
            directory = dataset
            dataset = None

    parent_dir = kwargs.get('parent_dir', None)
    directory = readdirname(directory, parent_dir=parent_dir)

    if not directory:
        # probably cancel has been chosen in the open dialog
        info_("No directory was selected.")
        return

    datasets = []

    recursive = kwargs.get('recursive', True)
    if recursive:
        for i, root in enumerate(os.walk(directory)):
            if i == 0:
                pass # debug_("reading root directory")
            else:
                pass # debug_("reading subdirectory")
            datasets += _read_single_dir(root[0])
    else:
        # debug_("reading root directory only")
        datasets += _read_single_dir(directory)

    if len(datasets) == 1:
        #debug_("finished read_dir()")
        return datasets[0]  # a single dataset is returned

    #debug_("finished read_dir()")
    return datasets  # several datasets returned


def _read_single_dir(directory):
    # lists all filenames of readable files in directory:
    filenames = [os.path.join(directory, f) for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f))]

    datasets = []

    if not filenames:
        #debug_("empty directory")
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
            datasets.append(NDDataset.read(files[extension], protocol='scp'))

        elif extension == '.mat':
            for filename in files[extension]:
                matlist = NDDataset.read_matlab(filename)
                for mat in matlist:
                    datasets.append(mat)
        # else the files are not (yet) readable
        else:
            pass
    # TODO: extend to other implemented readers (NMR !)
    return datasets


# function for reading data in a directory
# --------------------------------------
def read_carroucell(dataset=None, directory=None, **kwargs):
    """
    Open .spa files in a directory after a carroucell experiment.
    The files for a given sample are grouped in NDDatasets (sorted by acquisition date).
    The NDDatasets are returned in a list sorted by sample number.
    When the file containing the temperature data is present, the temperature is read
    and assigned as a label to each spectrum.


    Notes
    ------
    All files are expected to be present in the same directory and their filenames
    are expected to be in the format : X_samplename_YYY.spa
    and for the backround files : X_BCKG_YYYBG.spa
    where X is the sample holder number and YYY the spectrum number.

    Parameters
    ----------
    dataset : `NDDataset`
        The dataset to store the data and metadata.
        If None, a NDDataset is created
    directory : str, optional.
        If not specified, opens a dialog box.
    parent_dir : str, optional.
        The parent directory where to look at
    spectra : arraylike of 2 int (min, max), optional, default=None
        The first and last spectrum to be loaded as determined by their number.
         If None all spectra are loaded
    discardbg : bool, optional, default=True
        If True : do not load background (sample #9)

    delta_clocks : int, optional, default=0
        Difference in seconds between the clocks used for spectra and temperature acquisition.
        Defined as t(thermocouple clock) - t(spectrometer clock).


    Returns
    --------
    nddataset : |NDDataset| or list of |NDDataset|

    Examples
    --------

    """

    # debug_("starting reading in a folder")
    # check if the first parameter is a dataset
    # because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameter must be the directory
        if isinstance(dataset, str) and dataset != '':
            directory = dataset
            dataset = None

    parent_dir = kwargs.get('parent_dir', None)
    directory = readdirname(directory, parent_dir=parent_dir)

    if not directory:
        # probably cancel has been chosen in the open dialog
        info_("No directory was selected.")
        return

    spectra = kwargs.get('spectra', None)
    discardbg = kwargs.get('discardbg', True)

    delta_clocks = datetime.timedelta(seconds=kwargs.get('delta_clocks', 0))

    datasets = []

    # get the sorted list of spa files in the directory
    spafiles = sorted([f for f in os.listdir(directory)
                       if (os.path.isfile(os.path.join(directory, f))
                           and f[-4:].lower() == '.spa')])

    # discard BKG files
    if discardbg:
        spafiles = sorted([f for f in spafiles if 'BCKG' not in f])

    # select files
    if spectra is not None:
        [min, max] = spectra
        if discardbg:
            spafiles = sorted([f for f in spafiles if min <= int(f.split('_')[2][:-4]) <= max
                               and 'BCKG' not in f])
        if not discardbg:
            spafilespec = sorted([f for f in spafiles if min <= int(f.split('_')[2][:-4]) <= max
                                  and 'BCKG' not in f])
            spafileback = sorted([f for f in spafiles if min <= int(f.split('_')[2][:-6]) <= max
                                  and 'BCKG' in f])
            spafiles = spafilespec + spafileback

    curfilelist = [spafiles[0]]
    curprefix = spafiles[0][::-1].split("_", 1)[1][::-1]

    for f in spafiles[1:]:
        if f[::-1].split("_", 1)[1][::-1] != curprefix:
            datasets.append(NDDataset.read_omnic(curfilelist, sortbydate=True, directory=directory))
            datasets[-1].name = os.path.basename(curprefix)
            curfilelist = [f]
            curprefix = f[::-1].split("_", 1)[1][::-1]
        else:
            curfilelist.append(f)

    datasets.append(NDDataset.read_omnic(curfilelist, sortbydate=True, directory=directory))
    datasets[-1].name = os.path.basename(curprefix)

    # Now manage temperature
    Tfile = sorted([f for f in os.listdir(directory)
                    if f[-4:].lower() == '.xls'])
    if len(Tfile) == 0:
        print_("no temperature file")
    elif len(Tfile) > 1:
        warnings.warn("several .xls/.csv files. The temperature will not be read")
    else:
        Tfile = Tfile[0]
        if Tfile[-4:].lower() == '.xls':
            book = xlrd.open_workbook(os.path.join(directory, Tfile))

            # determine experiment start and end time (thermocouple clock)
            ti = datasets[0].y.labels[0][0] + delta_clocks
            tf = datasets[-1].y.labels[-1][0] + delta_clocks

            # get thermocouple time and T information during the experiment
            t = []
            T = []
            sheet = book.sheet_by_index(0)
            for i in range(9, sheet.nrows):
                try:
                    time = datetime.datetime.strptime(sheet.cell(i, 0).value, '%d/%m/%y %H:%M:%S').replace(
                        tzinfo=datetime.timezone.utc)
                    if ti <= time <= tf:
                        t.append(time)
                        T.append(sheet.cell(i, 4).value)
                except ValueError:
                    # debug_('incorrect date or temperature format in row {}'.format(i))
                    pass
                except TypeError:
                    # debug_('incorrect date or temperature format in row {}'.format(i))
                    pass

            # interpolate T = f(timestamp)
            tstamp = [time.timestamp() for time in t]
            # interpolate, except for the first and last points that are extrapolated
            interpolator = scipy.interpolate.interp1d(tstamp, T, fill_value='extrapolate', assume_sorted=True)

            for ds in datasets:
                # timestamp of spectra for the thermocouple clock

                tstamp_ds = [(label[0] + delta_clocks).timestamp() for label in ds.y.labels]
                T_ds = interpolator(tstamp_ds)
                newlabels = np.hstack((ds.y.labels, T_ds.reshape((50,1))))
                ds.y = Coord(title=ds.y.title, data=ds.y.data, labels=newlabels)

    if len(datasets) == 1:
        # debug_("finished read_dir()")
        return datasets[0]  # a single dataset is returned

    # debug_("finished read_dir()")
    # several datasets returned, sorted by sample #
    return sorted(datasets, key=lambda ds: int(ds.name.split('_')[0]))


if __name__ == '__main__':
    pass
