# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
"""
This module provides methods for reading data in a directory.
"""
__all__ = ["read_dir", "read_carroucell"]
__dataset_methods__ = __all__

import os
import warnings
import datetime
import re

import scipy.interpolate
import numpy as np
import xlrd

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils import get_filename, readdirname
from spectrochempy.core import info_, print_
from spectrochempy.core.readers.importer import importermethod, Importer


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_dir(directory=None, **kwargs):
    """
    Read an entire directory.

    Open a list of readable files in a and store data/metadata in a dataset or a list of datasets according to the
    following rules :

    * 2D spectroscopic data (e.g. valid *.spg files or matlab arrays, etc...) from
      distinct files are stored in distinct NDdatasets.
    * 1D spectroscopic data (e.g., *.spa files) in a given directory are grouped
      into single NDDataset, providing their unique dimension are compatible. If not,
      an error is generated.

    Parameters
    ----------
    directory : str or pathlib
        Folder where are located the files to read.

    Returns
    --------
    read_dir
        |NDDataset| or list of |NDDataset|.

    Depending on the python version, the order of the datasets in the list may change.

    See Also
    --------
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Examples
    --------

    >>> scp.preferences.csv_delimiter = ','
    >>> A = scp.read_dir('irdata')
    >>> len(A)
    4

    >>> B = scp.NDDataset.read_dir()
    """
    kwargs["listdir"] = True
    importer = Importer()
    return importer(directory, **kwargs)


# TODO: make an importer function, when a test directory will be provided.
# ..............................................................................
def read_carroucell(dataset=None, directory=None, **kwargs):
    """
    Open .spa files in a directory after a carroucell experiment.

    The files for a given sample are grouped in NDDatasets (sorted by acquisition date).
    The NDDatasets are returned in a list sorted by sample number.
    When the file containing the temperature data is present, the temperature is read
    and assigned as a label to each spectrum.

    Parameters
    ----------
    dataset : `NDDataset`
        The dataset to store the data and metadata.
        If None, a NDDataset is created.
    directory : str, optional
        If not specified, opens a dialog box.
    spectra : arraylike of 2 int (min, max), optional, default=None
        The first and last spectrum to be loaded as determined by their number.
         If None all spectra are loaded.
    discardbg : bool, optional, default=True
        If True : do not load background (sample #9).
    delta_clocks : int, optional, default=0
        Difference in seconds between the clocks used for spectra and temperature acquisition.
        Defined as t(thermocouple clock) - t(spectrometer clock).

    Returns
    --------
    nddataset
        |NDDataset| or list of |NDDataset|.

    See Also
    --------
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Notes
    ------
    All files are expected to be present in the same directory and their filenames
    are expected to be in the format : X_samplename_YYY.spa
    and for the backround files : X_BCKG_YYYBG.spa
    where X is the sample holder number and YYY the spectrum number.

    Examples
    --------
    """

    # check if the first parameter is a dataset
    # because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameter must be the directory
        if isinstance(dataset, str) and dataset != "":
            directory = dataset

    directory = readdirname(directory)

    if not directory:
        # probably cancel has been chosen in the open dialog
        info_("No directory was selected.")
        return

    spectra = kwargs.get("spectra", None)
    discardbg = kwargs.get("discardbg", True)

    delta_clocks = datetime.timedelta(seconds=kwargs.get("delta_clocks", 0))

    datasets = []

    # get the sorted list of spa files in the directory
    spafiles = sorted(
        [
            f
            for f in os.listdir(directory)
            if (os.path.isfile(os.path.join(directory, f)) and f[-4:].lower() == ".spa")
        ]
    )

    # discard BKG files
    if discardbg:
        spafiles = sorted([f for f in spafiles if "BCKG" not in f])

    # select files
    if spectra is not None:
        [min, max] = spectra
        if discardbg:
            spafiles = sorted(
                [
                    f
                    for f in spafiles
                    if min <= int(f.split("_")[2][:-4]) <= max and "BCKG" not in f
                ]
            )
        if not discardbg:
            spafilespec = sorted(
                [
                    f
                    for f in spafiles
                    if min <= int(f.split("_")[2][:-4]) <= max and "BCKG" not in f
                ]
            )
            spafileback = sorted(
                [
                    f
                    for f in spafiles
                    if min <= int(f.split("_")[2][:-6]) <= max and "BCKG" in f
                ]
            )
            spafiles = spafilespec + spafileback

    curfilelist = [spafiles[0]]
    curprefix = spafiles[0][::-1].split("_", 1)[1][::-1]

    for f in spafiles[1:]:
        if f[::-1].split("_", 1)[1][::-1] != curprefix:
            datasets.append(
                NDDataset.read_omnic(curfilelist, sortbydate=True, directory=directory)
            )
            datasets[-1].name = os.path.basename(curprefix)
            curfilelist = [f]
            curprefix = f[::-1].split("_", 1)[1][::-1]
        else:
            curfilelist.append(f)

    datasets.append(
        NDDataset.read_omnic(curfilelist, sortbydate=True, directory=directory)
    )
    datasets[-1].name = os.path.basename(curprefix)

    # Now manage temperature
    Tfile = sorted([f for f in os.listdir(directory) if f[-4:].lower() == ".xls"])
    if len(Tfile) == 0:
        print_("no temperature file")
    elif len(Tfile) > 1:
        warnings.warn("several .xls/.csv files. The temperature will not be read")
    else:
        Tfile = Tfile[0]
        if Tfile[-4:].lower() == ".xls":
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
                    time = datetime.datetime.strptime(
                        sheet.cell(i, 0).value, "%d/%m/%y %H:%M:%S"
                    ).replace(tzinfo=datetime.timezone.utc)
                    if ti <= time <= tf:
                        t.append(time)
                        T.append(sheet.cell(i, 4).value)
                except ValueError:
                    pass
                except TypeError:
                    pass

            # interpolate T = f(timestamp)
            tstamp = [time.timestamp() for time in t]
            # interpolate, except for the first and last points that are extrapolated
            interpolator = scipy.interpolate.interp1d(
                tstamp, T, fill_value="extrapolate", assume_sorted=True
            )

            for ds in datasets:
                # timestamp of spectra for the thermocouple clock

                tstamp_ds = [
                    (label[0] + delta_clocks).timestamp() for label in ds.y.labels
                ]
                T_ds = interpolator(tstamp_ds)
                newlabels = np.hstack((ds.y.labels, T_ds.reshape((50, 1))))
                ds.y = Coord(title=ds.y.title, data=ds.y.data, labels=newlabels)

    if len(datasets) == 1:
        return datasets[0]  # a single dataset is returned

    # several datasets returned, sorted by sample #
    return sorted(datasets, key=lambda ds: int(re.split("-|_", ds.name)[0]))


# ======================================================================================================================
# Private functions
# ======================================================================================================================


@importermethod
def _read_dir(*args, **kwargs):
    _, directory = args
    directory = readdirname(directory)
    files = get_filename(directory, **kwargs)
    datasets = []
    for key in files.keys():
        if key:
            importer = Importer()
            nd = importer(files[key], **kwargs)
            if not isinstance(nd, list):
                nd = [nd]
            datasets.extend(nd)
    return datasets


if __name__ == "__main__":
    pass
