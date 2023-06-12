# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module provides methods for reading data in a directory after a carroucell experiment.
"""
__all__ = ["read_carroucell"]
__dataset_methods__ = __all__

import datetime
import os
import re
import warnings

import numpy as np
import scipy.interpolate
import xlrd

from spectrochempy.application import info_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer, _importer_method
from spectrochempy.core.readers.read_omnic import read_omnic
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.file import get_directory_name, get_filenames

_docstring.delete_params("Importer.see_also", "read_carroucell")


@_docstring.dedent
def read_carroucell(directory=None, **kwargs):
    """
    Open :file:`.spa` files in a directory after a :term:`carroucell` experiment.

    The files for a given sample are grouped in `NDDataset`\ s (sorted by
    acquisition date).
    The `NDDataset`\ s are returned in a list sorted by sample number.
    When the file containing the temperature data is present, the temperature is read
    and assigned as a label to each spectrum.

    Parameters
    ----------
    directory : `str`, optional
        If not specified, opens a dialog box.
    %(kwargs)s

    Returns
    -------
    %(Importer.returns)s

    Other Parameters
    ----------------
    spectra : :term:`array-like` of 2 `int` (``min`` , ``max`` ), optional, default: `None`
        The first and last spectrum to be loaded as determined by their number.
        If `None` all spectra are loaded.
    discardbg : `bool`, optional, default: `True`
        If `True` : do not load background (sample #9).
    delta_clocks : `int`, optional, default:  0
        Difference in seconds between the clocks used for spectra and temperature
        acquisition. Defined as ``t(thermocouple clock) - t(spectrometer clock)`` .

    See Also
    --------
    %(Importer.see_also.no_read_carroucell)s

    Notes
    ------
    All files are expected to be present in the same directory and their filenames
    are expected to be in the format : :file:`X_samplename_YYY.spa`
    and for the background files : :file:`X_BCKG_YYYBG.spa`
    where ``X`` is the sample holder number and ``YYY`` the spectrum number.

    Examples
    --------

    >>> scp.read_carroucell("irdata/carroucell_samp")
    no temperature file
    [NDDataset: [float64] a.u. (shape: (y:6, x:11098)), NDDataset: ...
    """
    kwargs["filetypes"] = ["Carroucell files (*.spa)"]
    kwargs["protocol"] = ["carroucell"]
    importer = Importer()

    return importer(directory, **kwargs)


# --------------------------------------------------------------------------------------
# Private methods
# --------------------------------------------------------------------------------------
@_importer_method
def _read_carroucell(*args, **kwargs):

    _, directory = args
    directory = get_directory_name(directory)

    if not directory:  # pragma: no cover
        # probably cancel has been chosen in the open dialog
        info_("No directory was selected.")
        return

    spectra = kwargs.get("spectra", None)
    discardbg = kwargs.get("discardbg", True)
    delta_clocks = datetime.timedelta(seconds=kwargs.get("delta_clocks", 0))

    datasets = []

    # get the sorted list of spa files in the directory
    spafiles = sorted(get_filenames(directory, **kwargs)[".spa"])
    spafilespec = [f for f in spafiles if "BCKG" not in f.stem]
    spafileback = [f for f in spafiles if "BCKG" in f.stem]

    # select files
    prefix = lambda f: f.stem.split("_")[0]
    number = lambda f: int(f.stem.split("_")[1])
    if spectra is not None:
        [min, max] = spectra
        spafilespec = [f for f in spafilespec if min <= number(f) <= max]
        spafileback = [f for f in spafileback if min <= number(f) <= max]

    # discard BKG files
    spafiles = spafilespec
    if not discardbg:
        spafiles += spafileback

    # merge dataset with the same number
    curfilelist = [spafiles[0]]
    curprefix = prefix(spafiles[0])
    for f in spafiles[1:]:
        if prefix(f) != curprefix:
            ds = read_omnic(
                curfilelist, sortbydate=True, directory=directory, name=curprefix
            )
            datasets.append(ds)
            curfilelist = [f]
            curprefix = prefix(f)
        else:
            curfilelist.append(f)
    ds = read_omnic(curfilelist, sortbydate=True, directory=directory, name=curprefix)
    datasets.append(ds)

    # Now manage temperature
    Tfile = sorted([f for f in os.listdir(directory) if f[-4:].lower() == ".xls"])
    if len(Tfile) == 0:
        info_("no temperature file")
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
    return sorted(datasets, key=lambda ds: re.split("-|_", ds.name)[0])
