#  -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

#
#
# author: Guillaume Clet (LCS)
#

"""
This module extend NDDataset with the import method for Labspec *.txt generated data files.
"""

__all__ = ["read_labspec"]
__dataset_methods__ = __all__

import numpy as np
from datetime import datetime

from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.core.readers.importer import importermethod, Importer
from spectrochempy.core.dataset.meta import Meta
from spectrochempy.utils.datetimeutils import strptime64

# from spectrochempy.utils.exceptions import deprecated


# ======================================================================================
# Public functions
# ======================================================================================
def read_labspec(*paths, **kwargs):
    """
    Read a single Raman spectrum or a series of Raman spectra.

    Files to open are *.txt file created by Labspec software. Non-labspec .txt files are ignored (return None)

    Parameters
    ----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    read
        |NDDataset| or list of |NDDataset| or None.

    Other Parameters
    ----------------
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False)
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True)
    comment : str, optional
        A Custom comment.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For examples on how to use this feature, one can look in the ``tests/tests_readers`` directory
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True)
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read : Generic read method.
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
    >>> A = scp.read_labspec('ramandata/Activation.txt')
    """

    kwargs["filetypes"] = ["LABSPEC exported files (*.txt)"]
    kwargs["protocol"] = ["labspec", "txt"]
    importer = Importer()
    return importer(*paths, **kwargs)


read_txt = read_labspec


# ======================================================================================
# Private functions
# ======================================================================================

# ..............................................................................
@importermethod
def _read_txt(*args, **kwargs):
    # read Labspec *txt files or series

    dataset, filename = args
    content = kwargs.get("content", False)

    if content:
        pass
        # fid = io.StringIO(content)
        # TODO: get the l list of string

    else:
        fid = open(filename, "r", encoding="utf-8")
        try:
            lines = fid.readlines()
        except UnicodeDecodeError:
            fid = open(filename, "r", encoding="latin-1")
            lines = fid.readlines()
            fid.close()

    if len(lines) == 0:
        return

    # Metadata
    meta = Meta()

    i = 0
    while lines[i].startswith("#"):
        key, val = lines[i].split("=")
        key = key[1:]
        if key in meta.keys():
            key = f"{key} {i}"
        meta[key] = val.strip()
        i += 1

    # .txt extension is fairly common. We determine non labspc files based
    # on the absence of few keys. Two types of files (1D or 2D) are considered:
    labspec_keys_1D = ["Acq. time (s)", "Dark correction"]
    labspec_keys_2D = ["Exposition", "Grating"]

    if all(keywd in meta.keys() for keywd in labspec_keys_1D):
        pass
    elif all(keywd in meta.keys() for keywd in labspec_keys_2D):
        pass
    else:
        # this is not a labspec txt file"
        return

    # read raw data
    rawdata = np.genfromtxt(lines[i:], delimiter="\t")

    # get some info for times
    acquired = strptime64(meta.get("Acquired", meta.Date))
    # date of acquisition
    delay = np.timedelta64(meta.get("Delay time (s)", 0), "s")
    acq = np.timedelta64(meta.get("Acq. time (s)", meta.Exposition), "s")
    accu = int(meta.get("Accumulations", meta.Accumulation))

    # delay between spectra (quite approximative as the delay,acq etc...
    # are int number representing second) - And this assume that the
    # acquisition of the spectra was very regular but it is not!)
    delayspectra = acq * accu + delay

    # populate the dataset
    if rawdata.shape[1] == 2:  # single dimension case
        data = rawdata[:, 1][np.newaxis]
        xdata = rawdata[:, 0]
        ydata = np.array([0]).astype("timedelta64[us]")

    else:
        data = rawdata[1:, 1:]
        xdata = rawdata[0, 1:]
        ydata = (rawdata[1:, 0] * 1.0e6).astype("timedelta64[us]")

    # transform y  timedata to datetime64 using the acquired date a basis
    ydata = (acquired - delayspectra) + ydata

    _x = Coord(xdata, long_name="Raman shift", units="1/cm")
    _y = Coord(ydata, long_name="Time", units=None)

    # try to transform to linear coord
    _x.linear = True

    # if success linear should still be True
    if _x.linear:
        _x = LinearCoord(_x)

    # set dataset metadata
    dataset.data = data
    dataset.set_coordset(y=_y, x=_x)
    dataset.long_name = "Count"
    dataset.units = None
    dataset.name = filename.stem
    dataset.meta = meta

    # date_acq is Acquisition date at start (first moment of acquisition)
    dataset.comment = f"Spectrum acquisition : " f"{acquired.astype('datetime64[m]')}"

    # Set the NDDataset date
    dataset._created = datetime.utcnow()
    dataset._modified = dataset._created

    # Set origin, comment and history
    dataset.history = f"Imported from LabSpec6 text file {filename}"

    return dataset


# TODO: save in this format ? See below.

#
# @deprecated('too inaccurate')
# def _transf_meta(y, meta):
#     # Reformats some of the metadata from Labspec6 information
#     # such as the acquisition date of the spectra and returns a list with the acquisition in datetime format,
#     # the list of effective dates for each spectrum
#
#     # IMPORTANT
#     # ---------
#     # because the number seems to be
#     # saved as integer ! this method gives a very bad precision - it seems
#     # better to base all this calculation on the y coordinates
#
#     if meta:
#
#         dateacq = strptime64(
#             meta.get("Acquired", meta.Date)
#         )
#
#         acq = np.timedelta64(meta.get("Acq. time (s)", meta.Exposition), "s")
#         delay = np.timedelta64(meta.get("Delay time (s)", 0), "s")
#
#         accu = int(meta.get("Accumulations", meta.Accumulation))
#         # total = val_from_key_wo_time_units('Full time')
#
#     else:
#         dateacq = np.datetime64("2000:01.01T00:00:00", "us")
#
#         acq = np.timedelta64(0, "s")
#         delay = np.timedelta64(0, "s")
#         accu = 0
#         # total = timedelta(0)
#
#     # Delay between spectra
#     delayspectra = acq * accu + delay   # <--- this seems to be very
#     # imprecise as acq is given in second (int. number)
#
#     # Acquisition date: the time indicated is when the acquisition is started
#     dateacq = dateacq - delayspectra
#
#     # Effective dates of each of the spectra in the series: the time
#     # indicated is the time when the acquisition is started
#     # Labels for time : dates with the initial time for each spectrum
#     try:
#         y.labels = np.arange(y.size) * delayspectra + dateacq
#     except Exception as e:
#         print(e)
#
#     return dateacq, y
#

# def rgp_series(lsdatasets, sortbydate=True):
#     """
#     Concatenation of individual spectra to an integrated series
#
#     :type lsdatasets: list
#         list of datasets (usually created after opening several spectra at once)
#
#     :type sortbydate: bool
#         to sort data by date order (default=True)
#
#     :type out: NDDataset
#         single dataset after grouping
#     """
#
#     # Test and initialize
#     if (type(lsdatasets) != list):
#         print('Error : A list of valid NDDatasets is expected')
#         return
#
#     lsfile = list(lsdatasets[i].name for i in range(len(lsdatasets)))
#
#     out = stack(lsdatasets)
#
#     lstime = out.y.labels
#     out.y.data = lstime
#
#     # Orders by date and calculates relative times
#     if sortbydate:
#         out = out.sort(dim='y')
#
#     lstime = []
#     ref = out[0].y.labels[0]
#     for i in range(out.shape[0]):
#         time = (out[i].y.labels[0] - ref).seconds
#         lstime.append((time))
#
#     # Formats the concatenated dataset
#     labels = out.y.labels
#     out.y = lstime
#     out.y.labels = labels, lsfile
#     out.y.long_name = 'time'
#     out.y.units = 's'
#     out.name = 'Series concatenated'
#
#     return out
#
#
# ## Saving data
#
# def reconstruct_data(dataset):
#     """
#     Recreates raw data matrix from the values of X,Y and data of a NDDataset
#     """
#     dim0, dim1 = dataset.shape
#     rawdata = np.zeros((dim0 + 1, dim1 + 1))
#     rawdata[0, 0] = None
#     rawdata[1::, 0] = dataset.y
#     rawdata[0, 1::] = dataset.x
#     rawdata[1::, 1::] = np.copy(dataset.data)
#
#     metalist = []
#     metakeysmod = {}
#     for ky in dataset.meta.keys():  # writes metadata in the same order as Labspec6
#         if ky != 'ordre Labspec6':
#             ind = dataset.meta['ordre Labspec6'][ky]
#             kymod = str(ind) + ky
#             metakeysmod[kymod] = dataset.meta[ky]
#     for ky2 in sorted(metakeysmod.keys()):
#         metalist.append(ky2[ky2.find('#'):] + '=' + metakeysmod[ky2])
#
#     return rawdata, metalist
#
#
# def save_txt(dataset, filename=''):
#     """ Exports dataset to txt format, aiming to be readable by Labspec software
#     Only partially efficient. Loss of metadata
#     """
#     # if no filename is provided, opens a dialog box to create txt file
#     if filename == '':
#         root = tk.Tk()
#         root.withdraw()
#         root.overrideredirect(True)
#         root.geometry('0x0+0+0')
#         root.deiconify()
#         root.lift()
#         root.focus_force()
#         f = filedialog.asksaveasfile(initialfile='dataset',
#                                      defaultextension=".txt",
#                                      filetypes=[("Text", "*.txt"), ("All Files", "*.*")],
#                                      confirmoverwrite=True)
#         if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
#             return
#         root.destroy()
#     else:
#         f = filename
#
#     rawdata, metalist = reconstruct_data(dataset)
#
#     with open('savetxt.txt', 'w') as f:
#         # After leaving the above block of code, the file is closed
#         f.writelines(metalist)
#         np.savetxt(f, rawdata, delimiter='\t')
#
#     return
#
#
# # Data treatment
#
# def elimspikes(self, seuil=0.03):
#     """Spikes removal tool
#
#     :seuil: float : minimal threshold for the detection(fraction)"""
#
#     out = self.copy()
#     self.plot(reverse=False)
#     for k in range(3):
#         outmd = out.data[:, 1:-1:]  # median point
#         out1 = out.data[:, 0:-2:]  # previous point
#         out2 = out.data[:, 2::]  # next point
#
#         test1 = (outmd > (1 + seuil) * out1)
#         test2 = (outmd > (1 + seuil) * out2)
#         test = (test1 & test2)
#         outmd[test] = (out1[test] + out2[test]) / 2
#         out.data[:, 1:-1:] = outmd
#     out.name = '*' + self.name
#     out.history = 'Spikes removed by elimspikes(), with a fraction threshold value=' + str(seuil)
#     out.plot(reverse=False)
#     return out
