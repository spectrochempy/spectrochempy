# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Extend NDDataset with the import method for Labspec *.txt generated data files."""

__all__ = ["read_labspec"]

import datetime

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.utils._logging import error_
from spectrochempy.utils.meta import Meta


# ======================================================================================
# Public functions
# ======================================================================================
def read_labspec(*paths, **kwargs):
    r"""
    Open Raman LABSPEC files.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ..., kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to ``False``.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    object : `NDDataset` or list of `NDDataset`
        The returned dataset(s).

    Other Parameters
    ----------------
    content : `bytes` object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly
        useful for a GUI Dash application to handle drag and drop of files into a
        Browser.
    csv_delimiter : `str`, optional, default: `~spectrochempy.preferences.csv_delimiter`
        Set the column delimiter in CSV file.
    description : `str`, optional
        A custom description.
    directory : `~pathlib.Path` object objects or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `True` and several filenames or a ``directory`` have been provided as
        arguments, then a single `NDDataset` with merged dataset (stacked along the first
        dimension) is returned. In the case not all datasets have compatible dimensions or types/origins,
        then several NDDatasets can be returned for different groups of compatible datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used, for instance, when reading a directory with different types of
        files and merging compatible datasets into separate groups by origin.

        It is also used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading, for example ``'scp'``, ``'omnic'``,
        ``'opus'``, ``'matlab'``, ``'jcamp'``, ``'csv'``, or ``'excel'``.
        If not provided, the correct protocol is inferred whenever possible
        from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic reader inferring protocol from the filename extension.
    read_zip : Read Zip archives (containing spectrochempy readable files)
    read_dir : Read an entire directory.
    read_opus : Read OPUS spectra.
    read_omnic : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    read_soc : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    read_galactic : Read Galactic files (:file:`.spc`).
    read_quadera : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.

    read_csv : Read CSV files (:file:`.csv`).
    read_matlab : Read Matlab files (:file:`.mat`, :file:`.dso`).
    read_jcamp : Read Infrared JCAMP-DX files (:file:`.jdx`, :file:`.dx`).
    read_wire : Read Renishaw Wire files (:file:`.wdf`).

    Examples
    --------
    Reading a single LABSPEC file

    >>> scp.read_labspec('irdata/labspec.txt')
    NDDataset: [float64] a.u. (shape: (y:1, x:1024))

    """
    kwargs["filetypes"] = ["Labspec files (*.txt)"]
    kwargs["protocol"] = ["labspec"]
    importer = Importer()
    return importer(*paths, **kwargs)


read_txt = read_labspec


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_txt(*args, **kwargs):
    # read Labspec *txt files or series

    dataset, filename = args
    content = kwargs.get("content", False)

    if content:
        try:
            lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            lines = content.decode("latin-1").splitlines()
    else:
        try:
            with open(filename, encoding="utf-8") as fid:
                lines = fid.readlines()
        except UnicodeDecodeError:
            with open(filename, encoding="latin-1") as fid:
                lines = fid.readlines()

    if len(lines) == 0:
        return None

    # Metadata
    meta = Meta()

    i = 0
    while lines[i].startswith("#"):
        key, val = lines[i].split("=")
        key = key[1:]
        if key in meta:
            key = f"{key} {i}"
        meta[key] = val.strip()
        i += 1

    # .txt extension is fairly common. We determine non labspc files based
    # on the absence of few keys. Two types of files (1D or 2D) are considered:
    labspec_keys_1D = ["Acq. time (s)", "Dark correction"]
    labspec_keys_2D = ["Exposition", "Grating"]

    if all(keywd in meta for keywd in labspec_keys_1D) or all(
        keywd in meta for keywd in labspec_keys_2D
    ):
        pass
    else:
        # this is not a labspec txt file"
        return None

    # read spec
    rawdata = np.genfromtxt(lines[i:], delimiter="\t")

    # populate the dataset
    if rawdata.shape[1] == 2:
        data = rawdata[:, 1][np.newaxis]
        _x = Coord(rawdata[:, 0], title="Raman shift", units="1/cm")
        _y = Coord(None, title="Time", units="s")
        date_acq, _y = _transf_meta(_y, meta)

    else:
        data = rawdata[1:, 1:]
        _x = Coord(rawdata[0, 1:], title="Raman shift", units="1/cm")
        _y = Coord(rawdata[1:, 0], title="Time", units="s")
        date_acq, _y = _transf_meta(_y, meta)

    # set dataset metadata
    dataset.data = data
    dataset.set_coordset(y=_y, x=_x)
    dataset.title = "Counts"
    dataset.units = None
    dataset.name = filename.stem
    dataset.filename = filename
    dataset.meta = meta

    # date_acq is Acquisition date at start (first moment of acquisition)
    dataset.description = "Spectrum acquisition : " + str(date_acq)

    # Set origin, description and history
    dataset.history = f"Imported from LabSpec6 text file {filename}"

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset


def _transf_meta(y, meta):
    # Reformats some of the metadata from Labspec6 information
    # such as the acquisition date of the spectra and returns a list with the acquisition in datetime format,
    # the list of effective dates for each spectrum

    # def val_from_key_wo_time_units(k):
    #     for key in meta:
    #         h, m, s = 0, 0, 0
    #         if k in key:
    #             _, units = key.split(k)
    #             units = units.strip()[1:-1]
    #             if units == 's':
    #                 s = meta[key]
    #             elif units == 'mm:ss':
    #                 m, s = meta[key].split(':')
    #             elif units == 'hh:mm':
    #                 h, m = meta[key].split(':')
    #             break
    #     return datetime.timedelta(seconds=int(s), minutes=int(m), hours=int(h))

    if meta:
        try:
            dateacq = datetime.datetime.strptime(meta["Acquired"], "%d.%m.%Y %H:%M:%S")
        except TypeError:
            dateacq = datetime.datetime.strptime(meta["Date"], "%d/%m/%y %H:%M:%S")

        acq = int(meta.get("Acq. time (s)", meta["Exposition"]))
        accu = int(meta.get("Accumulations", meta.get("Accumulation")))
        delay = int(meta.get("Delay time (s)", 0))
        # total = val_from_key_wo_time_units('Full time')

    else:
        dateacq = datetime.datetime(2000, 1, 1, 0, 0, 0)
        # datestr = '01/01/2000 00:00:00'
        acq = 0
        accu = 0
        delay = 0
        # total = datetime.timedelta(0)

    # delay between spectra
    delayspectra = datetime.timedelta(seconds=acq * accu + delay)

    # Date d'acquisition : le temps indiqué est celui où l'on démarre l'acquisition
    dateacq = dateacq - delayspectra

    # Dates effectives de chacun des spectres de la série : le temps indiqué est celui où l'on démarre l'acquisition
    # Labels for time : dates with the initial time for each spectrum
    try:
        y.labels = [dateacq + delayspectra * i for i in range(len(y))]
    except Exception as e:
        error_(e)

    return dateacq, y


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
#     out.y.title = 'time'
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
