#  -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#
# author: Guillaume Clet (LCS)
#

"""This module extend NDDataset with the import method for Labspec *.txt generated data files.

"""

__all__ = ['read_labspec', 'read_txt']
__dataset_methods__ = __all__

import io
import datetime
import numpy as np

from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.readers.importer import docstrings, importermethod, Importer
from spectrochempy.utils import Meta


# ======================================================================================================================
# Public functions
# ======================================================================================================================

# ......................................................................................................................
@docstrings.dedent
def read_labspec(*args, **kwargs):
    """
    Converts a single Raman spectrum or a series of Raman spectra (*.txt file created by Labspec software) to a valid
    SpectroChemPy NDDataset or a list of NDDataset

    Parameters
    ----------
    args
    kwargs

    Returns
    -------
    out : NDDataset| or list of |NDDataset|
        The dataset or a list of dataset corresponding to a (set of) .txt file(s).

    Examples
    --------

    >>> A=read_txt('ramandata/Activation.txt')


    See Also
    --------
    read : Generic read method
    read_dir : Read a set of data from a directory



    """

    kwargs['filetypes'] = ['LABSPEC exported files (*.txt)']
    kwargs['protocol'] = ['labspec', 'txt']
    importer = Importer()
    return importer(*args, **kwargs)


def read_txt(*args, **kwargs):
    return read_labspec(*args, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================

# ......................................................................................................................
@importermethod
def _read_txt(*args, **kwargs):
    # read Labspec *txt files or series

    dataset, filename = args
    content = kwargs.get('content', False)

    if content:
        fid = io.StringIO(content)
        # TODO: get the l list of string

    else:
        fid = open(filename, 'r', encoding='utf-8')
        try:
            lines = fid.readlines()
        except UnicodeDecodeError:
            fid = open(filename, 'r', encoding='latin-1')
            lines = fid.readlines()
            fid.close()

    # Metadata
    meta = Meta()

    i = 0
    while lines[i].startswith('#'):
        key, val = lines[i].split('=')
        key = key[1:]
        if key in meta.keys():
            key = f'{key} {i}'
        meta[key] = val.strip()
        i += 1

    # read spec
    rawdata = np.genfromtxt(lines[i:], delimiter='\t')

    # populate the dataset
    if rawdata.shape[1] == 2:
        data = rawdata[:, 1][np.newaxis]
        _x = Coord(rawdata[:, 0], title='Raman shift', units='1/cm')
        _y = Coord(None, title='Time', units='s')
        date_acq, _y = _transf_meta(_y, meta)

    else:
        data = rawdata[1:, 1:]
        _x = Coord(rawdata[0, 1:], title='Raman shift', units='1/cm')
        _y = Coord(rawdata[1:, 0], title='Time', units='s')
        date_acq, _y = _transf_meta(_y, meta)

    # set dataset metadata
    dataset.data = data
    dataset.set_coordset(y=_y, x=_x)
    dataset.title = 'Raman Intensity'
    dataset.units = 'absorbance'
    dataset.name = filename.stem
    dataset.meta = meta

    # date_acq is Acquisition date at start (first moment of acquisition)
    dataset.description = 'Spectrum acquisition : ' + str(date_acq)

    # Set the NDDataset date
    dataset._date = datetime.datetime.now(datetime.timezone.utc)
    dataset._modified = dataset.date

    # Set origin, description and history
    dataset.history = f'{dataset.date}:imported from LabSpec6 text file {filename}'

    return dataset


def _transf_meta(y, meta):
    # Reformats some of the metadata from Labspec6 informations
    # such as the acquisition date of the spectra and returns a list with the acquisition in datetime format,
    # the list of effective dates for each spectrum

    def val_from_key_wo_time_units(k):
        for key in meta:
            h, m, s = 0, 0, 0
            if k in key:
                _, units = key.split(k)
                units = units.strip()[1:-1]
                if units == 's':
                    s = meta[key]
                elif units == 'mm:ss':
                    m, s = meta[key].split(':')
                elif units == 'hh:mm':
                    h, m = meta[key].split(':')
                break
        return datetime.timedelta(seconds=int(s), minutes=int(m), hours=int(h))

    if meta:
        try:
            dateacq = datetime.datetime.strptime(meta['Acquired'], '%d.%m.%Y %H:%M:%S')
        except TypeError:
            dateacq = datetime.datetime.strptime(meta['Date'], '%d/%m/%y %H:%M:%S')

        acq = int(meta.get('Acq. time (s)', meta['Exposition']))
        accu = int(meta.get('Accumulations', meta.get('Accumulation')))
        delay = int(meta.get('Delay time (s)', 0))
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
        print(e)

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
#     out.y.title = 'Time'
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
