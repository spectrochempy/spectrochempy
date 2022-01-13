#  -*- coding: utf-8 -*-
#
#  =====================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie,
#  Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in
#  the root directory
#  =====================================================================================================================
#
"""
This module extend NDDataset with the import method for Thermo galactic (spc) data files.
"""
__all__ = ["read_spc"]
__dataset_methods__ = __all__

from datetime import datetime, timezone, timedelta
import io
import struct

import numpy as np

from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import importermethod, Importer
from spectrochempy.units import Quantity
from spectrochempy.utils import SpectroChemPyException, SpectroChemPyWarning


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_spc(*paths, **kwargs):
    """
    Open a Thermo Glactic spc file.

    Open spc file or a list of files with extension ``.spc``
    and set data/metadata in the current dataset.

    The collected metatdata are:
    - names of spectra
    - acquisition dates (UTC)
    -
    An error is generated if attempt is made to inconsistent datasets: units
    of spectra and xaxis, limits and number of points of the xaxis.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name
        for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e.
        no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs : dict
        See other parameters.

    Returns
    --------
    out
        The dataset or a list of dataset corresponding to a (set of) .spc file(s).

    Other Parameters
    -----------------
    directory : str, optional
        From where to read the specified `filename`. If not specified,
        read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been
        provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description : str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content
        can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is
        particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the
        ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided
        `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current
        reading protocol (default=True).
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read : Generic read method.
    read_dir : Read a set of data from a directory.
    read
    read_spg : Read Omnic files *.spg.
    read_spa : Read Omnic files *.spa.
    read_srs : Read Omnic files *.srs.
    read_opus : Read Bruker OPUS files.
    read_topspin : Read TopSpin NMR files.
    read_csv : Read *.csv.
    read_matlab : Read MATLAB files *.mat.
    read_zip : Read zipped group of files.

    Examples
    ---------
    Reading a single OMNIC file  (providing a windows type filename relative
    to the default ``datadir``)

    >>> scp.read_spc('irdata\\\\nh4y-activation.spg')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))


    """


def read_spc(*paths, **kwargs):
    """
    Open a Thermo Nicolet file or a list of files with extension ``.spg``.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name
        for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e.
        no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_spc
        The dataset or a list of dataset corresponding to a (set of) .spa
        file(s).

    Other Parameters
    -----------------
    directory : str, optional
        From where to read the specified `filename`. If not specified,
        read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been
        provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description: str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content
        can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is
        particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the
        ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided
        `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current
        reading protocol (default=True).
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read : Generic read method.
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spa : Read Omnic *.spa spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Examples
    ---------
    #todo:
    # >>> scp.read_spc('irdata/subdir/20-50/7_CZ0-100 Pd_21.SPA')
    # NDDataset: [float64] a.u. (shape: (y:1, x:5549))
    # >>> scp.read_spa(directory='irdata/subdir', merge=True)
    # NDDataset: [float64] a.u. (shape: (y:4, x:5549))
    """

    kwargs["filetypes"] = ["GRAMS/Thermo Galactic files (*.spc)"]
    kwargs["protocol"] = ["spc"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================

# constants

VERSION_SUPPORTED = (b'\x4b')   #this is the last version (1997)

# old_head_str = "<cchfffcchcccc8shh28s130s30s32s"             # 256
logstc_str = "<iiiii44s"

#
subhead_siz = 32
log_siz = 64


# ..............................................................................
@importermethod
def _read_spc(*args, **kwargs):
    dataset, filename = args
    content = kwargs.get("content", False)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, "rb")
        content = fid.read()

    # extract the header (see: Galactic Universal Data Format Specification 9/4/97)
    # from SPC.H Header File:
    # typedef struct
    # {
    # BYTE ftflgs; /* Flag bits defined below */
    # BYTE fversn; /* 0x4B=> new LSB 1st, 0x4C=> new MSB 1st, 0x4D=> old format */
    # BYTE fexper; /* Instrument technique code (see below) */
    # char fexp; /* Fraction scaling exponent integer (80h=>float) */
    # DWORD fnpts; /* Integer number of points (or TXYXYS directory position) */
    # double ffirst; /* Floating X coordinate of first point */
    # double flast; /* Floating X coordinate of last point */
    # DWORD fnsub; /* Integer number of subfiles (1 if not TMULTI) */
    # BYTE fxtype; /* Type of X axis units (see definitions below) */
    # BYTE fytype; /* Type of Y axis units (see definitions below) */
    # BYTE fztype; /* Type of Z axis units (see definitions below) */
    # BYTE fpost; /* Posting disposition (see GRAMSDDE.H) */
    # DWORD fdate; /* Date/Time LSB: min=6b,hour=5b,day=5b,month=4b,year=12b */
    # char fres[9]; /* Resolution description text (null terminated) */
    # char fsource[9]; /* Source instrument description text (null terminated) */
    # WORD fpeakpt; /* Peak point number for interferograms (0=not known) */
    # float fspare[8]; /* Used for Array Basic storage */
    # char fcmnt[130]; /* Null terminated comment ASCII text string */
    # char fcatxt[30]; /* X,Y,Z axis label strings if ftflgs=TALABS */
    # DWORD flogoff; /* File offset to log block or 0 (see above) */
    # DWORD fmods; /* File Modification Flags (see below: 1=A,2=B,4=C,8=D..) */
    # BYTE fprocs; /* Processing code (see GRAMSDDE.H) */
    # BYTE flevel; /* Calibration level plus one (1 = not calibration data) */
    # WORD fsampin; /* Sub-method sample injection number (1 = first or only ) */
    # float ffactor; /* Floating data multiplier concentration factor (IEEE-32) */
    # char fmethod[48]; /* Method/program/data filename w/extensions comma list */
    # float fzinc; /* Z subfile increment (0 = use 1st subnext-subfirst) */
    # DWORD fwplanes; /* Number of planes for 4D with W dimension (0=normal) */
    # float fwinc; /* W plane increment (only if fwplanes is not 0) */
    # BYTE fwtype; /* Type of W axis units (see definitions below) */
    # char freserv[187]; /* Reserved (must be set to zero) */
    # } SPCHDR;
    #define SPCHSZ sizeof(SPCHDR) /* Size of spectrum header for disk file. */

    Ftflgs, Fversn,  Fexper, Fexp, Fnpts, Ffirst, Flast, Fnsub, Fxtype, Fytype, Fztype, Fpost, Fdate, Fres, Fsource, \
    Fpeakpt, Fspare, Fcmnt, Fcatxt, Flogoff, Fmods, Fprocs, Flevel, Fsampin, Ffactor, Fmethod, Fzinc, Fwplanes, Fwinc, \
    Fwtype, Freserv = struct.unpack("<cccciddicccci9s9sh32s130s30siicchf48sfifc187s".encode('utf8'), content[:512])

    # extract bit flags
    tsprec, tcgram, tmulti, trandm, tordrd, talabs, txyxys, txvals \
        = [x == '1' for x in reversed(list('{0:08b}'.format(ord(Ftflgs))))]

    #  Flag      Value   Description
    # TSPREC     0x01h   Y data blocks are 16 bit integer (only if fexp is NOT 0x80h)
    # TCGRAM     0x02h   Enables fexper in older software (not used)
    # TMULTI     0x04h   Multifile data format (more than one subfile)
    # TRANDM     0x08h   If TMULTI and TRANDM then Z values in SUBHDR structures are in random order (not used)
    # TORDRD     0x10h   If TMULTI and TORDRD then Z values are in ascending or descending ordered but not evenly spaced.
    #                    Z values read from individual SUBHDR structures.
    # TALABS     0x20h   Axis label text stored in fcatxt separated by nulls. Ignore fxtype, fytype, fztype corresponding
    #                    to non-null text in fcatxt.
    # TXYXYS     0x40h   Each subfile has unique X array; can only be used if TXVALS is also used. Used exclusively
    #                     to flag as MS data for drawing as “sticks” rather than connected lines.
    # TXVALS     0x80h   Non-evenly spaced X data. File has X value array preceding Y data block(s).

    # check spc version
    if Fversn !=  b'\x4b':
        raise SpectroChemPyException(f"The version {Fversn} is not yet supported. "
              f"Current supported versions is {VERSION_SUPPORTED}.")


    techniques = ["General SPC", "Gas Chromatogram", "General Chromatogram", "HPLC Chromatogram", "FT-IR, FT-NIR, "
                  "FT-Raman Spectrum or Igram", "NIR Spectrum", "UV-VIS Spectrum", "X-ray Diffraction Spectrum",
                  "Mass Spectrum ", "NMR Spectrum or FID", "Raman Spectrum", "Fluorescence Spectrum", "Atomic Spectrum",
                 "Chromatography Diode Array Spectra"]

    technique = techniques[int.from_bytes(Fexper, "little")]

    x_or_z_title = ['axis title', 'Wavenbumbers', 'Wavelength', 'Wavelength', 'Time', 'Time', 'Frequency', 'Frequency',
                    'Frequency', 'm/z', 'Chemical shift', 'Time', 'Time', 'Raman shift', 'Energy', 'text_label',
                    'diode number', 'Channel', '2 theta', 'Temperature', 'Temperature', 'Temperature', 'Data Points',
                    'Time', 'Time', 'Time', 'Frequency', 'Wavelength', 'Wavelength', 'Wavelength', 'Time']

    x_or_z_unit = [None, "cm^-1", "um", "nm", "s", "min", "Hz", "kHz",
                   "MHz", "g/(mol * e)", "ppm", "days", "years", "cm^-1", "eV", None,
                   None, None, "degree", "fahrenheit", "celsius", "kelvin", None,
                   "ms", "us", "ns", "GHz", "cm", "m", "mm", "hour"]

    ixtype = int.from_bytes(Fxtype, "little")
    if ixtype != 255:
        x_unit = x_or_z_unit[ixtype]
        x_title = x_or_z_title[ixtype]
    else:
        x_unit = None
        x_title = 'Double interferogram'

    iztype = int.from_bytes(Fztype, 'little')
    if iztype != 255:
        z_unit = x_or_z_unit[iztype]
        z_title = x_or_z_title[iztype]
    else:
        z_unit = None
        z_title = 'Double interferogram'

    y_title = ['Arbitrary Intensity', 'Interferogram', 'Absorbance', 'Kubelka-Munk', 'Counts', 'Voltage', 'Angle',
              'Intensity', 'Length', 'Voltage', 'Log(1/R)', 'Transmittance', 'Intensity', 'Relative Intensity',
              'Energy', None, 'Decibel', None, None, 'Temperature', 'Temperature', 'Temperature',
              'Index of Refraction [N]', 'Extinction Coeff. [K]', 'Real', 'Imaginary', 'Complex']

    y_unit= [None, None, "absorbance", "Kubelka_Munk", None, "Volt", "degree",
             "mA", "mm", "mV", None, "percent", None, None,
             None, None, "dB", None, None, "fahrenheit", "celsius", "kelvin",
             None, None, None, None, None]

    iytype = int.from_bytes(Fytype, "little")
    if iytype < 128:
        y_unit = y_unit[iytype]
        y_title = y_title[iytype]

    elif iytype == 128:
        y_unit = None
        y_title = 'Transmission'

    elif iytype == 129:
        y_unit = None
        y_title = 'Reflectance'

    elif iytype == 130:
        y_unit = None
        y_title = 'Arbitrary or Single Beam with Valley Peaks'

    elif iytype == 131:
        y_unit_unit = None
        y_title = 'Emission'

    else:
        SpectroChemPyWarning('Wrong y unit label code in the SPC file. It will be set to arbitrary intensity')
        y_unit_unit = None
        y_title = 'Arbitrary Intensity'

    if Fexp == b'\x80':
        iexp = None     # floating Point Data
    else:
        iexp =  int.from_bytes(Fexp, "little")   # Datablock scaling Exponent


    # set date (from https://github.com/rohanisaac/spc/blob/master/spc/spc.py)
    year = Fdate >> 20
    month = (Fdate >> 16) % (2 ** 4)
    day = (Fdate >> 11) % (2 ** 5)
    hour = (Fdate >> 6) % (2 ** 5)
    minute = Fdate % (2 ** 6)

    acqdate = datetime(year, month, day, hour, minute)
    timestamp = acqdate.timestamp()

    sres = Fres.decode('utf-8')
    ssource = Fsource.decode('utf-8')

    scmnt = Fcmnt.decode('utf-8')
    scatxt = Fcatxt.decode('utf-8')

    if Fwplanes:
        iwtype = int.from_bytes(Fwtype, 'little')
        if iwtype != 255:
            w_unit = x_or_z_unit[ixtype]
            w_title = x_or_z_title[ixtype]
        else:
            w_unit = None
            w_title = 'Double interferogram'

    if Fnsub > 1:
        raise NotImplementedError('spc reader not implemented yet for multifiles. If you need it, please '
                                  'submit a feature request on spectrochempy repository :-)')

    if not txvals:  # evenly spaced x data
        spacing = (Flast - Ffirst) / (Fnpts - 1)
        _x = LinearCoord(
            offset=Ffirst,
            increment=spacing,
            size=Fnpts,
            title=x_title,
            units=x_unit,
        )
    else:
        raise NotImplementedError('spc reader not implemented yet for unevenly spaced X values. If you need it, please '
                                  'submit a feature request on spectrochempy repository :-)')


    if iexp is None:
        # 32-bit IEEE floating numbers
        floatY = np.frombuffer(content[544: 544 + Fnpts * 4], np.float32)
    else:
        # fixed point signed fractions
        if tsprec:
            integerY = np.frombuffer(content[544: 544 + Fnpts * 4], np.int16)
            floatY = (2**iexp) * integerY / (2**16)
        else:
            integerY = np.frombuffer(content[544: 544 + Fnpts * 4], np.int32)
            floatY = (2 ** iexp) * integerY / (2 ** 32)


    # Create NDDataset Object for the series
    dataset = NDDataset(floatY)
    dataset.name = 'name'
    dataset.units = y_unit
    dataset.title = y_title
    dataset.origin = "thermo galactic"

    # now add coordinates

    _y = Coord(
        [timestamp],
        title="acquisition timestamp (GMT)",
        units="s",
        labels=([acqdate], [filename]))

    dataset.set_coordset(y=_y, x=_x)

    dataset.description = kwargs.get("description", "Dataset from spc file.")

    dataset.history = str(
        datetime.now(timezone.utc)
    ) + ":imported from srs file {} ; ".format(filename)

    if y_unit == 'Interferogram':
        # interferogram
        dataset.meta.interferogram = True
        dataset.meta.td = list(dataset.shape)
        dataset.x._zpd = Fpeakpt
        dataset.meta.laser_frequency = Quantity("15798.26 cm^-1")
        dataset.x.set_laser_frequency()
        dataset.x._use_time_axis = (
            False  # True to have time, else it will be optical path difference
        )

    # uncomment below to load the last datafield
    # has the same dimension as the time axis
    # its function is not known. related to Grams-schmidt ?

    # pos = _nextline(pos)
    # found = False
    # while not found:
    #     pos += 16
    #     f.seek(pos)
    #     key = _fromfile(f, dtype='uint8', count=1)
    #     if key == 1:
    #         pos += 4
    #         f.seek(pos)
    #         X = _fromfile(f, dtype='float32', count=info['ny'])
    #         found = True
    #
    # X = NDDataset(X)
    # _x = Coord(np.around(np.linspace(0, info['ny']-1, info['ny']), 0),
    #            title='time',
    #            units='minutes')
    # X.set_coordset(x=_x)
    # X.name = '?'
    # X.title = '?'
    # X.origin = 'omnic'
    # X.description = 'unknown'
    # X.history = str(datetime.now(timezone.utc)) + ':imported from srs

    fid.close()

    return dataset

    return NDDataset()


# ..............................................................................
def _fromfile(fid, dtype, count):
    # to replace np.fromfile in case of io.BytesIO object instead of byte object
    t = {
        "uint8": "B",
        "int8": "b",
        "uint16": "H",
        "int16": "h",
        "uint32": "I",
        "int32": "i",
        "float32": "f",
    }
    typ = t[dtype] * count
    if dtype.endswith("16"):
        count = count * 2
    elif dtype.endswith("32"):
        count = count * 4

    out = struct.unpack(typ, fid.read(count))
    if len(out) == 1:
        return out[0]
    return np.array(out)


# ..............................................................................
def _readbtext(fid, pos):
    # Read some text in binary file, until b\0\ is encountered.
    # Returns utf-8 string
    fid.seek(pos)  # read first byte, ensure entering the while loop
    btext = fid.read(1)
    while not (btext[len(btext) - 1] == 0):  # while the last byte of btext differs from
        # zero
        btext = btext + fid.read(1)  # append 1 byte

    btext = btext[0 : len(btext) - 1]  # cuts the last byte
    try:
        text = btext.decode(encoding="utf-8")  # decode btext to string
    except UnicodeDecodeError:
        try:
            text = btext.decode(encoding="latin_1")
        except UnicodeDecodeError:
            text = btext.decode(encoding="utf-8", errors="ignore")
    return text



# ..............................................................................
def _nextline(pos):
    # reset current position to the begining of next line (16 bytes length)
    return 16 * (1 + pos // 16)


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
