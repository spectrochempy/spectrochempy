# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Extend NDDataset with the import method for Thermo galactic (spc) data files."""

__all__ = ["read_spc"]
__dataset_methods__ = __all__

import struct
from datetime import datetime
from warnings import warn

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid
from spectrochempy.core.units import Quantity
from spectrochempy.utils.docutils import docprocess

# ======================================================================================
# Public functions
# ======================================================================================
docprocess.delete_params("Importer.see_also", "read_spc")


@docprocess.dedent
def read_spc(*paths, **kwargs):
    r"""
    Read GRAMS/Thermo Scientific Galactic files or a list of files with extension :file:`.spc`.

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    -------
    %(Importer.returns)s

    Other Parameters
    ----------------
    %(Importer.other_parameters)s

    See Also
    --------
    %(Importer.see_also.no_read_spc)s

    Examples
    --------
    >>> scp.read_spc("galacticdata/BENZENE.spc")
    NDDataset: [float64] a.u. (shape: (y:1, x:1842))

    """
    kwargs["filetypes"] = ["GRAMS/Thermo Galactic files (*.spc)"]
    kwargs["protocol"] = ["spc"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_spc(*args, **kwargs):
    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    content = fid.read()

    # extract version
    _, Fversn = struct.unpack(b"cc", content[:2])

    # check spc version
    if Fversn == b"\x4b":
        endian = "little"
        head_format = "<cccciddicccci9s9sh32s130s30siicchf48sfifc187s"
        logstc_format = "<iiiiic"
        float32_dtype = "<f4"
        int16_dtype = "<i2"
        int32_dtype = "<i4"
    elif Fversn == b"\x4c":
        endian = "big"
        head_format = ">cccciddicccci9s9sh32s130s30siicchf48sfifc187s"
        logstc_format = ">iiiiic"
        float32_dtype = ">f4"
        int16_dtype = ">i2"
        int32_dtype = ">i4"
    else:
        raise NotImplementedError(
            f"The version {Fversn} is not yet supported. "
            f"Currently supported versions are b'\x4b' and b'\x4c'.",
        )

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

    (
        Ftflgs,
        Fversn,
        Fexper,
        Fexp,
        Fnpts,
        Ffirst,
        Flast,
        Fnsub,
        Fxtype,
        Fytype,
        Fztype,
        Fpost,
        Fdate,
        Fres,
        Fsource,
        Fpeakpt,
        Fspare,
        Fcmnt,
        Fcatxt,
        Flogoff,
        Fmods,
        Fprocs,
        Flevel,
        Fsampin,
        Ffactor,
        Fmethod,
        Fzinc,
        Fwplanes,
        Fwinc,
        Fwtype,
        Freserv,
    ) = struct.unpack(head_format.encode("utf8"), content[:512])

    # check compatibility with current implementation
    if Fnsub > 1:
        raise NotImplementedError(
            "spc reader not implemented yet for multifiles. If you need it, please "
            "submit a feature request on spectrochempy repository :-)",
        )

    # extract bit flags
    tsprec, tcgram, tmulti, trandm, tordrd, talabs, txyxys, txvals = (
        x == "1" for x in reversed(list(f"{ord(Ftflgs):08b}"))
    )

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

    techniques = [
        "General SPC",
        "Gas Chromatogram",
        "General Chromatogram",
        "HPLC Chromatogram",
        "FT-IR, FT-NIR, FT-Raman Spectrum",
        "NIR Spectrum",
        "UV-VIS Spectrum",
        None,
        "X-ray Diffraction Spectrum",
        "Mass Spectrum ",
        "NMR Spectrum or FID",
        "Raman Spectrum",
        "Fluorescence Spectrum",
        "Atomic Spectrum",
        "Chromatography Diode Array Spectra",
    ]

    technique = techniques[int.from_bytes(Fexper, endian)]

    if talabs:
        warn(
            "The SPC file has custom Unit Labels, but spc_reader does not yet take them into account "
            "and will use defaults. "
            "If needed let us know and submit a feature request :) ",
            stacklevel=2,
        )

    x_or_z_title = [
        "axis title",
        "Wavenbumbers",
        "Wavelength",
        "Wavelength",
        "Time",
        "Time",
        "Frequency",
        "Frequency",
        "Frequency",
        "m/z",
        "Chemical shift",
        "Time",
        "Time",
        "Raman shift",
        "Energy",
        "text_label",
        "diode number",
        "Channel",
        "2 theta",
        "Temperature",
        "Temperature",
        "Temperature",
        "Data Points",
        "Time",
        "Time",
        "Time",
        "Frequency",
        "Wavelength",
        "Wavelength",
        "Wavelength",
        "Time",
    ]

    x_or_z_unit = [
        None,
        "cm^-1",
        "um",
        "nm",
        "s",
        "min",
        "Hz",
        "kHz",
        "MHz",
        "g/(mol * e)",
        "ppm",
        "days",
        "years",
        "cm^-1",
        "eV",
        None,
        None,
        None,
        "degree",
        "fahrenheit",
        "celsius",
        "kelvin",
        None,
        "ms",
        "us",
        "ns",
        "GHz",
        "cm",
        "m",
        "mm",
        "hour",
    ]

    ixtype = int.from_bytes(Fxtype, endian)
    if ixtype != 255:
        x_unit = x_or_z_unit[ixtype]
        x_title = x_or_z_title[ixtype]
    else:
        x_unit = None
        x_title = "Double interferogram"

    # if Fnsub > 1:
    #     iztype = int.from_bytes(Fztype, endian)
    #     if iztype != 255:
    #         z_unit = x_or_z_unit[iztype]
    #         z_title = x_or_z_title[iztype]
    #     else:
    #         z_unit = None
    #         z_title = "Double interferogram"

    y_title = [
        "Arbitrary Intensity",
        "Interferogram",
        "Absorbance",
        "Kubelka-Munk",
        "Counts",
        "Voltage",
        "Angle",
        "Intensity",
        "Length",
        "Voltage",
        "Log(1/R)",
        "Transmittance",
        "Intensity",
        "Relative Intensity",
        "Energy",
        None,
        "Decibel",
        None,
        None,
        "Temperature",
        "Temperature",
        "Temperature",
        "Index of Refraction [N]",
        "Extinction Coeff. [K]",
        "Real",
        "Imaginary",
        "Complex",
    ]

    y_unit = [
        None,
        None,
        "absorbance",
        "Kubelka_Munk",
        None,
        "Volt",
        "degree",
        "mA",
        "mm",
        "mV",
        None,
        "percent",
        None,
        None,
        None,
        None,
        "dB",
        None,
        None,
        "fahrenheit",
        "celsius",
        "kelvin",
        None,
        None,
        None,
        None,
        None,
    ]

    iytype = int.from_bytes(Fytype, endian)
    if iytype < 128:
        y_unit = y_unit[iytype]
        y_title = y_title[iytype]

    elif iytype == 128:
        y_unit = None
        y_title = "Transmission"

    elif iytype == 129:
        y_unit = None
        y_title = "Reflectance"

    elif iytype == 130:
        y_unit = None
        y_title = "Arbitrary or Single Beam with Valley Peaks"

    elif iytype == 131:
        y_unit = None
        y_title = "Emission"

    else:
        warn(
            "Wrong y unit label code in the SPC file. It will be set to arbitrary intensity",
            stacklevel=2,
        )
        y_unit = None
        y_title = "Arbitrary Intensity"

    if Fexp == b"\x80":  # noqa: SIM108
        iexp = None  # floating Point Data
    else:
        iexp = int.from_bytes(Fexp, endian)  # Datablock scaling Exponent

    # set date (from https://github.com/rohanisaac/spc/blob/master/spc/spc.py)
    year = Fdate >> 20
    month = (Fdate >> 16) % (2**4)
    day = (Fdate >> 11) % (2**5)
    hour = (Fdate >> 6) % (2**5)
    minute = Fdate % (2**6)

    if (
        year == 0 or month == 0 or day == 0
    ):  # occurs when acquision time is not reported
        timestamp = 0
        acqdate = datetime.fromtimestamp(0, tz=None)
        warn(f"No collection time found. Arbitrarily set to {acqdate}", stacklevel=2)
    else:
        acqdate = datetime(year, month, day, hour, minute)
        timestamp = acqdate.timestamp()

    sres = Fres.decode("utf-8")
    ssource = Fsource.decode("utf-8")

    scmnt = Fcmnt.decode("utf-8")

    # if Fwplanes:
    #     iwtype = int.from_bytes(Fwtype, endian)
    #     if iwtype != 255:
    #         w_unit = x_or_z_unit[ixtype]
    #         w_title = x_or_z_title[ixtype]
    #     else:
    #         w_unit = None
    #         w_title = "Double interferogram"

    if not txvals:  # evenly spaced x data
        _x = Coord.linspace(
            Ffirst,
            Flast,
            Fnpts,
            title=x_title,
            units=x_unit,
        )

    else:
        _x = Coord(
            data=np.frombuffer(content, offset=512, dtype=float32_dtype, count=Fnpts),
            title=x_title,
            units=x_unit,
        )

    if iexp is None:
        # 32-bit IEEE floating numbers
        floatY = np.frombuffer(
            content,
            offset=544 + txvals * Fnpts * 4,
            dtype=float32_dtype,
            count=Fnpts,
        )
    elif tsprec:
        integerY = np.frombuffer(
            content,
            offset=544 + txvals * Fnpts * 4,
            dtype=int16_dtype,
            count=Fnpts,
        )
        floatY = (2**iexp) * (integerY / (2**16))
    else:
        integerY = np.frombuffer(
            content,
            offset=544 + txvals * Fnpts * 4,
            dtype=int32_dtype,
            count=Fnpts,
        )
        floatY = (2**iexp) * (integerY / (2**32))

    if Flogoff:  # read log data header
        (
            Logsizd,
            Logsizm,
            Logtxto,
            Logbins,
            Logdsks,
            Logspar,
        ) = struct.unpack(
            logstc_format.encode("utf-8"),
            content[Flogoff : Flogoff + 21],
        )

        logtxt = str(content[Flogoff + Logtxto : len(content)].decode("utf-8"))

    # Create NDDataset Object for the series
    dataset = NDDataset(np.expand_dims(floatY, axis=0))
    dataset.name = str(filename)
    dataset.filename = filename
    dataset.units = y_unit
    dataset.title = y_title
    dataset.origin = "thermo galactic"

    # now add coordinates
    _y = Coord(
        [timestamp],
        title="acquisition timestamp (GMT)",
        units="s",
        labels=([acqdate], [filename]),
    )

    dataset.set_coordset(y=_y, x=_x)

    dataset.description = kwargs.get("description", "Dataset from spc file.\n")
    if ord(Fexper) != 0 and ord(Fexper) != 7:
        dataset.description += "Instrumental Technique: " + technique + "\n"
    if Fres != b"\x00\x00\x00\x00\x00\x00\x00\x00\x00":
        dataset.description += "Resolution: " + sres + "\n"
    if Fsource != b"\x00\x00\x00\x00\x00\x00\x00\x00\x00":
        dataset.description += "Source Instrument: " + ssource + "\n"
    if (
        Fcmnt
        != b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00"
    ):
        dataset.description += "Memo: " + scmnt + "\n"
    if Flogoff:
        if Logtxto:
            dataset.description += "Log Text: \n---------\n"
            dataset.description += logtxt
            dataset.description += "---------\n"
        if Logbins or Logsizd:
            if Logtxto:
                dataset.description += (
                    "Note: The Log block of the spc file also contains: \n"
                )
            else:
                dataset.description += (
                    "Note: The Log block of the spc file contains: \n"
                )
            if Logbins:
                dataset.description += f"a Log binary block of size {Logbins} bytes "
            if Logsizd:
                dataset.description += f"a Log disk block of size {Logsizd} bytes "

    dataset.history = f"Imported from spc file {filename}."

    if y_unit == "Interferogram":
        # interferogram
        dataset.meta.interferogram = True
        dataset.meta.td = list(dataset.shape)
        dataset.x._zpd = Fpeakpt
        dataset.meta.laser_frequency = Quantity("15798.26 cm^-1")
        dataset.x.set_laser_frequency()
        dataset.x._use_time_axis = (
            False  # True to have time, else it will be optical path difference
        )

    fid.close()
    return dataset
