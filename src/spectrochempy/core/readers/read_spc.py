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

import numpy as np

from spectrochempy.application.application import debug_
from spectrochempy.application.application import warning_
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
    kwargs["merge"] = False
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private functions
# ======================================================================================
class _SpcFile:
    # File header (see: Galactic Universal Data Format Specification 9/4/97)
    # http://ensembles-eu.metoffice.gov.uk/met-res/aries/technical/GSPC_UDF.PDF

    # Define some dicionaries for the SPC file format
    xz_info = {
        0: ("axis title", None),
        1: ("Wavenumbers", "cm^-1"),
        2: ("Wavelength", "um"),
        3: ("Wavelength", "nm"),
        4: ("Time", "s"),
        5: ("Time", "min"),
        6: ("Frequency", "Hz"),
        7: ("Frequency", "kHz"),
        8: ("Frequency", "MHz"),
        9: ("m/z", "g/(mol * e)"),
        10: ("Chemical shift", "ppm"),
        11: ("Time", "days"),
        12: ("Time", "years"),
        13: ("Raman shift", "cm^-1"),
        14: ("Energy", "eV"),
        15: ("text_label", None),
        16: ("diode number", None),
        17: ("Channel", None),
        18: ("2 theta", "degree"),
        19: ("Temperature", "fahrenheit"),
        20: ("Temperature", "celsius"),
        21: ("Temperature", "kelvin"),
        22: ("Data Points", None),
        23: ("Time", "ms"),
        24: ("Time", "us"),
        25: ("Time", "ns"),
        26: ("Frequency", "GHz"),
        27: ("Wavelength", "cm"),
        28: ("Wavelength", "m"),
        29: ("Wavelength", "mm"),
        30: ("Time", "hour"),
        255: ("Double interferogram", None),
    }

    y_info = {
        0: ("Arbitrary Intensity", None),
        1: ("Interferogram", None),
        2: ("Absorbance", "absorbance"),
        3: ("Kubelka-Munk", "Kubelka_Munk"),
        4: ("Counts", None),
        5: ("Voltage", "Volt"),
        6: ("Angle", "degree"),
        7: ("Intensity", "mA"),
        8: ("Length", "mm"),
        9: ("Voltage", "mV"),
        10: ("Log(1/R)", None),
        11: ("Transmittance", "percent"),
        12: ("Intensity", None),
        13: ("Relative Intensity", None),
        14: ("Energy", None),
        15: (None, None),
        16: ("Decibel", "dB"),
        17: (None, None),
        18: (None, None),
        19: ("Temperature", "fahrenheit"),
        20: ("Temperature", "celsius"),
        21: ("Temperature", "kelvin"),
        22: ("Index of Refraction [N]", None),
        23: ("Extinction Coeff. [K]", None),
        24: ("Real", None),
        25: ("Imaginary", None),
        26: ("Complex", None),
        128: ("Transmission", None),
        129: ("Reflectance", None),
        130: ("Arbitrary or Single Beam with Valley Peaks", None),
        131: ("Emission", None),
    }

    instrumental_techniques = {
        0: "General SPC",
        1: "Gas Chromatogram",
        2: "General Chromatogram",
        3: "HPLC Chromatogram",
        4: "FT-IR, FT-NIR, FT-Raman Spectrum",
        5: "NIR Spectrum",
        6: "Unknown",
        7: "UV-VIS Spectrum",
        8: "X-ray Diffraction Spectrum",
        9: "Mass Spectrum ",
        10: "NMR Spectrum or FID",
        11: "Raman Spectrum",
        12: "Fluorescence Spectrum",
        13: "Atomic Spectrum",
        14: "Chromatography Diode Array Spectra",
    }

    def __init__(self, content):
        # extract flag data type and version
        Ftflgs, Fversn = struct.unpack(b"cc", content[:2])

        # extract bit flags
        self._extract_bitflag(Ftflgs)

        # set endianness
        self._endian = "little"
        s = ">" if self._endian == "big" else "<"
        self.float32_dtype = s + "f4"
        self.int16_dtype = s + "i2"
        self.int32_dtype = s + "i4"

        # extract header
        if Fversn != b"\x4d":
            # New format (0x4B or 0x4C)
            version = "new LSB 1st" if Fversn == b"\x4b" else "new MSB 1st"
            if version == "new MSB 1st":
                self._endian = "big"
            self.head_size = 512
            self._extract_new_format_header(content, s)
        elif Fversn == b"\x4d":
            # old format (0x4D)
            version = "old format"
            self.head_size = 256
            self._extract_old_format_header(content)
        else:
            raise ValueError(
                f"Unknown SPC format: `{Fversn}`. Please add an issue on Github."
            )

        # create public attributes
        self.version = version
        exper = (
            int.from_bytes(self._Fexper, self._endian)
            if hasattr(self, "_Fexper")
            else 0
        )
        self.technique = self.instrumental_techniques[exper]
        self.cmnt = str(self._Fcmnt.decode("utf-8")).strip("\x00")
        self.res = str(self._Fres.decode("utf-8")).strip("\x00")
        self.source = (
            str(self._Fsource.decode("utf-8")).strip("\x00")
            if hasattr(self, "_Fsource")
            else None
        )
        self.peakpt = self._Fpeakpt

        (
            self.acqdate,
            self.timestamp,
        ) = self._date()  # for old format must be located BEFORE defining ztype
        self.npts = int(self._Fnpts)
        self.first = float(self._Ffirst)
        self.last = float(self._Flast)
        xtype = int.from_bytes(self._Fxtype, self._endian)
        ytype = int.from_bytes(self._Fytype, self._endian)
        ztype = int.from_bytes(self._Fztype, self._endian)
        self.x_title, self.x_units = self.xz_info.get(xtype, (None, None))
        self.y_title, self.y_units = self.y_info.get(ytype, (None, None))
        self.z_title, self.z_units = self.xz_info.get(ztype, (None, None))

        if self._talabs:
            # extract axis labels from fcatxt
            ll = self._Fcatxt.split(b"\x00")
            if len(ll) > 2:
                xl, yl, zl = ll[:3]
                # overwrite only if non zero
                if len(xl) > 0:
                    self.x_title = xl
                if len(yl) > 0:
                    self.y_title = yl
                if len(zl) > 0:
                    self.z_title = zl

        # data format
        # -----------
        # EX-Y: single file, evenly spaced x data
        if Ftflgs == b"\x00":
            self.format = "XE-Y"

        # EX-MY: Multifile, evenly spaced x data
        if self._tmulti and not self._txvals:
            self.format = "XE-MY"

        # X-Y: single file, non-evenly spaced x data
        if not self._tmulti and self._txvals:
            self.format = "X-Y"

        # X-MY: Multifile, non-evenly spaced x data, common x data
        if self._tmulti and self._txvals:
            self.format = "X-MY"

        # MXY: Multifile, non-evenly spaced x data, unique x data
        if self._tmulti and self._txvals and self._txyxys:
            self.format = "MXY"

        # for old format nsub is not known
        self.nsub = int(self._Fnsub) if hasattr(self, "_Fnsub") else None

        # prepare list of data
        self.nds = nds = []
        offset = self.head_size  # start after header
        if self.version == "old format":
            offset -= (
                32  # for old format, the first subfile header is included in the header
            )

        if not (self.format == "MXY" and self.npts > 0):  # not a directory reading
            npts = self.npts
            # read first block
            x, y, z, offset = self._get_sub_file(content, offset, npts, xdata=True, s=s)
            nds.append((x, y, z))

            if "M" in self.format:
                xinit = None
                if "XE" in self.format:
                    # save x
                    xinit = x
                if self.nsub:
                    # read subfiles
                    xdata = self.format == "MXY"
                    for _ in range(1, self.nsub):
                        x, y, z, offset = self._get_sub_file(
                            content, offset, npts, xdata, s
                        )
                        if xinit is not None:
                            x = xinit
                        nds.append((x, y, z))
                else:  # old format (we need to iterate until the end of file)
                    while offset < len(content):
                        x, y, z, offset = self._get_sub_file(content, offset, npts)
                        if xinit is not None:
                            x = xinit
                        nds.append((x, y, z))
        else:
            # read the directory
            # /* This structure defines the entries in the XY subfile directory. */
            # /* Its size is guaranteed to be 12 bytes long. */
            # typedef struct
            # {
            # DWORD ssfposn; /* disk file position of beginning of subfile (subhdr)*/
            # DWORD ssfsize; /* byte size of subfile (subhdr+X+Y) */
            # float ssftime; /* floating Z time of subfile (subtime) */
            # } SSFSTC;
            if self.version == "old format":
                raise NotImplementedError(
                    "Reading multifile using directory is not implemented for old SPC file format"
                )

            offset = self.npts
            ssftc_format = s + "iif"
            for _i in range(self.nsub):
                ssfposn, ssfsize, ssftime = struct.unpack(
                    ssftc_format.encode("utf8"),
                    content[offset : offset + 12],
                )
                npts = int((ssfsize - 32) / 8)
                self.xxx = ssfposn, ssfsize, npts, ssftime
                offset += 12

                x, y, z, _ = self._get_sub_file(content, ssfposn, npts, s=s)
                nds.append((x, y, ssftime))

        # check consistency
        if offset > len(content):
            warning_(f"File size {len(content)} is smaller than expected {offset}")

        self.logtxt = None
        if offset < len(content):
            # reading content not complete, look for log info
            if self.version != "old format" and self._Flogoff >= offset:
                self.logtxt = self._extract_log_info(content, s)
            else:
                warning_("Reading contents seems not complete")

        # LOG FOR DEBUG
        self._debug_info()

    def _get_sub_file(self, content, offset, npts, xdata=False, s="<"):
        """Extract x, y data and update offset for a subfile."""
        x = None

        # Handle x data based on format
        if "XE" in self.format and xdata:
            x = np.linspace(self.first, self.last, num=npts)
        elif "X-" in self.format and xdata:
            x = self._extract_x_data(offset, content, npts)
            offset += npts * 4

        # Extract subfile header (common to all formats)
        (
            subflgs,
            self._subexp,
            subindx,
            subtime,
            subnext,
            subnois,
            subnpts,
            subscan,
            subwlevel,
            subresv,
        ) = self._extract_subfile_header(content[offset : offset + 32], s)
        offset += 32

        if subnpts != 0 and subnpts != npts:
            warning_(f"Number of points mismatch: {subnpts} != {npts}")
        z = 0
        if "M" in self.format:
            if subindx == 0 and not self._tordrd:
                # store first subtime value
                self._subfirst = subtime
                self._subinc = subnext - subtime

            # Determine z value based on ordered/unordered status
            z = subtime if self._tordrd else subindx * self._subinc + self._subfirst

        # Handle MXY format x data
        if "MXY" in self.format:
            x = self._extract_x_data(offset, content, npts)
            offset += npts * 4

        # Extract y data (common to all formats)
        y = self._extract_y_data(offset, content, npts)
        offset += npts * 4

        return x, y, z, offset

    def _extract_x_data(self, offset, content, npts):
        # extract x data
        # --------------
        # X values FNPTS 32-bit floating X values
        x = np.frombuffer(
            content,
            offset=self.head_size,
            dtype=self.float32_dtype,
            count=npts,
        )
        np.frombuffer(content, offset=offset, dtype=self.float32_dtype, count=npts)
        return x

    def _extract_y_data(self, offset, content, npts):
        iexp = self._exp()
        ydata = content[offset : offset + npts * 4]
        if self.version != "old format":
            # select content

            if iexp is None:
                # 32-bit IEEE floating numbers
                floatY = np.frombuffer(
                    ydata,
                    dtype=self.float32_dtype,
                    count=npts,
                )
            elif self._tsprec:
                integerY = np.frombuffer(
                    ydata,
                    dtype=self.int16_dtype,
                    count=npts,
                )
                floatY = (2**iexp) * (integerY / (2**16))
            else:
                integerY = np.frombuffer(
                    ydata,
                    dtype=self.int32_dtype,
                    count=npts,
                )
                floatY = (2**iexp) * (integerY / (2**32))

        else:
            # old format
            if iexp is None:
                # 32-bit floating numbers
                # TODO: check if this is correct (not sure about inversion of bytes as documented in spc.h)
                floatY = np.frombuffer(
                    content,
                    offset=offset,
                    dtype=self.float32_dtype,
                    count=npts,
                )
            else:
                # Adapted from https://github.com/velexi-research/spc-spectra/blob/main/src/spc_spectra/subfile.py#L12
                # for old format, extract the entire array out as 1 bit unsigned
                # integers, swap 1st and 2nd byte, as well as 3rd and 4th byte to get
                # the final integer then scale by the exponent
                data_format = ">" + "B" * 4 * npts
                y_raw = struct.unpack(data_format.encode("utf8"), ydata)

                y_int = []
                for i in range(0, len(y_raw), 4):
                    y_int.append(
                        y_raw[i + 1] * (256**3)
                        + y_raw[i] * (256**2)
                        + y_raw[i + 3] * (256)
                        + y_raw[i + 2]
                    )

                # convert y-data to signed ints
                y_int = np.array(y_int).astype("int32", copy=False)

                # convert y-data to floats
                floatY = y_int / (2 ** (32 - iexp))

        return floatY

    def _date(self):
        # set date (from https://github.com/rohanisaac/spc/blob/master/spc/spc.py)
        if hasattr(self, "_Fdate"):  # new format
            Fdate = self._Fdate
            year = Fdate >> 20
            month = (Fdate >> 16) % (2**4)
            day = (Fdate >> 11) % (2**5)
            hour = (Fdate >> 6) % (2**5)
            minute = Fdate % (2**6)
        else:  # old format
            # /* Year collected (0=no date/time) - MSB 4 bits are Z type */
            year = self._Fyear & 0x0FFF  # Get lower 12 bits for year
            self._Fztype = ((self._Fyear >> 12) & 0x0F).to_bytes(
                1, self._endian
            )  # Get upper 4 bits for Z type

            month = int.from_bytes(self._Fmonth, self._endian)
            day = int.from_bytes(self._Fday, self._endian)
            hour = int.from_bytes(self._Fhour, self._endian)
            minute = int.from_bytes(self._Fminute, self._endian)

        if (
            year == 0 or month == 0 or day == 0
        ):  # occurs when acquision time is not reported
            timestamp = 0
            acqdate = datetime.fromtimestamp(0, tz=None)
            warning_(f"No collection time found. Arbitrarily set to {acqdate}")
        else:
            acqdate = datetime(year, month, day, hour, minute)
            timestamp = acqdate.timestamp()

        return acqdate, timestamp

    def _exp(self):
        _exp = self._subexp if self.format == "MXY" else self._Fexp

        if _exp == b"\x80":  # noqa: SIM108
            iexp = None  # floating Point Data
        elif isinstance(_exp, int):
            iexp = _exp
        else:
            iexp = int.from_bytes(_exp, self._endian)  # Datablock scaling Exponent
        return iexp

    def _extract_log_info(self, content, s):
        # typedef struct /* log block header format */
        # {
        # DWORD logsizd; /* byte size of disk block */
        # DWORD logsizm; /* byte size of memory block */
        # DWORD logtxto; /* byte offset to text */
        # DWORD logbins; /* byte size of binary area (after logstc*/
        # DWORD logdsks; /* byte size of disk area (after logbins)*/
        # char logspar[44]; /* reserved (must be zero) */
        # } LOGSTC;
        offset = self._Flogoff
        logstc_format = s + "iiiii44s"
        siz = struct.calcsize(logstc_format)
        sel = content[offset : offset + siz]
        (
            Logsizd,
            Logsizm,
            Logtxto,
            Logbins,
            Logdsks,
            Logspar,
        ) = struct.unpack(
            logstc_format.encode("utf-8"),
            sel,
        )

        return str(content[offset + Logtxto : len(content)].decode("utf-8"))

    def _extract_subfile_header(self, content, s):
        # **************************************************************************
        # * This structure defines the subfile headers that preceed each trace in a
        # * multi-type file. Note that for evenly-spaced files, subtime and subnext are
        # * optional (and ignored) for all but the first subfile. The (subnext-subtime)
        # * for the first subfile determines the Z spacing for all evenly-spaced subfiles.
        # * For ordered and random multi files, subnext is normally set to match subtime.
        # * However, for all types, the subindx must be correct for all subfiles.
        # * This header must must always be present even if there is only one subfile.
        # * However, if TMULTI is not set, then the subexp is ignored in favor of fexp.
        # * Normally, subflgs and subnois are set to zero and are used internally.
        # ***************************************************************************
        # #define SUBCHGD 1/* Subflgs bit if subfile changed */
        # #define SUBNOPT 8 /* Subflgs bit if peak table file should not be used */
        # #define SUBMODF 128 /* Subflgs bit if subfile modified by arithmetic */
        # typedef struct
        # {
        # BYTE subflgs; /* Flags as defined above */
        # char subexp; /* Exponent for sub-file's Y values (80h=>float) */
        # WORD subindx; /* Integer index number of trace subfile (0=first) */
        # float subtime; /* Floating time for trace (Z axis coordinate) */
        # float subnext; /* Floating time for next trace (May be same as beg) */
        # float subnois; /* Floating peak pick noise level if high byte nonzero */
        # DWORD subnpts; /* Integer number of subfile points for TXYXYS type */
        # DWORD subscan; /* Integer number of co-added scans or 0 (for collect) */
        # float subwlevel; /* Floating W axis value (if fwplanes non-zero) */
        # char subresv[4]; /* Reserved area (must be set to zero) */
        # } SUBHDR;
        # #define FSNOIS fsubh1+subnois+3 /* Byte which is non-zero if subnois valid */

        subhead_format = s + "cchfffiif4s"
        if struct.calcsize(subhead_format) != 32:
            raise ValueError(
                f"Subheader size mismatch: {struct.calcsize(subhead_format)} != 32"
            )

        return struct.unpack(subhead_format.encode("utf8"), content)

    def _extract_bitflag(self, Ftflgs):
        # extract bit flags
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

        (
            self._tsprec,
            self._tcgram,
            self._tmulti,
            self._trandm,
            self._tordrd,
            self._talabs,
            self._txyxys,
            self._txvals,
        ) = (x == "1" for x in reversed(list(f"{ord(Ftflgs):08b}")))

    def _extract_new_format_header(self, content, s):
        # New format (0x4B or 0x4C)
        # -------------------------
        # * The new format allows X,Y pairs data to be stored when the TXVALS flag is set.
        # * The main header is immediately followed by any array of fnpts 32-bit floating
        # * numbers giving the X values for all points in the file or subfiles. Note
        # * that for multi files, there is normally a single X values array which applies
        # * to all subfiles. The X values are followed by a subfile header and fixed
        # * point Y values array or, for multi files, by the subfiles which each consist
        # * of a subfile header followed by a fixed-point Y values array. Note that
        # * the software may be somewhat slower when using X-values type files.
        # * Another X,Y mode allows for separate X arrays and differing numbers of
        # * points for each subfile. This mode is normally used for Mass Spec Data.
        # * If the TXYXYS flag is set along with TXVALS, then each subfile has a
        # * separate X array which follows the subfile header and preceeds the Y array.
        # * An additional subnpts subfile header entry gives the number of X,Y values
        # * for the subfile (rather than the fnpts entry in the main header). Under
        # * this mode, there may be a directory subfile pointers whose offset is
        # * stored in the fnpts main header entry. This directory consists of an
        # * array of ssfstc structures, one for each of the subfiles. Each ssfstc
        # * gives the byte file offset of the begining of the subfile (that is, of
        # * its subfile header) and also gives the Z value (subtime) for the subfile
        # * and is byte size. This directory is normally saved at the end of the
        # * file after the last subfile. If the fnpts entry is zero, then no directory
        # * is present and GRAMS/32 automatically creates one (by scanning through the
        # * subfiles) when the file is opened. Otherwise, fnpts should be the byte
        # * offset into the file to the first ssfstc for the first subfile. Note
        # * that when the directory is present, the subfiles may not be sequentially
        # * stored in the file. This allows GRAMS/32 to add points to subfiles by
        # * moving them to the end of the file.
        # * Y values are represented as fixed-point signed fractions (which are similar
        # * to integers except that the binary point is above the most significant bit
        # * rather than below the least significant) scaled by a single exponent value.
        # * For example, 0x40000000 represents 0.25 and 0xC0000000 represents -0.25 and
        # * if the exponent is 2 then they represent 1 and -1 respectively. Note that
        # * in the old 0x4D format, the two words in a 4-byte DP Y value are reversed.
        # * To convert the fixed Y values to floating point:
        # *     FloatY = (2^Exponent)*FractionY
        # * or: FloatY = (2^Exponent)*IntegerY/(2^32) -if 32-bit values
        # * or: FloatY = (2^Exponent)*IntegerY/(2^16) -if 16-bit values
        #
        # * Optionally, the Y values on the disk may be 32-bit IEEE floating numbers.
        # * In this case the fexp value (or subexp value for multifile subfiles)
        # * must be set to 0x80 (-128 decimal). Floating Y values are automatically
        # * converted to the fixed format when read into memory and are somewhat slower.
        # * GRAMS/32 never saves traces with floating Y values but can read them.
        #
        # * Thus an SPC trace file normally has these components in the following order:
        # * SPCHDR Main header (512 bytes in new format, 224 or 256 in old)
        # * [X Values] Optional FNPTS 32-bit floating X values if TXVALS flag
        # * SUBHDR Subfile Header for 1st subfile (32 bytes)
        # * Y Values FNPTS 32 or 16 bit fixed Y fractions scaled by exponent
        # * [SUBHDR ] Optional Subfile Header for 2nd subfile if TMULTI flag
        # * [Y Values] Optional FNPTS Y values for 2nd subfile if TMULTI flag
        # *
        # ... Additional subfiles if TMULTI flag (up to FNSUB total)
        # * [Log Info] Optional LOGSTC and log data if flogoff is non-zero
        # * However, files with the TXYXYS ftflgs flag set have these components:
        # * SPCHDR Main header (512 bytes in new format)
        # * SUBHDR Subfile Header for 1st subfile (32 bytes)
        # * X Values FNPTS 32-bit floating X values
        # * Y Values FNPTS 32 or 16 bit fixed Y fractions scaled by exponent
        # * [SUBHDR ] Subfile Header for 2nd subfile
        # * [X Values] FNPTS 32-bit floating X values for 2nd subfile
        # * [Y Values] FNPTS Y values for 2nd subfile
        # *
        # ... Additional subfiles (up to FNSUB total)
        # * [Directory] Optional FNSUB SSFSTC entries pointed to by FNPTS
        # * [Log Info] Optional LOGSTC and log data if flogoff is non-zero
        # * Note that the fxtype, fytype, and fztype default axis labels can be
        # * overridden with null-terminated strings at the end of fcmnt. If the
        # * TALABS bit is set in ftflgs (or Z=ZTEXTL in old format), then the labels
        # * come from the fcatxt offset of the header. The X, Y, and Z labels
        # * must each be zero-byte terminated and must occure in the stated (X,Y,Z)
        # * order. If a label is only the terminating zero byte then the fxtype,
        # * fytype, or fztype (or Arbitrary Z) type label is used instead. The
        # * labels may not exceed 20 characters each and all three must fit in 30 bytes.
        # * The fpost, fprocs, flevel, fsampin, ffactor, and fmethod offsets specify
        # * the desired post collect processing for the data. Zero values are used
        # * for unspecified values causing default settings to be used. See GRAMSDDE.INC
        # * Normally fpeakpt is zero to allow the centerburst to be automatically located.
        # * If flogoff is non-zero, then it is the byte offset in the SPC file to a
        # * block of memory reserved for logging changes and comments. The beginning
        # * of this block holds a logstc structure which gives the size of the
        # * block and the offset to the log text. The log text must be at the block's
        # * end. The log text consists of lines, each ending with a carriage return
        # * and line feed. After the final line's CR and LF must come a zero character
        # * (which must be the first in the text). Log text requires V1.10 or later.
        # * The log is normally after the last subfile (or after the TXYXYS directory).
        # * The fwplanes allows a series of subfiles to be interpreted as a volume of
        # * data with ordinate Y values along three dimensions (X,Z,W). Volume data is
        # * also known as 4D data since plots can have X, Z, W, and Y axes. When
        # * fwplanes is non-zero, then groups of subfiles are interpreted as planes
        # * along a W axis. The fwplanes value gives the number of planes (groups of
        # * subfiles) and must divide evenly into the total number of subfiles (fnsub).
        # * If the fwinc is non-zero, then the W axis values are evenly spaced beginning
        # * with subwlevel for the first subfile and incremented by fwinc after each
        # * group of fwplanes subfiles. If fwinc is zero, then the planes may have
        # * non-evenly-spaced W axis values as given by the subwlevel for the first
        # * subfile in the plane's group. However, the W axis values must be ordered so
        # * that the plane values always increase or decrease. Also all subfiles in the
        # * plane should have the same subwlevel. Equally-spaced W planes are recommended
        # * and some software may not handle fwinc=0. The fwtype gives the W axis type
        #
        #
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

        head_format = s + "cccciddicccci9s9sh32s130s30siicchf48sfifc187s"
        if struct.calcsize(head_format) != self.head_size:
            raise ValueError(
                f"Header size mismatch: {struct.calcsize(head_format)} != {self.head_size}"
            )

        (
            self._Ftflgs,
            self._Fversn,
            self._Fexper,
            self._Fexp,
            self._Fnpts,
            self._Ffirst,
            self._Flast,
            self._Fnsub,
            self._Fxtype,
            self._Fytype,
            self._Fztype,
            self._Fpost,
            self._Fdate,
            self._Fres,
            self._Fsource,
            self._Fpeakpt,
            self._Fspare,
            self._Fcmnt,
            self._Fcatxt,
            self._Flogoff,
            self._Fmods,
            self._Fprocs,
            self._Flevel,
            self._Fsampin,
            self._Ffactor,
            self._Fmethod,
            self._Fzinc,
            self._Fwplanes,
            self._Fwinc,
            self._Fwtype,
            self._Freserv,
        ) = struct.unpack(head_format.encode("utf8"), content[: self.head_size])

    def _extract_old_format_header(self, content):
        # /**************************************************************************
        # * In the old 0x4D format, fnpts is floating point rather than a DP integer,
        # * ffirst and flast are 32-bit floating point rather than 64-bit, and fnsub
        # * fmethod, and fextra do not exist. (Note that in the new formats, the
        # * fcmnt text may extend into the fcatxt and fextra areas if the TALABS flag
        # * is not set. However, any text beyond the first 130 bytes may be
        # * ignored in future versions if fextra is used for other purposes.)
        # * Also, in the old format, the date and time are stored differently.
        # * Note that the new format header has 512 bytes while old format headers
        # * have 256 bytes and in memory all headers use 288 bytes. Also, the
        # * new header does not include the first subfile header but the old does.
        # * The following constants define the offsets in the old format header:
        # * Finally, the old format 32-bit Y values have the two words reversed from
        # * the Intel least-significant-word-first order. Within each word, the
        # * least significant byte comes first, but the most significant word is first.
        # ***************************************************************************/
        #
        # typedef struct
        # {
        # BYTE oftflgs;
        # BYTE oversn; /* 0x4D rather than 0x4C or 0x4B */
        # short oexp; /* Word rather than byte */
        # float onpts; /* Floating number of points */
        # float ofirst; /* Floating X coordinate of first pnt (SP rather than DP) */
        # float olast; /* Floating X coordinate of last point (SP rather than DP) */
        # BYTE oxtype; /* Type of X units */
        # BYTE oytype; /* Type of Y units */
        # WORD oyear; /* Year collected (0=no date/time) - MSB 4 bits are Z type */
        # BYTE omonth; /* Month collected (1=Jan) */
        # BYTE oday; /* Day of month (1=1st) */
        # BYTE ohour; /* Hour of day (13=1PM) */
        # BYTE ominute; /* Minute of hour */
        # char ores[8]; /* Resolution text (null terminated unless 8 bytes used) */
        # WORD opeakpt;
        # WORD onscans;
        # float ospare[7];
        # char ocmnt[130];
        # char ocatxt[30];
        # char osubh1[32]; /* Header for first (or main) subfile included in main header */
        # } OSPCHDR;

        head_format = "<cchfffcchcccc8shh28s130s30s32s"

        if struct.calcsize(head_format) != self.head_size:
            raise ValueError(
                f"Header size mismatch: {struct.calcsize(head_format)} != {self.head_size}"
            )

        (
            self._Ftflgs,
            self._Fversn,
            self._Fexp,
            self._Fnpts,
            self._Ffirst,
            self._Flast,
            self._Fxtype,
            self._Fytype,
            self._Fyear,
            self._Fmonth,
            self._Fday,
            self._Fhour,
            self._Fminute,
            self._Fres,
            self._Fpeakpt,
            self._Fnscans,
            self._Fspare,
            self._Fcmnt,
            self._Fcatxt,
            self._Fsubh1,
        ) = struct.unpack(head_format.encode("utf8"), content[: self.head_size])

    def _debug_info(self):
        debug_(f"Version: {self.version}")
        debug_(f"format: {self.format}")
        debug_(f"Number of subfiles: {self.nsub}")
        # Flag bits
        if self._tsprec:
            debug_("16-bit y data")
        if self._tcgram:
            debug_("enable fexper")
        if self._tmulti:
            debug_("multiple traces")
        if self._trandm:
            debug_("arb time (z) values")
        if self._tordrd:
            debug_("ordered but uneven subtimes")
        if self._talabs:
            debug_("use fcatxt axis not fxtype")
        if self._txyxys:
            debug_("each subfile has own x's")
        if self._txvals:
            debug_("floating x-value array preceeds y's")

        # subfiles
        try:
            if self._Fnsub == 1:
                debug_("Single file only")
            else:
                debug_(f"Multiple subfiles: {self._Fnsub}")
        except AttributeError:
            debug_("Fnsub not defined")

        # multiple y values
        if self._tmulti:
            debug_("Multiple y-values")
        else:
            debug_("Single set of y-values")


@_importer_method
def _read_spc(*args, **kwargs):
    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    content = fid.read()

    # Read content
    spcf = _SpcFile(content)

    # files
    nds = spcf.nds

    # Create NDDataset Object for the file / subfiles
    if len(nds) < 0:
        raise ValueError(f"No data found in the {filename} SPC file")

    datasets = []

    if len(nds) == 1:
        # single file
        x, y, z = nds[0]
        dataset = NDDataset([y], title=spcf.y_title, units=spcf.y_units)
        coordx = Coord(x, title=spcf.x_title, units=spcf.x_units)
        coordy = Coord(
            [spcf.timestamp],
            title="acquisition timestamp (GMT)",
            units="s",
            labels=([spcf.acqdate]),
        )
        dataset.set_coordset(y=coordy, x=coordx)
        datasets.append(dataset)

    else:
        # multiple files
        # First check if all x arrays are the same length and values
        all_x_same = False
        if spcf.format != "MXY":
            all_x_same = True
        else:
            # Check if all x arrays have the same shape and values
            first_x = nds[0][0]
            if all(np.array_equal(first_x, nd[0]) for nd in nds):
                all_x_same = True

        if all_x_same:
            # Create a 2D NDDataset with common x coordinates
            ys = np.array([nd[1] for nd in nds])
            zs = np.array([nd[2] for nd in nds])

            dataset = NDDataset(ys, title=spcf.y_title, units=spcf.y_units)
            coordx = Coord(nds[0][0], title=spcf.x_title, units=spcf.x_units)
            coordy = Coord(zs, title=spcf.z_title or "z", units=spcf.z_units)
            dataset.set_coordset(y=coordy, x=coordx)

            datasets.append(dataset)

        else:
            # Create a list of 1D NDDatasets for the case where x coordinates differ
            for i, (x, y, z) in enumerate(nds):
                ds = NDDataset([y], title=spcf.y_title, units=spcf.y_units)
                coordx = Coord(x, title=spcf.x_title, units=spcf.x_units)
                coordy = Coord([z], title=spcf.z_title or "z", units=spcf.z_units)
                dataset.set_coordset(y=coordy, x=coordx)
                ds.name = f"{str(filename)}_{i}"
                datasets.append(ds)

    for dataset in datasets:
        dataset.name = str(filename)
        dataset.filename = filename
        dataset.origin = kwargs.get(
            "origin", "thermo galactic"
        )  # TODO: use log info or comment to determine this
        dataset.description = kwargs.get("description", "Dataset from spc file.\n")
        if spcf.cmnt:
            dataset.description += "Memo: " + spcf.cmnt + "\n"
        if spcf.logtxt:
            dataset.description += "Log Text: \n---------\n"
            dataset.description += spcf.logtxt + "\n"
        # metadata
        if spcf.technique != "Unknown":
            dataset.meta.technique = spcf.technique
        if spcf.res:
            dataset.meta.resolution = spcf.res
        if spcf.source:
            dataset.meta.source = spcf.source
        dataset.meta.fileformat = spcf.format
        dataset.meta.scpversion = spcf.version

        dataset.history = f"Imported from spc file {filename}."

        if spcf.y_units == "Interferogram":
            # interferogram
            dataset.meta.interferogram = True
            dataset.meta.td = list(dataset.shape)
            dataset.x._zpd = spcf.peakpt
            dataset.meta.laser_frequency = Quantity("15798.26 cm^-1")
            dataset.x.set_laser_frequency()
            dataset.x._use_time_axis = (
                False  # True to have time, else it will be optical path difference
            )

    fid.close()
    return datasets[0] if len(datasets) == 1 else datasets
