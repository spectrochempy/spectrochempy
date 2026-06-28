# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Module extends NDDataset with the import method for Renishaw WiRe generated data files.

Notes
-----
Code incorporated from py_wdf_reader package (MIT License)
(see https://github.com/alchem0x2A/py-wdf-reader).

The code has been modified to be adapted to the needs of SpectroChemPy

# MIT License
#
# Copyright (c) 2022 T.Tian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

See Also
--------
The original code was inspired by Henderson, Alex DOI:10.5281/zenodo.495477
(see https://bitbucket.org/AlexHenderson/renishaw-file-formats/src/master/)

See also gwyddion for the DATA types
https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/file/renishaw.c

"""

__all__ = ["read_wdf", "read_wire"]

import io
import struct
from enum import Enum
from enum import IntEnum

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import debug_
from spectrochempy.utils._logging import error_
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.datetimeutils import windows_time_to_dt64
from spectrochempy.utils.file import fromfile
from spectrochempy.utils.objects import Adict

try:
    import PIL
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational
except ImportError:
    PIL = None


class _wdfReader:
    """
    Reader for Renishaw(TM) WiRE Raman spectroscopy files (.wdf format).

    The wdf file format is separated into several DataBlocks, with starting 4-char
    strings such as (incomplete list):
    `WDF1`: File header for information
    `DATA`: Spectra data
    `XLST`: Data for X-axis of data, usually the Raman shift or wavelength
    `YLST`: Data for Y-axis of data, possibly not important
    `WMAP`: Information for mapping, e.g. StreamLine or StreamLineHR mapping
    `MAP `: Mapping information(?)
    `ORGN`: Data for stage origin
    `TEXT`: Annotation text etc
    `WXDA`: ? TODO
    `WXDM`: ? TODO
    `ZLDC`: ? TODO
    `BKXL`: ? TODO
    `WXCS`: ? TODO
    `WXIS`: ? TODO
    `WHTL`: Whilte light image

    Following the block name, there are two indicators:
    Block uid: int32
    Block size: int64

    Parameters
    ----------
    fid : BytesIO object
        File object for the wdf file.
    dataset : `NDDataset`
        Dataset to fill with the data from the wdf file.

    Notes
    -----
    Metadata :
    title (str) : Title of measurement
    username (str) : Username
    application_name (str) : Default WiRE
    application_version (int,) x 4 : Version number, e.g. [4, 4, 0, 6602]
    measurement_type (int) : Type of measurement
                             0=unknown, 1=single, 2=multi, 3=mapping
    scan_type (int) : Scan of type, see values in scan_types
    laser_wavenumber (float32) : Wavenumber in cm^-1
    count (int) : Numbers of experiments (same type), can be smaller than capacity
    data (numpy.array) : Spectral data
    spectral_units (int) : Unit of spectra, see unit_types
    xlist_type (int) : See unit_types
    xlist_unit (int) : See unit_types
    xlist_length (int): Size for the xlist
    xdata (numpy.array): x-axis data
    ylist_type (int): Same as xlist_type
    ylist_unit (int): Same as xlist_unit
    ylist_length (int): Same as xlist_length
    ydata (numpy.array): y-data, possibly not used
    point_per_spectrum (int): Should be identical to xlist_length
    data_origin_count (int) : Number of rows in data origin list
    capacity (int) : Max number of spectra
    accumulation_count (int) : Single or multiple measurements
    block_info (dict) : Info block at least with following keys
                        DATA, XLST, YLST, ORGN
                        # TODO types?

    """

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(self, fid, dataset):
        # The content to read
        self._fid = fid

        # The dataset to fill
        self._dataset = dataset

        # The metadata
        self._meta = self._dataset.meta

        # Parse the header section in the wdf file
        self._block_info = self._locate_all_blocks()

        # Parse individual blocks
        self._parse_header()  # File header -> metadata
        username = self._meta.username.replace("\x00", "").strip()
        if username:
            self._dataset.author = username
        coord_x = self._parse_dimension("X")
        coord_meta = self._parse_dimension("Y")
        other_dimensions = self._parse_others()
        acq = getattr(self._meta, "acquisition_time", None)
        if acq is not None:
            self._dataset.acquisition_date = acq.item()
        data = self._parse_data()
        # self._parse_img()

        if self._meta.measurement_type == MeasurementType.Mapping:
            # Reshape spectra after reading mapping information
            map_shape = self._parse_mapping(other_dimensions)
            data = self._reshape_data(data, map_shape)
        else:  # Single or Series
            data = self._reshape_data(data)

        # Fill the dataset with the data
        dataset.data = data
        dataset.title = "count"

        # Fill the dataset with the coordinates
        odim = list(other_dimensions.values())
        if self._meta.measurement_type == MeasurementType.Single:
            dataset.set_coordset(x=coord_x, y=odim[0], m=coord_meta)
        elif self._meta.measurement_type == MeasurementType.Series:
            dataset.set_coordset(x=coord_x, y=odim[::-1], m=coord_meta)
        elif self._meta.measurement_type == MeasurementType.Mapping:
            if self._meta.map_area_type == MapAreaType.Unspecified:
                self._meta.map_area_type = MapAreaType.ColumnMajor
                warning_(
                    "Map area type is not specified, "
                    "will assume a xy (column major) scan for the mapping data.",
                )
            # line scan
            if self._meta.map_area_type == MapAreaType.XYLine:
                # create a new coordinate distance
                X, Y, Time = odim[0], odim[1], odim[2]
                dist = np.sqrt(X.data**2 + Y.data**2)
                dist = dist - dist[0]
                distance = Coord(dist, units=X.units, title="distance")
                coord_y = CoordSet(distance, Time, Y, X)
                dataset.set_coordset(x=coord_x, y=coord_y, m=coord_meta)

            # xy column major scan
            elif self._meta.map_area_type == MapAreaType.ColumnMajor:
                # extract the coordinates
                X, Y, Time = odim[0], odim[1], odim[2]
                if np.all(np.array(map_shape) > 1):
                    X.data = X.data.reshape(map_shape[::-1])[0]
                    Y.data = Y.data.reshape(map_shape[::-1])[:, 0]
                dataset.set_coordset(x=coord_x, y=X, m=coord_meta, z=Y)

            # not implemented yet
            else:
                error_(
                    f"Map area type {self._meta.map_area_type.name} not implemented yet!",
                )

        # Finally close the fid
        self._close()

    # ----------------------------------------------------------------------------------
    # Public properties
    # ----------------------------------------------------------------------------------
    @property
    def dataset(self):
        return self._dataset

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _locate_all_blocks(self):
        """Get information for all data blocks and store them inside self._block_info."""
        block_info = {}
        curpos = 0
        finished = False
        while not finished:
            try:
                block_name, block_uid, block_size = self._locate_single_block(curpos)
                block_info[block_name] = (block_uid, curpos, block_size)
                curpos += block_size
            except (EOFError, UnicodeDecodeError):
                finished = True
        return block_info

    def _locate_single_block(self, pos):
        """Get block information starting at pos."""
        self._fid.seek(pos)
        block_name = self._fid.read(0x4).decode("ascii")
        if len(block_name) < 4:
            raise EOFError
        block_uid = self._read_type("int32")
        block_size = self._read_type("int64")
        return block_name, block_uid, block_size

    def _parse_header(self):
        """Solve block WDF1."""
        self._fid.seek(0)  # return to the head

        block_ID = self._fid.read(Offsets.block_id).decode("ascii")
        block_UID = self._read_type("int32")
        block_len = self._read_type("int64")

        # First block must be "WDF1"
        if (
            (block_ID != "WDF1")
            or (block_UID != 0 and block_UID != 1)
            or (block_len != Offsets.data_block)
        ):
            raise ValueError("The wdf file format is incorrect!")
        # TODO what are the digits in between?

        # The keys from the header
        self._fid.seek(Offsets.measurement_info)  # space

        self._meta.point_per_spectrum = self._read_type("int32")
        self._meta.capacity = self._read_type("int64")
        self._meta.count = self._read_type("int64")
        self._meta.accumulation_count = self._read_type("int32")
        self._y_size = self._read_type("int32")
        self._x_size = self._read_type("int32")
        self._other_data_count = self._read_type("int32")
        application_name = self._read_type("utf8", 24)  # Must be "WiRE"
        application_version = [0] * 4
        for i in range(4):
            application_version[i] = str(self._read_type("int16"))
        self._dataset.origin = f"{application_name} {'.'.join(application_version)}"
        self._meta.scan_type = ScanType(self._read_type("int32"))
        self._meta.measurement_type = MeasurementType(self._read_type("int32"))

        # For the units
        self._fid.seek(Offsets.spectral_info)
        self._dataset.units = str(UnitType(self._read_type("int32")))
        self._meta.laser_frequency = self._read_type("float") * ur("cm^-1")

        # Username and title
        self._fid.seek(Offsets.file_info)
        self._meta.username = self._read_type(
            "utf8",
            Offsets.usr_name - Offsets.file_info,
        )
        self._dataset.description = self._read_type(
            "utf8",
            Offsets.data_block - Offsets.usr_name,
        )

    def _parse_others(self):
        """
        Get information from ORGN block.

        Additional dimensions information is stored in the ORGN block.
        """
        try:
            _, pos, _ = self._block_info["ORGN"]
        except KeyError:
            debug_("Current measurement does not contain dimension information!")
            return None

        count = self._meta.count
        capacity = self._meta.capacity
        list_increment = Offsets.origin_increment + LenType.l_double.value * capacity
        curpos = pos + Offsets.origin_info

        dimensions = Adict()
        for i in range(self._other_data_count):
            self._fid.seek(curpos)

            # First index: don't know how to use this!
            p1 = self._read_type("int32")
            flag = (p1 >> 31 & 0b1) == 1
            if flag:
                debug_("Flag is set to True, don't know how to use this!")

            # Second: Data type of the row
            datatype = DataType(p1 & ~(0b1 << 31))
            if datatype == DataType.Checksum or datatype == DataType.Flags:
                continue  # skip these two types which are not useful (as of now)

            # Third: Unit
            units = str(UnitType(self._read_type("int32")))
            # Fourth: annotation
            title = self._read_type("utf8", 0x10)
            # Last: the actual data
            if datatype == DataType.Time:
                data = np.array(
                    [
                        windows_time_to_dt64(self._read_type("int64"))
                        for i in range(count)
                    ],
                )
                # set the acquisition time from the first time stamp
                self._meta.acquisition_time = data[0]
                data = data - data[0]
            else:
                data = np.array([self._read_type("double") for i in range(count)])

            # Now build the corresponding coordinates
            coord = Coord(data.astype(float), units=units, title=title)
            # TODO: leave timedeltas when coordinates can handle this

            dimensions[str(datatype)] = coord

            curpos += list_increment

        return dimensions

    def _parse_dimension(self, dim):
        """Get information from XLST or YLST blocks."""
        if dim.upper() not in ["X", "Y"]:
            raise ValueError("Direction argument `dir` must be X or Y!")

        block_name = dim.upper() + "LST"
        _, pos, size = self._block_info[block_name]
        offset = Offsets.block_data
        self._fid.seek(pos + offset)
        dimtype = DataType(self._read_type("int32"))
        units = str(UnitType(self._read_type("int32")))
        size = getattr(self, f"_{dim.lower()}_size")
        if size == 0:  # Possibly not started
            raise ValueError(f"{dim.upper()} array possibly not yet initialized!")

        data = fromfile(self._fid, dtype="float32", count=size)
        data = np.array(data, dtype=float, ndmin=1)

        # now build corresponding coordinates
        coord = Coord(data, units=units)
        coord.name = dim.lower()
        coord.title = str(dimtype).replace("_", " ").lower()

        return coord

    def _parse_mapping(self, others):
        """Get information about mapping in StreamLine and StreamLineHR."""
        try:
            _, pos, _ = self._block_info["WMAP"]
        except KeyError:
            debug_("Current measurement does not contain mapping information!")
            return None

        self._fid.seek(pos + Offsets.wmap_origin)
        self._meta.map_area_type = MapAreaType(self._read_type("int32"))
        _ = self._read_type("int32")
        x_offset = self._read_type("float")
        y_offset = self._read_type("float")
        z_offset = self._read_type("float")
        x_increment = self._read_type("float")
        y_increment = self._read_type("float")
        z_increment = self._read_type("float")
        x_size = self._read_type("int32")
        y_size = self._read_type("int32")
        z_size = self._read_type("int32")
        linefocus_size = self._read_type("int32")

        self._meta.map_info = Adict(
            x=Adict(offset=x_offset, increment=x_increment, size=x_size),
            y=Adict(offset=y_offset, increment=y_increment, size=y_size),
            z=Adict(offset=z_offset, increment=z_increment, size=z_size),
            linefocus_size=linefocus_size,
        )

        # return map shape
        return x_size, y_size

    def _parse_data(self, start=0, end=-1):
        """Get information from DATA block."""
        if end == -1:  # take all spectra
            end = self._meta.count - 1
        if (start not in range(self._meta.count)) or (
            end not in range(self._meta.count)
        ):
            raise ValueError("Wrong start and end indices of spectra!")
        if start > end:
            raise ValueError("Start cannot be larger than end!")

        # Determine how many points to read
        points = self._meta.point_per_spectrum

        # Determine start position
        _, pos, _ = self._block_info["DATA"]
        pos_start = pos + Offsets.block_data + LenType["l_float"].value * start * points
        n_row = end - start + 1
        self._fid.seek(pos_start)
        data = fromfile(self._fid, dtype="float32", count=n_row * points)
        return np.array(data, dtype=float, ndmin=2)

    def _parse_img(self):
        """
        Extract the white-light JPEG image.

        The size of while-light image is coded in its EXIF.
        Use PIL to parse the EXIF information.
        """
        try:
            _, pos, size = self._block_info["WHTL"]
        except KeyError:
            debug_("The wdf file does not contain an image")
            return

        # Read the bytes. `self._meta.img` is a wrapped IO object mimicking a file
        self._fid.seek(pos + Offsets.jpeg_header)
        img_bytes = self._fid.read(size - Offsets.jpeg_header)
        self._meta.img = io.BytesIO(img_bytes)
        # Handle image dimension if PIL is present
        if PIL is not None:
            pil_img = Image.open(self._meta.img)
            # Weird missing header keys when Pillow >= 8.2.0.
            # see https://pillow.readthedocs.io/en/stable/releasenotes/8.2.0.html#image-getexif-exif-and-gps-ifd
            # Use fall-back _getexif method instead
            exif_header = dict(pil_img._getexif())
            try:
                # Get the width and height of image
                w_ = exif_header[ExifTags.FocalPlaneXResolution]
                h_ = exif_header[ExifTags.FocalPlaneYResolution]
                x_org_, y_org_ = exif_header[ExifTags.FocalPlaneXYOrigins]

                def rational2float(v):
                    """Pillow<7.2.0 returns tuple, Pillow>=7.2.0 returns IFDRational."""
                    if not isinstance(v, IFDRational):
                        return v[0] / v[1]
                    return float(v)

                w_, h_ = rational2float(w_), rational2float(h_)
                x_org_, y_org_ = rational2float(x_org_), rational2float(y_org_)

                # The dimensions (width, height)
                # with unit `img_dimension_unit`
                self._meta.img_dimensions = np.array([w_, h_])
                # Origin of image is at upper right corner
                self._meta.img_origins = np.array([x_org_, y_org_])
                # Default is microns (5)
                self._meta.img_dimension_unit = UnitType(
                    exif_header[ExifTags.FocalPlaneResolutionUnit],
                )
                # Give the box for cropping
                # Following the PIL manual
                # (left, upper, right, lower)
                self._meta.img_cropbox = self.__calc_crop_box()

            except KeyError:
                error_("Some keys in white light image header cannot be read!")
        return

    def _close(self):
        self._fid.close()
        if hasattr(self._meta, "img") and self._meta.img is not None:
            self._meta.img.close()

    def _get_type_string(self, attr, data_type):
        """Get the enumerated-data_type as string."""
        val = getattr(self, attr)  # No error checking
        if data_type is None:
            return val
        return data_type(val).name

    def _read_type(self, type, size=1):
        """Unpack struct data for certain type."""
        if type in ["int16", "int32", "int64", "float", "double"]:
            if size > 1:
                raise NotImplementedError(
                    "Does not support read number type with size >1",
                )
            # unpack into unsigned values
            fmt_out = LenType["s_" + type].value
            fmt_in = LenType["l_" + type].value

            return struct.unpack(fmt_out, self._fid.read(fmt_in * size))[0]

        if type == "utf8":
            # Read utf8 string with determined size block
            return self._fid.read(size).decode("utf8").replace("\x00", "")

        raise ValueError("Unknown data length format!")

    @property
    def _is_completed(self):
        # If count < capacity, this measurement is not completed
        return self._meta.count == self._meta.capacity

    def __calc_crop_box(self):
        """Calculate crop box."""

        def _proportion(x, minmax, pixels):
            """Get proportional pixels."""
            min, max = minmax
            return int(pixels * (x - min) / (max - min))

        pil_img = PIL.Image.open(self._meta.img)
        w_, h_ = self._meta.img_dimensions
        x0_, y0_ = self._meta.img_origins
        pw = pil_img.width
        ph = pil_img.height
        map_xl = self.xpos.min()
        map_xr = self.xpos.max()
        map_yt = self.ypos.min()
        map_yb = self.ypos.max()
        left = _proportion(map_xl, (x0_, x0_ + w_), pw)
        right = _proportion(map_xr, (x0_, x0_ + w_), pw)
        top = _proportion(map_yt, (y0_, y0_ + h_), ph)
        bottom = _proportion(map_yb, (y0_, y0_ + h_), ph)
        return (left, top, right, bottom)

    def _reshape_data(self, data, map_shape=None):
        """Reshape spectra into w x h x points if mapping data else count x points."""
        count = self._meta.count
        points = self._meta.point_per_spectrum
        if not self._is_completed:
            warning_(
                "The measurement is not completed, "
                "will try to reshape spectra into count x pps.",
            )
            try:
                data = np.reshape(data, (count, points))
            except ValueError:
                error_("Reshaping spectra array failed..")
                return None

        elif map_shape is not None:
            # Is a mapping
            w, h = map_shape
            if w * h != count:
                debug_(
                    "Mapping information from WMAP block not"
                    " corresponding to ORGN block! ",
                )
                error_("Can't reshape the spectra with the given mapping information.")
                return None

            if w * h * points != data.size:
                debug_(
                    "Mapping information from WMAP"
                    " not corresponding to DATA! "
                    "Will not reshape the spectra",
                )
                error_("Reshaping spectra array failed.")
                return None

            # Should be h rows * w columns. np.ndarray is row first
            # Reshape to 3D matrix when doing 2D mapping
            if (h > 1) and (w > 1):
                data = np.reshape(data, (h, w, points))
            # otherwise it is a line scan
            else:
                data = np.reshape(data, (count, points))

        # For any other type of measurement, reshape into (counts, point_per_spectrum)
        # example: series scan
        elif count > 1:
            data = np.reshape(data, (count, points))

        return data


# --------------------------------------------------------------------------------------
# Declaration of DATA types
# --------------------------------------------------------------------------------------
class LenType(Enum):
    l_int16 = 2
    l_int32 = 4
    l_int64 = 8
    s_int16 = "<H"  # unsigned short int
    s_int32 = "<I"  # unsigned int32
    s_int64 = "<Q"  # unsigned int64
    l_float = 4
    s_float = "<f"
    l_double = 8
    s_double = "<d"


class MeasurementType(IntEnum):
    Unspecified = 0
    Single = 1
    Series = 2
    Mapping = 3

    def __str__(self):
        return self._name_


class ScanType(IntEnum):
    Unspecified = 0
    Static = 1
    Continuous = 2
    StepRepeat = 3
    FilterScan = 4
    FilterImage = 5
    StreamLine = 6
    StreamLineHR = 7
    PointDetector = 8

    def __str__(self):
        return self._name_


class UnitType(IntEnum):
    Arbitrary = 0
    RamanShift = 1  # cm^-1 by default
    Wavelength = 2  # nm
    Nanometre = 3
    ElectronVolt = 4
    Micron = 5  # same for EXIF units
    Counts = 6
    Electrons = 7
    Millimetres = 8
    Metres = 9
    Kelvin = 10
    Pascal = 11
    Seconds = 12
    Milliseconds = 13
    Hours = 14
    Days = 15
    Pixels = 16
    Intensity = 17
    RelativeIntensity = 18
    Degrees = 19
    Radians = 20
    Celsius = 21
    Fahrenheit = 22
    KelvinPerMinute = 23
    AcquisitionTime = 24
    Microseconds = 25

    def __str__(self):
        """Rewrite the unit name output."""
        unit_str = {
            "Arbitrary": "",
            "RamanShift": "1/cm",  # cm^-1 by default
            "Wavelength": "nm",  # nm
            "Nanometre": "nm",
            "ElectronVolt": "eV",
            "Micron": "um",  # same for EXIF units
            "Counts": "counts",
            "Electrons": "electrons",
            "Millimetres": "mm",
            "Metres": "m",
            "Kelvin": "K",
            "Pascal": "Pa",
            "Seconds": "s",
            "Milliseconds": "ms",
            "Hours": "h",
            "Days": "d",
            "Pixels": "px",
            "Intensity": "",
            "RelativeIntensity": "",
            "Degrees": "°",
            "Radians": "rad",
            "Celsius": "°C",
            "Fahrenheit": "°F",
            "KelvinPerMinute": "K/min",
            "AcquisitionTime": "us",
            "Microseconds": "us",
        }
        return unit_str[self._name_]


class MapAreaType(IntEnum):
    Unspecified = 0  # (find in some test files)
    RandomPoints = 1  # rectangle area
    ColumnMajor = 2  # X first then Y.
    Alternating = 4  # raster or snake
    LineFocusMapping = 8  # see also linefocus_height
    SurfaceProfile = 64  # Z data is non-regular (surface maps)
    XYLine = 128  # line or depth slice forming a single line along
    # the XY plane


class DataType(IntEnum):
    Arbitrary = 0
    Raman_Shift = 1
    Intensity = 2
    X = 3
    Y = 4
    Z = 5
    R = 6
    Theta = 7
    Phi = 8
    Temperature = 9
    Pressure = 10
    Time = 11
    Derived = 12
    Polarization = 13
    FocusTrack = 14
    RampRate = 15
    Checksum = 16
    Flags = 17
    ElapsedTime = 18
    Spectral = 19
    Mp_Well_Spatial_X = 22
    Mp_Well_Spatial_Y = 23
    Mp_LocationIndex = 24
    Mp_WellReference = 25
    EndMarker = 26
    ExposureTime = 27

    def __str__(self):
        return self._name_


class Offsets(IntEnum):
    """Offsets to the start of block."""

    # General offsets
    block_name = 0x0
    block_id = 0x4
    block_data = 0x10
    # offsets in WDF1 block
    measurement_info = 0x3C  #
    spectral_info = 0x98
    file_info = 0xD0
    usr_name = 0xF0
    data_block = 0x200
    # offsets in ORGN block
    origin_info = 0x14
    origin_increment = 0x18
    # offsets in WMAP block
    wmap_origin = 0x10
    wmap_wh = 0x30
    # offsets in WHTL block
    jpeg_header = 0x10


class ExifTags(IntEnum):
    """Customized EXIF TAGS."""

    # Standard EXIF TAGS
    FocalPlaneXResolution = 0xA20E
    FocalPlaneYResolution = 0xA20F
    FocalPlaneResolutionUnit = 0xA210
    # Customized EXIF TAGS from Renishaw
    FocalPlaneXYOrigins = 0xFEA0
    FieldOfViewXY = 0xFEA1


# ======================================================================================
# Public functions
# ======================================================================================
def read_wire(*paths, **kwargs):
    r"""
    Open Renishaw Wire files.

    This is the explicit WiRE reader in the public import API. Use
    :func:`spectrochempy.read` for generic format autodetection and
    ``scp.wire.read(...)`` or :func:`spectrochempy.read_wire` when the WiRE
    format is already known.

    Non-merged multi-file reads may return a list-like `ScpObjectList`
    exposing helper methods for dataset selection. See
    :func:`spectrochempy.read` for the complete description of the generic
    import convention and multi-object return behavior.

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
    object : `NDDataset` or `ScpObjectList` of `NDDataset`
        The returned dataset(s). When several datasets are returned, the
        result is a list-like `ScpObjectList` with helper attributes such as
        ``.names``, ``.select_largest()``, ``.select_by_name()``,
        ``.filter_by_ndim()``, and ``.filter_by_shape()``.

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
    :func:`spectrochempy.read_zip` : Read Zip archives (containing spectrochempy readable files)
    :func:`spectrochempy.read_dir` : Read an entire directory.
    :func:`spectrochempy.read_opus` : Read OPUS spectra.
    :func:`spectrochempy.read_labspec` : Read Raman LABSPEC spectra (:file:`.txt`).
    :func:`spectrochempy.read_omnic` : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    :func:`spectrochempy.read_soc` : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_jcamp` : Read Infrared JCAMP-DX files (:file:`.jdx`, :file:`.dx`).

    Examples
    --------
    Reading a single Wire file

    >>> scp.read_wire('irdata/wire/white_light.wdf')
    NDDataset: [float64] a.u. (shape: (y:1, x:261))

    Using the explicit namespace API

    >>> scp.wire.read('irdata/wire/white_light.wdf')
    NDDataset: [float64] a.u. (shape: (y:1, x:261))

    """
    kwargs["filetypes"] = ["Renishaw Wire files (*.wdf)"]
    kwargs["protocol"] = ["wire"]
    importer = Importer()
    return importer(*paths, **kwargs)


read_wdf = read_wire


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_wdf(*args, **kwargs):
    # read WiRe *.wdf files or series

    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    reader = _wdfReader(fid, dataset)
    dataset = reader.dataset
    if dataset is None:
        error_(f"The {filename.stem} file is not readable!")
        return None
    dataset.name = filename.stem
    dataset.filename = filename
    dataset.history = f"Imported from {filename.name}"
    return dataset
