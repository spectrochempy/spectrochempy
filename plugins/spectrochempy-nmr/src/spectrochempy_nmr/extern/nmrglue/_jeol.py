# ======================================================================================
# Vendored from nmrglue.fileio.jeol (BSD 3-Clause, Jonathan J. Helmus).
#
# Functions for reading JEOL JDF files.  Adapted for vendoring into
# spectrochempy-nmr to avoid a runtime dependency on the full nmrglue package.
#
# Original: https://github.com/jjhelmus/nmrglue
# License: see NMRGLUE_LICENSE.rst in this plugin directory.
# ======================================================================================

import struct
from warnings import warn

import numpy as np

from ._base import create_blank_udic
from ._base import reorder_submatrix


def read_jeol(fname):
    """
    Read a JDF (JEOL Data Format) file.

    Parameters
    ----------
    fname : str or path
        Path to the .jdf file.

    Returns
    -------
    dic : dict
        Dictionary with ``header`` and ``parameters`` sub-dicts.
    data : numpy.ndarray
        NMR data array.

    """
    with open(fname, "rb") as f:
        buffer = f.read()

    dic = parse_jeol(buffer)
    data = read_bin_data(dic, buffer)
    data = reorganize(dic, data)
    if isinstance(data, np.ndarray):
        data = truncate_data(dic, data)

    return dic, data


def guess_udic(dic, data):
    """
    Guess parameters of universal dictionary from dic, data pair.

    Parameters
    ----------
    dic : dict
        Dictionary with JEOL header and parameters.
    data : numpy.ndarray
        Array of NMR data.

    Returns
    -------
    udic : dict
        Universal dictionary of spectral parameters.

    """
    _dims = ndims(dic)
    udic = create_blank_udic(_dims)

    for b_dim in range(_dims):
        udic[b_dim]["size"] = data.shape[b_dim]
        try:
            udic = add_axis_to_udic(udic, dic, b_dim)
        except Exception as e:
            error_message = "\n".join(e.args)
            warn(
                f"Failed to determine udic parameters for dim: {b_dim} "
                f"with the following error: {error_message}",
                stacklevel=2,
            )

    return udic


def add_axis_to_udic(udic, dic, udim):
    """Add dimension information to a universal dictionary."""
    axis_prefix = dimension_names(dic)[udim]
    udic[udim]["sw"] = dic["parameters"][f"{axis_prefix}_sweep"]
    udic[udim]["obs"] = dic["parameters"][f"{axis_prefix}_freq"] / 1e6
    udic[udim]["car"] = dic["parameters"][f"{axis_prefix}_offset"] * udic[udim]["obs"]
    udic[udim]["label"] = dic["parameters"][f"{axis_prefix}_domain"]
    udic[udim]["encoding"] = dic["header"]["data_axis_type"][udim]
    return udic


def reorganize(dic, bin_data):
    """Reorganize raw binary data for different dimension counts."""
    dim = ndims(dic)
    if dim == 1:
        return reorganize_1d(dic, bin_data)
    if dim == 2 and dic["header"]["data_format"] == "two_d":
        return reorganize_2d(dic, bin_data)
    if dim == 2 and dic["header"]["data_format"] == "small_two_d":
        raise NotImplementedError(
            "Datasets of the type 'small_two_d' have not been tested yet."
        )
    if dim > 2:
        raise NotImplementedError(
            "Datasets with dimensions greater than 2 have not been tested yet."
        )
    return None


def reorganize_1d(dic, bin_data):
    """Reorganize 1D data into correctly ordered numpy array."""
    dic, sections = split_sections(dic, bin_data)
    _out_shape = dic["header"]["data_points"][:2][::-1]

    sections = [
        reorder_submatrix(
            data=s, shape=_out_shape, submatrix_shape=submatrix_shape(dic)
        )
        for s in sections
    ]

    _type = dic["header"]["data_axis_type"][0]
    _ls = len(sections)

    if (_ls == 1) and (_type == "real"):
        return sections[0]
    if (_ls == 2) and (_type == "complex"):
        return sections[0] - 1j * sections[1]
    warn(
        f"Inconsistent data found ({_ls} sections and type {_type}). "
        f"Returning data as {_ls} sections instead of an array.",
        stacklevel=2,
    )
    return sections


def reorganize_2d(dic, bin_data):
    """Reorganize 2D data into correctly ordered numpy array."""
    dic, sections = split_sections(dic, bin_data)
    _out_shape = dic["header"]["data_points"][:2][::-1]

    sections = [
        reorder_submatrix(
            data=s, shape=_out_shape, submatrix_shape=submatrix_shape(dic)
        )
        for s in sections
    ]

    _type = dic["header"]["data_axis_type"][:2]
    _ls = len(sections)

    if (_ls == 1) and (_type == ["real", "real"]):
        return sections[0]
    if (_ls == 2) and (
        (_type == ["real_complex", "real_complex"]) or (_type == ["complex", "real"])
    ):
        return sections[0] - 1j * sections[1]
    if (_ls == 4) and (_type == ["complex", "complex"]):
        real = sections[0] - 1j * sections[1]
        imag = sections[2] - 1j * sections[3]

        result = np.zeros((_out_shape[0] * 2, _out_shape[1]), dtype="complex")
        result[0::2] = real
        result[1::2] = -imag
        return result
    warn(
        f"Inconsistent data found ({_ls} sections and type {_type}). "
        f"Returning data as {_ls} sections instead of an array.",
        stacklevel=2,
    )
    return sections


def truncate_data(dic, data):
    """Truncate data to the declared data points per axis."""
    header = dic["header"]
    slices = []
    for i, (start, end) in enumerate(
        zip(header["data_offset_start"], header["data_offset_stop"], strict=False)
    ):
        if i > 0:
            end = (end + 1) * (nsections(dic) // 2)
            slices.append(slice(start, end))
        else:
            slices.append(slice(start, end + 1))
        if i == (ndims(dic) - 1):
            break
    slices = slices[::-1]
    return data[tuple(slices)]


def get_data_shape(dic):
    """Return the effective data shape (excluding singleton axes)."""
    shape = dic["header"]["data_points"]
    shape = [i for i in shape if i > 1]
    return shape[::-1]


def parse_jeol(buffer):
    """Parse the binary JDF buffer into header and parameters."""
    buffer = IOBuffer(buffer, conversion_table=ConversionTable)
    buffer, header = read_header(buffer)
    buffer, params = read_parameters(buffer, header["param_start"], header["endian"])
    return {"header": header, "parameters": params}


def read_header(buffer):
    """Read the JDF file header."""
    t = buffer.conversion_table
    header = {}

    header["file_identifier"] = buffer.read_chars(8)
    header["endian"] = t.endianness[buffer.read_int8()]
    header["major_version"] = buffer.read_uint8()
    header["minor_version"] = buffer.read_uint16()
    header["data_dimension_number"] = buffer.read_uint8()

    info = buffer.read_byte()
    header["data_dimension_exist"] = [bool(int(x)) for x in format(info, "08b")]

    info = buffer.read_byte()
    header["data_type"] = t.data_type[info >> 6]
    header["data_format"] = t.data_format[info & 0b00111111]
    header["submatrix_edge"] = t.submatrix_edge[header["data_format"]]
    header["instrument"] = t.instruments[buffer.read_int8()]
    header["translate"] = [buffer.read_int8() for i in range(8)]
    header["data_axis_type"] = [
        t.axis_type[i] for i in buffer.get_array("read_int8", 8)
    ]
    header["units"] = buffer.get_unit(8)
    header["title"] = buffer.get_string(124)

    info = []
    for i in buffer.get_array("read_uint8", 4):
        info.append(t.data_axis_ranged[i >> 4])
        info.append(t.data_axis_ranged[i & 0b00001111])
    header["data_axis_ranged"] = info

    header["data_points"] = buffer.get_array("read_uint32", 8)
    header["data_offset_start"] = buffer.get_array("read_uint32", 8)
    header["data_offset_stop"] = buffer.get_array("read_uint32", 8)
    header["data_axis_start"] = buffer.get_array("read_float64", 8)
    header["data_axis_stop"] = buffer.get_array("read_float64", 8)

    info = buffer.get_array("read_byte", 4)
    header["creation_time"] = {
        "year": 1990 + (info[0] >> 1),
        "month": ((info[0] << 3) & 0b00001000) + (info[1] >> 5),
        "day": info[2] & 0b00011111,
    }

    info = buffer.get_array("read_byte", 4)
    header["revision_time"] = {
        "year": 1990 + (info[0] >> 1),
        "month": ((info[0] << 3) & 0b00001000) + (info[1] >> 5),
        "day": info[2] & 0b00011111,
    }

    header["node_name"] = buffer.get_string(16)
    header["site"] = buffer.get_string(128)
    header["author"] = buffer.get_string(128)
    header["comment"] = buffer.get_string(128)
    header["data_axis_titles"] = buffer.get_array("get_string", 8, 32)
    header["base_freq"] = buffer.get_array("read_float64", 8)
    header["zero_point"] = buffer.get_array("read_float64", 8)
    header["reversed"] = buffer.get_array("read_boolean", 8)

    buffer.skip(3)
    header["annotation_ok"] = bool(buffer.read_byte() >> 7)
    header["history_used"] = buffer.read_uint32()
    header["history_length"] = buffer.read_uint32()
    header["param_start"] = buffer.read_uint32()
    header["param_length"] = buffer.read_uint32()
    header["list_start"] = buffer.get_array("read_uint32", 8)
    header["list_length"] = buffer.get_array("read_uint32", 8)
    header["data_start"] = buffer.read_uint32()

    header["data_length"] = (buffer.read_uint32() << 32) | (buffer.read_uint32())
    header["context_start"] = (buffer.read_uint32() << 32) | (buffer.read_uint32())
    header["context_length"] = buffer.read_uint32()
    header["annote_start"] = (buffer.read_uint32() << 32) | (buffer.read_uint32())
    header["annote_length"] = buffer.read_uint32()
    header["total_size"] = (buffer.read_uint32() << 32) | (buffer.read_uint32())
    header["unit_location"] = buffer.get_array("read_uint8", 8)

    info = []
    for _ in range(2):
        unit = []
        scaler = buffer.read_int16()
        for _ in range(5):
            byte = buffer.read_int16()
            unit.append(byte)
        info.append({"scaler": scaler, "unit": unit})
    header["compound_units"] = info

    return buffer, header


def read_parameters(buffer, param_start, endianness):
    """Read the JDF parameter block."""
    t = buffer.conversion_table

    if endianness == "little_endian":
        buffer.set_little_endian()

    buffer.position = param_start

    params = {
        "parameter_size": buffer.read_uint32(),
        "low_index": buffer.read_uint32(),
        "high_index": buffer.read_uint32(),
        "total_size": buffer.read_uint32(),
    }

    param_array = {}
    for _p in range(params["high_index"]):
        _class = buffer.get_array("read_byte", 4)
        unit_scaler = buffer.read_int16()
        unit = buffer.get_unit(5)
        buffer.skip(16)
        value_type = t.value_type[buffer.read_int32()]
        buffer.position -= 20

        value = None
        if value_type == "string":
            value = buffer.get_string(16).replace(" ", "")
        elif value_type == "integer":
            value = buffer.read_int32()
            buffer.skip(12)
        elif value_type == "float":
            value = buffer.read_float64()
            buffer.skip(8)
        elif value_type == "complex":
            value = buffer.read_float64() + 1j * buffer.read_float64()
        elif value_type == "infinity":
            value = buffer.read_int32()
            buffer.skip(12)
        else:
            buffer.skip(16)

        buffer.skip(4)

        name = buffer.get_string(28)
        bname = name.lower().replace(" ", "")

        param_array[bname] = {
            "_class": _class,
            "name": name,
            "unit_scaler": unit_scaler,
            "units": unit,
            "value": value,
            "value_type": value_type,
        }

        params[bname] = value * (10**unit_scaler)

    params["info"] = param_array

    return buffer, params


def read_bin_data(dic, buffer):
    """Read the raw binary data block."""
    buffer = IOBuffer(buffer, conversion_table=ConversionTable)

    if dic["header"]["endian"] == "little_endian":
        buffer.set_little_endian()
    elif dic["header"]["endian"] == "big_endian":
        buffer.set_big_endian()

    start = dic["header"]["data_start"]
    length = np.prod(dic["header"]["data_points"]) * nsections(dic)

    buffer.position = start
    data = buffer.get_array(f"read_{dic['header']['data_type']}", length)

    return np.array(data)


def ndims(dic):
    """Return the number of active dimensions."""
    return sum(bool(i) for i in dic["header"]["data_axis_type"])


def dimension_names(dic):
    """Return the axis letter names (x, y, ...) for existing dimensions."""
    return [
        i
        for i, j in zip("xyzabcde", dic["header"]["data_dimension_exist"], strict=False)
        if j
    ]


def nsections(dic):
    """Return the number of data sections (2^complex_dims)."""
    return 2 ** num_complex_dims(dic)


def split_sections(dic, data):
    """Split data into sections based on complex dimensions."""
    sections = data.reshape(nsections(dic), -1)
    return dic, list(sections)


def submatrix_shape(dic):
    """Return the submatrix shape for reordering."""
    submatrix_edge = ConversionTable.submatrix_edge[dic["header"]["data_format"]]
    return [submatrix_edge] * ndims(dic)


def num_complex_dims(dic):
    """Count the number of complex dimensions."""
    complex_dims = 0
    for dim in dic["header"]["data_axis_type"]:
        if dim == "complex":
            complex_dims += 1
        elif dim == "real_complex":
            return 1
    return complex_dims


class IOBuffer:
    """Low-level binary buffer reader with endianness support."""

    def __init__(self, buffer, conversion_table, endian=None):
        self.buffer = buffer
        self.conversion_table = conversion_table
        self.position = 0
        self.set_big_endian()

    def set_big_endian(self):
        self.endian = "big"
        self.e = ">"

    def set_little_endian(self):
        self.endian = "little"
        self.e = "<"

    def skip(self, count):
        self.position += count

    def read_chars(self, count):
        chars = struct.unpack_from(f"{self.e}{count}s", self.buffer, self.position)
        self.position += count
        return chars[0].decode("utf-8")

    def read_int8(self):
        value = struct.unpack_from(f"{self.e}b", self.buffer, self.position)
        self.position += 1
        return value[0]

    def read_uint8(self):
        value = struct.unpack_from(f"{self.e}B", self.buffer, self.position)
        self.position += 1
        return value[0]

    def read_uint16(self):
        value = struct.unpack_from(f"{self.e}H", self.buffer, self.position)
        self.position += 2
        return value[0]

    def read_int16(self):
        value = struct.unpack_from(f"{self.e}h", self.buffer, self.position)
        self.position += 2
        return value[0]

    def read_uint32(self):
        value = struct.unpack_from(f"{self.e}I", self.buffer, self.position)
        self.position += 4
        return value[0]

    def read_int32(self):
        value = struct.unpack_from(f"{self.e}i", self.buffer, self.position)
        self.position += 4
        return value[0]

    def read_float32(self):
        value = struct.unpack_from(f"{self.e}f", self.buffer, self.position)
        self.position += 4
        return value[0]

    def read_float64(self):
        value = struct.unpack_from(f"{self.e}d", self.buffer, self.position)
        self.position += 8
        return value[0]

    def read_byte(self):
        value = struct.unpack_from("B", self.buffer, self.position)
        self.position += 1
        return value[0]

    def read_boolean(self):
        value = struct.unpack_from("B", self.buffer, self.position)
        self.position += 1
        return bool(value[0])

    def get_array(self, read_func, count, *args, **kwargs):
        return [getattr(self, read_func)(*args, **kwargs) for _ in range(count)]

    def get_unit(self, size):
        unit = []
        for _i in range(size):
            byte = self.read_byte()
            prefix = self.conversion_table.prefix[byte >> 4]
            power = byte & 0b00001111
            base = self.conversion_table.base[self.read_int8()]
            unit.append((prefix, power, base))
        return unit

    def get_string(self, count):
        return self.read_chars(count).replace("\x00", "")


# Conversion Table for JEOL Data Format
class ConversionTable:
    endianness = {0: "big_endian", 1: "little_endian"}

    instruments = {
        0: None,
        1: "gsx",
        2: "alpha",
        3: "eclipse",
        4: "mass_spec",
        5: "compiler",
        6: "other_nmr",
        7: "unknown",
        8: "gemini",
        9: "unity",
        10: "aspect",
        11: "ux",
        12: "felix",
        13: "lambda",
        14: "ge_1280",
        15: "ge_omega",
        16: "chemagnetics",
        17: "cdff",
        18: "galactic",
        19: "triad",
        20: "generic_nmr",
        21: "gamma",
        22: "jcamp_dx",
        23: "amx",
        24: "dmx",
        25: "eca",
        26: "alice",
        27: "nmrpipe",
        28: "simpson",
    }

    data_type = {0: "float64", 1: "float32", 2: "reserved", 3: "reserved"}

    data_format = {
        1: "one_d",
        2: "two_d",
        3: "three_d",
        4: "four_d",
        5: "five_d",
        6: "six_d",
        7: "seven_d",
        8: "eight_d",
        9: "not for NMR data formats",
        10: "not for NMR data formats",
        11: "not for NMR data formats",
        12: "small_two_d",
        13: "small_three_d",
        14: "small_four_d",
    }

    axis_type = {
        0: None,
        1: "real",
        2: "tppi",
        3: "complex",
        4: "real_complex",
        5: "envelope",
    }

    prefix = {
        -8: "yotta",
        -7: "zetta",
        -6: "exa",
        -5: "pecta",
        -4: "tera",
        -3: "giga",
        -2: "mega",
        -1: "kilo",
        0: "none",
        1: "milli",
        2: "micro",
        3: "nano",
        4: "pico",
        5: "femto",
        6: "atto",
        7: "zepto",
        15: "None",
    }

    base = {
        0: None,
        1: "abundance",
        2: "ampere",
        3: "candela",
        4: "celsius",
        5: "coulomb",
        6: "degree",
        7: "electronvolt",
        8: "farad",
        9: "sievert",
        10: "gram",
        11: "gray",
        12: "henry",
        13: "hertz",
        14: "kelvin",
        15: "joule",
        16: "liter",
        17: "lumen",
        18: "lux",
        19: "meter",
        20: "mole",
        21: "newton",
        22: "ohm",
        23: "pascal",
        24: "percent",
        25: "point",
        26: "ppm",
        27: "radian",
        28: "second",
        29: "siemens",
        30: "steradian",
        31: "tesla",
        32: "volt",
        33: "watt",
        34: "weber",
        35: "decibel",
        36: "dalton",
        37: "thompson",
        38: "ugeneric",
        39: "lpercent",
        40: "ppt",
        41: "ppb",
        42: "index",
    }

    data_axis_ranged = {
        0: "ranged",
        1: "listed",
        2: "sparse",
        3: "listed",
    }

    value_type = {
        0: "string",
        1: "integer",
        2: "float",
        3: "complex",
        4: "infinity",
    }

    submatrix_edge = {
        "one_d": 8,
        "two_d": 32,
        "three_d": 8,
        "four_d": 8,
        "five_d": 4,
        "six_d": 4,
        "seven_d": 2,
        "eight_d": 2,
        "small_two_d": 4,
        "small_three_d": 4,
        "small_four_d": 4,
    }
