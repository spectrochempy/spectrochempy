# ruff: noqa
import errno
import io
import os
import struct

import numpy as np

from spectrochempy.extern.brukeropus.file.constants import STRUCT_3D_INFO_BLOCK
from spectrochempy.extern.brukeropus.file.constants import SUBREPORT_TYPE_FMT

__docformat__ = "google"


def read_opus_file_bytes(filepath) -> bytes:
    """Returns `bytes` of an OPUS file specified by `filepath` (or `None`).

    Function determines if `filepath` points to an OPUS file by reading the first four bytes which are always the same
    for OPUS files.  If `filepath` is not a file, or points to a non-OPUS file, the function returns `None`.  Otherwise
    the function returns the entire file as raw `bytes`.

    Args:
        filepath (str or Path): full filepath to OPUS file

    Returns:
        **filebytes (bytes):** raw bytes of OPUS file or `None` (if filepath does not point to an OPUS file)
    """
    filebytes = None
    if isinstance(filepath, (str, os.PathLike)):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
        with open(filepath, "rb") as f:
            first_four = f.read(4)
            if first_four == b"\n\n\xfe\xfe":
                filebytes = first_four + f.read()
        return filebytes
    if not isinstance(filepath, (io.BufferedReader, io.BytesIO)):
        raise TypeError("filepath must be a path or a binary file object")

    try:
        first_four = filepath.read(4)
        if first_four == b"\n\n\xfe\xfe":
            filebytes = first_four + filepath.read()
    except (OSError, ValueError):
        pass  # Empty file or a closed file object.
    return filebytes


def get_block_type(type_int: int) -> tuple:
    """Converts an int32 block type code to a six-integer tuple `block_type`.

    This function is used to decode the `type_int` from the directory block of an OPUS file into a tuple of integers.
    Each integer in the tuple provides information about the associated data block.

    Args:
        type_int: 32-bit integer decoded from file directory block

    Returns:
        **block_type (tuple):** six-integer tuple which specifies the block type
    """
    type_bit_str = format(type_int, "#034b")  # binary representation as string
    block_type = (
        int(type_bit_str[-2:], 2),
        int(type_bit_str[-4:-2], 2),
        int(type_bit_str[-10:-4], 2),
        int(type_bit_str[-17:-10], 2),
        int(type_bit_str[-19:-17], 2),
        int(type_bit_str[-22:-19], 2),
    )
    return block_type


def decode_str(size: int, blockbytes: bytes, offset: int) -> str:
    """Decode string that is packed as bytes in `blockbytes` starting from `offset`.

    Strings are frequently stored in OPUS files with a size designation that is larger than the actual string. The end
    of the string is designated by a terminator byte: b'\x00'. This function unpacks the string using the size
    designator, truncates at the terminator byte if found, and decodes as "latin-1"

    Args:
        size: size (number of bytes) of the string
        blockbytes: raw bytes of an OPUS file block
        offset: offset location where string begins in blockbytes

    Returns:
        string: decoded string
    """
    fmt = "<" + str(size) + "s"
    try:
        val = struct.unpack_from(fmt, blockbytes, offset)[0]
        x00_pos = val.find(b"\x00")
        if x00_pos != -1:
            val = val[:x00_pos].decode("latin-1")
        else:
            val = val.decode("latin-1")
    except Exception as e:
        val = "Failed to decode: " + str(e)
    return val


def parse_header(filebytes: bytes) -> tuple:
    """Parses the OPUS file header.

    The header of an OPUS file contains some basic information about the file including the version number, location of
    the directory block, and number of blocks in the file. This header is first to be parsed as it specifies how to
    read the file directory block (which contains information about each block in the file)

    Args:
        filebytes: raw bytes of OPUS file (all bytes)

    Returns:
        **header_info (tuple):**
            (
                **version (float64):** program version number as a floating-point date (later versions always greater)
                **directory_start (int32):** pointer to start location of directory block (number of bytes)
                **max_blocks (int32):** maximum number of blocks supported by the directory block (this should only be
                    relevant when trying to edit an OPUS file, i.e. when adding data blocks to a file)
                **num_blocks (int32):** total number of blocks in the opus file
            )
    """
    version = struct.unpack_from("<d", filebytes, 4)[0]
    directory_start = struct.unpack_from("<i", filebytes, 12)[0]
    max_blocks = struct.unpack_from("<i", filebytes, 16)[0]
    num_blocks = struct.unpack_from("<i", filebytes, 20)[0]
    return version, directory_start, max_blocks, num_blocks


def parse_directory(blockbytes: bytes) -> list:
    """Parses directory block of OPUS file and returns a list of block info tuples: (type, size, start).

    The directory block of an OPUS file contains information about every block in the file. The block information is
    stored as three int32 values: `type_int`, `size_int`, `start`.  `type_int` is an integer representation of the block
    type. The bits of this `type_int` have meaning and are parsed into a tuple using `get_block_type`. The `size_int` is
    the size of the block in 32-bit words. `start` is the starting location of the block (in number of bytes).

    Args:
        blockbytes: raw bytes of an OPUS file directory block

    Returns:
        **blocks (list):** list of block_info tuples
            **block_info (tuple):**
                (
                    **block_type (tuple):** six-integer tuple which specifies the block type (see: `get_block_type`)
                    **size (int):** size (number of bytes) of the block
                    **start (int):** pointer to start location of the block (number of bytes)
                )
    """
    loc = 0
    blocks = []
    while loc < len(blockbytes):
        type_int, size_int, start = struct.unpack_from("<3i", blockbytes, loc)
        loc = loc + 12
        if start > 0:
            block_type = get_block_type(type_int)
            size = size_int * 4
            blocks.append((block_type, size, start))
        else:
            break
    return blocks


def parse_params(blockbytes: bytes) -> dict:
    """Parses the bytes in a parameter block and returns a dict containing the decoded keys and vals.

    Parameter blocks are in the form: `XXX`, `dtype_code`, `size`, `val`.  `XXX` is a three char abbreviation of the
    parameter (key). The value of the parameter is decoded according to the `dtype_code` and size integers to be either:
    `int`, `float`, or `string`.

    Args:
        blockbytes: raw bytes of an OPUS file parameter block

    Returns:
        **items (tuple):** (key, value) pairs where key is three char string (lowercase) and value can be `int`, `float`
            or `string`.
    """
    loc = 0
    params = dict()
    while loc < len(blockbytes):
        key = blockbytes[loc : loc + 3].decode("utf-8")
        if key == "END":
            break
        dtype_code, val_size = struct.unpack_from("<2h", blockbytes[loc + 4 : loc + 8])
        val_size = val_size * 2
        if dtype_code == 0:
            val = struct.unpack_from("<i", blockbytes, loc + 8)[0]
        elif dtype_code == 1:
            val = struct.unpack_from("<d", blockbytes, loc + 8)[0]
        else:
            val = decode_str(val_size, blockbytes, loc + 8)
        params[key.lower()] = val
        loc = loc + val_size + 8
    return params


def get_dpf_dtype_count(dpf: int, size: int) -> tuple:
    """Returns numpy dtype and array count from the data point format (dpf) and block size (in bytes).

    Args:
        dpf: data point format integer stored in data status block.
            dpf = 1 -> array of float32
            dpf = 2 -> array of int32
        size: Block size in bytes.

    Returns:
        **dtype (numpy.dtype):** `numpy` dtype for defining an `ndarray` to store the data
        **count (int):** length of array calculated from the block size and byte size of the dtype.
    """
    if dpf == 1:
        dtype = np.float32
        count = round(size / 4)
    elif dpf == 2:
        dtype = np.int32
        count = round(size / 4)
    else:
        print("Unknown Data Point Format Requested:", dpf, "[using default: `float32`]")
        dtype = np.float32
        count = round(size / 4)
    return dtype, count


def parse_data(blockbytes: bytes, dpf: int = 1) -> np.ndarray:
    """Parses the bytes in a data block and returns a `numpy` array.

    Data blocks contain no metadata, only the y-values of a data array. Data arrays include: single-channel sample,
    reference, phase, interferograms, and a variety of resultant data (transmission, absorption, etc.).  Every data
    block should have a corresponding data status parameter block which can be used to generate the x-array values for
    the data block. The data status block also specifies the data type of the data array with the `DPF` parameter. It
    appears that OPUS currently exclusively stores data blocks as 32-bit floats, but has a reservation for 32-bit
    integers when `DPF` = 2.

    Args:
        blockbytes: raw bytes of data block
        dpf: data-point-format integer stored in corresponding data status block.

    Returns:
        **y_array (numpy.ndarray):** `numpy` array of y values contained in the data block
    """
    dtype, count = get_dpf_dtype_count(dpf=dpf, size=len(blockbytes))
    return np.frombuffer(blockbytes, dtype=dtype, count=count)


def parse_data_compact(blockbytes: bytes, npt: int, dpf: int = 1) -> np.ndarray:
    """Parses the bytes in a data compact block and returns a `numpy` array.

    Some data blocks are stored in the "Compact" block format that includes some metadata that preceeds the raw data. At
    this time, the metadata is ignored, and the compact spectra is extracted from the last bytes of the block that fit
    the known array size (specified with "npt" in corresponding data status block).

    Args:
        blockbytes: raw bytes of data block
        npt: number of data points in the spectra (from data status block)
        dpf: data-point-format integer stored in corresponding data status block.

    Returns:
        **y_array (numpy.ndarray):** `numpy` array of y values contained in the data block
    """
    dtype, count = get_dpf_dtype_count(dpf=dpf, size=len(blockbytes))
    return np.frombuffer(blockbytes, dtype=dtype, count=count)


def parse_data_series(blockbytes: bytes, dpf: int = 1) -> dict:
    """Parses the bytes in a 3D data block (series of spectra) and returns a data `dict` containing data and metadata.

    3D data blocks are structured differently than standard data blocks. In addition to the series of spectra, they
    include metadata for each of the spectrum.  This function returns a `dict` containing all the extracted information
    from the data block.  The series spectra is formed into a 2D array while metadata captured for each spectra is
    formed into a 1D array (length = number of spectral measurements in the series).

    Args:
        blockbytes: raw bytes of the data series block
        dpf: data-point-format integer stored in corresponding data status block.

    Returns:
        **data_dict (dict):** `dict` containing all extracted information from the data block
            {
                **version:** file format version number (should be 0)
                **num_blocks:** number of sub blocks; each sub block features a data spectra and associated metadata
                **offset:** offset in bytes to the first sub data block
                **data_size:** size in bytes of each sub data block
                **info_size:** size in bytes of the metadata info block immediately following the sub data block
                **store_table:** run numbers of the first and last blocks to keep track of skipped spectra
                **y:** 2D `numpy` array containing all spectra (C-order)
                **metadata arrays:** series of metadata arrays in 1D array format (e.g. `npt`, `mny`, `mxy`, `ert`).
                    The most useful one is generally `ert`, which can be used as the time axis for 3D data plots.
            }
    """
    header = struct.unpack_from("<6i", blockbytes, 0)
    data = {
        "version": header[0],
        "num_blocks": header[1],
        "offset": header[2],
        "data_size": header[3],
        "info_size": header[4],
    }
    data["store_table"] = [
        struct.unpack_from("<2i", blockbytes, 24 + i * 8) for i in range(header[5])
    ]
    dtype, count = get_dpf_dtype_count(dpf, data["data_size"])
    data["y"] = np.zeros((data["num_blocks"], count), dtype=dtype)
    for entry in STRUCT_3D_INFO_BLOCK:
        data[entry["key"]] = np.zeros((data["num_blocks"]), dtype=entry["dtype"])
    offset = data["offset"]
    num_spectra = 0
    for i in range(data["num_blocks"]):
        data["y"][i] = np.frombuffer(blockbytes[offset:], dtype=dtype, count=count)
        offset = offset + data["data_size"]
        info_vals = struct.unpack_from(
            "<" + "".join([e["fmt"] for e in STRUCT_3D_INFO_BLOCK]), blockbytes, offset
        )
        for j, entry in enumerate(STRUCT_3D_INFO_BLOCK):
            data[entry["key"]][i] = info_vals[j]
        offset = offset + data["info_size"]
        num_spectra = i + 1
        if offset >= len(blockbytes):
            break  # Not all blocks are necessarily stored (see Store Table)
    data["y"] = data["y"][:num_spectra]
    for entry in STRUCT_3D_INFO_BLOCK:
        data[entry["key"]] = data[entry["key"]][:num_spectra]
    return data


def parse_text(block_bytes: bytes, encoding="utf-8", backup_encoding="utf-16") -> str:
    """Parses and OPUS file block as text (e.g. history or file-log block).

    The history (aka file-log) block of an OPUS file contains some information about how the file was generated and
    edits that have been performed on the file.  This function parses the text block but does not take any steps to
    parameterizing what is contained in the text.  The history block is generally not needed to retrieve the file data
    and metadata, but might be useful for inspecting the file.

    Args:
        blockbytes: raw bytes of the text block (e.g. history or file-log)
        encoding: string represting text encoding type. Can be set to "utf-16" for chinese character support
        backup_encoding: if default encoding fails, this encoding string will be attempted.

    Returns:
        text: string of text contained in the file block.
    """
    byte_string = struct.unpack("<" + str(len(block_bytes)) + "s", block_bytes)[0]
    byte_strings = byte_string.split(b"\x00")
    strings = []
    for entry in byte_strings:
        if entry != b"":
            try:
                strings.append(entry.decode(encoding))
            except Exception:
                try:
                    strings.append(entry.decode(backup_encoding))
                except Exception:
                    try:
                        strings.append(entry.decode("latin-1"))
                    except Exception as e:
                        strings.append("<Decode Exception>: " + str(e))
    return "\n".join(strings)


def parse_subreport(subreport_bytes: bytes) -> dict:
    """Parses the bytes of a subreport and returns the extracted data as a dictionary

    Subreports are contained within a report block (e.g. Multi-Evaluation Test Report). A report can contain multiple
    subreports, and they generally follow a table format. This sub-block is organized with a mini parameter block
    followed by packed data. The mini parameter block contains information about how to read the packed data:
        nco: number of columns
        nln: number of rows
        siz: size of mini parameter block (number of bytes)
        src: size in bytes of entire row of data (offset for extracting column data from row 2, 3 ...)
        f00, f01 ... fxx: start position of data in column 0, 1 ... xx (relative to end of mini param block)
        t00, t01 ... txx: integer representing type of data (e.g. int32, float32, float64, str, etc.)
        s00, s01 ... sxx: column header label
        p00, p01 ... pxx: formatting string for numbers in column 0, 1 ... xx (not included for every column)

    Args:
        subreport_bytes: raw bytes of the subreport. Needs to start precisely where subreport begins, but can include
        data beyond the end of the subreport (i.e. end of subreport does not need to be determined a priori).

    Returns:
        **subreport (dict):** `dict` containing subreport data and extraction/formatting parameters
            {
                **info:** `dict` of parameters extracted directly from subreport that describes how to read the data
                    table and provides some basic metadata about the table (e.g. column header labels).
                **data:** `list` of lists of data (table format) contained in the subreport
            }
    """
    info = parse_params(subreport_bytes)
    data = []
    for row in range(info["nln"]):
        data.append([])
        for col in range(info["nco"]):
            offset = info["siz"] + row * info["src"] + info["f" + f"{col:02}"]
            type_int = info["t" + f"{col:02}"]
            if col < info["nco"] - 1:
                size = min(
                    [
                        type_int - 1000,
                        info["f" + f"{col + 1:02}"] - info["f" + f"{col:02}"],
                    ]
                )
            else:
                size = info["src"] - info["f" + f"{col:02}"]
            if type_int > 1000:
                val = decode_str(size, subreport_bytes, offset)
            elif type_int in SUBREPORT_TYPE_FMT.keys():
                fmt = SUBREPORT_TYPE_FMT[type_int]
                val = struct.unpack_from(fmt, subreport_bytes, offset)[0]
            else:
                val = subreport_bytes[offset : offset + size]
            data[row].append(val)
    return {"info": info, "data": data}


def parse_report(blockbytes: bytes) -> dict:
    """Parses the report block of an OPUS file, such as Multi-Evaluation test reports, returning the report as a dict.

    Report blocks are formatted in a very general way, potentially enabling a variety of different report structures.
    This algorithm was developed using several OPUS files with a variety of different Multi-Evaluation Test Reports.
    It is possible that other classes of test reports could be generated by OPUS that might change the structure, but
    the overal organization and decoding methods should be similar.  In particular, the report block structure might
    support multiple reports, but no such file has been available for testing to date.  This algorithm will extract a
    single report and all the associated subreports.

    Report blocks start with a mini parameter block that begins after the 12th byte.  It contains the following:
        tit: Title of the report
        f00: Starting position of the report summary data
        Known unused parameters: bid, nrp, siz, e00, z00
    This is followed by the report summary. For a multi-evaluation test report, this is a pair of tables summarizing the
    methods applied to the spectrum.  It also specifies the number of subreports that follow, and the starting position
    and title of each subreport. Some of the keys in this parameter set are described in the `parse_subreport` method.
    Other parameters in the report summary include:
        sub: Number of subreports
        h00, h01 ... hxx: header labels of first summary table
        v00, v01 ... vxx: corresponding values of first summary table
        g00, g01 ... gxx: starting positions of each subreport relative to the start of this report summary
        u00, u02 ... uxx: titles of each subreport
    It should be noted that the only class of reports used for testing this algorithm were a variety of multi-evaluation
    test reports. It is possible there are other similar report blocks OPUS writes that follow a similar structure but
    could vary in some way that is incompatible with some of the assumptions made by this algorithm.

    Args:
        blockbytes: raw bytes of an OPUS file report block

    Returns:
        **report (dict):** `dict` containing report and subreport data
            {
                **header:** `dict` of parameters from first mini param block
                **info:** `dict` of parameters extracted from second mini param block
                **data:** `list` of data that comprises second summary table
                **subreports:** `list` of subreport `dict`s beneath the main report.
            }
    """
    header_ints = struct.unpack("<3i", blockbytes[:12])
    header_info = parse_params(blockbytes[12:])
    header_info["ints"] = header_ints
    report_info = parse_subreport(blockbytes[header_info["f00"] :])
    report = {
        "header": header_info,
        "info": report_info["info"],
        "data": report_info["data"],
    }
    subreports = []
    for idx in range(report_info["info"]["sub"]):
        offset = header_info["f00"] + report_info["info"]["g" + f"{idx:02}"]
        subreports.append(parse_subreport(blockbytes[offset:]))
    report["subreports"] = subreports
    return report
