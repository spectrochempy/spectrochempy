import errno
import io
import os
import struct

import numpy as np

from spectrochempy.extern.brukeropus.file.constants import STRUCT_3D_INFO_BLOCK

__docformat__ = "google"


def read_opus_file_bytes(filepath) -> bytes:
    """
    Returns `bytes` of an OPUS file specified by `filepath` (or `None`).

    Function determines if `filepath` points to an OPUS file by reading the first four bytes which are always the same
    for OPUS files.  If `filepath` is not a file, or points to a non-OPUS file, the function returns `None`.  Otherwise
    the function returns the entire file as raw `bytes`.

    Args:
    ----
        filepath (str or Path): full filepath to OPUS file

    Returns:
    -------
        **filebytes (bytes):** raw bytes of OPUS file or `None` (if filepath does not point to an OPUS file)
    """
    # Modified original function to accept a BufferedReader or BytesIO object
    filebytes = None
    if isinstance(filepath, str):
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                try:
                    first_four = f.read(4)
                    if first_four == b"\n\n\xfe\xfe":
                        filebytes = first_four + f.read()
                except:  # noqa: E722, S110
                    pass  # Empty file (or file with fewer than 4 bytes)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
    elif isinstance(filepath, io.BufferedReader | io.BytesIO):
        try:
            first_four = filepath.read(4)
            if first_four == b"\n\n\xfe\xfe":
                filebytes = first_four + filepath.read()
        except:  # noqa: E722, S110
            pass  # Empty file (or file with fewer than 4 bytes)
    else:
        raise TypeError("filepath must be a string or a BufferedReader")
    return filebytes


def get_block_type(type_int: int) -> tuple:
    """
    Converts an int32 block type code to a six-integer tuple `block_type`.

    This function is used to decode the `type_int` from the directory block of an OPUS file into a tuple of integers.
    Each integer in the tuple provides information about the associated data block.

    Args:
    ----
        type_int: 32-bit integer decoded from file directory block

    Returns:
    -------
        **block_type (tuple):** six-integer tuple which specifies the block type
    """
    type_bit_str = format(type_int, "#034b")  # binary representation as string
    return (
        int(type_bit_str[-2:], 2),
        int(type_bit_str[-4:-2], 2),
        int(type_bit_str[-10:-4], 2),
        int(type_bit_str[-17:-10], 2),
        int(type_bit_str[-19:-17], 2),
        int(type_bit_str[-22:-19], 2),
    )


def parse_header(filebytes: bytes) -> tuple:
    """
    Parses the OPUS file header.

    The header of an OPUS file contains some basic information about the file including the version number, location of
    the directory block, and number of blocks in the file. This header is first to be parsed as it specifies how to
    read the file directory block (which contains information about each block in the file)

    Args:
    ----
        filebytes: raw bytes of OPUS file (all bytes)

    Returns:
    -------
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
    """
    Parses directory block of OPUS file and returns a list of block info tuples: (type, size, start).

    The directory block of an OPUS file contains information about every block in the file. The block information is
    stored as three int32 values: `type_int`, `size_int`, `start`.  `type_int` is an integer representation of the block
    type. The bits of this `type_int` have meaning and are parsed into a tuple using `get_block_type`. The `size_int` is
    the size of the block in 32-bit words. `start` is the starting location of the block (in number of bytes).

    Args:
    ----
        blockbytes: raw bytes of an OPUS file directory block

    Returns:
    -------
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
    """
    Parses the bytes in a parameter block and returns a dict containing the decoded keys and vals.

    Parameter blocks are in the form: `XXX`, `dtype_code`, `size`, `val`.  `XXX` is a three char abbreviation of the
    parameter (key). The value of the parameter is decoded according to the `dtype_code` and size integers to be either:
    `int`, `float`, or `string`.

    Args:
    ----
        blockbytes: raw bytes of an OPUS file parameter block

    Returns:
    -------
        **items (tuple):** (key, value) pairs where key is three char string (lowercase) and value can be `int`, `float`
            or `string`.
    """
    loc = 0
    params = {}
    while loc < len(blockbytes):
        key = blockbytes[loc : loc + 3].decode("utf-8")
        if key == "END":
            break
        dtype_code, val_size = struct.unpack_from("<2h", blockbytes[loc + 4 : loc + 8])
        val_size = val_size * 2
        if dtype_code == 0:
            fmt_str = "<i"
        elif dtype_code == 1:
            fmt_str = "<d"
        else:
            fmt_str = "<" + str(val_size) + "s"
        try:
            val = struct.unpack_from(fmt_str, blockbytes, loc + 8)[0]
            if "s" in fmt_str:
                x00_pos = val.find(b"\x00")
                if x00_pos != -1:
                    val = val[:x00_pos].decode("latin-1")
                else:
                    val = val.decode("latin-1")
        except Exception as e:
            val = "Failed to decode: " + str(e)
        params[key.lower()] = val
        loc = loc + val_size + 8
    return params


def get_dpf_dtype_count(dpf: int, size: int) -> tuple:
    """
    Returns numpy dtype and array count from the data point format (dpf) and block size (in bytes).

    Args:
    ----
        dpf: data point format integer stored in data status block.
            dpf = 1 -> array of float32
            dpf = 2 -> array of int32
        size: Block size in bytes.

    Returns:
    -------
        **dtype (numpy.dtype):** `numpy` dtype for defining an `ndarray` to store the data
        **count (int):** length of array calculated from the block size and byte size of the dtype.
    """
    if dpf == 2:
        dtype = np.int32
        count = round(size / 4)
    else:
        dtype = np.float32
        count = round(size / 4)
    return dtype, count


def parse_data(blockbytes: bytes, dpf: int = 1) -> np.ndarray:
    """
    Parses the bytes in a data block and returns a `numpy` array.

    Data blocks contain no metadata, only the y-values of a data array. Data arrays include: single-channel sample,
    reference, phase, interferograms, and a variety of resultant data (transmission, absorption, etc.).  Every data
    block should have a corresponding data status parameter block which can be used to generate the x-array values for
    the data block. The data status block also specifies the data type of the data array with the `DPF` parameter. It
    appears that OPUS currently exclusively stores data blocks as 32-bit floats, but has a reservation for 32-bit
    integers when `DPF` = 2.

    Args:
    ----
        blockbytes: raw bytes of data block
        dpf: data-point-format integer stored in corresponding data status block.

    Returns:
    -------
        **y_array (numpy.ndarray):** `numpy` array of y values contained in the data block
    """
    dtype, count = get_dpf_dtype_count(dpf=dpf, size=len(blockbytes))
    return np.frombuffer(blockbytes, dtype=dtype, count=count)


def parse_data_series(blockbytes: bytes, dpf: int = 1) -> dict:
    """
    Parses the bytes in a 3D data block (series of spectra) and returns a data `dict` containing data and metadata.

    3D data blocks are structured differently than standard data blocks. In addition to the series of spectra, they
    include metadata for each of the spectrum.  This function returns a `dict` containing all the extracted information
    from the data block.  The series spectra is formed into a 2D array while metadata captured for each spectra is
    formed into a 1D array (length = number of spectral measurements in the series).

    Args:
    ----
        blockbytes: raw bytes of the data series block
        dpf: data-point-format integer stored in corresponding data status block.

    Returns:
    -------
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
    for i in range(data["num_blocks"]):
        data["y"][i] = np.frombuffer(blockbytes[offset:], dtype=dtype, count=count)
        offset = offset + data["data_size"]
        info_vals = struct.unpack_from(
            "<" + "".join([e["fmt"] for e in STRUCT_3D_INFO_BLOCK]), blockbytes, offset
        )
        for j, entry in enumerate(STRUCT_3D_INFO_BLOCK):
            data[entry["key"]][i] = info_vals[j]
        offset = offset + data["info_size"]
    return data


def parse_text(block_bytes: bytes) -> str:
    """
    Parses and OPUS file block as text (e.g. history or file-log block).

    The history (aka file-log) block of an OPUS file contains some information about how the file was generated and
    edits that have been performed on the file.  This function parses the text block but does not take any steps to
    parameterizing what is contained in the text.  The history block is generally not needed to retrieve the file data
    and metadata, but might be useful for inspecting the file.

    Args:
    ----
        blockbytes: raw bytes of the text block (e.g. history or file-log)

    Returns:
    -------
        text: string of text contained in the file block.
    """
    byte_string = struct.unpack("<" + str(len(block_bytes)) + "s", block_bytes)[0]
    byte_strings = byte_string.split(b"\x00")
    strings = []
    for entry in byte_strings:
        if entry != b"":
            try:
                strings.append(entry.decode("latin-1"))
            except Exception:
                try:
                    strings.append(entry.decode("utf-8"))
                except Exception as e:
                    strings.append("<Decode Exception>: " + str(e))
    return "\n".join(strings)
