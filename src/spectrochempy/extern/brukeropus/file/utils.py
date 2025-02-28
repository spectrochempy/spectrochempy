# ruff: noqa: T201

import os
import re

from spectrochempy.extern.brukeropus.file.block import FileBlock
from spectrochempy.extern.brukeropus.file.constants import CODE_3_ABR  # noqa: F401
from spectrochempy.extern.brukeropus.file.constants import PARAM_LABELS  # noqa: F401
from spectrochempy.extern.brukeropus.file.constants import (
    TYPE_CODE_LABELS,  # noqa: F401
)
from spectrochempy.extern.brukeropus.file.labels import get_block_type_label
from spectrochempy.extern.brukeropus.file.labels import get_param_label
from spectrochempy.extern.brukeropus.file.parse import parse_directory
from spectrochempy.extern.brukeropus.file.parse import parse_header
from spectrochempy.extern.brukeropus.file.parse import read_opus_file_bytes

__all__ = ["find_opus_files", "parse_file_and_print"]


__docformat__ = "google"


def find_opus_files(directory, recursive: bool = False):
    """
    Finds all files in a directory with a strictly numeric extension (OPUS file convention).

    Returns a list of all files in directory that end in .# (e.g. file.0, file.1, file.1001, etc.). Setting recursive
    to true will search directory and all sub directories recursively. No attempt is made to verify the files are
    actually OPUS files (requires opening the file); the function simply looks for files that match the naming pattern.

    Args:
    ----
        directory (str or Path): path indicating directory to search
        recursive: Set to True to recursively search sub directories as well

    Returns:
    -------
        filepaths (list): list of filepaths that match OPUS naming convention (numeric extension)
    """
    pattern = re.compile(r".+\.[0-9]+$")
    file_list = []
    for root, _dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if pattern.match(filename):
                file_list.append(os.path.join(root, filename))
        if not recursive:
            break
    return file_list


def parse_file_and_print(filepath, width=120):
    """
    Parses an OPUS file and prints the block information as it goes along to the console.

    This function demonstrates the basic usage and interaction of the parsing functions.  It
    can also be used to diagnose a file parsing issue if one comes up.

    Args:
    ----
        filepath (str or Path): filepath to an OPUS file.
    """
    filebytes = read_opus_file_bytes(filepath)
    if filebytes is not None:
        width = 120
        info_col_widths = (28, 15, 16, 61)
        info_col_labels = (
            "Block Type",
            "Size (bytes)",
            "Start (bytes)",
            "Friendly Name",
        )
        _print_block_header(filepath, width)
        version, dir_start, max_blocks, num_blocks = parse_header(filebytes)
        h_text = "    ".join(
            [
                "Version: " + str(version),
                "Directory start: " + str(dir_start),
                "Max Blocks: " + str(max_blocks),
                "Num Blocks: " + str(num_blocks),
            ]
        )
        _print_centered(h_text, width)
        _print_block_header("Directory", width)
        block_infos = []
        _print_cols(info_col_labels, info_col_widths)
        for b_type, size, start in parse_directory(
            filebytes[dir_start : dir_start + num_blocks * 3 * 4]
        ):
            try:
                vals = [b_type, size, start, get_block_type_label(b_type)]
                _print_cols(vals, info_col_widths)
                block_infos.append((b_type, size, start))
            except Exception as e:
                print("Exception parsing block: ", e)
        for b_type, size, start in block_infos:
            try:
                block = FileBlock(filebytes, block_type=b_type, size=size, start=start)
                block.parse()
                _print_block(block, width=width)
            except Exception as e:
                print(
                    "Exception parsing block:", block.get_label(), "\n\tException:", e
                )
    else:
        print("Selected file is not an OPUS file: ", filepath)


def _print_block_header(label, width, sep="="):
    """Helper function for: parse_file_and_print"""
    print("\n" + sep * width)
    _print_centered(label, width)


def _print_centered(text, width):
    """Helper function for: parse_file_and_print"""
    print(" " * int((width - len(text)) / 2) + text)


def _print_block(block: FileBlock, width: int):
    """Helper function for: parse_file_and_print"""
    param_col_widths = (10, 45, 45)
    key_width = 10
    key_label_width = 45
    param_col_widths = (key_width, key_label_width, width - key_width - key_label_width)
    param_col_labels = ("Key", "Friendly Name", "Value")
    if not block.is_directory():
        _print_block_header(get_block_type_label(block.type), width)
        if block.is_param():
            _print_cols(param_col_labels, param_col_widths)
            if isinstance(block.data, dict):
                for key, val in block.data.items():
                    _print_cols(
                        (key.upper(), get_param_label(key), val), param_col_widths
                    )
            else:
                print(block.data)
        elif block.is_data():
            print(block.data)
        elif block.is_data_series():
            if isinstance(block.data, dict):
                _print_centered("Num Blocks: " + str(block.data["num_blocks"]), width)
                _print_centered("Store Table: " + str(block.data["store_table"]), width)
                print(block.data["y"])
            else:
                print(block.data)
        elif block.is_file_log():
            print(block.data)
        else:
            _print_centered(
                "Undefined Block Type: " + str(block.type) + " [Raw Bytes]", width
            )
            print(block.bytes)


def _print_cols(
    vals,
    col_widths,
):
    """Helper function for: parse_file_and_print"""
    string = ""
    for i, val in enumerate(vals):
        col_width = col_widths[i]
        val = str(val)
        if len(val) <= col_width - 2:
            string = string + val + " " * (col_width - len(val))
        else:
            string = string + val[: col_width - 5] + "...  "
    print(string)
