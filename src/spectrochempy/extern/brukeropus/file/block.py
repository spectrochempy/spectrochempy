# ruff: noqa
from spectrochempy.extern.brukeropus.file.constants import CODE_3_ABR
from spectrochempy.extern.brukeropus.file.constants import TYPE_CODE_LABELS
from spectrochempy.extern.brukeropus.file.parse import parse_data
from spectrochempy.extern.brukeropus.file.parse import parse_data_series
from spectrochempy.extern.brukeropus.file.parse import parse_directory
from spectrochempy.extern.brukeropus.file.parse import parse_params
from spectrochempy.extern.brukeropus.file.parse import parse_report
from spectrochempy.extern.brukeropus.file.parse import parse_text

__docformat__ = "google"


class BlockType(tuple):
    """Six-integer tuple representing the category (type) of block within an OPUS file.

    Each block in an OPUS file is categorized with six integers, for example (3, 1, 1, 2, 0, 0). This class stores the
    integers as a `tuple`, but extends the `tuple` class to provide a few useful functions/attributes.

    Args:
        block_type: six integers found in the OPUS file directory that describe the block type.

    Attributes:
        label: human-readable label that describes the block category
    """

    def get_label(self):
        """Converts a six-integer tuple block type into a human readable label.

        This package includes the majority of type codes that OPUS uses, but in the event a type code label is not known,
        this function will return: "Unknown 0 4" where the first number is the position index, and the second is the
        unknown value integer.

        Args:
            block_type: six integer tuple found in the OPUS file directory that describes the block type

        Returns:
            label (str): human-readable string label
        """
        labels = [
            self._get_sub_type_label(idx)
            for idx in range(len(self))
            if self[idx] > 0 and self._get_sub_type_label(idx) != ""
        ]
        if len(labels) > 0:
            return " ".join(labels)
        return "Undefined"

    def _get_sub_type_label(self, pos_idx: int):
        """Returns the sub-type label of a file block given the position index and value of the type code.

        Args:
            pos_idx: positional index of the type code (0 - 5)

        Returns:
            label (str): human-readable string label that describes the type code at that index.
        """
        if pos_idx == 3:
            channels, type_idx = divmod(self[3], 32)
        else:
            channels, type_idx = 0, self[pos_idx]
        try:
            label = TYPE_CODE_LABELS[pos_idx][type_idx]
        except KeyError:
            label = "Unknown " + str(pos_idx) + " " + str(type_idx)
        if channels > 0:
            label = label + " " + str(channels + 1) + "-Channel"
        return label

    def get_aligned_tuple_str(self, pad=1):
        return (
            f"{self[0]}"
            + f"{self[1]:2}"
            + f"{self[2]:3}"
            + f"{self[3]:3}"
            + f"{self[4]:2}"
            + f"{self[5]:2}"
        )

    def __repr__(self):
        return "BlockType((" + ", ".join([str(i) for i in self]) + "))"

    def __str__(self):
        return self.get_aligned_tuple_str() + "   " + self.get_label()

    def __new__(cls, iterable):
        instance = super().__new__(cls, iterable)
        if len(instance) != 6 or any(type(i) != int for i in instance):
            raise ValueError(
                "BlockType input must be a 6-integer iterable, but a value of:"
                + str(iterable)
                + " was given"
            )
        return instance

    def __init__(self, iterable):
        super().__init__()
        self.label = self.get_label()


class FileBlock:
    """Generic OPUS file block.

    This class initializes from the block info stored in the file directory (type, size, and start location) as well as
    the raw bytes from the file (which can subsequently be parsed).

    Args:
        filebytes: raw bytes of the file
        block_type: six integer tuple that describes the type of data in the file block
        size: size of block in number of bytes
        start: pointer to start location of the block within the file
        debug: whether to read the block in debug mode (default: False)

    Attributes:
        type: six integer tuple that describes the type of data in the file block
        size: size of block in number of bytes
        start: pointer to start location of the block within the file
        bytes: raw bytes of file block (set to zero bytes if successfully parsed unless in debug mode)
        data: parsed data if successful. Could be: `list`, `str`, `np.ndarray` or `dict` depending on the block type.
        parser: name of parsing function if parsing was successful
    """

    __slots__ = (
        "type",
        "size",
        "start",
        "bytes",
        "data",
        "parser",
        "parse_error",
        "keys",
        "debug",
    )

    def __init__(
        self, filebytes: bytes, block_type: tuple, size: int, start: int, debug=False
    ):
        self.bytes = filebytes[start : start + size]
        self.type = BlockType(block_type)
        self.size = size
        self.start = start
        self.debug = debug
        self.data = None
        self.parser = None
        self.parse_error = None

    def __str__(self):
        label = self.get_label()
        return (
            "FileBlock: "
            + label
            + " (size: "
            + str(self.size)
            + " bytes; start: "
            + str(self.start)
            + ")"
        )

    def _try_parser(self, parser):
        try:
            self.data = parser(self.bytes)
            if type(self.data) is dict:
                self.keys = list(self.data.keys())
            self._clear_parsed_bytes(parser=parser)
        except Exception as e:
            self.parse_error = "Error parsing (" + parser.__name__ + "): " + str(e)

    def _clear_parsed_bytes(self, parser):
        """Clear raw bytes that have been parsed (and log the parser for reference)"""
        self.parser = parser.__name__
        if not self.debug:
            self.bytes = b""

    def is_data_status(self):
        """Returns True if `FileBlock` is a data status parameter block"""
        return self.type[2] == 1

    def is_rf_param(self):
        """Returns True if `FileBlock` is a parameter block associated with the reference measurement (not including
        data status blocks)"""
        return self.type[2] > 1 and self.type[1] == 2

    def is_param(self):
        """Returns True if `FileBlock` is any parameter block (could be data status, rf param, sample param, etc.)"""
        return self.type[2] > 0 or self.type == (0, 0, 0, 0, 0, 1)

    def is_sm_param(self):
        """Returns True if `FileBlock` is a parameter block associated with sample/result measurement (not including
        data status blocks)"""
        return self.is_param() and not self.is_data_status() and not self.is_rf_param()

    def is_directory(self):
        """Returns True if `FileBlock` is the directory block"""
        return self.type == (0, 0, 0, 13, 0, 0)

    def is_file_log(self):
        """Returns True if `FileBlock` is the file log (aka 'history') block"""
        return self.type == (0, 0, 0, 0, 0, 5)

    def is_report(self):
        """Returns True if `FileBlock` is a test report"""
        return self.type in [
            (0, 0, 0, 0, 0, 2),
            (0, 0, 0, 0, 0, 3),
            (0, 0, 0, 0, 0, 4),
        ] or (self.type[2] == 0 and self.type[3] not in [0, 13] and self.type[5] == 5)

    def is_data(self):
        """Returns True if `FileBlock` is a 1D data block (not a data series)"""
        return (
            self.type[2] == 0
            and self.type[3] not in [0, 13]
            and self.type[5] not in [2, 5]
        )

    def is_data_series(self):
        """Returns True if `FileBlock` is a data series block (i.e. 3D data)"""
        return self.type[2] == 0 and self.type[3] not in [0, 13] and self.type[5] == 2

    def is_compact_data(self):
        """Returns True if `FileBlock` is a compact 1D data block. These data blocks have metadata preceeding the data
        array (currently ignored)."""
        return self.is_data() and self.type[5] == 4

    def get_label(self):
        """Returns a friendly string label that describes the block type"""
        return self.type.label

    def get_data_key(self):
        """If block is a data block, this function will return a shorthand key to reference that data.

        e.g. t: transmission, a: absorption, sm: sample, rf: reference, phsm: sample phase etc. If the block is not
        a data block, it will return `None`."""
        if self.is_data() or self.is_data_series():
            channels = self.type[3] // 32 + 1
            type_idx = self.type[3] % 32
            if type_idx in CODE_3_ABR.keys():
                key = CODE_3_ABR[type_idx]
            else:
                key = "_" + str(self.type[3])
            if self.type[1] == 1:
                key = key + "sm"
            elif self.type[1] == 2:
                key = key + "rf"
            elif self.type[1] > 3:
                key = key + "_" + str(self.type[1])
            if channels > 1:
                key = key + "_" + str(channels) + "ch"
            if self.type[5] == 4:
                key = key + "_c"
            return key
        return None

    def get_parser(self):
        """Returns the appopriate file block parser based on the type code (None if not recognized)"""
        if self.is_directory():
            return parse_directory
        if self.is_file_log():
            return parse_text
        if self.is_param():
            return parse_params
        if self.is_report():
            return parse_report
        if self.is_data_series():
            return parse_data_series
        if self.is_data():
            return parse_data
        return None

    def parse(self):
        """Determines the appropriate parser for the block and parses the raw bytes.  Parsed data is stored in `data`
        attribute and `bytes` attribute is set empty to save memory"""
        parser = self.get_parser()
        if parser is not None:
            self._try_parser(parser)


def is_data_status_type_match(
    data_block: FileBlock, data_status_block: FileBlock
) -> bool:
    """Checks if data and data status blocks are a match based soley on the block type.

    This check correctly and accurately matches blocks most of the time, but is occasionally not sufficient on its own
    (e.g. when multiple spectra of the same exact type are stored in a single file)"""
    t1 = data_status_block.type
    t2 = data_block.type
    return t1[:2] == t2[:2] and t1[3:] == t2[3:]


def is_data_status_val_match(
    data_block: FileBlock, data_status_block: FileBlock
) -> bool:
    """Checks if min(data) and max(data) match up with the data status parameters: MNY and MXY.

    When multiple spectra of the same type exist in a file, this is used to distinguish if the data and data status
    blocks are a good match.  This can reduce the number of duplicate matches, but is not generally sufficient to
    fully eliminate duplicate matches.

    See test file: `Test Vit C_Glass.0000_comp.0`"""
    if data_block.is_data():
        try:
            ds = data_status_block.data
            if len(data_block.data) < ds["npt"]:
                return False
            y = ds["csf"] * data_block.data[: ds["npt"]]
            return y.min() == ds["mny"] and y.max() == ds["mxy"]
        except:
            return True  # If error, can't rule out the match
    else:
        return (
            True  # Don't rule out data series at this time (no example files to test)
        )


def is_valid_match(data_block: FileBlock, data_status_block: FileBlock) -> bool:
    """Checks that number of points in data status are less than or equal to length of parsed data block.

    This does not apply to data series. While rare, it is occasionally necessary to remove these bad matches.

    See test file: `unreadable.0000`"""
    if data_block.is_data() and len(data_block.data) < data_status_block.data["npt"]:
        return False
    return True


def pair_data_and_status_blocks(blocks: list) -> list:
    """Takes a list of `FileBlock` and returns a list of matching (data, data status) blocks for further processing.

    All valid data blocks have an associated data status parameter block that contains y-scaling and x-axis info.
    Generally, these blocks can be easily paired with one another by using the block type. However, some files can
    contain multiple data blocks with the same exact type, which leads to duplicate matches and requires further
    inspection to accurately pair.  This function uses the following logical sequence to accurately pair these blocks:

        1. Pair by type and isolate singular matches from duplicate matches
        2. For duplicate matches, check min(data) and max(data) match data status MNY and MXY (including CSF scaling)
        3. Again isolate singular matches from duplicate matches
        4. For remaining duplicate matches, remove any matches that are already in the singular match list.
        5. Again isolate singular matches (ideally no remaining duplicate matches at this point)
        6. Remove invalid matches from the singular match list if len(data) is < data status NPT (invalid condition)
        7. Sort the matches in reverse order of where the data blocks are stored in the file (presume last is most
           recently added, and therefore final spectra which was true for limited test files).

    For the very limited test files available, this is sufficient to accurately pair all the blocks, but it seems
    plausible that it will not be sufficient for all files that have duplicate data type entries. More test files
    required for thorough testing.
    """
    data_status = [b for b in blocks if b.is_data_status() and type(b.data) is not str]
    data = [
        b
        for b in blocks
        if b.is_data() or b.is_data_series() and type(b.data) is not str
    ]
    type_matches = []
    for d in data:
        type_matches.append(
            (d, [b for b in data_status if is_data_status_type_match(d, b)])
        )
    single_matches = [(m[0], m[1][0]) for m in type_matches if len(m[1]) == 1]
    multi_matches = [match for match in type_matches if len(match[1]) > 1]
    val_matches = []
    for d, matches in multi_matches:
        val_matches.append((d, [b for b in matches if is_data_status_val_match(d, b)]))
    single_matches = single_matches + [
        (m[0], m[1][0]) for m in val_matches if len(m[1]) == 1
    ]
    multi_matches = [match for match in val_matches if len(match[1]) > 1]
    reduced_matches = []
    single_starts = [m[1].start for m in single_matches]
    for d, matches in multi_matches:
        reduced_matches.append(
            (d, [b for b in matches if b.start not in single_starts])
        )
    single_matches = single_matches + [
        (m[0], m[1][0]) for m in reduced_matches if len(m[1]) == 1
    ]
    multi_matches = [match for match in reduced_matches if len(match[1]) > 1]

    single_matches = [
        match for match in single_matches if is_valid_match(match[0], match[1])
    ]  # remove invalid

    single_matches.sort(
        key=lambda pairs: pairs[0].start, reverse=True
    )  # last spec seems to be OPUS preference
    return single_matches
