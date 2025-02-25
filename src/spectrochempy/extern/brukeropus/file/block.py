from spectrochempy.extern.brukeropus.file.labels import get_block_type_label
from spectrochempy.extern.brukeropus.file.labels import get_data_key
from spectrochempy.extern.brukeropus.file.parse import parse_data
from spectrochempy.extern.brukeropus.file.parse import parse_data_series
from spectrochempy.extern.brukeropus.file.parse import parse_directory
from spectrochempy.extern.brukeropus.file.parse import parse_header
from spectrochempy.extern.brukeropus.file.parse import parse_params
from spectrochempy.extern.brukeropus.file.parse import parse_text

__docformat__ = "google"


class FileBlock:
    """
    Generic OPUS file block.

    This class initializes with the most basic file block info from the file directory: type, size, and start location
    as well as the raw bytes from the file (which can subsequently be parsed).

    Args:
    ----
        filebytes: raw bytes of the file
        block_type: six integer tuple that describes the type of data in the file block
        size: size of block in number of bytes
        start: pointer to start location of the block within the file

    Attributes:
    ----------
        type: six integer tuple that describes the type of data in the file block
        size: size of block in number of bytes
        start: pointer to start location of the block within the file
        bytes: raw bytes of file block (set to zero bytes if successfully parsed)
        data: parsed data if successful. Could be: `list`, `str`, `np.ndarray` or `dict` depending on the block type.
        parser: name of parsing function if parsing was successful
    """

    __slots__ = ("type", "size", "start", "bytes", "data", "parser", "keys")

    def __init__(self, filebytes: bytes, block_type: tuple, size: int, start: int):  # type: ignore
        self.bytes = filebytes[start : start + size]
        self.type = block_type
        self.size = size
        self.start = start
        self.data = None
        self.parser = None

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
            if isinstance(self.data, dict):
                self.keys = list(self.data.keys())
            self._clear_parsed_bytes(parser=parser)
        except Exception as e:
            self.data = "Error parsing: " + str(e)

    def _clear_parsed_bytes(self, parser):
        """Clear raw bytes that have been parsed (and log the parser for reference)"""
        self.parser = parser.__name__
        self.bytes = b""

    def is_data_status(self):
        """Returns True if `FileBlock` is a data status parameter block"""
        return self.type[2] == 1

    def is_rf_param(self):
        """
        Returns True if `FileBlock` is a parameter block associated with the reference measurement (not including
        data status blocks)
        """
        return self.type[2] > 1 and self.type[1] == 2

    def is_param(self):
        """Returns True if `FileBlock` is any parameter block (could be data status, rf param, sample param, etc.)"""
        return self.type[2] > 0 or self.type == (0, 0, 0, 0, 0, 1)

    def is_sm_param(self):
        """
        Returns True if `FileBlock` is a parameter block associated with sample/result measurement (not including
        data status blocks)
        """
        return self.is_param() and not self.is_data_status() and not self.is_rf_param()

    def is_directory(self):
        """Returns True if `FileBlock` is the directory block"""
        return self.type == (0, 0, 0, 13, 0, 0)

    def is_file_log(self):
        """Returns True if `FileBlock` is the file log (aka 'history') block"""
        return self.type == (0, 0, 0, 0, 0, 5)

    def is_data(self):
        """Returns True if `FileBlock` is a 1D data block (not a data series)"""
        return self.type[2] == 0 and self.type[3] not in [0, 13] and self.type[5] != 2

    def is_data_series(self):
        """Returns True if `FileBlock` is a data series block (i.e. 3D data)"""
        return self.type[2] == 0 and self.type[5] == 2

    def get_label(self):
        """Returns a friendly string label that describes the block type"""
        return get_block_type_label(self.type)

    def get_data_key(self):
        """
        If block is a data block, this function will return a shorthand key to reference that data.

        e.g. t: transmission, a: absorption, sm: sample, rf: reference, phsm: sample phase etc. If the block is not
        a data block, it will return `None`.
        """
        if self.is_data() or self.is_data_series():
            return get_data_key(self.type)
        return None

    def get_parser(self):
        """Returns the appopriate file block parser based on the type code (None if not recognized)"""
        if self.is_directory():
            return parse_directory
        if self.is_file_log():
            return parse_text
        if self.is_param():
            return parse_params
        if self.is_data_series():
            return parse_data_series
        if self.is_data():
            return parse_data
        return None

    def parse(self):
        """
        Determines the appropriate parser for the block and parses the raw bytes.  Parsed data is stored in `data`
        attribute and `bytes` attribute is set empty to save memory
        """
        parser = self.get_parser()
        if parser is not None:
            self._try_parser(parser)


class FileDirectory:
    """
    Contains type and pointer information for all blocks of data in an OPUS file.

    `FileDirectory` information is decoded from the raw file bytes of an OPUS file. First the header is read which
    provides the start location of the directory block, number of blocks in file, and maximum number of blocks the file
    supports. Then it decodes the block pointer information from each entry of the file's directory block to create a
    `FileBlock` instance, initiates the block parsing, and adds the parsed block to the `blocks` attribute.

    Args:
    ----
        filebytes: raw bytes from OPUS file. see: `brukeropus.file.parser.read_opus_file_bytes`

    Attributes:
    ----------
        start: pointer to start location of the directory block
        max_blocks: maximum number of blocks supported by file
        num_blocks: total number of blocks in the file
        blocks: list of `FileBlock` from the file. The class parses these blocks upon initilization of the class.
    """

    def __init__(self, filebytes: bytes):
        self.version, self.start, self.max_blocks, self.num_blocks = parse_header(
            filebytes
        )
        size = self.max_blocks * 3 * 4
        blocks = []
        for block_type, block_size, start in parse_directory(
            filebytes[self.start : self.start + size]
        ):
            block = FileBlock(
                filebytes=filebytes, block_type=block_type, size=block_size, start=start
            )
            block.parse()
            blocks.append(block)
        self.blocks = blocks


def is_data_status_type_match(
    data_block: FileBlock, data_status_block: FileBlock
) -> bool:
    """
    Checks if data and data status blocks are a match based soley on the block type.

    This check correctly and accurately matches blocks most of the time, but is occasionally not sufficient on its own
    (e.g. when multiple spectra of the same exact type are stored in a single file)
    """
    t1 = data_status_block.type
    t2 = data_block.type
    return t1[:2] == t2[:2] and t1[3:] == t2[3:]


def is_data_status_val_match(
    data_block: FileBlock, data_status_block: FileBlock
) -> bool:
    """
    Checks if min(data) and max(data) match up with the data status parameters: MNY and MXY.

    When multiple spectra of the same type exist in a file, this is used to distinguish if the data and data status
    blocks are a good match.  This can reduce the number of duplicate matches, but is not generally sufficient to
    fully eliminate duplicate matches.

    See test file: `Test Vit C_Glass.0000_comp.0`
    """
    if data_block.is_data():
        try:
            ds = data_status_block.data
            if len(data_block.data) < ds["npt"]:
                return False
            y = ds["csf"] * data_block.data[: ds["npt"]]
            return y.min() == ds["mny"] and y.max() == ds["mxy"]
        except:  # noqa: E722
            return True  # If error, can't rule out the match
    else:
        return (
            True  # Don't rule out data series at this time (no example files to test)
        )


def is_valid_match(data_block: FileBlock, data_status_block: FileBlock) -> bool:
    """
    Checks that number of points in data status are less than or equal to length of parsed data block.

    This does not apply to data series. While rare, it is occasionally necessary to remove these bad matches.

    See test file: `unreadable.0000`
    """
    return not (
        data_block.is_data() and len(data_block.data) < data_status_block.data["npt"]
    )


def pair_data_and_status_blocks(blocks: list) -> list:
    """
    Takes a list of `FileBlock` and returns a list of matching (data, data status) blocks for further processing.

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
    data_status = [
        b for b in blocks if b.is_data_status() and not isinstance(b.data, str)
    ]
    data = [
        b
        for b in blocks
        if b.is_data() or b.is_data_series() and not isinstance(b.data, str)
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
