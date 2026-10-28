# ruff: noqa
from spectrochempy.extern.brukeropus.file.block import FileBlock
from spectrochempy.extern.brukeropus.file.parse import parse_directory
from spectrochempy.extern.brukeropus.file.parse import parse_header

__docformat__ = "google"


class FileDirectory:
    """Contains type and pointer information for all blocks of data in an OPUS file.

    `FileDirectory` information is decoded from the raw file bytes of an OPUS file. First the header is read which
    provides the start location of the directory block, number of blocks in file, and maximum number of blocks the file
    supports. Then it decodes the block pointer information from each entry of the file's directory block to create a
    `FileBlock` instance, initiates the block parsing, and adds the parsed block to the `blocks` attribute.

    Args:
        filebytes: raw bytes from OPUS file. see: `brukeropus.file.parser.read_opus_file_bytes`
        debug: whether to read the file in debug mode (default: False)

    Attributes:
        start: pointer to start location of the directory block
        max_blocks: maximum number of blocks supported by file
        num_blocks: total number of blocks in the file
        blocks: list of `FileBlock` from the file. The class parses these blocks upon initilization of the class.
        parse_error_blocks: list of `FileBlock` that encountered an error when attempting to parse the binary data.
    """

    def __init__(self, filebytes: bytes, debug: bool = False):
        self.version, self.start, self.max_blocks, self.num_blocks = parse_header(
            filebytes
        )
        size = self.max_blocks * 3 * 4
        self.blocks = []
        self.parse_error_blocks = []
        self.toc = []
        for block_type, size, start in parse_directory(
            filebytes[self.start : self.start + size]
        ):
            block = FileBlock(
                filebytes=filebytes,
                block_type=block_type,
                size=size,
                start=start,
                debug=debug,
            )
            block.parse()
            if block.parse_error:
                self.parse_error_blocks.append(block)
            else:
                self.blocks.append(block)

    def __str__(self):
        if len(self.toc) == 0:
            return "Blank OPUS file directory"
        label_width = max([len(e["type"].get_label()) for e in self.toc])
        type_width = 13
        start_width = 7
        size_width = 6
        attr_width = max([len(e["attr"]) for e in self.toc])
        info = "    ".join(
            (
                f"Version: {self.version}",
                f"Num Blocks: {len(self.toc)}/{self.num_blocks}[{self.max_blocks}]",
                f"Start: {self.start}",
            )
        )
        space = "   "
        header = space.join(
            (
                f"{'File Block Name':<{label_width}}",
                f"{'Type Code':^{type_width}}",
                f"{'Start':>{start_width}}",
                f"{'Size':>{size_width}}",
                f"{'Attribute':<{attr_width}}",
            )
        )
        lines = [
            f"{'  Directory  ':=^{len(header)}}",
            f"{info:^{len(header)}}",
            "―" * len(header),
            header,
            "―" * len(header),
        ]
        for entry in self.toc:
            string = space.join(
                (
                    f"{entry['type'].get_label():<{label_width}}",
                    f"{entry['type'].get_aligned_tuple_str():^{type_width}}",
                    f"{entry['start']:>{start_width}}",
                    f"{entry['size']:>{size_width}}",
                    f"{entry['attr']:<{attr_width}}",
                )
            )
            lines.append(string)
        return "\n".join(lines)
