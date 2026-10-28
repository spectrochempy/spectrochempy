# ruff: noqa
import os

from spectrochempy.extern.brukeropus.file.block import FileBlock
from spectrochempy.extern.brukeropus.file.block import pair_data_and_status_blocks
from spectrochempy.extern.brukeropus.file.data import Data
from spectrochempy.extern.brukeropus.file.data import DataSeries
from spectrochempy.extern.brukeropus.file.directory import FileDirectory
from spectrochempy.extern.brukeropus.file.params import Parameters
from spectrochempy.extern.brukeropus.file.parse import read_opus_file_bytes
from spectrochempy.extern.brukeropus.file.report import Report
from spectrochempy.extern.brukeropus.file.utils import _print_block_header
from spectrochempy.extern.brukeropus.file.utils import _print_cols
from spectrochempy.extern.brukeropus.file.utils import get_param_label

__docformat__ = "google"

"""
The `OPUSFile` class attempts to abstract away some of the complexity and rigid organization structure of Bruker's OPUS
files while providing full access to the data contained in them.  This way, the user does not have to memorize the
organization structure of an OPUS file (e.g. which parameter block contains the beamsplitter parameter) to access the
information.
"""


class OPUSFile:
    """Class that contains the data and metadata contained in a bruker OPUS file.

    Args:
        filepath: full path to the OPUS file to be parsed. Can be a string or Path object and is required to initilize
            an `OPUSFile` object.
        debug: whether to read the file in debug mode (default: False)

    Attributes:
        is_opus: True if filepath points to an OPUS file, False otherwise. Also returned for dunder `__bool__()`
        filepath: full path pointing to the OPUS file
        name: base filename of the OPUS file
        params: class containing all general parameter metadata for the OPUS file. To save typing, the
            three char parameters from params also become attributes of the `OPUSFile` class (e.g. bms, apt, src).
        rf_params: class containing all reference parameter metadata for the OPUS file
        data_keys: list of all `Data` attributes stored in the file (e.g. sm, rf, t, a, r, igsm, igrf, phsm, etc.).
            This only includes 1D data (i.e. omits `DataSeries`).
        series_keys: list of all `DataSeries` attributes stored in the file (e.g. sm, rf, t, a, igsm, phsm, etc.).
            This only includes data series (i.e. omits 1D `Data`).
        all_data_keys: list of all `Data` and `DataSeries` attributes stored in the file (1D and series comingled).
        datetime: Returns the most recent datetime of all the data blocks stored in the file (typically result spectra)
        reports: list of `Report` class containing all reports in the file.
        directory: `FileDirectory` class containing information about all the various data blocks in the file.
        history: History (file-log) containing text about how the file was generated/edited (not always saved)
        unmatched_data_blocks: list of data `FileBlock` that were not uniquely matched to a data status block
        unmatched_data_status_blocks: list of data status `FileBlock` that were not uniquely matched to a data block
        unknown_blocks: list of `FileBlock` that were not parsed and/or assigned to attributes into the class
        parse_error_blocks: list of `FileBlock` that raised an error while attempting to parse

    Data Attributes:
        **sm:** Single-channel sample spectra
        **rf:** Single-channel reference spectra
        **igsm:** Sample interferogram
        **igrf:** Reference interferogram
        **phsm:** Sample phase
        **phrf:** Reference phase
        **a:** Absorbance
        **t:** Transmittance
        **r:** Reflectance
        **km:** Kubelka-Munk
        **tr:** Trace (Intensity over Time)
        **gcig:** gc File (Series of Interferograms)
        **gcsc:** gc File (Series of Spectra)
        **ra:** Raman
        **e:** Emission
        **pw:** Power
        **logr:** log(Reflectance)
        **atr:** ATR
        **pas:** Photoacoustic
    """

    def __str__(self):
        c1_width = 15
        c2_width = 20
        width = 120
        c3_width = width - c1_width - c2_width
        if self.is_opus:
            lines = ["=" * width, f"{'OPUS File: ' + self.filepath:^{width}}"]
            lines.append(
                f"{'Attribute':<{c1_width}}{'Class type':<{c2_width}}" + "Description"
            )
            lines.append("―" * width)
            for attr in ("params", "rf_params"):
                val = getattr(self, attr)
                if val.keys():
                    lines.append(
                        f"{attr:<{c1_width}}{'Parameters':<{c2_width}}"
                        + val.label[:c3_width]
                    )
            for attr in self.data_keys:
                val = getattr(self, attr)
                lines.append(
                    f"{attr:<{c1_width}}{'Data':<{c2_width}}" + val.label[:c3_width]
                )
            for attr in self.series_keys:
                val = getattr(self, attr)
                lines.append(
                    f"{attr:<{c1_width}}{'DataSeries':<{c2_width}}"
                    + val.label[:c3_width]
                )
            for i, report in enumerate(self.reports):
                lines.append(
                    f"{'reports[' + str(i) + ']':<{c1_width}}{'Report':<{c2_width}}"
                    + report.title
                )
            if self.history:
                lines.append(
                    f"{'history':<{c1_width}}{'str':<{c2_width}}"
                    + "History log of file"
                )
            lines.append("―" * width)
            return "\n".join(lines)
        return "Not an OPUS file: " + str(self.filepath)

    def __bool__(self):
        return self.is_opus

    def __getattr__(self, name):
        if name == "blocks":
            return self.directory.blocks
        if name.lower() in self.params.keys():
            return getattr(self.params, name)
        if name.lower() in self.rf_params.keys():
            return getattr(self.rf_params, name)

    def __init__(self, filepath, debug: bool = False):
        """Note: a list of `FileBlock` is initially loaded and parsed using the `FileDirectory` class.  This list is
        located in `OPUSFile.directory.blocks`. After parsing all the file blocks (performed by the `FileBlock` class),
        data from those blocks are saved to various attributes within the `OPUSFile` class.  Subsequently, the block is
        removed from `OPUSFile.directory.blocks` to eliminate redundant data and reduce memory footprint."""
        self.filepath = filepath
        self.name = (
            os.path.basename(filepath)
            if isinstance(filepath, (str, os.PathLike))
            else "<file object>"
        )
        self.is_opus = False
        self.data_keys = []
        self.series_keys = []
        self.all_data_keys = []
        self.reports = []
        self.unknown_blocks = []
        self.unmatched_data_blocks = []
        self.unmatched_data_status_blocks = []
        filebytes = read_opus_file_bytes(filepath)
        if debug:
            self.bytes = filebytes
        if filebytes:
            self.is_opus = True
            self.directory = FileDirectory(filebytes, debug=debug)
            self._init_directory()
            self._init_params("rf_params", "is_rf_param")
            self._init_params("params", "is_sm_param")
            self._init_data()
            self._init_reports()
            self._init_history()
            self.unknown_blocks = [block for block in self.directory.blocks]
            self._remove_blocks(self.unknown_blocks, "unknown_blocks")
            self.parse_error_blocks = [
                block for block in self.directory.parse_error_blocks
            ]
            self._remove_blocks(self.parse_error_blocks, "parse_error_blocks")

    def _init_directory(self):
        """Moves the directory `FileBlock` into the directory attribute."""
        try:
            dir_block = [b for b in self.directory.blocks if b.is_directory()][0]
            self._remove_blocks([dir_block], "directory")
        except:
            dir_block = None
        self.directory.block = dir_block

    def _init_params(self, attr: str, is_param: str):
        """Sets `Parameter` attributes (`self.params`, `self.rf_params`) from directory blocks and removes them from
        the directory."""
        blocks = [
            b
            for b in self.directory.blocks
            if getattr(b, is_param)() and type(b.data) is dict
        ]
        setattr(self, attr, Parameters(blocks))
        self._remove_blocks(blocks, attr)

    def _init_reports(self):
        """Adds all reports (`Report`) from the file to the `.reports` attribute."""
        report_blocks = [b for b in self.directory.blocks if b.is_report()]
        for b in report_blocks:
            try:
                self.reports.append(Report(b))
            except Exception as e:
                b.parse_error = e
                self.directory.parse_error_blocks.append(b)
        self._remove_blocks(report_blocks, "reports")

    def _init_history(self):
        """Sets the history attribute to the parsed history (file_log) data and removes the block."""
        hist_blocks = [b for b in self.directory.blocks if b.is_file_log()]
        if len(hist_blocks) > 0:
            self.history = "\n\n".join([b.data for b in hist_blocks])
            self.history_blocks = hist_blocks
            self._remove_blocks(hist_blocks, "history")

    def _get_unused_data_key(self, data_block: FileBlock):
        """Returns a shorthand attribute key for the data_block type. If key already exists"""
        key = data_block.get_data_key()
        if key in self.all_data_keys:
            for i in range(10):
                sub_key = key + "_" + str(i + 2)
                if sub_key not in self.all_data_keys:
                    key = sub_key
                    break
        return key

    def _get_data_vel(self, data_block: FileBlock):
        """Get the mirror velocity setting for the data `Fileblock` (based on whether it is reference or sample)"""
        if data_block.type[1] == 2 and "vel" in self.rf_params.keys():
            return self.rf_params.vel
        if data_block.type[1] != 2 and "vel" in self.params.keys():
            return self.params.vel
        return 0

    def _init_data(self):
        """Pairs data and data_series `Fileblock`, sets all `Data` and `DataSeries` attributes, and removes the blocks
        from the directory. Unmatched blocks are moved to `unmached_data_blocks` or `unmatched_data_status_blocks`"""
        matches = pair_data_and_status_blocks([b for b in self.directory.blocks])
        for data, status in matches:
            key = self._get_unused_data_key(data)
            vel = self._get_data_vel(data)
            if data.is_data():
                data_class = Data
                self.data_keys.append(key)
            elif data.is_data_series():
                data_class = DataSeries
                self.series_keys.append(key)
            setattr(self, key, data_class(data, status, key=key, vel=vel))
            self.all_data_keys.append(key)
            self._remove_blocks([data, status], key)
        self.unmatched_data_blocks = [
            b for b in self.directory.blocks if b.is_data() or b.is_data_series()
        ]
        self._remove_blocks(self.unmatched_data_blocks, "unmatched_data_blocks")
        self.unmatched_data_status_blocks = [
            b for b in self.directory.blocks if b.is_data_status()
        ]
        self._remove_blocks(
            self.unmatched_data_status_blocks, "unmatched_data_status_blocks"
        )

    def _get_toc_entry(self, block: FileBlock, attr_name: str):
        entry = {
            "type": block.type,
            "attr": attr_name,
            "start": block.start,
            "size": block.size,
        }
        return entry

    def _remove_blocks(self, blocks: list, attr_name: str):
        """Removes blocks from the directory whose data has been stored elsewhere in class (e.g. params, data, etc.)."""
        starts = [b.start for b in blocks]
        for b in blocks:
            self.directory.toc.append(self._get_toc_entry(b, attr_name))
        self.directory.blocks = [
            b for b in self.directory.blocks if b.start not in starts
        ]

    def iter_data(self):
        """Generator that yields the various Data classes from the OPUSFile (excluding DataSeries)"""
        for key in self.data_keys:
            yield getattr(self, key)

    def iter_series(self):
        """Generator that yields the various DataSeries classes from the OPUSFile (excluding Data)"""
        for key in self.series_keys:
            yield getattr(self, key)

    def iter_all_data(self):
        """Generator that yields all the various Data and DataSeries classes from the OPUSFile"""
        for key in self.all_data_keys:
            yield getattr(self, key)

    def print_parameters(self, key_width=7, label_width=40, value_width=53):
        """Prints all the parameter metadata to the console (organized by block)"""
        width = key_width + label_width + value_width
        col_widths = (key_width, label_width, value_width)
        param_infos = [
            ("Sample/Result Parameters", "params"),
            ("Reference Parameters", "rf_params"),
        ]
        for title, attr in param_infos:
            _print_block_header(title + " (" + attr + ")", width=width, sep="=")
            blocks = getattr(self, attr).blocks
            for block in blocks:
                label = block.type.label
                _print_block_header(label, width=width, sep=".")
                _print_cols(("Key", "Label", "Value"), col_widths=col_widths)
                for key in block.keys:
                    label = get_param_label(key)
                    value = getattr(getattr(self, attr), key)
                    _print_cols((key.upper(), label, value), col_widths=col_widths)


def read_opus(filepath: str, debug: bool = False) -> OPUSFile:
    """Return an `OPUSFile` object from an OPUS file filepath.

    The following produces identical results:
        ```python
        data = read_opus(filepath)
        data = OPUSFile(filepath)
        ```
    Args:
        filepath (str or Path): filepath of an OPUS file (typically *.0)
        debug: whether to read the file in debug mode (default: False)

    Returns:
        opus_file: an instance of the `OPUSFile` class containing all data/metadata extracted from the file.
    """
    return OPUSFile(filepath, debug=debug)
