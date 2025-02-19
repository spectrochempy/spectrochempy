import datetime

import numpy as np

from spectrochempy.extern.brukeropus.file.block import FileBlock
from spectrochempy.extern.brukeropus.file.block import FileDirectory
from spectrochempy.extern.brukeropus.file.block import pair_data_and_status_blocks
from spectrochempy.extern.brukeropus.file.parse import read_opus_file_bytes
from spectrochempy.extern.brukeropus.file.utils import _print_block_header
from spectrochempy.extern.brukeropus.file.utils import _print_cols
from spectrochempy.extern.brukeropus.file.utils import get_block_type_label
from spectrochempy.extern.brukeropus.file.utils import get_param_label

__docformat__ = "google"

"""
The `OPUSFile` class attempts to abstract away some of the complexity and rigid organization structure of Bruker's OPUS
files while providing full access to the data contained in them.  This way, the user does not have to memorize the
organization structure of an OPUS file (e.g. which parameter block contains the beamsplitter parameter) to access the
information.
"""


class OPUSFile:
    """
    Class that contains the data and metadata contained in a bruker OPUS file.

    Args:
    ----
        filepath: full path to the OPUS file to be parsed. Can be a string or Path object and is required to initilize
            an `OPUSFile` object.

    Attributes:
    ----------
        is_opus: True if filepath points to an OPUS file, False otherwise. Also returned for dunder `__bool__()`
        filepath: full path pointing to the OPUS file
        params: class containing all general parameter metadata for the OPUS file. To save typing, the
            three char parameters from params also become attributes of the `OPUSFile` class (e.g. bms, apt, src)
        rf_params: class containing all reference parameter metadata for the OPUS file
        data_keys: list of all `Data` attributes stored in the file (e.g. sm, rf, t, a, r, igsm, igrf, phsm, etc.).
            This only includes 1D data (i.e. omits `DataSeries`).
        series_keys: list of all `DataSeries` attributes stored in the file (e.g. sm, rf, t, a, igsm, phsm, etc.).
            This only includes data series (i.e. omits 1D `Data`).
        all_data_keys: list of all `Data` and `DataSeries` attributes stored in the file (1D and series comingled).
        datetime: Returns the most recent datetime of all the data blocks stored in the file (typically result spectra)
        directory: `FileDirectory` class containing information about all the various data blocks in the file.
        history: History (file-log) containing text about how the file was generated/edited (not always saved)
        unmatched_data_blocks: list of data `FileBlock` that were not uniquely matched to a data status block
        unmatched_data_status_blocks: list of data status `FileBlock` that were not uniquely matched to a data block
        unknown_blocks: list of `FileBlock` that were not parsed and/or assigned to attributes into the class

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
        if self.is_opus:
            data_str = ", ".join(self.all_data_keys)
            return "OPUS File: " + str(self.filepath) + "\n\tSpectra: " + data_str
        return "Not an OPUS file: " + str(self.filepath)

    def __bool__(self):
        return self.is_opus

    def __getattr__(self, name):
        if name == "blocks":
            return self.directory.blocks
        return None

    def __init__(self, filepath):
        """
        Note: a list of `FileBlock` is initially loaded and parsed using the `FileDirectory` class.  This list is
        located in `OPUSFile.directory.blocks`. After parsing all the file blocks (performed by the `FileBlock` class),
        data from those blocks are saved to various attributes within the `OPUSFile` class.  Subsequently, the block is
        removed from `OPUSFile.directory.blocks` to eliminate redundant data and reduce memory footprint.
        """
        self.filepath = filepath
        self.is_opus = False
        self.data_keys = []
        self.series_keys = []
        self.all_data_keys = []
        self.unknown_blocks = []
        self.special_blocks = []
        self.unmatched_data_blocks = []
        self.unmatched_data_status_blocks = []
        filebytes = read_opus_file_bytes(filepath)
        if filebytes:
            self.is_opus = True
            self.directory = FileDirectory(filebytes)
            self._init_directory()
            self._init_params("rf_params", "is_rf_param")
            self._init_params("params", "is_sm_param")
            self._init_history()
            self._init_data()
            self.unknown_blocks = list(self.directory.blocks)
            self._remove_blocks(self.unknown_blocks)

    def _init_directory(self):
        """Moves the directory `FileBlock` into the directory attribute."""
        dir_block = [b for b in self.directory.blocks if b.is_directory()][0]
        self.directory.block = dir_block
        self._remove_blocks([dir_block])

    def _init_params(self, attr: str, is_param: str):
        """
        Sets `Parameter` attributes (`self.params`, `self.rf_params`) from directory blocks and removes them from
        the directory.
        """
        blocks = [
            b
            for b in self.directory.blocks
            if getattr(b, is_param)() and isinstance(b.data, dict)
        ]
        setattr(self, attr, Parameters(blocks))
        self._remove_blocks(blocks)

    def _init_history(self):
        """Sets the history attribute to the parsed history (file_log) data and removes the block."""
        hist_blocks = [b for b in self.directory.blocks if b.is_file_log()]
        if len(hist_blocks) > 0:
            self.special_blocks = self.special_blocks + hist_blocks
            self.history = "\n\n".join([b.data for b in hist_blocks])
        self._remove_blocks(hist_blocks)

    def _get_unused_data_key(self, data_block: FileBlock):
        """Returns a shorthand attribute key for the data_block type. If key already exists"""
        key = data_block.get_data_key()
        if key in self.all_data_keys:
            for i in range(10):
                sub_key = key + "_" + str(i + 1)
                if sub_key not in self.all_data_keys:
                    key = sub_key
                    break
        return key

    def _get_data_vel(self, data_block: FileBlock):
        """Get the mirror velocity setting for the data `Fileblock` (based on whether it is reference or sample)"""
        if data_block.type[1] == 2 and "vel" in self.rf_params.keys():  # noqa: SIM118
            return self.rf_params.vel
        if data_block.type[1] != 2 and "vel" in self.params.keys():  # noqa: SIM118
            return self.params.vel
        return 0

    def _init_data(self):
        """
        Pairs data and data_series `Fileblock`, sets all `Data` and `DataSeries` attributes, and removes the blocks
        from the directory. Unmatched blocks are moved to `unmached_data_blocks` or `unmatched_data_status_blocks`
        """
        matches = pair_data_and_status_blocks(list(self.directory.blocks))
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
            self._remove_blocks([data, status])
        self.unmatched_data_blocks = [
            b for b in self.directory.blocks if b.is_data() or b.is_data_series()
        ]
        self._remove_blocks(self.unmatched_data_blocks)
        self.unmatched_data_status_blocks = [
            b for b in self.directory.blocks if b.is_data_status()
        ]
        self._remove_blocks(self.unmatched_data_status_blocks)

    def _remove_blocks(self, blocks: list):
        """Removes blocks from the directory whose data has been stored elsewhere in class (e.g. params, data, etc.)."""
        starts = [b.start for b in blocks]
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
                label = get_block_type_label(block.type)
                _print_block_header(label, width=width, sep=".")
                _print_cols(("Key", "Label", "Value"), col_widths=col_widths)
                for key in block.keys:
                    label = get_param_label(key)
                    value = getattr(getattr(self, attr), key)
                    _print_cols((key.upper(), label, value), col_widths=col_widths)


class Parameters:
    """
    Class containing parameter metadata of an OPUS file.

    Parameters of an OPUS file are stored as key, val pairs, where the key is always three chars.  For example, the
    beamsplitter is stored in the `bms` attribute, source in `src` etc.  A list of known keys, with friendly label can
    be found in `brukeropus.file.constants.PARAM_LABELS`.  The keys in an OPUS file are not case sensitive, and stored
    in all CAPS (i.e. `BMS`, `SRC`, etc.) but this class uses lower case keys to follow python convention.  The class is
    initialized from a list of `FileBlock` parsed as parameters.  The key, val items in blocks of the list are combined
    into one parameter class, so care must be taken not to pass blocks that will overwrite each others keys.  Analagous
    to a dict, the keys, values, and (key, val) can be iterated over using the functions: `keys()`, `values()`, and
    `items()` respectively.

    Args:
    ----
        blocks: list of `FileBlock`; that has been parsed as parameters.

    Attributes:
    ----------
        xxx: parameter attributes are stored as three char keys. Which keys are generated depends on the list of
            `FileBlock` that is used to initialize the class. If input list contains a single data status
            `FileBlock`, attributes will include: `fxv`, `lxv`, `npt` (first x-val, last x-val, number of points),
            etc. Other blocks produce attributes such as: `bms`, `src`, `apt` (beamsplitter, source, aperture) etc. A
            full list of keys available in a given `Parameters` instance are given by the `keys()` method.
        datetime: if blocks contain the keys: `dat` (date) and `tim` (time), the `datetime` attribute of this class will
            be set to a python `datetime` object. Currently, only data status blocks are known to have these keys. If
            `dat` and `tim` are not present in the class, the `datetime` attribute will return `None`.
        blocks: list of `FileBlock` with data removed to save memory (keys saved for reference)
    """

    __slots__ = ("_params", "datetime", "blocks")

    def __init__(self, blocks: list):
        self._params = {}
        if isinstance(blocks, FileBlock):
            blocks = [blocks]
        for block in blocks:
            self._params.update(block.data)
            block.data = None
        self.blocks = blocks
        self._set_datetime()

    def __getattr__(self, name):
        if name.lower() in self._params:
            return self._params[name.lower()]
        text = (
            str(name)
            + " not a valid attribute. For list of valid parameter keys, use: .keys()"
        )
        raise AttributeError(text)

    def __getitem__(self, item):
        return self._params.__getitem__(item)

    def _set_datetime(self):
        if "dat" in self.keys() and "tim" in self.keys():
            try:
                date_str = self.dat
                time_str = self.tim
                dt_str = date_str + "-" + time_str[: time_str.index(" (")]
                try:
                    fmt = "%d/%m/%Y-%H:%M:%S.%f"
                    dt = datetime.datetime.strptime(dt_str, fmt)
                except:  # noqa: E722
                    try:
                        fmt = "%Y/%m/%d-%H:%M:%S.%f"
                        dt = datetime.datetime.strptime(dt_str, fmt)
                    except:  # noqa: E722
                        self.datetime = None
                self.datetime = dt
            except:  # noqa: E722
                self.datetime = None
        else:
            self.datetime = None

    def keys(self):
        """Returns a `dict_keys` class of all valid keys in the class (i.e. dict.keys())"""
        return self._params.keys()

    def values(self):
        """Returns a `dict_values` class of all the values in the class (i.e. dict.values())"""
        return self._params.values()

    def items(self):
        """Returns a `dict_items` class of all the values in the class (i.e. dict.items())"""
        return self._params.items()


class Data:
    """
    Class containing array data and associated parameter/metadata from an OPUS file.

    Args:
    ----
        data_block: parsed `FileBlock` instance of a data block
        data_status_block: `parsed FileBlock` instance of a data status block which contains metadata about the data
            block. This block is a parameter block.
        key: attribute name (string) assigned to the data
        vel: mirror velocity setting for the measurement (from param or rf_param block as appropriate)

    Attributes:
    ----------
        params: `Parameter` class with metadata associated with the data block such as first x point: `fxp`, last x
            point: `lxp`, number of points: `npt`, date: `dat`, time: `tim` etc.
        y: 1D `numpy` array containing y values of data block
        x: 1D `numpy` array containing x values of data block. Units of x array are given by `dxu` parameter.
        label: human-readable string label describing the data block (e.g. Sample Spectrum, Absorbance, etc.)
        key: attribute name (string) assigned to the data
        vel: mirror velocity setting for the measurement (used to calculate modulation frequency)
        block: data `FileBlock` used to generate the `Data` class
        blocks: [data, data_status] `FileBlock` used to generate the `Data` class

    Extended Attributes:
        **wn:** Returns the x array in wavenumber (cm⁻¹) units regardless of what units the x array was originally
            saved in. This is only valid for spectral data blocks such as sample, reference, transmission, etc., not
            interferogram or phase blocks.
        **wl:** Returns the x array in wavelength (µm) units regardless of what units the x array was originally
            saved in. This is only valid for spectral data blocks such as sample, reference, transmission, etc., not
            interferogram or phase blocks.
        **f:** Returns the x array in modulation frequency units (Hz) regardless of what units the x array was
            originally saved in. This is only valid for spectral data blocks such as sample, reference, transmission,
            etc., not interferogram or phase blocks.
        **datetime:** Returns a `datetime` class of when the data was taken (extracted from data status parameter
            block).
        **xxx:** the various three char parameter keys from the `params` attribute can be directly called from the
            `Data` class for convenience. Common parameters include `dxu` (x units), `mxy` (max y value), `mny` (min y
            value), etc.
    """

    __slots__ = ("key", "params", "y", "x", "label", "vel", "block", "blocks")

    def __init__(
        self, data_block: FileBlock, data_status_block: FileBlock, key: str, vel: float
    ):
        self.key = key
        self.params = Parameters(data_status_block)
        y = data_block.data
        self.y = (
            self.params.csf * y[: self.params.npt]
        )  # Trim extra values on some spectra
        self.x = np.linspace(self.params.fxv, self.params.lxv, self.params.npt)
        self.label = data_block.get_label()
        self.vel = vel
        data_block.data = None
        self.block = data_block
        self.blocks = self.params.blocks + [self.block]

    def __getattr__(self, name):
        if name.lower() == "wn" and self.params.dxu in ("WN", "MI", "LGW"):
            return self._get_wn()
        if name.lower() == "wl" and self.params.dxu in ("WN", "MI", "LGW"):
            return self._get_wl()
        if name.lower() == "f" and self.params.dxu in ("WN", "MI", "LGW"):
            return self._get_freq()
        if name.lower() in self.params:
            return getattr(self.params, name.lower())
        if name == "datetime":
            return self.params.datetime
        text = str(name) + " is not a valid attribute for Data: " + str(self.key)
        raise AttributeError(text)

    def _get_wn(self):
        if self.params.dxu == "WN":
            return self.x
        if self.params.dxu == "MI":
            return 10000.0 / self.x
        if self.params.dxu == "LGW":
            return np.exp(self.x)
        return None

    def _get_wl(self):
        if self.params.dxu == "WN":
            return 10000.0 / self.x
        if self.params.dxu == "MI":
            return self.x
        if self.params.dxu == "LGW":
            return 10000 / np.exp(self.x)
        return None

    def _get_freq(self):
        vel = 1000 * np.float(self.vel) / 7900  # cm/s
        return vel * self.wn


class DataSeries(Data):
    """
    Class containing a data series (3D specra) and associated parameter/metadata from an OPUS file.

    Args:
    ----
        data_block: parsed `FileBlock` instance of a data block
        data_status_block: `parsed FileBlock` instance of a data status block which contains metadata about the data
            block. This block is a parameter block.
        key: attribute name (string) assigned to the data
        vel: mirror velocity setting for measurement (from param or rf_param block as appropriate)

    Attributes:
    ----------
        params: `Parameter` class with metadata associated with the data block such as first x point: `fxp`, last x
            point: `lxp`, number of points: `npt`, date: `dat`, time: `tim` etc.
        y: 2D numpy array containing y values of data block
        x: 1D numpy array containing x values of data block. Units of x array are given by `.dxu` attribute.
        num_spectra: number of spectra in the series (i.e. length of y)
        label: human-readable string label describing the data block (e.g. Sample Spectrum, Absorbance, etc.)
        key: attribute name (string) assigned to the data
        vel: mirror velocity setting for measurement (used to calculate modulation frequency)
        block: data `FileBlock` used to generate the `DataSeries` class
        blocks: [data, data_status] `FileBlock` used to generate the `DataSeries` class

    Extended Attributes:
        **wn:** Returns the x array in wavenumber (cm⁻¹) units regardless of what units the x array was originally saved
            in. This is only valid for spectral data blocks such as sample, reference, transmission, etc., not
            interferogram or phase blocks.
        **wl:** Returns the x array in wavelength (µm) units regardless of what units the x array was originally saved
            in. This is only valid for spectral data blocks such as sample, reference, transmission, etc., not
            interferogram or phase blocks.
        **datetime:** Returns a `datetime` class of when the data was taken (extracted from data status parameter
            block).
        **xxx:** the various three char parameter keys from the `params` attribute can be directly called from the data
            class for convenience. Several of these parameters return arrays, rather than singular values because they
            are recorded for every spectra in the series, e.g. `npt`, `mny`, `mxy`, `srt`, 'ert', `nsn`.
    """

    __slots__ = (
        "key",
        "params",
        "y",
        "x",
        "label",
        "vel",
        "block",
        "blocks",
        "num_spectra",
    )

    def __init__(
        self, data_block: FileBlock, data_status_block: FileBlock, key: str, vel: float
    ):
        self.key = key
        self.params = Parameters(data_status_block)
        data = data_block.data
        self.y = data["y"][:, : self.params.npt]  # Trim extra values on some spectra
        self.x = np.linspace(self.params.fxv, self.params.lxv, self.params.npt)
        self.num_spectra = data["num_blocks"]
        for key, val in data.items():
            if key not in [
                "y",
                "version",
                "offset",
                "num_blocks",
                "data_size",
                "info_size",
            ]:
                self.params._params[key] = val
        self.label = data_block.get_label()
        self.vel = vel
        data_block.data = None
        self.block = data_block
        self.blocks = self.params.blocks + [self.block]


def read_opus(filepath: str) -> OPUSFile:
    """
    Return an `OPUSFile` object from an OPUS file filepath.

    The following produces identical results:
        ```python
        data = read_opus(filepath)
        data = OPUSFile(filepath)
        ```
    Args:
        filepath (str or Path): filepath of an OPUS file (typically *.0)

    Returns
    -------
        opus_file: an instance of the `OPUSFile` class containing all data/metadata extracted from the file.
    """
    return OPUSFile(filepath)
