# ruff: noqa
import datetime

from spectrochempy.extern.brukeropus.file.block import FileBlock
from spectrochempy.extern.brukeropus.file.constants import PARAM_LABELS

__docformat__ = "google"


class Parameters:
    """Class containing parameter metadata of an OPUS file.

    Parameters of an OPUS file are stored as key, val pairs, where the key is always three chars.  For example, the
    beamsplitter is stored in the `bms` attribute, source in `src` etc.  A list of known keys, with friendly label can
    be found in `brukeropus.file.constants.PARAM_LABELS`.  The keys in an OPUS file are not case sensitive, and stored
    in all CAPS (i.e. `BMS`, `SRC`, etc.) but this class uses lower case keys to follow python convention.  The class is
    initialized from a list of `FileBlock` parsed as parameters.  The key, val items in blocks of the list are combined
    into one parameter class, so care must be taken not to pass blocks that will overwrite each others keys.  Analagous
    to a dict, the keys, values, and (key, val) can be iterated over using the functions: `keys()`, `values()`, and
    `items()` respectively.

    Args:
        blocks: list of `FileBlock`; that has been parsed as parameters.

    Attributes:
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

    __slots__ = ("_params", "datetime", "blocks", "label")

    def __init__(self, blocks: list):
        self._params = dict()
        if type(blocks) is FileBlock:
            blocks = [blocks]
        for block in blocks:
            self._params.update(block.data)
            block.data = None
        self.blocks = blocks
        self._set_datetime()
        self.label = self._get_label()

    def __getattr__(self, name):
        if name.lower() in self._params.keys():
            return self._params[name.lower()]
        text = (
            str(name)
            + " not a valid attribute. For list of valid parameter keys, use: .keys()"
        )
        raise AttributeError(text)

    def __getitem__(self, item):
        return self._params.__getitem__(item)

    def __str__(self):
        return self._get_label()

    def _get_label(self):
        labels = [block.type.label for block in self.blocks]
        labels = [label.replace(" Parameters", "") for label in labels]
        if len(labels) > 0 and all("Reference " in label for label in labels):
            labels = [label.replace("Reference ", "") for label in labels]
            labels[-1] = labels[-1] + " Reference"
        if len(labels) > 0:
            return ", ".join(labels) + " Parameters"
        return "Parameters"

    def _set_datetime(self):
        if "dat" in self.keys() and "tim" in self.keys():
            try:
                date_str = self.dat
                time_str = self.tim
                dt_str = date_str + "-" + time_str[: time_str.index(" (")]
                try:
                    fmt = "%d/%m/%Y-%H:%M:%S.%f"
                    dt = datetime.datetime.strptime(dt_str, fmt)
                except:
                    try:
                        fmt = "%Y/%m/%d-%H:%M:%S.%f"
                        dt = datetime.datetime.strptime(dt_str, fmt)
                    except:
                        self.datetime = None
                self.datetime = dt
            except:
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


def get_param_label(param: str):
    """Returns a short but descriptive label for 3-letter parameters. For example, bms returns Beamsplitter.

    The 3-letter parameter input is not case sensitive.  This package includes the majority of parameters that OPUS
    uses, but in the event a parameter label is not known, this function will return: "Unknown XXX" where XXX is the
    unknown 3-letter parameter.

    Args:
        param: three letter parameter code (e.g. bms, src, npt, etc.) [not case sensitive]

    Returns:
        label (str): Human-readable string label for the parameter.
    """
    try:
        label = PARAM_LABELS[param.upper()]
        if param.upper() == "BMS":
            return "Beamsplitter"
        return label
    except KeyError:
        return "Unknown " + param.upper()
