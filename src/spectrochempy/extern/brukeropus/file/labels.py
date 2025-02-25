from spectrochempy.extern.brukeropus.file.constants import CODE_3_ABR
from spectrochempy.extern.brukeropus.file.constants import PARAM_LABELS
from spectrochempy.extern.brukeropus.file.constants import TYPE_CODE_LABELS

__docformat__ = "google"


def get_param_label(param: str):
    """
    Returns a short but descriptive label for 3-letter parameters. For example, bms returns Beamsplitter.

    The 3-letter parameter input is not case sensitive.  This package includes the majority of parameters that OPUS
    uses, but in the event a parameter label is not known, this function will return: "Unknown XXX" where XXX is the
    unknown 3-letter parameter.

    Args:
    ----
        param: three letter parameter code (e.g. bms, src, npt, etc.) [not case sensitive]

    Returns:
    -------
        label (str): Human-readable string label for the parameter.
    """
    try:
        return PARAM_LABELS[param.upper()]
    except KeyError:
        return "Unknown " + param.upper()


def get_type_code_label(pos_idx: int, val: int):
    """
    Returns the type code label of a file block given the position index and value of the type code.

    The file blocks on an OPUS file feature six-integer type codes, for example (3, 1, 1, 2, 0, 0), that categorize the
    contents of the file block. The positional index defines the category, while the value at that index defines the
    specific type of that category.  For example, the first integer (pos_idx=0), describes the type of data in the
    block, if applicable:

        0: Undefined or N/A,
        1: Real Part of Complex Data,
        2: Imaginary Part of Complex Data,
        3: Amplitude

    This package includes the majority of type codes that OPUS uses, but in the event a type code label is not known,
    this function will return: "Unknown 0 4" where the first number is the position index, and the second is the
    unknown value integer.

    Args:
    ----
        pos_idx: positional index of the type code (0 - 5)
        val: value of the type code

    Returns:
    -------
        label (str): human-readable string label that describes the type code.
    """
    try:
        return TYPE_CODE_LABELS[pos_idx][val]
    except KeyError:
        return "Unknown " + str(pos_idx) + " " + str(val)


def get_block_type_label(block_type: tuple):
    """
    Converts a six-integer tuple block type into a human readable label.

    Args:
    ----
        block_type: six integer tuple found in the OPUS file directory that describes the block type

    Returns:
    -------
        label (str): human-readable string label
    """
    labels = [
        get_type_code_label(idx, val)
        for idx, val in enumerate(block_type)
        if val > 0 and get_type_code_label(idx, val) != ""
    ]
    return " ".join(labels)


def get_data_key(block_type: tuple):
    """
    Returns a shorthand key for a given data block type: sm, rf, igsm, a, t, r, etc.

    Determines if the data block type is an interferogram, single-channel, absorption, etc. and whether it is associated
    with the sample or reference channel and returns a shortand key-like label: sm, rf, igsm, igrf, a, t, r, etc.  For
    the full data label (e.g. Sample Spectrum, Absorbance) use: get_block_type_label.
    This package includes the majority of type codes that OPUS uses, but in the event a type code label is not known,
    this function will return: "_33" or "sm_33" where 33 will change to the unkown block_type integer value.

    Args:
    ----
        block_type: six integer tuple found in the OPUS file directory that describes the block type

    Returns:
    -------
    key (str): shorthand string label that can be utilized as a data key (e.g. "sm", "igrf", "a")
    """
    if block_type[3] in CODE_3_ABR:
        key = CODE_3_ABR[block_type[3]]
        if block_type[1] == 1:
            key = merge_key(key, "sm")
        elif block_type[1] == 2:
            key = merge_key(key, "rf")
        elif block_type[1] > 3:
            key = key + "_" + str(block_type[1])
    else:
        key = "_" + str(block_type[3])
        if block_type[1] == 1:
            key = "sm" + key
        elif block_type[1] == 2:
            key = "rf" + key
        elif block_type[1] > 3:
            key = "_" + str(block_type[1]) + key
    return key


def merge_key(key: str, sm: str):
    """
    Merges "sm" or "rf" into an abreviated data key.  For special cases like ig or pw, the addition is appended
    (e.g. igsm, phrf), but for other cases, the addition is prepended (e.g. sm_2ch, rf_3ch)
    """
    if key[:2] in ["ig", "ph", "pw"]:
        return key[:2] + sm + key[2:]
    return sm + key
