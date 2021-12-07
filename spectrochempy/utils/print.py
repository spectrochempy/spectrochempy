import numpy as np
import re
from colorama import Fore, Style

from . import NOMASK
from .misc import TYPE_INTEGER, TYPE_COMPLEX, TYPE_FLOAT

__all__ = [
    "numpyprintoptions",
    "insert_masked_print",
    "TBold",
    "TRed",
    "TGreen",
    "TBlue",
    "TCyan",
    "TMagenta",
    "TYellow",
    "colored",
    "colored_output",
    "pstr",
    "convert_to_html",
]


def pstr(object, **kwargs):
    if hasattr(object, "implements") and object.implements() in [
        "NDArray",
        "NDComplexArrray",
        "NDDataset",
        "LinearCoord",
        "Coord",
        "CoordSet",
    ]:
        return object._cstr(**kwargs).strip()
    else:
        return str(object).strip()


# ======================================================================================================================
# Terminal colors and styles
# ======================================================================================================================


def TBold(text):
    return Style.BRIGHT + str(text) + Style.RESET_ALL


def TRed(text):
    return Fore.RED + str(text) + Fore.RESET


def TGreen(text):
    return Fore.GREEN + str(text) + Fore.RESET


def TBlue(text):
    return Fore.BLUE + str(text) + Fore.RESET


def TMagenta(text):
    return Fore.MAGENTA + str(text) + Fore.RESET


def TYellow(text):
    return Fore.YELLOW + str(text) + Fore.RESET


def TCyan(text):
    return Fore.CYAN + str(text) + Fore.RESET


def TBlack(text):
    return Fore.BLACK + str(text) + Fore.RESET


def colored(text, color):
    c = getattr(Fore, color)
    return c + str(text) + Fore.RESET


def colored_output(out):
    regex = r"^(\W*(DIMENSION|DATA).*)$"
    subst = TBold(r"\1")
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"^(\W{10}\(_\d{1}\))"
    subst = TBold(r"\1")
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"\0{3}([\w\W]*?)\0{3}"
    subst = TBlack(r"\1")
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"^(\W{0,12}\w+\W?\w+)(:\W{1}.*$)"
    subst = TGreen(r"\1") + r"\2"
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"\0{2}([\w\W]*?)\0{2}"
    subst = TCyan(r"\1")
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"\0{1}([\w\W]*?)\0{1}"
    subst = TBlue(r"\1")
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    return out


#
# html output
#


def html_output(out):
    return out


def convert_to_html(obj):
    tr = (
        "<tr>"
        "<td style='padding-right:5px; padding-bottom:0px; padding-top:0px; width:124px'>{0}</td>"
        "<td style='text-align:left; padding-bottom:0px; padding-top:0px; {2} '>{1}</td><tr>\n"
    )

    obj._html_output = True

    out = obj._cstr()

    regex = r"\0{3}[\w\W]*?\0{3}"

    # noinspection PyPep8
    def subst(match):
        return "<div>{}</div>".format(
            match.group(0).replace("\n", "<br/>").replace("\0", "")
        )

    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"^(\W{0,12}\w+\W?\w+)(:\W{1}.*$)"  # r"^(\W*\w+\W?\w+)(:.*$)"
    subst = r"<font color='green'>\1</font> \2"
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"^(.*(DIMENSION|DATA).*)$"
    subst = r"<strong>\1</strong>"
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"^(\W{10}\(_\d{1}\)).*$"
    subst = r"<strong>\1</strong>"
    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"\0{2}[\w\W]*?\0{2}"

    # noinspection PyPep8
    def subst(match):
        return "<div><font color='darkcyan'>{}</font></div>".format(
            match.group(0).replace("\n", "<br/>").replace("\0", "")
        )

    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"\0{1}[\w\W]*?\0{1}"

    # noinspection PyPep8
    def subst(match):
        return "<div><font color='blue'>{}</font></div>".format(
            match.group(0).replace("\n", "<br/>").replace("\0", "")
        )

    out = re.sub(regex, subst, out, 0, re.MULTILINE)

    regex = r"\.{3}\s+\n"
    out = re.sub(regex, "", out, 0, re.MULTILINE)

    html = "<table style='background:transparent'>\n"
    for line in out.splitlines():
        if "</font> :" in line:
            # keep only first match
            parts = line.split(":")
            html += tr.format(
                parts[0], ":".join(parts[1:]), "border:.5px solid lightgray; "
            )
        elif "<strong>" in line:
            html += tr.format(line, "<hr/>", "padding-top:10px;")
    html += "</table>"

    obj._html_output = False

    return html


# ======================================================================================================================
#  Printing options
#  copied from numpy.ma.core to avoid using
#  the non-public interface of numpy.ma
#  see the header of numpy.ma.core.py for the license
# ======================================================================================================================


class _MaskedPrintOption(object):

    # """
    # Handle the string used to represent missing data in a masked array.
    # copied from numpy.ma.core
    # """

    def __init__(self, display):
        # """
        # Create the masked_print_option object.
        #
        # """
        self._display = display
        self._enabled = True

    def display(self):
        # """
        # Display the string to print for masked values.
        #
        # """
        return self._display

    def set_display(self, s):
        # """
        # Set the string to print for masked values.
        #
        # """
        self._display = s

    def enabled(self):
        # """
        # Is the use of the display value enabled?
        #
        # """
        return self._enabled

    def enable(self, shrink=1):
        # """
        # Set the enabling shrink to `shrink`.
        #
        # """
        self._enabled = shrink

    def __str__(self):
        return str(self._display)

    __repr__ = __str__


def _replace_dtype_fields_recursive(dtype, primitive_dtype):
    # Private function allowing recursion in _replace_dtype_fields.
    # -- copied from numpy.ma.core

    _recurse = _replace_dtype_fields_recursive

    # Do we have some name fields ?
    if dtype.names:
        descr = []
        for name in dtype.names:
            field = dtype.fields[name]
            if len(field) == 3:
                # Prepend the title to the name
                name = (field[-1], name)
            descr.append((name, _recurse(field[0], primitive_dtype)))
        new_dtype = np.dtype(descr)

    # Is this some kind of composite a la (float,2)
    elif dtype.subdtype:
        descr = list(dtype.subdtype)
        descr[0] = _recurse(dtype.subdtype[0], primitive_dtype)
        new_dtype = np.dtype(tuple(descr))

    # this is a primitive type, so do a direct replacement
    else:
        new_dtype = primitive_dtype

    # preserve identity of dtypes
    if new_dtype == dtype:
        new_dtype = dtype

    return new_dtype


def _replace_dtype_fields(dtype, primitive_dtype):
    # """
    # Construct a dtype description list from a given dtype.
    #
    # Returns a new dtype object, with all fields and subtypes in the given type
    # recursively replaced with `primitive_dtype`.
    #
    # Arguments are coerced to dtypes first.
    #
    # -- copied from numpy.ma.core
    # """
    dtype = np.dtype(dtype)
    primitive_dtype = np.dtype(primitive_dtype)
    return _replace_dtype_fields_recursive(dtype, primitive_dtype)


def _recursive_printoption(result, mask, printopt):
    # """
    # Puts printoptions in result where mask is True.
    #
    # Private function allowing for recursion
    #
    # copied from numpy.ma.core
    # """
    names = result.dtype.names
    if names:
        for name in names:
            curdata = result[name]
            curmask = mask[name]
            _recursive_printoption(curdata, curmask, printopt)
    else:
        np.copyto(result, printopt, where=mask)
    return


def insert_masked_print(ds, mask_string="--"):
    """
    Replace masked values with mask_string.

    -- copied from numpy.ma.core and modified

    Parameters
    -----------
    ds : |NDDataset| instance
    mask_string : str
    """

    mask = ds._mask
    if mask is NOMASK:
        res = ds._data
    else:
        # convert to object array to make filled work
        data = ds._data
        # For big arrays, to avoid a costly conversion to the
        # object dtype, extract the corners before the conversion.
        print_width = ds._print_width if ds.ndim > 1 else ds._print_width_1d
        for axis in range(ds.ndim):
            if data.shape[axis] > print_width:
                ind = print_width // 2
                arr = np.split(data, (ind, -ind), axis=axis)
                data = np.concatenate((arr[0], arr[2]), axis=axis)
                arr = np.split(mask, (ind, -ind), axis=axis)
                mask = np.concatenate((arr[0], arr[2]), axis=axis)

        rdtype = _replace_dtype_fields(ds.dtype, "O")
        res = data.astype(rdtype)

        masked_print_option = _MaskedPrintOption(mask_string)
        _recursive_printoption(res, mask, masked_print_option)

    return res


# ======================================================================================================================
# numpy printoptions
# ======================================================================================================================


def numpyprintoptions(
    precision=4,
    threshold=6,
    edgeitems=2,
    suppress=True,
    formatter=None,
    spc=4,
    linewidth=150,
    **kargs
):
    """
    Method to control array printing.

    Parameters
    ----------
    precision
    threshold
    edgeitems
    suppress
    formatter
    spc
    linewidth
    kargs
    """

    def _format_object(x):

        if isinstance(x, _MaskedPrintOption):
            # a workaround to format masked values
            # the problem is that is depends on the type
            # so we have add a type to the passed object: '--' ->  '--int64'
            el = str(x)[2:].split("_")
            typ = el[0].lower()
            if "int" in typ:
                fmt = "{:>{lspace}s}".format("--", lspace=precision + spc)
            elif "float" in typ:
                fmt = "{:>{lspace}s}".format("--", lspace=2 * precision - 4 + spc)
            elif "complex" in typ:
                fmt = "{:>{lspace}s}".format("--", lspace=4 * precision - 8 + spc)
            else:
                fmt = "n.d."

        elif isinstance(x, TYPE_FLOAT):
            fmt = "{:{lspace}.0{prec}g}".format(
                x, prec=precision, lspace=precision + spc  # - 1,
            )

        elif isinstance(x, TYPE_COMPLEX):
            fmt = "{:{lspace}.0{prec}g}{:+{lc}.0{prec}g}j".format(
                x.real, x.imag, prec=precision - 1, lspace=precision + spc, lc=precision
            )

        elif isinstance(x, TYPE_INTEGER):
            fmt = "{:>{lspace}d}".format(x, lspace=precision + spc)

        else:
            fmt = "  {}".format(x)

        return fmt

    if not formatter:
        spc = 4
        formatter = {
            "all": _format_object,
        }

    np.set_printoptions(
        precision=precision,
        threshold=threshold,
        edgeitems=edgeitems,
        suppress=suppress,
        formatter=formatter,
        linewidth=linewidth,
        **kargs
    )
