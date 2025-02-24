# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import re

import numpy as np
from colorama import Fore
from colorama import Style


def pstr(object, **kwargs):
    if hasattr(object, "_implements") and object._implements() in [
        "NDArray",
        "NDComplexArrray",
        "NDDataset",
        "Coord",
        "CoordSet",
    ]:
        return object._cstr(**kwargs).strip()
    return str(object).strip()


# ======================================================================================
# Terminal colors and styles
# ======================================================================================
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
    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"^(\W{10}\(_\d{1}\))"
    subst = TBold(r"\1")
    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"\0{3}([\w\W]*?)\0{3}"
    subst = TBlack(r"\1")
    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"^(\W{0,12}\w+\W?\w+)(:\W{1}.*$)"
    subst = TGreen(r"\1") + r"\2"
    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"\0{2}([\w\W]*?)\0{2}"
    subst = TCyan(r"\1")
    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"\0{1}([\w\W]*?)\0{1}"
    subst = TBlue(r"\1")

    return re.sub(regex, subst, out, count=0, flags=re.MULTILINE)


#
# html output
#


def html_output(out):
    return out


def convert_to_html(obj):
    """Convert object representation to HTML with collapsible sections."""
    # Style for regular table rows
    tr = (
        "<tr>"
        "<td style='padding-right:5px; padding-bottom:0px; "
        "padding-top:0px; width:124px'>{0}</td>"
        "<td style='text-align:left; padding-bottom:0px; "
        "padding-top:0px; {2} '>{1}</td><tr>\n"
    )

    # CSS for accordion
    accordion_css = """
    <style>
    .scp-accordion {
        margin-bottom: 1rem;
    }
    .scp-accordion-header {
        background-color: #f8f9fa;
        padding: 0.5rem;
        cursor: pointer;
        border: 1px solid #dee2e6;
        border-radius: 4px;
    }
    .scp-accordion-header:hover {
        background-color: #e9ecef;
    }
    .scp-accordion-content {
        padding: 0.5rem;
        border: 1px solid #dee2e6;
        border-top: none;
        display: none;
    }
    .scp-accordion-header.active + .scp-accordion-content {
        display: block;
    }
    </style>
    """

    # JavaScript for accordion functionality
    accordion_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const headers = document.querySelectorAll('.scp-accordion-header');
        headers.forEach(header => {
            header.addEventListener('click', function() {
                this.classList.toggle('active');
                const content = this.nextElementSibling;
                if (content.style.display === 'block') {
                    content.style.display = 'none';
                } else {
                    content.style.display = 'block';
                }
            });
        });
    });
    </script>
    """

    obj._html_output = True
    out = obj._cstr()

    # Process regular content
    regex = r"\0{3}[\w\W]*?\0{3}"

    # noinspection PyPep8
    def subst(match):
        return "<div>{}</div>".format(
            match.group(0).replace("\n", "<br/>").replace("\0", ""),
        )

    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"^(\W{0,12}\w+\W?\w+)(:\W{1}.*$)"  # r"^(\W*\w+\W?\w+)(:.*$)"
    subst = r"<font color='#28A745'>\1</font> \2"  # accent-green (attribute names)
    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"^(.*(DIMENSION|DATA).*)$"
    subst = r"<strong>\1</strong>"
    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"^(\W{10}\(_\d{1}\)).*$"
    subst = r"<strong>\1</strong>"
    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"\0{2}[\w\W]*?\0{2}"

    def subst(match):  # (labels)
        return "<div><font color='darkcyan'>{}</font></div>".format(
            match.group(0).replace("\n", "<br/>").replace("\0", ""),
        )

    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"\0{1}[\w\W]*?\0{1}"

    def subst(match):  # accent-blue (numeric data)
        return "<div><font color='#2D7FF9'>{}</font></div>".format(
            match.group(0).replace("\n", "<br/>").replace("\0", ""),
        )

    out = re.sub(regex, subst, out, count=0, flags=re.MULTILINE)

    regex = r"\.{3}\s+\n"
    out = re.sub(regex, "", out, count=0, flags=re.MULTILINE)

    # Build HTML with accordion sections
    html = accordion_css
    html += "<div class='scp-container'>\n"

    current_section = None
    section_content = ""

    for line in out.splitlines():
        if "DIMENSION" in line or "DATA" in line:
            # Close previous section if exists
            if current_section:
                html += f"""
                    <div class='scp-accordion'>
                        <div class='scp-accordion-header'>{current_section}</div>
                        <div class='scp-accordion-content'>
                            <table style='background:transparent'>{section_content}</table>
                        </div>
                    </div>
                """
                section_content = ""

            current_section = line.strip()

        elif "</font> :" in line:
            parts = line.split(":")
            section_content += tr.format(
                parts[0], ":".join(parts[1:]), "border:.5px solid lightgray; "
            )

        elif "<strong>" in line:
            section_content += tr.format(line, "<hr/>", "padding-top:10px;")

    # Close last section
    if current_section and section_content:
        html += f"""
            <div class='scp-accordion'>
                <div class='scp-accordion-header'>{current_section}</div>
                <div class='scp-accordion-content'>
                    <table style='background:transparent'>{section_content}</table>
                </div>
            </div>
        """

    html += "</div>\n"
    html += accordion_js

    obj._html_output = False
    return html


# ======================================================================================
#  Printing options
#  copied from numpy.ma.core to avoid using
#  the non-public interface of numpy.ma
#  see the header of numpy.ma.core.py for the license
# ======================================================================================
class _MaskedPrintOption:
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
        # Set the enabling shrink to `shrink` .
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
    # recursively replaced with `primitive_dtype` .
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


def insert_masked_print(ds, mask_string="--"):
    """
    Replace masked values with mask_string.

    -- copied from numpy.ma.core and modified

    Parameters
    ----------
    ds : `NDDataset` instance
    mask_string : str

    """
    from spectrochempy.utils.constants import NOMASK

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


# ======================================================================================
# numpy printoptions
# ======================================================================================
def numpyprintoptions(
    precision=4,
    threshold=6,
    edgeitems=2,
    suppress=True,
    formatter=None,
    spc=4,
    linewidth=150,
    **kargs,
):
    """
    Control array printing.

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
        from spectrochempy.utils.misc import TYPE_COMPLEX
        from spectrochempy.utils.misc import TYPE_FLOAT
        from spectrochempy.utils.misc import TYPE_INTEGER

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
                x,
                prec=precision,
                lspace=precision + spc,  # - 1,
            )

        elif isinstance(x, TYPE_COMPLEX):
            fmt = "{:{lspace}.0{prec}g}{:+{lc}.0{prec}g}j".format(
                x.real,
                x.imag,
                prec=precision - 1,
                lspace=precision + spc,
                lc=precision,
            )

        elif isinstance(x, TYPE_INTEGER):
            fmt = "{:>{lspace}d}".format(x, lspace=precision + spc)

        else:
            fmt = f"  {x}"

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
        **kargs,
    )
