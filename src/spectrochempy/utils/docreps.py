# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Analyze docstrings to detect errors.

Adapted from Pandas (see License in the root directory)
"""

import functools
import re
import textwrap

import docrep

_common_doc = """
out : `object`
    Input object or a newly allocated object, depending on the `inplace` flag.
new : `object`
    Newly allocated object.
copy : `bool`, optional, default: `True`
    Perform a copy of the passed object.
inplace : `bool`, optional, default: `False`
    By default, the method returns a newly allocated object.
    If `inplace` is set to `True`, the input object is returned.
dataset : `NDDataset` or :term:`array-like` of shape (`n_observations` , `n_features`)
    Input data, where :term:`n_observations` is the number of observations
    and :term:`n_features` is the number of features.
dim : `int` or `str`, optional, default: -1,
    Dimension along which the method is applied.
    By default, the method is applied to the last dimension.
    If `dim` is specified as an integer it is equivalent to the usual `axis` numpy
    parameter.
**kwargs : keyword parameters, optional
    See Other Parameters.
"""


class DocstringProcessor(docrep.DocstringProcessor):
    param_like_sections = ["See Also"] + docrep.DocstringProcessor.param_like_sections

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        regex = re.compile(r"(?=^[*]{0,2}\b\w+\b\s?:?\s?)", re.MULTILINE | re.DOTALL)
        plist = regex.split(_common_doc.strip())[1:]
        params = {
            k.strip("*"): f"{k.strip()} : {v.strip()}"
            for k, v in (re.split(r"\s?:\s?", p, maxsplit=1) for p in plist)
        }
        self.params.update(params)

    def dedent(self, s, stacklevel=3):
        s_ = s
        start = ""
        end = ""
        string = True
        if not isinstance(s, str) and hasattr(s, "__doc__"):
            string = False
            s_ = s.__doc__
        if s_.startswith("\n"):  # restore the first blank line
            start = "\n"
        if s_.strip(" ").endswith("\n"):  # restore the last return before quote
            end = "\n"
        s_mod = super().dedent(s, stacklevel=stacklevel)
        if string:
            s_mod = f"{start}{s_mod}{end}"
        else:
            s_mod.__doc__ = f"{start}{s_mod.__doc__}{end}"
        return s_mod


# Docstring substitution (docrep)
# --------------------------------------------------------------------------------------
_docstring = DocstringProcessor()


# TODO replace this in module where it is used by docrep
def add_docstring(*args):
    """Add a docstring to the actual function docstring."""

    def new_doc(func):
        for item in args:
            item.strip()

        func.__doc__ = textwrap.dedent(func.__doc__).format(*args)
        return func

    return new_doc


def getdocfrom(origin):
    def decorated(func):
        func.__doc__ = origin.__doc__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorated


def htmldoc(text):
    """
    Format docstring in html for a nice display in IPython.

    Parameters
    ----------
    text : str
        The string to convert to html.

    Returns
    -------
    out : str
        The html string.

    """
    p = re.compile("^(?P<name>.*:)(.*)", re.MULTILINE)  # To get the keywords
    html = p.sub(r"<b>\1</b>\2", text)
    html = html.replace("-", "")
    html = html.split("\n")
    while html[0].strip() == "":
        html = html[1:]  # suppress initial blank lines

    for i in range(len(html)):
        html[i] = html[i].strip()
        if i == 0:
            html[i] = f"<h3>{html[i]}</h3>"
        html[i] = html[i].replace("Parameters", "<h4>Parameters</h4>")
        html[i] = html[i].replace("Properties", "<h4>Properties</h4>")
        html[i] = html[i].replace("Methods", "<h4>Methods</h4>")
        if html[i] != "":
            if "</h" not in html[i]:
                html[i] = html[i] + "<br/>"
            if not html[i].strip().startswith("<"):
                html[i] = "&nbsp;&nbsp;&nbsp;&nbsp;" + html[i]
    return "".join(html)
