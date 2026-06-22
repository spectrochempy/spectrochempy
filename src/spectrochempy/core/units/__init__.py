# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# lazy_stub: skip
"""The core interface to the Pint library."""

__all__ = [
    "Unit",
    "Quantity",
    "ur",
    "DimensionalityError",
]

import re
import warnings

from pint import __version__

# check pint version
pint_version = int(__version__.split(".")[1])
if pint_version < 24:
    raise ImportError(
        "Current pint version is {__version__} but must be 0.24 or higher. Please consider upgrading it "
        "(e.g. `> pip install pint --upgrade` or `> conda update pint` )\n",
    )


def _register_dotted_symbols(registry):
    """
    Allow parsing SpectroChemPy custom dotted symbols.

    pint interprets ``.`` as a multiplication operator, so SpectroChemPy's
    dotted display symbols (``a.u.`` for absorbance, ``K.M.`` for Kubelka-Munk)
    round-trip on output but cannot be read back from a string. This installs a
    registry preprocessor that rewrites those symbols, with or without their
    trailing dot, to the canonical unit name before parsing. See issue #913.
    """
    mapping = {
        "a.u.": "absorbance",
        "a.u": "absorbance",
        "K.M.": "Kubelka_Munk",
        "K.M": "Kubelka_Munk",
    }

    pattern = re.compile(
        r"(?<![\w.])("
        + "|".join(re.escape(s) for s in sorted(mapping, key=len, reverse=True))
        + r")(?![\w.])"
    )

    registry.preprocessors.append(
        lambda string: pattern.sub(lambda m: mapping[m.group(1)], string)
    )


from functools import wraps

from pint import DimensionalityError
from pint import Unit
from pint import UnitRegistry
from pint import set_application_registry

# utilities
from pint.delegates.formatter._compound_unit_helpers import localize_per
from pint.delegates.formatter._compound_unit_helpers import prepare_compount_unit
from pint.delegates.formatter._format_helpers import formatter
from pint.delegates.formatter._format_helpers import pretty_fmt_exponent

# formatters to be  subclassed
from pint.delegates.formatter.full import FullFormatter
from pint.delegates.formatter.html import HTMLFormatter
from pint.delegates.formatter.latex import LatexFormatter
from pint.delegates.formatter.latex import latex_escape
from pint.delegates.formatter.plain import CompactFormatter
from pint.delegates.formatter.plain import DefaultFormatter
from pint.delegates.formatter.plain import PrettyFormatter


####################################################################################
# SpectroChemPy specific formatters
# ##################################################################################
class ScpDefaultFormatter(DefaultFormatter):
    """Subclasses the DefaultFormatter to provide a specific formatting for SpectroChemPy."""

    def format_unit(self, unit, uspec, sort_func, **babel_kwds) -> str:
        numerator, denominator = prepare_compount_unit(
            unit,
            uspec,
            sort_func=sort_func,
            **babel_kwds,
            registry=self._registry,
        )

        return formatter(
            numerator,
            denominator,
            as_ratio=False,
            single_denominator=False,
            product_fmt=" ",
            division_fmt="/",
            power_fmt="{}^{}",
            parentheses_fmt=r"({})",
        )


class ScpCompactFormatter(CompactFormatter):
    """Subclasses the CompactFormatter to provide a specific formatting for SpectroChemPy."""

    def format_unit(self, unit, uspec, sort_func, **babel_kwds) -> str:
        numerator, denominator = prepare_compount_unit(
            unit,
            uspec,
            sort_func=sort_func,
            **babel_kwds,
            registry=self._registry,
        )

        return formatter(
            numerator,
            denominator,
            as_ratio=False,
            single_denominator=False,
            product_fmt="*",
            division_fmt="/",
            power_fmt="{}**{}",
            parentheses_fmt=r"({})",
        )


class ScpPrettyFormatter(PrettyFormatter):
    """Subclasses the PretyFormatter to provide a specific formatting for SpectroChemPy."""

    def format_unit(self, unit, uspec, sort_func, **babel_kwds) -> str:
        numerator, denominator = prepare_compount_unit(
            unit,
            uspec,
            sort_func=sort_func,
            **babel_kwds,
            registry=self._registry,
        )

        return formatter(
            numerator,
            denominator,
            as_ratio=False,
            single_denominator=False,
            product_fmt="⋅",
            division_fmt="/",
            power_fmt="{}{}",
            parentheses_fmt=r"({})",
            exp_call=pretty_fmt_exponent,
        )


class ScpHTMLFormatter(HTMLFormatter):
    """Subclasses the HTMLFormatter to provide a specific formatting for SpectroChemPy."""

    def format_unit(self, unit, uspec, sort_func, **babel_kwds) -> str:
        numerator, denominator = prepare_compount_unit(
            unit,
            uspec,
            sort_func=sort_func,
            **babel_kwds,
            registry=self._registry,
        )

        if babel_kwds.get("locale"):
            length = babel_kwds.get("length") or ("short" if "~" in uspec else "long")
            division_fmt = localize_per(length, babel_kwds.get("locale"), "{}/{}")
        else:
            division_fmt = "{}/{}"

        return formatter(
            numerator,
            denominator,
            as_ratio=False,
            single_denominator=True,
            product_fmt=r"⋅",
            division_fmt=division_fmt,
            power_fmt=r"{}<sup>{}</sup>",
            parentheses_fmt=r"({})",
        )


class ScpLatexFormatter(LatexFormatter):
    """Subclasses the LatexFormatter to provide a specific formatting for SpectroChemPy."""

    def format_unit(self, unit, uspec, sort_func, **babel_kwds) -> str:
        numerator, denominator = prepare_compount_unit(
            unit,
            uspec,
            sort_func=sort_func,
            **babel_kwds,
            registry=self._registry,
        )

        numerator = ((rf"\mathrm{{{latex_escape(u)}}}", p) for u, p in numerator)
        denominator = ((rf"\mathrm{{{latex_escape(u)}}}", p) for u, p in denominator)

        formatted = formatter(
            numerator,
            denominator,
            as_ratio=False,
            single_denominator=True,
            product_fmt=r" \cdot ",
            division_fmt=r"\frac[{}][{}]",
            power_fmt="{}^[{}]",
            parentheses_fmt=r"\left({}\right)",
        )

        return formatted.replace("[", "{").replace("]", "}")


class ScpFullFormatter(FullFormatter):
    """Subclasses the Formatter to provide a specific formatting for SpectroChemPy."""

    default_format: str = "~P"

    def __init__(self, registry):
        super().__init__(registry)

        self._formatters = {}
        self._formatters["D"] = ScpDefaultFormatter(registry)
        self._formatters["C"] = ScpCompactFormatter(registry)
        self._formatters["P"] = ScpPrettyFormatter(registry)
        self._formatters["H"] = ScpHTMLFormatter(registry)
        self._formatters["L"] = ScpLatexFormatter(registry)


####################################################################################
# Spectrochempy UnitRegistry
####################################################################################

if globals().get("ur", None) is None:
    ur = UnitRegistry(on_redefinition="ignore")
    ur.formatter = ScpFullFormatter(ur)
    ur.formatter._registry = ur

    Quantity = ur.Quantity

    ur.define(
        "__wrapped__ = 1",
    )  # <- hack to avoid an error with pytest (doctest activated)
    ur.define("@alias count = point")
    ur.define("transmittance = 1. / 100. = %")
    ur.define("absolute_transmittance = 1.")
    ur.define("absorbance = 1. = a.u.")
    ur.define("Kubelka_Munk = 1. = K.M.")
    ur.define("ppm = 1. = ppm")

    # pint reads "." as multiplication, so custom display symbols that
    # contain dots (e.g. "a.u." for absorbance, "K.M." for Kubelka-Munk)
    # cannot be parsed back from their string form. Register a preprocessor
    # mapping those symbols (with or without the trailing dot) to their
    # canonical unit name before parsing. See #913.
    _register_dotted_symbols(ur)

    set_application_registry(ur)
    del UnitRegistry  # to avoid importing it

else:
    warnings.warn(
        "Unit registry was already set up. Bypassed the new loading",
        stacklevel=2,
    )

ur.enable_contexts("spectroscopy", "boltzmann", "chemistry")

########################################################################################
# utilities
########################################################################################


def remove_args_units(func):
    """Remove units of arguments of a function."""

    def _remove_units(val):
        if isinstance(val, Quantity):
            val = val.m
        elif isinstance(val, list | tuple):
            val = type(val)([_remove_units(v) for v in val])
        return val

    @wraps(func)
    def new_func(*args, **kwargs):
        args = tuple([_remove_units(arg) for arg in args])
        kwargs = {key: _remove_units(val) for key, val in kwargs.items()}
        return func(*args, **kwargs)

    return new_func
