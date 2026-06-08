"""NMR-specific unit-context helpers."""

from __future__ import annotations

from pint import Context

from spectrochempy.core.units import ur

__all__ = ["set_nmr_context"]


def set_nmr_context(larmor):
    """
    Set the NMR Pint context for ppm/frequency conversions.

    Parameters
    ----------
    larmor : `Quantity` or `float`
        Larmor frequency of the current nucleus. Bare numbers are
        interpreted as MHz.
    """
    if not isinstance(larmor, ur.Quantity):
        larmor = larmor * ur.MHz

    if "nmr" not in ur._contexts:
        context = Context("nmr", defaults={"larmor": larmor})
        context.add_transformation(
            "[]",
            "[frequency]",
            lambda _ur, value, **kwargs: value * kwargs.get("larmor") / 1.0e6,
        )
        context.add_transformation(
            "[frequency]",
            "[]",
            lambda _ur, value, **kwargs: value * 1.0e6 / kwargs.get("larmor"),
        )
        ur.add_context(context)
    else:
        ur._contexts["nmr"].defaults["larmor"] = larmor
