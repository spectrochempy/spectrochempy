"""NMR-specific post-processing for FFT results (axis labels, ppm conversion)."""

from __future__ import annotations

import re

from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415
from spectrochempy.core.units import ur  # noqa: PLC0415


def _meta_dim_index(new, dim):
    """Resolve the metadata index matching the target dataset dimension."""
    if isinstance(dim, str):
        return new.dims.index(dim)
    axis, _ = new.get_axis(dim, negative_axis=False)
    return axis


def _fft_postprocess_result(new, dim=-1, inv=False, **kwargs):
    """
    Apply NMR-specific coordinate creation and unit conversion after FFT.

    This handler is registered on ``fft.postprocess_result`` and recreates
    the spectral axis when the dataset carries NMR metadata (``sfo1``,
    ``bf1``, ``sw_h``, ``nuc1``).
    """
    # Only act when NMR metadata is present
    if not hasattr(new.meta, "sfo1") or new.meta.sfo1 is None:
        return new

    x = new.coordset[dim]
    size = x.size
    meta_idx = _meta_dim_index(new, dim)

    sfo1 = new.meta.sfo1[meta_idx]
    bf1 = new.meta.bf1[meta_idx]
    sf = new.meta.sf[meta_idx]
    sw = new.meta.sw_h[meta_idx]

    if new.meta.nuc1 is not None:
        nuc1 = new.meta.nuc1[meta_idx]
        m = re.match(r"([^a-zA-Z]+)([a-zA-Z]+)", nuc1)
        nucleus = "^{" + m[1] + "}" + m[2] if m is not None else ""
    else:
        nucleus = ""

    if not inv:
        # time → frequency
        sizem = max(size - 1, 1)
        deltaf = -sw / sizem
        first = sfo1 - sf - deltaf * sizem / 2.0

        newcoord = Coord.arange(size) * deltaf + first
        newcoord.show_datapoints = False
        newcoord.name = x.name
        newcoord.title = f"${nucleus}$ frequency"
        newcoord.ito("Hz")

        # Store acquisition frequency for ppm conversion context
        newcoord.meta["acquisition_frequency"] = bf1

        ppm = kwargs.get("ppm", True)
        if ppm:
            newcoord.ito("ppm")
            newcoord.title = rf"$\delta\ {nucleus}$"

        new.coordset[dim] = newcoord
    else:
        # frequency/ppm → time
        # Replace the generic core-created coordinate with a proper NMR time axis.
        sw_val = abs(x.data[-1] - x.data[0])
        if x.units == "ppm":
            sw_val = (bf1.to("Hz").magnitude * sw_val) / 1.0e6
        deltat = (1.0 / sw_val) * ur.us

        newcoord = Coord.arange(size) * deltat
        newcoord.name = x.name
        newcoord.title = "time"
        newcoord.ito("us")
        new.coordset[dim] = newcoord

    if getattr(new.meta, "isfreq", None) is not None:
        new.meta.isfreq[meta_idx] = not inv

    return new
