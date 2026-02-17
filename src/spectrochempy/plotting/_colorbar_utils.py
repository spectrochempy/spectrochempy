# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Colorbar utilities for SpectroChemPy.

This module provides helpers for creating and configuring colorbars
in a deterministic, architecture-compliant way.
"""

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullLocator


def _nice_step(raw_step):
    """
    Return a 'nice' step using 1-2-5 scaling.

    Parameters
    ----------
    raw_step : float
        Raw step size to round to a nice value.

    Returns
    -------
    float
        A nice step size (1, 2, 5, 10 multiplied by a power of 10).
    """
    if raw_step <= 0:
        return 1.0

    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10**exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * (10**exponent)


def _apply_colorbar_tick_policy(cbar, norm, vmin=None, vmax=None):
    """
    Apply deterministic tick policy based on norm and data range.

    This function enforces a publication-grade tick strategy:
    - Diverging norms (TwoSlopeNorm): symmetric ticks around 0
    - Sequential norms (vmin >= 0): evenly spaced ticks

    Parameters
    ----------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar to configure.
    norm : matplotlib.colors.Normalize
        The normalization object used for the colormap.
    vmin : float, optional
        Minimum data value. If None, derived from norm.
    vmax : float, optional
        Maximum data value. If None, derived from norm.

    Notes
    -----
    This function uses FixedLocator with deterministic nice ticks.
    No MaxNLocator, AutoLocator, or update_ticks() are used.
    """
    if vmin is None:
        vmin = norm.vmin
    if vmax is None:
        vmax = norm.vmax

    # Force canvas draw to get accurate geometry
    fig = cbar.ax.figure
    fig.canvas.draw()

    # Compute colorbar height in pixels
    bbox = cbar.ax.get_window_extent()
    height_px = bbox.height

    # Define minimum pixel spacing per tick label
    MIN_LABEL_SPACING = 22  # px

    # Compute maximum allowed ticks based on height
    max_ticks = max(3, int(height_px / MIN_LABEL_SPACING))
    max_ticks = min(max_ticks, 9)

    is_diverging = isinstance(norm, TwoSlopeNorm)

    if is_diverging:
        maxabs = max(abs(vmin), abs(vmax))
        raw_step = (2 * maxabs) / (max_ticks - 1)
        step = _nice_step(raw_step)
        maxabs_rounded = np.ceil(maxabs / step) * step

        ticks = np.arange(-maxabs_rounded, maxabs_rounded + step * 0.5, step)
        ticks = np.unique(np.round(ticks, decimals=10))
    else:
        if vmin == vmax:
            ticks = np.array([vmin])
        else:
            raw_step = (vmax - vmin) / (max_ticks - 1)
            step = _nice_step(raw_step)

            start = np.floor(vmin / step) * step
            end = np.ceil(vmax / step) * step

            ticks = np.arange(start, end + step * 0.5, step)
            ticks = np.unique(np.round(ticks, decimals=10))
            ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]

    cbar.locator = FixedLocator(ticks)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    formatter.set_scientific(False)
    cbar.ax.yaxis.set_major_formatter(formatter)

    # Disable minor ticks completely
    cbar.ax.yaxis.set_minor_locator(NullLocator())
