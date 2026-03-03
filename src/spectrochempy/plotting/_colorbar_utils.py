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
from matplotlib.ticker import NullLocator


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

    This function lets Matplotlib's locator handle tick generation.

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
    This function uses MaxNLocator for automatic tick selection.
    """
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import ScalarFormatter

    # Get actual normalization limits from the mappable
    actual_vmin = cbar.mappable.norm.vmin
    actual_vmax = cbar.mappable.norm.vmax

    # Override with explicit vmin/vmax if provided
    if vmin is not None:
        actual_vmin = vmin
    if vmax is not None:
        actual_vmax = vmax

    # Set the y-axis limits to match the data range
    cbar.ax.set_ylim(actual_vmin, actual_vmax)

    # Use MaxNLocator for automatic tick selection
    locator = MaxNLocator(nbins=9)
    cbar.ax.yaxis.set_major_locator(locator)

    # Update ticks to reflect the new locator
    cbar.update_ticks()

    # Set formatter for nice scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    formatter.set_scientific(False)
    cbar.ax.yaxis.set_major_formatter(formatter)

    # Disable minor ticks completely
    cbar.ax.yaxis.set_minor_locator(NullLocator())
