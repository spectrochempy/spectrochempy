# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
L1 Style Resolution Layer for SpectroChemPy.

This module provides centralized style resolution logic for plotting functions.
It is Layer 1 (L1) - pure, deterministic, side-effect free.

This module must:
- Have NO pyplot usage
- Not create figures
- Not clear axes
- Not modify rcParams
- Only read matplotlib defaults if needed via `import matplotlib as mpl`

Priority order for all style resolution:
    kwargs > preferences > mpl_style > matplotlib defaults
"""

__all__ = [
    "resolve_line_style",
    "resolve_colormap",
    "resolve_stack_colors",
    "detect_diverging",
    "detect_stack_semantics",
    "_relative_luminance",
    "_contrast_ratio",
    "_get_categorical_cmap",
    "_ensure_min_contrast",
]

import numpy as np

# ======================================================================================
# Utility functions (copied from plot2d.py - no changes)
# ======================================================================================


def _relative_luminance(rgb):
    """
    Compute relative luminance of an sRGB color.

    Implements WCAG 2.1 relative luminance formula:
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B

    where R, G, B are linearized (gamma-corrected) sRGB values.

    Parameters
    ----------
    rgb : tuple
        RGB color tuple (r, g, b) with values in [0, 1].

    Returns
    -------
    float
        Relative luminance in [0, 1].
    """
    r, g, b = rgb[:3]

    def linearize(c):
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r_lin = linearize(r)
    g_lin = linearize(g)
    b_lin = linearize(b)

    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def _contrast_ratio(rgb1, rgb2):
    """
    Compute WCAG 2.1 contrast ratio between two colors.

    Contrast ratio = (L1 + 0.05) / (L2 + 0.05)
    where L1 is the lighter color's luminance.

    Parameters
    ----------
    rgb1, rgb2 : tuple
        RGB color tuples (r, g, b) with values in [0, 1].

    Returns
    -------
    float
        Contrast ratio (>= 1).
    """
    l1 = _relative_luminance(rgb1)
    l2 = _relative_luminance(rgb2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)


def _get_categorical_cmap(n):
    """
    Get categorical colormap based on number of categories.

    Parameters
    ----------
    n : int
        Number of categories.

    Returns
    -------
     Returns a ListedColormap with deterministic cycling behavior.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if n <= 10:
        base = plt.get_cmap("tab10").colors
    else:
        base = plt.get_cmap("tab20").colors

    # Deterministic cycling
    colors = [base[i % len(base)] for i in range(n)]

    return ListedColormap(colors, name="categorical_cycled")


def _ensure_min_contrast(cmap, background_rgb, min_contrast=1.5, samples=256):
    """
    Trim colormap to ensure minimum contrast with background.

    Samples the colormap and finds the smallest interval [start, end]
    where all colors meet the minimum contrast threshold with the background.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The colormap to potentially trim.
    background_rgb : tuple
        RGB tuple (r, g, b) of background color, values in [0, 1].
    min_contrast : float, optional, default: 1.5
        Minimum WCAG contrast ratio required.
    samples : int, optional, default: 256
        Number of samples to check across the colormap.

    Returns
    -------
    matplotlib.colors.Colormap
        Original colormap if no valid interval exists, otherwise
        a truncated colormap using LinearSegmentedColormap.from_list.
    """
    import matplotlib.colors as mcolors

    x = np.linspace(0, 1, samples)
    colors = cmap(x)

    contrasts = np.array([_contrast_ratio(c[:3], background_rgb) for c in colors])

    valid_mask = contrasts >= min_contrast

    if not np.any(valid_mask):
        return cmap

    valid_indices = np.where(valid_mask)[0]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]

    if start_idx == 0 and end_idx == samples - 1:
        return cmap

    start_frac = x[start_idx]
    end_frac = x[end_idx]

    truncated_colors = cmap(np.linspace(start_frac, end_frac, 256))
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated", truncated_colors, N=256
    )

    return truncated_cmap


# ======================================================================================
# Semantic detection functions (copied from plot2d.py - no changes)
# ======================================================================================


def detect_diverging(data, margin=0.05):
    """
    Detect if data should use diverging colormap.

    Diverging is selected only if:
    1) vmin < 0 < vmax
    2) min(abs(vmin), abs(vmax)) / (vmax - vmin) > margin

    Parameters
    ----------
    data : array-like
        The data to check.
    margin : float, optional, default: 0.05
        Minimum ratio threshold for smaller lobe.

    Returns
    -------
    bool
        True if diverging colormap should be used.
    """
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    if not (vmin < 0 < vmax):
        return False

    span = vmax - vmin
    if span == 0:
        return False

    smaller_lobe = min(abs(vmin), abs(vmax))
    ratio = smaller_lobe / span

    return ratio > margin


def detect_stack_semantics(dataset):
    """
    Detect semantic type for stack plots.

    Returns "categorical" if and only if:
    - dataset has no valid y coordinate
    - OR y coordinate is numeric, integer dtype, strictly consecutive,
      has no duplicates, and starts at 0 or 1

    Otherwise returns "sequential".

    Parameters
    ----------
    dataset : NDDataset
        The dataset being plotted as a stack.

    Returns
    -------
    str
        "categorical" or "sequential".
    """
    if dataset._squeeze_ndim < 2:
        return "categorical"

    dimy = dataset.dims[-2]
    y = getattr(dataset, dimy, None)

    if y is None or not hasattr(y, "data") or y.data is None:
        return "categorical"

    y_data = np.asarray(y.data)

    if not np.issubdtype(y_data.dtype, np.number):
        return "categorical"

    if not np.issubdtype(y_data.dtype, np.integer):
        return "sequential"

    if len(y_data) < 2:
        return "categorical"

    unique_sorted = np.unique(y_data)

    if len(unique_sorted) != len(y_data):
        return "sequential"

    diffs = np.diff(unique_sorted)

    if not np.all(diffs == 1):
        return "sequential"

    if unique_sorted[0] not in (0, 1):
        return "sequential"

    return "categorical"


# ======================================================================================
# Style resolution functions
# ======================================================================================


def resolve_line_style(
    dataset=None, geometry="line", kwargs=None, prefs=None, method=None
):
    """
    Resolve line/marker styles with priority: kwargs > preferences > mpl_style > matplotlib defaults.

    This function centralizes style resolution for 1D and 2D plotting.
    The priority order is strictly maintained:
        1. kwargs (explicitly provided)
        2. preferences (PlotPreferences)
        3. mpl_style defaults
        4. matplotlib defaults

    Parameters
    ----------
    dataset : NDDataset, optional
        The dataset being plotted. Used for geometry detection if not provided.
    geometry : str, optional, default: "line"
        Plot geometry: "line", "image", "contour", "surface".
    kwargs : dict, optional
        Explicit keyword arguments from caller.
    prefs : PlotPreferences, optional
        Preferences object. If None, will be fetched from spectrochempy.
    method : str, optional
        Plotting method name (e.g., "scatter", "scatter_pen", "pen").
        Used to determine marker fallback for scatter plots.

    Returns
    -------
    dict
        Resolved style values with keys:
        - color
        - linewidth (lw)
        - linestyle (ls)
        - marker
        - markersize (ms)
        - markerfacecolor (mfc)
        - markeredgecolor (mec)
        - alpha

        Each value is the resolved value OR the string "auto" if should use auto-detection.
    """
    if kwargs is None:
        kwargs = {}

    if prefs is None:
        from spectrochempy.application.preferences import preferences

        prefs = preferences

    result = {}

    result["color"] = kwargs.get("color", kwargs.get("c", "auto"))

    result["linewidth"] = kwargs.get("linewidth", kwargs.get("lw", "auto"))

    result["linestyle"] = kwargs.get("linestyle", kwargs.get("ls", "auto"))

    result["marker"] = kwargs.get("marker", kwargs.get("m", "auto"))

    # Deterministic fallback for scatter methods
    if result["marker"] == "auto" and method in ("scatter", "scatter_pen"):
        result["marker"] = getattr(prefs, "scatter_marker", "o")

    # For 2D plotting (stack), marker should default to None, not "auto"
    # The caller (plot2d.py) will handle None appropriately
    # For 1D plotting, "auto" is used to detect scatter vs pen

    result["markersize"] = kwargs.get(
        "markersize", kwargs.get("ms", prefs.lines_markersize)
    )

    result["markerfacecolor"] = kwargs.get("markerfacecolor", kwargs.get("mfc", "auto"))

    result["markeredgecolor"] = kwargs.get("markeredgecolor", kwargs.get("mec", "k"))

    result["alpha"] = kwargs.get("alpha")

    return result


def resolve_stack_colors(
    dataset, palette=None, n=None, geometry="line", contrast_safe=True, min_contrast=1.5
):
    """
    Resolve colors for stack plot with auto-detection.

    Parameters
    ----------
    dataset : NDDataset
        The 2D dataset being plotted.
    palette : str or list, optional
        If None: auto-detect based on dataset characteristics.
        If str:
            - "continuous": force continuous colormap (viridis)
            - "categorical": force categorical colors
            - colormap name: use that colormap
        If list: use as explicit categorical colors.
    n : int, optional
        Number of colors needed. If None, derived from dataset shape.
    geometry : str, optional, default: "line"
        Plot geometry. Affects contrast trimming.
    contrast_safe : bool, optional, default: True
        Whether to apply contrast trimming.
    min_contrast : float, optional, default: 1.5
        Minimum contrast ratio for trimming.

    Returns
    -------
    tuple
        (colors, is_categorical, mappable) where:
        - colors: list of color values (RGB tuples) or None
        - is_categorical: bool indicating discrete color cycling
        - mappable: ScalarMappable for continuous colormaps, None for categorical
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if n is None:
        n = dataset.shape[-2]

    if palette is not None:
        if palette == "continuous":
            cmap = plt.get_cmap("viridis")
            colors_data = cmap(np.linspace(0, 1, n))
            if contrast_safe and geometry in ("line", "contour"):
                background_rgb = (1.0, 1.0, 1.0)
                cmap = _ensure_min_contrast(cmap, background_rgb, min_contrast)
                colors_data = cmap(np.linspace(0, 1, n))
            # Return colors and mappable for continuous
            norm = Normalize(vmin=0, vmax=n - 1)
            mappable = ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(np.arange(n))
            return colors_data, False, mappable
        if palette == "categorical":
            cmap = _get_categorical_cmap(n)
            return list(cmap.colors), True, None

        if isinstance(palette, (list, tuple)):
            colors = list(palette)
            while len(colors) < n:
                colors.extend(palette)
            return colors[:n], True, None
        cmap = plt.get_cmap(palette)
        norm = Normalize(vmin=0, vmax=n - 1)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(np.arange(n))
        return cmap(np.linspace(0, 1, n)), False, mappable

    semantic = detect_stack_semantics(dataset)

    if semantic == "categorical":
        cmap = _get_categorical_cmap(n)
        return list(cmap.colors), True, None

    cmap = plt.get_cmap("viridis")
    colors_data = cmap(np.linspace(0, 1, n))

    if contrast_safe and geometry in ("line", "contour"):
        background_rgb = (1.0, 1.0, 1.0)
        cmap = _ensure_min_contrast(cmap, background_rgb, min_contrast)
        colors_data = cmap(np.linspace(0, 1, n))

    # Return colors and mappable for continuous
    norm = Normalize(vmin=0, vmax=n - 1)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.arange(n))
    return colors_data, False, mappable


def resolve_colormap(
    data=None,
    semantic="auto",
    cmap=None,
    norm=None,
    center=None,
    diverging_margin=0.05,
    contrast_safe=True,
    min_contrast=1.5,
    background_rgb=None,
    n=None,
    geometry=None,
    dataset=None,
):
    """
    Unified colormap resolver with semantic color system.

    This is the single entry point for colormap resolution across all
    plotting functions. It applies geometry-aware contrast trimming
    and enforces strict priority order.

    Priority order (strict):
        1. norm explicitly provided -> use as-is
        2. cmap explicitly provided -> use as-is
        3. semantic != "auto" -> respect it
        4. semantic="auto":
               - For stacks (dataset provided) -> stack detection rule
               - For fields -> diverging auto-detection rule
        5. Contrast trimming: ONLY if semantic==sequential AND
           geometry in ("line","contour") AND contrast_safe==True

    Parameters
    ----------
    data : array-like, optional
        The data to be plotted. Used for auto-detection.
    semantic : str, optional, default: "auto"
        Color semantics: "auto", "sequential", "diverging", or "categorical".
    cmap : str or Colormap, optional
        Explicit colormap. If str, will be resolved to Colormap.
    norm : Normalize, optional
        Explicit normalization. If provided, skips all auto-detection.
    center : numeric or str, optional
        Center value for diverging colormaps.
    diverging_margin : float, optional, default: 0.05
        Minimum ratio threshold for diverging auto-detection.
    contrast_safe : bool, optional, default: True
        Whether to apply contrast trimming.
    min_contrast : float, optional, default: 1.5
        Minimum WCAG contrast ratio for trimming.
    background_rgb : tuple, optional
        RGB tuple (r,g,b) of background. Defaults to white (1,1,1).
    n : int, optional
        Number of colors needed for categorical.
    geometry : str, optional
        Plot geometry: "line", "contour", "image", "surface".
    dataset : NDDataset, optional
        Dataset for stack semantic detection.

    Returns
    -------
    tuple
        (cmap, norm) resolved values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.colors import TwoSlopeNorm

    if background_rgb is None:
        background_rgb = (1.0, 1.0, 1.0)

    if norm is not None:
        if cmap is None:
            cmap = plt.get_cmap("viridis")
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        return cmap, norm

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        if norm is None:
            if data is not None:
                vmin = np.nanmin(data)
                vmax = np.nanmax(data)
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = Normalize(vmin=0, vmax=1)
        return cmap, norm

    if semantic == "categorical":
        if n is None:
            n = 10
        cmap = _get_categorical_cmap(n)
        if data is not None:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=0, vmax=n - 1)
        return cmap, norm

    if semantic == "sequential":
        use_diverging = False
    elif semantic == "diverging":
        use_diverging = True
    else:
        if dataset is not None:
            semantic = detect_stack_semantics(dataset)
            if semantic == "categorical":
                if n is None:
                    n = dataset.shape[-2] if dataset._squeeze_ndim >= 2 else 10
                cmap = _get_categorical_cmap(n)
                vmin = np.nanmin(data) if data is not None else 0
                vmax = np.nanmax(data) if data is not None else n - 1
                norm = Normalize(vmin=vmin, vmax=vmax)
                return cmap, norm
            use_diverging = False
        elif data is not None:
            use_diverging = detect_diverging(data, diverging_margin)
        else:
            use_diverging = False

    if use_diverging:
        cmap = plt.get_cmap("RdBu_r")
        if data is not None:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            if center is None:
                center_value = 0
            elif center == "auto":
                center_value = 0 if vmin < 0 < vmax else (vmin + vmax) / 2
            else:
                center_value = center

            if center_value <= vmin:
                center_value = vmin + (vmax - vmin) / 2
            if center_value >= vmax:
                center_value = vmin + (vmax - vmin) / 2

            norm = TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
        else:
            norm = Normalize(vmin=-1, vmax=1)
    else:
        cmap = plt.get_cmap("viridis")
        if data is not None:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=0, vmax=1)

    should_trim = (
        contrast_safe and semantic == "sequential" and geometry in ("line", "contour")
    )

    if should_trim:
        cmap = _ensure_min_contrast(cmap, background_rgb, min_contrast)

    return cmap, norm


def resolve_2d_colormap(
    data,
    cmap=None,
    cmap_mode="auto",
    center=None,
    norm=None,
    vmin=None,
    vmax=None,
    contrast_safe=True,
    min_contrast=1.5,
    background_rgb=None,
    geometry=None,
    diverging_margin=0.05,
):
    """
    Resolve colormap and normalization for 2D plots with auto-detection.

    This function implements a consistent colormap API for all 2D plotting functions
    (plot_image, plot_contour, plot_surface). It provides:
    - Auto-detection of sequential vs diverging colormaps based on data
    - Centered normalization for bipolar data
    - Explicit overrides via parameters
    - Geometry-aware contrast safety to ensure visibility on background

    Priority order (strict):
        1. norm (if explicitly provided) -> use as-is, no auto-detection
        2. vmin/vmax (if explicitly provided) -> override data range
        3. cmap (if explicitly provided) -> use as-is
        4. cmap_mode (if not "auto") -> force sequential or diverging
        5. auto-detection -> sequential if all positive, diverging if bipolar with margin
        6. contrast_safe -> trim colormap ends for visibility (geometry-aware)

    Parameters
    ----------
    data : array-like
        The 2D data array to be plotted.
    cmap : str, optional
        Colormap name. If None, will be determined based on cmap_mode.
    cmap_mode : str, optional, default: "auto"
        "auto", "sequential", or "diverging".
    center : numeric or str, optional
        Center value for diverging colormaps.
    norm : matplotlib.colors.Normalize, optional
        Explicit normalization. If provided, overrides all other normalization.
    vmin : float, optional
        Minimum value for normalization. If provided, overrides data-derived minimum.
    vmax : float, optional
        Maximum value for normalization. If provided, overrides data-derived maximum.
    contrast_safe : bool, optional, default: True
        If True, trim colormap ends to ensure minimum contrast with background.
    min_contrast : float, optional, default: 1.5
        Minimum WCAG contrast ratio.
    background_rgb : tuple, optional
        RGB tuple (r, g, b) of background color. Defaults to white (1,1,1).
    geometry : str, optional
        Plot geometry: "line", "contour", "image", "surface".
    diverging_margin : float, optional, default: 0.05
        Minimum ratio threshold for diverging auto-detection.

    Returns
    -------
    tuple
        (cmap, norm) resolved values for plotting.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.colors import TwoSlopeNorm

    if background_rgb is None:
        background_rgb = (1.0, 1.0, 1.0)

    if norm is not None:
        if cmap is None:
            cmap = plt.get_cmap("viridis")
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        return cmap, norm

    data_min = np.nanmin(data)
    data_max = np.nanmax(data)

    if vmin is not None:
        data_min = vmin
    if vmax is not None:
        data_max = vmax

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        if norm is None:
            norm = Normalize(vmin=data_min, vmax=data_max)
        return cmap, norm

    if cmap_mode == "diverging":
        semantic = "diverging"
    elif cmap_mode == "sequential":
        semantic = "sequential"
    else:
        semantic = "auto"

    if semantic == "sequential":
        use_diverging = False
    elif semantic == "diverging":
        use_diverging = True
    else:
        use_diverging = detect_diverging(data, diverging_margin)

    if use_diverging:
        cmap = plt.get_cmap("RdBu_r")
        if center is None:
            center_value = 0
        elif center == "auto":
            center_value = 0 if data_min < 0 < data_max else (data_min + data_max) / 2
        else:
            center_value = center

        if center_value <= data_min:
            center_value = data_min + (data_max - data_min) / 2
        if center_value >= data_max:
            center_value = data_min + (data_max - data_min) / 2

        norm = TwoSlopeNorm(vmin=data_min, vcenter=center_value, vmax=data_max)
    else:
        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=data_min, vmax=data_max)

    should_trim = contrast_safe and geometry in ("line", "contour")

    if should_trim:
        cmap = _ensure_min_contrast(cmap, background_rgb, min_contrast)

    return cmap, norm
