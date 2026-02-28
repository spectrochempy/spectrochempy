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
    "build_font_rc_overrides",
]

import matplotlib as mpl
import numpy as np

_MPL_DEFAULT_IMAGE_CMAP = mpl.rcParamsDefault["image.cmap"]

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


def build_font_rc_overrides(prefs):
    """
    Build a dictionary of matplotlib rcParams overrides from font preferences.

    This function is L1-safe: it does NOT mutate global rcParams.
    It returns a dictionary that can be used with mpl.rc_context().

    Parameters
    ----------
    prefs : PlotPreferences
        The plot preferences object containing font settings.

    Returns
    -------
    dict
        A dictionary of matplotlib rcParams overrides for font settings.
        Only includes keys that have non-default values in prefs.

    Notes
    -----
    Priority order: plot kwargs > prefs.font.* > style sheet > matplotlib defaults

    This function only handles the prefs.font.* level. Higher precedence
    (plot kwargs) should be handled separately in L3.
    """
    overrides = {}

    # Font family
    if hasattr(prefs, "font_family"):
        family = prefs.font_family
        if family is not None:
            overrides["font.family"] = family

    # Font size
    if hasattr(prefs, "font_size"):
        size = prefs.font_size
        if size is not None and size != 10.0:  # 10.0 is matplotlib default
            overrides["font.size"] = float(size)

    # Font style
    if hasattr(prefs, "font_style"):
        style = prefs.font_style
        if style is not None and style != "normal":
            overrides["font.style"] = style

    # Font weight
    if hasattr(prefs, "font_weight"):
        weight = prefs.font_weight
        if weight is not None and weight != "normal":
            overrides["font.weight"] = (
                str(weight) if isinstance(weight, int) else weight
            )

    # Font variant
    if hasattr(prefs, "font_variant"):
        variant = prefs.font_variant
        if variant is not None and variant != "normal":
            overrides["font.variant"] = variant

    # Font stretch (if implemented in prefs)
    if hasattr(prefs, "font_stretch"):
        stretch = prefs.font_stretch
        if stretch is not None and stretch != "normal":
            overrides["font.stretch"] = stretch

    return overrides


def _get_categorical_cmap(
    n, default_small="tab10", default_large="tab20", threshold=10
):
    """
    Get categorical colormap based on number of categories.

    Parameters
    ----------
    n : int
        Number of categories.
    default_small : str, optional
        Colormap for n <= threshold. Default: "tab10".
    default_large : str, optional
        Colormap for n > threshold. Default: "tab20".
    threshold : int, optional
        Threshold for switching between small and large. Default: 10.

    Returns
    -------
    ListedColormap
        A ListedColormap with deterministic cycling behavior.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if n <= threshold:
        base = plt.get_cmap(default_small).colors
    else:
        base = plt.get_cmap(default_large).colors

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
    return mcolors.LinearSegmentedColormap.from_list(
        "truncated", truncated_colors, N=256
    )


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
    dataset,
    palette="auto",
    n=None,
    geometry="line",
    contrast_safe=True,
    min_contrast=1.5,
    default_categorical_small="tab10",
    default_categorical_large="tab20",
    prefs=None,
):
    """
    Resolve colors for stack plot based on palette parameter.

    This function implements a deterministic palette API for stack plots.
    All semantic detection is delegated to detect_stack_semantics().

    Parameters
    ----------
    dataset : NDDataset
        The 2D dataset being plotted.
    palette : {"auto", "categorical", "continuous"} or str or list, optional
        Controls how multiple curves are colored in stack plots.

        - "auto" (default): detect from dataset semantics using detect_stack_semantics().
        - "categorical": force categorical mode, use matplotlib color cycle.
        - "continuous": force continuous mode, use sequential colormap.
        - colormap name string: force continuous mode using that colormap.
        - list of colors: explicit categorical colors.

        This parameter applies only to plot_lines.
    n : int, optional
        Number of colors needed. If None, derived from dataset shape.
    geometry : str, optional, default: "line"
        Plot geometry. Affects contrast trimming.
    contrast_safe : bool, optional, default: True
        Whether to apply contrast trimming.
    min_contrast : float, optional, default: 1.5
        Minimum contrast ratio for trimming.
    default_categorical_small : str, optional
        Default colormap for categorical with <=10 categories. Default: "tab10".
    default_categorical_large : str, optional
        Default colormap for categorical with >10 categories. Default: "tab20".
    prefs : object, optional
        Preferences object. If None, will be fetched from spectrochempy.

    Returns
    -------
    tuple
        (colors, is_categorical, mappable) where:
        - colors: list of color values (RGB tuples) or None
        - is_categorical: bool indicating discrete color cycling
        - mappable: ScalarMappable for continuous colormaps, None for categorical
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if prefs is None:
        from spectrochempy.application.preferences import preferences

        prefs = preferences

    if n is None:
        n = dataset.shape[-2]

    # Determine mode based on palette value
    mode = None
    explicit_cmap = None
    explicit_colors = None

    if palette is None or palette == "auto":
        mode = detect_stack_semantics(dataset)

    elif palette == "categorical":
        mode = "categorical"

    elif palette == "continuous":
        mode = "continuous"

    elif isinstance(palette, str):
        mode = "continuous"
        explicit_cmap = palette

    elif isinstance(palette, (list, tuple)):
        mode = "categorical"
        explicit_colors = list(palette)

    else:
        raise ValueError(
            f"Invalid palette value: {palette!r}. "
            "Expected 'auto', 'categorical', 'continuous', colormap name, or list of colors."
        )

    # Execute based on mode
    if mode == "categorical":
        if explicit_colors is not None:
            colors = list(explicit_colors)
            while len(colors) < n:
                colors.extend(explicit_colors)
            return colors[:n], True, None

        _MPL_DEFAULT_PROP_CYCLE = mpl.rcParamsDefault["axes.prop_cycle"]
        current_cycle = mpl.rcParams["axes.prop_cycle"]
        if current_cycle != _MPL_DEFAULT_PROP_CYCLE:
            cycle_colors = [c["color"] for c in current_cycle]
            colors = [cycle_colors[i % len(cycle_colors)] for i in range(n)]
            return colors, True, None

        if n <= 10:
            cmap = plt.get_cmap(default_categorical_small)
        else:
            cmap = plt.get_cmap(default_categorical_large)
        colors = [cmap(i) for i in np.linspace(0, 1, cmap.N)[:n]]
        return colors, True, None

    # mode == "continuous"
    if explicit_cmap is not None:
        cmap = plt.get_cmap(explicit_cmap)
    else:
        style_cmap = mpl.rcParams.get("image.cmap")
        if style_cmap is not None and style_cmap != _MPL_DEFAULT_IMAGE_CMAP:
            cmap = plt.get_cmap(style_cmap)
        else:
            cmap = plt.get_cmap(prefs.colormap_sequential)

    colors_data = cmap(np.linspace(0, 1, n))

    if contrast_safe and geometry in ("line", "contour"):
        background_rgb = (1.0, 1.0, 1.0)
        cmap = _ensure_min_contrast(cmap, background_rgb, min_contrast)
        colors_data = cmap(np.linspace(0, 1, n))

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
    default_categorical_small="tab10",
    default_categorical_large="tab20",
    categorical_threshold=10,
    prefs=None,
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
    default_categorical_small : str, optional
        Default colormap for categorical with n <= threshold. Default: "tab10".
    default_categorical_large : str, optional
        Default colormap for categorical with n > threshold. Default: "tab20".
    categorical_threshold : int, optional
        Threshold for small vs large categorical. Default: 10.
    prefs : object, optional
        Preferences object. If None, will be fetched from spectrochempy.

    Returns
    -------
    tuple
        (cmap, norm) resolved values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.colors import TwoSlopeNorm

    if prefs is None:
        from spectrochempy.application.preferences import preferences

        prefs = preferences

    if background_rgb is None:
        background_rgb = (1.0, 1.0, 1.0)

    if norm is not None:
        if cmap is None:
            cmap = plt.get_cmap(prefs.colormap_sequential)
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
        cmap = _get_categorical_cmap(
            n,
            default_small=default_categorical_small,
            default_large=default_categorical_large,
            threshold=categorical_threshold,
        )
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
                cmap = _get_categorical_cmap(
                    n,
                    default_small=default_categorical_small,
                    default_large=default_categorical_large,
                    threshold=categorical_threshold,
                )
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
        cmap = plt.get_cmap(prefs.colormap_diverging)
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
        cmap = plt.get_cmap(prefs.colormap_sequential)
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
    cmap_explicit=False,
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
    prefs=None,
):
    """
    Resolve colormap and normalization for 2D plots with auto-detection.

    This function implements a consistent colormap API for all 2D plotting functions
    (plot_image, plot_contour, plot_surface). It provides:
    - Auto-detection of sequential vs diverging colormaps based on data
    - Centered normalization for bipolar data
    - Explicit overrides via parameters
    - Geometry-aware contrast safety to ensure visibility on background
    - Categorical colormap support when cmap=None is explicitly passed

    Priority order (strict):
        1. norm (if explicitly provided) -> use as-is, no auto-detection
        2. vmin/vmax (if explicitly provided) -> override data range
        3. cmap_explicit=True, cmap=None -> categorical colormap (discrete)
        4. cmap_explicit=True, cmap=string -> use as-is
        5. cmap_explicit=False, prefs.colormap != "auto" -> use prefs.colormap
        6. cmap_mode (if not "auto") -> force sequential or diverging
        7. auto-detection -> sequential if all positive, diverging if bipolar with margin
        8. contrast_safe -> trim colormap ends for visibility (geometry-aware)

    Parameters
    ----------
    data : array-like
        The 2D data array to be plotted.
    cmap : str, optional
        Colormap name. If None, will be determined based on cmap_explicit and cmap_mode.
    cmap_explicit : bool, optional, default: False
        If True, cmap was explicitly provided by user (including None).
        If False, cmap was not provided at all.
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
    prefs : object, optional
        Preferences object with colormap, colormap_sequential, colormap_diverging,
        colormap_categorical_small, colormap_categorical_large, and
        colormap_categorical_threshold attributes. If None, will be fetched from
        spectrochempy.application.preferences.

    Returns
    -------
    tuple
        (cmap, norm) resolved values for plotting.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.colors import TwoSlopeNorm

    if prefs is None:
        from spectrochempy.application.preferences import preferences

        prefs = preferences

    if background_rgb is None:
        background_rgb = (1.0, 1.0, 1.0)

    if norm is not None:
        if cmap is None:
            cmap = plt.get_cmap(prefs.colormap_sequential)
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        return cmap, norm

    data_min = np.nanmin(data)
    data_max = np.nanmax(data)

    if vmin is not None:
        data_min = vmin
    if vmax is not None:
        data_max = vmax

    # Handle explicit categorical request (cmap_explicit=True, cmap=None)
    if cmap_explicit and cmap is None:
        # User explicitly passed cmap=None -> categorical colormap request
        # Get number of unique finite values in data
        finite_data = data[np.isfinite(data)]
        unique_values = np.unique(finite_data)
        n_unique = len(unique_values)

        # Get categorical preferences
        if prefs is not None:
            small_map = getattr(prefs, "colormap_categorical_small", "tab10")
            large_map = getattr(prefs, "colormap_categorical_large", "tab20")
            threshold = getattr(prefs, "colormap_categorical_threshold", 10)
        else:
            small_map = "tab10"
            large_map = "tab20"
            threshold = 10

        # Select base map based on threshold
        base = small_map if n_unique <= threshold else large_map

        # Build categorical colormap with exact number of colors
        cmap = _get_categorical_cmap(n_unique, base, threshold=threshold)

        # Create discrete-compatible normalization (no interpolation)
        norm = Normalize(vmin=0, vmax=n_unique - 1, clip=False)

        return cmap, norm

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        if norm is None:
            norm = Normalize(vmin=data_min, vmax=data_max)
        return cmap, norm

    # Handle implicit cmap (not provided) with prefs.colormap
    if cmap_explicit is False and prefs is not None:
        prefs_colormap = getattr(prefs, "colormap", "auto")
        if prefs_colormap != "auto":
            # Use fixed colormap from preferences
            cmap = plt.get_cmap(prefs_colormap)
            norm = Normalize(vmin=data_min, vmax=data_max)
            return cmap, norm

    # Check matplotlib style override - read rcParams dynamically at call time
    # This ensures we pick up any style-driven rcParams changes when called
    # within a plt.style.context() block
    current_cmap = mpl.rcParams.get("image.cmap", None)
    if current_cmap is not None and current_cmap != _MPL_DEFAULT_IMAGE_CMAP:
        cmap = plt.get_cmap(current_cmap)
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
        cmap = plt.get_cmap(prefs.colormap_diverging)
        # Use symmetric normalization around 0 for scientific standard
        maxabs = max(abs(data_min), abs(data_max))
        norm = TwoSlopeNorm(vmin=-maxabs, vcenter=0.0, vmax=+maxabs)
    else:
        cmap = plt.get_cmap(prefs.colormap_sequential)
        norm = Normalize(vmin=data_min, vmax=data_max)

    should_trim = contrast_safe and geometry in ("line", "contour")

    if should_trim:
        cmap = _ensure_min_contrast(cmap, background_rgb, min_contrast)

    return cmap, norm
