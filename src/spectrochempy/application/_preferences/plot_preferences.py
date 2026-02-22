# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
PlotPreferences.

This module implements SpectroChemPy's plotting configuration system.

It provides a *typed, observable, reversible* interface to Matplotlib's
global rcParams, using traitlets as the configuration backbone.

Key ideas
---------
- Matplotlib uses a global dict (rcParams) with side effects
- SpectroChemPy exposes plotting options as traitlets
- Trait changes are propagated to rcParams in a controlled way
- Matplotlib style sheets are *parsed*, not blindly applied

This file is intentionally verbose and explicit to avoid hidden plotting
side effects and make rcParams restoration possible.

IMPORTANT:
----------
This module lazy-loads matplotlib to avoid importing it at module import time.
All matplotlib imports are inside functions that are only called during
actual plotting, not at import/init time.
"""

# Static values for Line2D traits (lazy-loaded, not imported at module level)
# These values are unlikely to change across matplotlib versions
LINESTYLE_KEYS = ["-", "--", "-.", ":", "None", " ", ""]
MARKER_KEYS = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "d",
    "|",
    "_",
    "P",
    "X",
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    "None",
    "none",
    " ",
    "",
]

# traitlets below are used to:
# - provide type validation (Float, Bool, Enum, etc.)
# - allow observation of changes (@observe)
# - synchronize preferences → rcParams
#
# This is very different from NumPy/Pandas, but closer to Jupyter/IPython,
# where configuration is dynamic and user-facing.
from traitlets import All
from traitlets import Bool
from traitlets import Enum
from traitlets import Float
from traitlets import Integer
from traitlets import List
from traitlets import Set
from traitlets import TraitError
from traitlets import Tuple
from traitlets import Unicode
from traitlets import Union as TraitUnion
from traitlets import default
from traitlets import observe
from traitlets import validate

from spectrochempy.utils.metaconfigurable import MetaConfigurable


# Helper function for lazy preference management
def _is_mpl_initialized():
    """Check if matplotlib has been initialized by SpectroChemPy."""
    try:
        from spectrochempy.core.plotters.plot_setup import (
            _is_mpl_initialized as _is_initialized,
        )

        return _is_initialized()
    except ImportError:
        return False


# --------------------------------------------------------------------------------------
# available matplotlib styles (equivalent of plt.style.available)
# --------------------------------------------------------------------------------------
# This function extends matplotlib's own style discovery by also
# checking the user's matplotlib config directory.
#
# It is *not* cached on purpose:
# - users may add styles at runtime
# - styles are filesystem-based


def available_styles():
    """
    All matplotlib `styles <https://matplotlib.org/users/style_sheets.html>`_ which are available in  `SpectroChemPy`.

    Returns
    -------
    A list of matplotlib styles
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    from spectrochempy.utils.file import pathclean

    # Todo: Make this list extensible programmatically (adding files to stylelib)
    cfgdir = mpl.get_configdir()
    stylelib = pathclean(cfgdir) / "stylelib"
    styles = plt.style.available
    if stylelib.is_dir():
        listdir = stylelib.glob("*.mplstyle")
        for style in listdir:
            styles.append(style.stem)
    return list(set(styles))  # in order to remove possible duplicates


def _canonical_cmap_name(name: str) -> str:
    """Return Matplotlib's canonical colormap name, case-insensitively."""

    # Normalize colormap names in a case-insensitive way.
    #
    # Matplotlib itself is case-sensitive in many places, which leads
    # to confusing user errors. This helper ensures:
    #
    #   "Viridis" → "viridis"
    #   "VIRIDIS" → "viridis"
    #
    # while still rejecting invalid names early.

    if not isinstance(name, str):
        raise TraitError("colormap must be a string")

    # Lazy import to avoid loading matplotlib at module import time
    from matplotlib import pyplot as plt

    # Build a case-insensitive map from available colormaps
    cmaps = plt.colormaps()
    cmap_map = {c.lower(): c for c in cmaps}

    key = name.lower()
    if key not in cmap_map:
        raise TraitError(
            f"Invalid colormap '{name}'. "
            f"Available colormaps include: {', '.join(cmaps[:10])}..."
        )
    return cmap_map[key]


class PlotPreferences(MetaConfigurable):
    """
    Typed representation of Matplotlib rcParams + SpectroChemPy extensions.

    This class mirrors most matplotlib.rcParams entries as traitlets.
    Changing a trait updates rcParams automatically via observers.

    Important:
    ---------
    - rcParams are NOT the source of truth
    - traitlets are the source of truth
    - rcParams are a projection of the current PlotPreferences state
    """

    name = Unicode("PlotPreferences")
    description = Unicode("Options for Matplotlib")
    updated = Bool(False)
    _groups = Set(Unicode())
    _subgroups = Set(Unicode())
    _members = Set(Unicode())

    # ----------------------------------------------------------------------------------
    # Matplotlib rcParams mirrored as traitlets
    # ----------------------------------------------------------------------------------
    #
    # Naming convention:
    #   rcParams key:      "lines.linewidth"
    #   trait name:        lines_linewidth
    #
    # The dot is replaced by an underscore.
    # Mapping back to rcParams is handled by to_rc_key().
    #
    # Most defaults here match Matplotlib defaults or SpectroChemPy style defaults.

    # NOTE:
    # Some defaults differ from Matplotlib's defaults.
    # This is intentional and reflects SpectroChemPy's
    # "scpy" visual identity.
    #
    # These defaults are applied ONLY when the scpy style
    # is selected, not at import time.

    # ---------
    # 1. LINES
    # ---------
    # See http://matplotlib.org/api/artist_api.html#module-matplotlib.lines for more
    # information on line properties.
    #
    lines_linewidth = Float(0.75, help=r"""line width in points""").tag(
        config=True,
        gui=True,
        kind="",
    )
    lines_linestyle = Enum(
        list(LINESTYLE_KEYS),
        default_value="-",
        help=r"""solid line""",
    ).tag(config=True, gui=True, kind="")
    lines_marker = Enum(
        list(MARKER_KEYS),
        default_value="None",
        help=r"""the default marker""",
    ).tag(config=True, kind="")
    lines_markerfacecolor = Unicode(
        "auto",
        help=r"""the default markerfacecolor""",
    ).tag(config=True, kind="color")
    lines_markeredgecolor = Unicode(
        "auto",
        help=r"""the default markeredgecolor""",
    ).tag(config=True, kind="color")
    lines_markeredgewidth = Float(
        0.0,
        help=r"""the line width around the marker symbol""",
    ).tag(config=True, kind="")
    lines_markersize = Float(7.0, help=r"""markersize, in points""").tag(
        config=True,
        kind="",
    )
    lines_dash_joinstyle = Enum(
        [
            "miter",
            "round",
            "bevel",
            "miter",
            "round",
            "bevel",
        ],
        default_value="round",
        help=r"""miter|round|bevel""",
    ).tag(config=True, kind="")
    lines_dash_capstyle = Enum(
        [
            "butt",
            "round",
            "projecting",
            "butt",
            "round",
            "projecting",
        ],
        default_value="butt",
        help=r"""butt|round|projecting""",
    ).tag(config=True, kind="")
    lines_solid_joinstyle = Enum(
        [
            "miter",
            "round",
            "bevel",
        ],
        default_value="round",
        help=r"""miter|round|bevel""",
    ).tag(config=True, kind="")
    lines_solid_capstyle = Enum(
        [
            "butt",
            "round",
            "projecting",
        ],
        default_value="round",
        help=r"""butt|round|projecting""",
    ).tag(config=True, kind="")
    lines_antialiased = Bool(
        True,
        help=r"""render lines in antialiased (no jaggies)""",
    ).tag(config=True, kind="")
    lines_dashed_pattern = Tuple((6.0, 6.0), help=r"""""").tag(config=True, kind="")
    lines_dashdot_pattern = Tuple((3.0, 5.0, 1.0, 5.0), help=r"""""").tag(
        config=True,
        kind="",
    )
    lines_dotted_pattern = Tuple((1.0, 3.0), help=r"""""").tag(config=True, kind="")
    lines_scale_dashes = Bool(False, help=r"""""").tag(config=True, kind="")
    #
    # Marker props
    #
    markers_fillstyle = Enum(
        "full|left|right|bottom|top|none".split("|"),
        default_value="full",
        help=r"""full|left|right|bottom|top|none""",
    ).tag(config=True, kind="")

    # ----------
    # 2. PATCHES
    # ----------
    # Patches are graphical objects that fill 2D space, like polygons or
    # circles.  See
    # http://matplotlib.org/api/artist_api.html#module-matplotlib.patches
    # information on patch properties
    #
    patch_linewidth = Float(0.3, help=r"""edge width in points.""").tag(
        config=True,
        kind="",
    )
    patch_facecolor = Unicode("4C72B0", help=r"""""").tag(config=True, kind="color")
    patch_edgecolor = Unicode(
        "black",
        help=r"""if forced, or patch is not filled""",
    ).tag(config=True, kind="color")
    patch_force_edgecolor = Bool(False, help=r"""True to always use edgecolor""").tag(
        config=True,
        kind="",
    )
    patch_antialiased = Bool(
        True,
        help=r"""render patches in antialiased (no jaggies)""",
    ).tag(config=True, kind="")
    hatch_color = Unicode("black", help=r"""""").tag(config=True, kind="color")
    hatch_linewidth = Float(1.0, help=r"""""").tag(config=True, kind="")

    # --------
    # 3. FONTS
    # --------
    # font properties used by text.Text.  See
    # http://matplotlib.org/api/font_manager_api.html for more
    # information on font properties.  The 6 font properties used for font
    # matching are given below with their default values.
    #
    # The font.family property has five values: 'serif' (e.g., Times),
    # 'sans-serif' (e.g., Helvetica), 'cursive' (e.g., Zapf-Chancery),
    # 'fantasy' (e.g., Western), and 'monospace' (e.g., Courier).  Each of
    # these font families has a default list of font names in decreasing
    # order of priority associated with them.  When text.usetex is False,
    # font.family may also be one or more concrete font names.
    #
    # The font.style property has three values: normal (or roman), italic
    # or oblique.  The oblique style will be used for italic, if it is not
    # present.
    #
    # The font.variant property has two values: normal or small-caps.  For
    # TrueType fonts, which are scalable fonts, small-caps is equivalent
    # to using a font size of 'smaller', or about 83% of the current font
    # size.
    #
    # The font.weight property has effectively 13 values: normal, bold,
    # bolder, lighter, 100, 200, 300, ..., 900.  Normal is the same as
    # 400, and bold is 700.  bolder and lighter are relative values with
    # respect to the current weight.
    #
    # The font.stretch property has 11 values: ultra-condensed,
    # extra-condensed, condensed, semi-condensed, normal, semi-expanded,
    # expanded, extra-expanded, ultra-expanded, wider, and narrower.  This
    # property is not currently implemented.
    #
    # The font.size property is the default font size for text, given in pts.
    # 12pt is the standard value.

    font_family = Enum(
        ["sans-serif", "serif", "cursive", "monospace", "fantasy"],
        default_value="sans-serif",
        help=r"""sans-serif|serif|cursive|monospace|fantasy""",
    ).tag(config=True, kind="")
    font_style = Enum(
        ["normal", "roman", "italic", "oblique"],
        default_value="normal",
        help=r"""normal (or roman), italic or oblique""",
    ).tag(config=True, kind="")
    font_variant = Enum(
        ["normal", "small-caps"],
        default_value="normal",
        help=r"""""",
    ).tag(config=True, kind="")
    font_weight = Enum(
        [
            100,
            200,
            300,
            "normal",
            400,
            500,
            600,
            "bold",
            700,
            800,
            900,
            "bolder",
            "lighter",
        ],
        default_value="normal",
        help=r"""100|200|300|normal or 400|500|600|bold or 700|800|900|bolder|lighter""",
    ).tag(config=True, kind="")
    font_stretch = Unicode("normal", help=r"""""")  # not implemented
    # note that font.size controls default text sizes.  To configure
    # special text sizes tick labels, axes, labels, title, etc, see the
    # settings for axes and ticks. Special text sizes can be defined
    # relative to font.size, using the following values: xx-small, x-small,
    # small, medium, large, x-large, xx-large, larger, or smaller
    font_size = Float(
        10.0,
        help=r"""The default fontsize. Special text sizes can be defined relative to font.size,
                      using the following values: xx-small, x-small, small, medium, large, x-large, xx-large,
                      larger, or smaller""",
    ).tag(config=True, kind="")

    # --------
    # 4. TEXT
    # --------
    # text properties used by text.Text.  See
    # http://matplotlib.org/api/artist_api.html#module-matplotlib.text for more
    # information on text properties
    #
    text_color = Unicode(".15", help=r"""""").tag(config=True, kind="color")
    #
    # LaTeX customizations. See http://www.scipy.org/Wiki/Cookbook/Matplotlib/UsingTex
    #
    text_usetex = Bool(
        False,
        help=r"""use latex for all text handling. The following fonts
                       are supported through the usual rc parameter settings: new century schoolbook, bookman, times,
                       palatino, zapf chancery, charter, serif, sans-serif, helvetica, avant garde, courier, monospace,
                       computer modern roman, computer modern sans serif, computer modern typewriter.
                       If another font is desired which can loaded using the LaTeX \usepackage command, please inquire
                       at the matplotlib mailing list""",
    ).tag(config=True, kind="")
    latex_preamble = Unicode(
        r"""\usepackage{siunitx}
                            \sisetup{detect-all}
                            \usepackage{times} # set the normal font here
                            \usepackage{sansmath}
                            # load up the sansmath so that math -> helvet
                            \sansmath
                            """,
        help=r"""Latex preamble for matplotlib outputs

                             IMPROPER USE OF THIS FEATURE WILL LEAD TO LATEX FAILURES.
                             preamble is a comma separated
                             list of LaTeX statements that are included in the LaTeX document preamble.
                             An example:
                             text.latex.preamble : \usepackage{bm},\usepackage{euler}
                             The following packages are always loaded with usetex, so beware of package collisions:
                             color, geometry, graphicx, type1cm, textcomp. Adobe Postscript (PSSNFS) font packages
                             may also be loaded, depending on your font settings.""",
    ).tag(config=True, kind="")
    text_hinting = Enum(
        ["none", "auto", "native", "either"],
        default_value="auto",
        help=r"""May be one of the
    following: 'none': Perform no hinting
                         * 'auto': Use freetype's autohinter
                         * 'native': Use the hinting information in the font file, if available, and if your freetype
                            library supports it
                         * 'either': Use the native hinting information or the autohinter if none is available.
                         For backward compatibility, this value may also be True === 'auto' or False ===
                         'none'.""",
    ).tag(config=True)
    text_hinting_factor = Float(
        8,
        help=r"""Specifies the amount of softness for hinting in the horizontal
    direction. A value of 1 will hint to full pixels. A value of 2 will hint to half pixels etc.""",
    ).tag(config=True, kind="")
    text_antialiased = Bool(
        True,
        help=r"""If True (default), the text will be antialiased.
                            This only affects the Agg backend.""",
    ).tag(config=True, kind="")
    # The following settings allow you to select the fonts in math mode.
    # They map from a TeX font name to a fontconfig font pattern.
    # These settings are only used if mathtext.fontset is 'custom'.
    # Note that this "custom" mode is unsupported and may go away in the
    # future.
    mathtext_cal = Unicode("cursive", help=r"""""").tag(config=True, kind="")
    mathtext_rm = Unicode("dejavusans", help=r"""""").tag(config=True, kind="")
    mathtext_tt = Unicode("monospace", help=r"""""").tag(config=True, kind="")
    mathtext_it = Unicode("dejavusans:italic", help=r"""italic""").tag(
        config=True,
        kind="",
    )
    mathtext_bf = Unicode("dejavusans:bold", help=r"""bold""").tag(config=True, kind="")
    mathtext_sf = Unicode(r"sans\-serif", help=r"""""").tag(config=True, kind="")
    mathtext_fontset = Unicode(
        "dejavusans",
        help=r'''Should be "dejavusans" (default),
                               "dejavuserif", "cm" (Computer Modern), "stix", "stixsans" or "custom"''',
    ).tag(config=True, kind="")
    mathtext_fallback_to_cm = Bool(
        False,
        help=r"""When True, use symbols from the Computer Modern fonts when a
    symbol
                                       can not be found in one of the custom math fonts.""",
    ).tag(config=True, kind="")
    mathtext_default = Unicode(
        "regular",
        help=r"""The default font to use for math. Can be any of the LaTeX font
    names, including the special name "regular" for the same font used in regular text.""",
    ).tag(config=True, kind="")

    # -------
    # 5. AXES
    # -------
    # Axes are the area on which data is plotted, including default face and edge color, default tick sizes,
    # default fontsizes for ticklabels, and so on.
    # See http://matplotlib.org/api/axes_api.html#module-matplotlib.axes

    axes_facecolor = Unicode("F0F0F0", help=r"""axes background color""").tag(
        config=True,
        kind="color",
    )
    axes_edgecolor = Unicode("black", help=r"""axes edge color""").tag(
        config=True,
        kind="color",
    )
    axes_linewidth = Float(0.8, help=r"""edge linewidth""").tag(config=True, kind="")
    axes_grid = Bool(False, help=r"""display grid or not""").tag(config=True, kind="")
    axes_grid_which = Unicode("major").tag(config=True, kind="")
    axes_grid_axis = Unicode("both").tag(config=True, kind="")
    axes_titlesize = Float(14.0, help=r"""fontsize of the axes title""").tag(
        config=True,
        kind="",
    )
    axes_titley = Float(1.0, help=r"""at the top, no autopositioning.""").tag(
        config=True,
        kind="",
    )
    axes_titlepad = Float(5.0, help=r"""pad between axes and title in points""").tag(
        config=True,
        kind="",
    )
    axes_titleweight = Unicode("normal", help=r"""font weight for axes title""").tag(
        config=True,
        kind="",
    )
    axes_labelsize = Float(10.0, help=r"""fontsize of the x any y labels""").tag(
        config=True,
        kind="",
    )
    axes_labelpad = Float(4.0, help=r"""space between label and axis""").tag(
        config=True,
        kind="",
    )
    axes_labelweight = Unicode("normal", help=r"""weight of the x and y labels""").tag(
        config=True,
        kind="",
    )
    axes_labelcolor = Unicode("black", help=r"""""").tag(config=True, kind="color")
    axes_axisbelow = Bool(
        True,
        help=r"""whether axis gridlines and ticks are below
                          the axes elements (lines, text, etc)""",
    ).tag(config=True, kind="")
    axes_formatter_limits = Tuple(
        (-5, 6),
        help=r"use scientific notation if log10 of the axis range is smaller than the "
        r"first or larger than the second",
    ).tag(config=True, kind="")
    axes_formatter_use_locale = Bool(
        False,
        help=r"""When True, format tick labels according to the user"s locale.
                                       For example, use "," as a decimal separator in the fr_FR locale.""",
    ).tag(config=True, kind="")
    axes_formatter_use_mathtext = Bool(
        False,
        help=r"""When True, use mathtext for scientific notation.""",
    ).tag(config=True, kind="")
    axes_formatter_useoffset = Bool(
        False,
        help=r"""If True, the tick label formatter will default to labeling ticks
                                    relative to an offset when the data range is small compared to the minimum
                                    absolute value of the data.""",
    ).tag(config=True, kind="")
    axes_formatter_offset_threshold = Integer(
        4,
        help=r"""When useoffset is True, the offset will be used when it can
                                              remove at least this number of significant digits from tick labels.""",
    ).tag(config=True, kind="")
    axes_unicode_minus = Bool(
        True,
        help=r"""use unicode for the minus symbol rather than hyphen. See
                                http://en.wikipedia.org/wiki/Plus_and_minus_signs#Character_codes""",
    ).tag(config=True, kind="")
    axes_prop_cycle = Unicode(
        "cycler('color', ['007200', '009E73', 'D55E00', 'CC79A7', 'F0E442', '56B4E9'])",
        help=r"""color cycle for plot lines as list of string colorspecs: single letter,
                            long name, or web-style hex""",
    ).tag(config=True, kind="function")
    axes_autolimit_mode = Unicode(
        "data",
        help=r"""How to scale axes limits to the data. Use "data" to use data
    limits,
                                    plus some margin. Use "round_number" move to the nearest "round" number""",
    ).tag(config=True, kind="")
    axes_xmargin = Float(0.05, help=r"""x margin. See `axes.Axes.margins`""").tag(
        config=True,
        kind="",
    )
    axes_ymargin = Float(0.05, help=r"""y margin See `axes.Axes.margins`""").tag(
        config=True,
        kind="",
    )
    axes_spines_bottom = Bool(True).tag(config=True, kind="")
    axes_spines_left = Bool(True).tag(config=True, kind="")
    axes_spines_right = Bool(True).tag(config=True, kind="")
    axes_spines_top = Bool(True).tag(config=True, kind="")
    polaraxes_grid = Bool(True, help=r"""display grid on polar axes""").tag(
        config=True,
        kind="",
    )
    axes3d_grid = Bool(True, help=r"""display grid on 3d axes""").tag(
        config=True,
        kind="",
    )
    # -------
    # 6. DATE
    # -------
    timezone = Unicode(
        "UTC",
        help=r"""a IANA timezone string, e.g., US/Central or Europe/Paris""",
    ).tag(config=True, kind="")
    date_autoformatter_year = Unicode("%Y").tag(config=True, kind="")
    date_autoformatter_month = Unicode("%b %Y").tag(config=True, kind="")
    date_autoformatter_day = Unicode("%b %d %Y").tag(config=True, kind="")
    date_autoformatter_hour = Unicode("%H:%M:%S").tag(config=True, kind="")
    date_autoformatter_minute = Unicode("%H:%M:%S.%f").tag(config=True, kind="")
    date_autoformatter_second = Unicode("%H:%M:%S.%f").tag(config=True, kind="")
    date_autoformatter_microsecond = Unicode("%H:%M:%S.%f").tag(config=True, kind="")

    # -------
    # 7 TICKS
    # -------
    # see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
    #
    xtick_top = Bool(False, help=r"""draw ticks on the top side""").tag(
        config=True,
        kind="",
    )
    xtick_bottom = Bool(True, help=r"""draw ticks on the bottom side""").tag(
        config=True,
        kind="",
    )
    xtick_major_size = Float(3.5, help=r"""major tick size in points""").tag(
        config=True,
        kind="",
    )
    xtick_minor_size = Float(2.0, help=r"""minor tick size in points""").tag(
        config=True,
        kind="",
    )
    xtick_major_width = Float(0.8, help=r"""major tick width in points""").tag(
        config=True,
        kind="",
    )
    xtick_minor_width = Float(0.6, help=r"""minor tick width in points""").tag(
        config=True,
        kind="",
    )
    xtick_major_pad = Float(
        3.5,
        help=r"""distance to major tick label in points""",
    ).tag(config=True, kind="")
    xtick_minor_pad = Float(
        3.4,
        help=r"""distance to the minor tick label in points""",
    ).tag(config=True, kind="")
    xtick_color = Unicode(".15", help=r"""color of the tick labels""").tag(
        config=True,
        kind="color",
    )
    xtick_labelsize = Float(10.0, help=r"""fontsize of the tick labels""").tag(
        config=True,
        kind="",
    )
    xtick_direction = Unicode("out", help=r"""direction""").tag(config=True, kind="")
    xtick_minor_visible = Bool(
        False,
        help=r"""visibility of minor ticks on x-axis""",
    ).tag(config=True, kind="")
    xtick_major_top = Bool(True, help=r"""draw x axis top major ticks""").tag(
        config=True,
        kind="",
    )
    xtick_major_bottom = Bool(True, help=r"""draw x axis bottom major ticks""").tag(
        config=True,
        kind="",
    )
    xtick_minor_top = Bool(True, help=r"""draw x axis top minor ticks""").tag(
        config=True,
        kind="",
    )
    xtick_minor_bottom = Bool(True, help=r"""draw x axis bottom minor ticks""").tag(
        config=True,
        kind="",
    )
    ytick_left = Bool(True, help=r"""draw ticks on the left side""").tag(
        config=True,
        kind="",
    )
    ytick_right = Bool(False, help=r"""draw ticks on the right side""").tag(
        config=True,
        kind="",
    )
    ytick_major_size = Float(3.5, help=r"""major tick size in points""").tag(
        config=True,
        kind="",
    )
    ytick_minor_size = Float(2.0, help=r"""minor tick size in points""").tag(
        config=True,
        kind="",
    )
    ytick_major_width = Float(0.8, help=r"""major tick width in points""").tag(
        config=True,
        kind="",
    )
    ytick_minor_width = Float(0.6, help=r"""minor tick width in points""").tag(
        config=True,
        kind="",
    )
    ytick_major_pad = Float(
        3.5,
        help=r"""distance to major tick label in points""",
    ).tag(config=True, kind="")
    ytick_minor_pad = Float(
        3.4,
        help=r"""distance to the minor tick label in points""",
    ).tag(config=True, kind="")
    ytick_color = Unicode(".15", help=r"""color of the tick labels""").tag(
        config=True,
        kind="color",
    )
    ytick_labelsize = Float(10.0, help=r"""fontsize of the tick labels""").tag(
        config=True,
        kind="",
    )
    ytick_direction = Unicode("out", help=r"""direction""").tag(config=True, kind="")
    ytick_minor_visible = Bool(
        False,
        help=r"""visibility of minor ticks on y-axis""",
    ).tag(config=True, kind="")
    ytick_major_left = Bool(True, help=r"""draw y axis left major ticks""").tag(
        config=True,
        kind="",
    )
    ytick_major_right = Bool(True, help=r"""draw y axis right major ticks""").tag(
        config=True,
        kind="",
    )
    ytick_minor_left = Bool(True, help=r"""draw y axis left minor ticks""").tag(
        config=True,
        kind="",
    )
    ytick_minor_right = Bool(True, help=r"""draw y axis right minor ticks""").tag(
        config=True,
        kind="",
    )
    #
    # --------
    # 8. GRIDS
    # --------
    #
    grid_color = Unicode(".85", help=r"""grid color""").tag(config=True, kind="color")
    grid_linestyle = Enum(
        list(LINESTYLE_KEYS),
        default_value="-",
        help=r"""solid""",
    ).tag(config=True, kind="")
    grid_linewidth = Float(0.85, help=r"""in points""").tag(config=True, kind="")
    grid_alpha = Float(1.0, help=r"""transparency, between 0.0 and 1.0""").tag(
        config=True,
        kind="",
    )
    legend_loc = Unicode("best", help=r"""""").tag(config=True, kind="")
    legend_frameon = Bool(
        False,
        help=r"""if True, draw the legend on a background patch""",
    ).tag(config=True, kind="")

    # ---------
    # 9. LEGEND
    # ----------

    legend_framealpha = TraitUnion(
        trait_types=[Float(), Unicode()],
        default_value=0.8,
        help="Legend patch transparency (float or 'None')",
    ).tag(config=True)
    legend_facecolor = Unicode(
        "inherit",
        help=r"""inherit from axes.facecolor; or color spec""",
    ).tag(config=True, kind="color")
    legend_edgecolor = Unicode("0.8", help=r"""background patch boundary color""").tag(
        config=True,
        kind="color",
    )
    legend_fancybox = Bool(
        True,
        help=r"""if True, use a rounded box for the legend background, else a rectangle""",
    ).tag(config=True, kind="")
    legend_shadow = Bool(
        False,
        help=r"""if True, give background a shadow effect""",
    ).tag(config=True, kind="")
    legend_numpoints = Integer(
        1,
        help=r"""the number of marker points in the legend line""",
    ).tag(config=True, kind="")
    legend_scatterpoints = Integer(1, help=r"""number of scatter points""").tag(
        config=True,
        kind="",
    )
    legend_markerscale = Float(
        1.0,
        help=r"""the relative size of legend markers vs. original""",
    ).tag(config=True, kind="")
    legend_fontsize = Float(9.0, help=r"""""").tag(config=True, kind="")
    legend_borderpad = Float(0.4, help=r"""border whitespace""").tag(
        config=True,
        kind="",
    )
    legend_labelspacing = Float(
        0.2,
        help=r"""the vertical space between the legend entries""",
    ).tag(config=True, kind="")
    legend_handlelength = Float(2.0, help=r"""the length of the legend lines""").tag(
        config=True,
        kind="",
    )
    legend_handleheight = Float(0.7, help=r"""the height of the legend handle""").tag(
        config=True,
        kind="",
    )
    legend_handletextpad = Float(
        0.1,
        help=r"""the space between the legend line and legend text""",
    ).tag(config=True, kind="")
    legend_borderaxespad = Float(
        0.5,
        help=r"""the border between the axes and legend edge""",
    ).tag(config=True, kind="")
    legend_columnspacing = Float(0.5, help=r"""column separation""").tag(
        config=True,
        kind="",
    )
    figure_titlesize = Float(
        12.0,
        help=r"""size of the figure title (Figure.suptitle())""",
    ).tag(config=True, kind="")
    figure_titleweight = Unicode("normal", help=r"""weight of the figure title""").tag(
        config=True,
        kind="",
    )
    figure_figsize = Tuple((6, 4), help=r"""figure size in inches""").tag(
        config=True,
        kind="",
    )
    figure_dpi = Float(96.0, help=r"""figure dots per inch""").tag(config=True, kind="")
    figure_facecolor = Unicode(
        "white",
        help=r"""figure facecolor; 0.75 is scalar gray""",
    ).tag(config=True, kind="color")
    figure_edgecolor = Unicode("white", help=r"""figure edgecolor""").tag(
        config=True,
        kind="color",
    )
    figure_autolayout = Bool(
        True,
        help=r"""When True, automatically adjust subplot parameters to make the plot fit the
                             figure""",
    ).tag(config=True, kind="")

    # ----------
    # 10. FIGURE
    # ----------
    # See http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
    #
    figure_max_open_warning = Integer(
        30,
        help=r"""The maximum number of figures to open through the pyplot
    interface before emitting a warning. If less than one this feature is disabled.""",
    ).tag(config=True, kind="")
    figure_subplot_left = Float(
        0.15,
        help=r"""the left side of the subplots of the figure""",
    ).tag(config=True, kind="")
    figure_subplot_right = Float(
        0.95,
        help=r"""the right side of the subplots of the figure""",
    ).tag(config=True, kind="")
    figure_subplot_bottom = Float(
        0.12,
        help=r"""the bottom of the subplots of the figure""",
    ).tag(config=True, kind="")
    figure_subplot_top = Float(
        0.98,
        help=r"""the top of the subplots of the figure""",
    ).tag(config=True, kind="")
    figure_subplot_wspace = Float(
        0.0,
        help=r"""the amount of width reserved for blank space between subplots,
                                  expressed as a fraction of the average axis width""",
    ).tag(config=True, kind="")
    figure_subplot_hspace = Float(
        0.0,
        help=r"""the amount of height reserved for white space between subplots,
                                  expressed as a fraction of the average axis height""",
    ).tag(config=True, kind="")
    figure_frameon = Bool(True, help="Show figure frame").tag(config=True)
    figure_window_position = Tuple(
        Integer(),
        Integer(),
        default_value=None,
        allow_none=True,
        help="Position of figure window (x, y) in screen pixels for TkAgg backend. "
        "None = let window manager decide. Negative values allowed for multi-monitor.",
    ).tag(config=True)
    #
    # IMAGES
    #
    image_interpolation = Unicode(
        "antialiased",
        help=r"""see help(imshow) for options""",
    ).tag(config=True, kind="")

    image_lut = Integer(256, help=r"""the size of the colormap lookup table""").tag(
        config=True,
        kind="",
    )
    image_origin = Unicode("upper", help=r"""lower | upper""").tag(config=True, kind="")
    image_resample = Bool(True, help=r"""""").tag(config=True, kind="")
    image_composite_image = Bool(
        True,
        help=r"""When True, all the images on a set of axes are
     combined into a single composite image before
     saving a figure as a vector graphics file,
     such as a PDF.""",
    ).tag(config=True, kind="")

    # -----------------
    # 11. CONTOUR PLOTS
    # -----------------

    contour_negative_linestyle = Enum(
        ["dashed", "solid"],
        default_value="dashed",
        help=r"""dashed | solid""",
    ).tag(config=True, kind="")
    contour_corner_mask = Enum(
        [True, False, "legacy"],
        default_value=True,
        help=r"""True | False | legacy""",
    ).tag(config=True, kind="")

    # ------------
    # 12. ERRORBAR
    # -------------
    errorbar_capsize = Float(
        1.0,
        help=r"""length of end cap on error bars in pixels""",
    ).tag(config=True, kind="")
    #
    # HIST
    #
    hist_bins = TraitUnion(
        trait_types=[Integer(), Unicode()],
        default_value="auto",
        help=("The default number of histogram bins. May be an integer or 'auto'."),
    ).tag(config=True)

    # -----------
    # 13. SCATTER
    # -----------
    scatter_marker = Enum(
        list(MARKER_KEYS),
        default_value="o",
        help=r"""The default marker type for scatter plots.""",
    ).tag(config=True, kind="")
    #
    # SAVING FIGURES
    #
    savefig_dpi = Unicode("300", help=r'''figure dots per inch or "figure"''').tag(
        config=True,
        kind="",
    )
    savefig_facecolor = Unicode("white", help=r"""figure facecolor when saving""").tag(
        config=True,
        kind="color",
    )
    savefig_edgecolor = Unicode("white", help=r"""figure edgecolor when saving""").tag(
        config=True,
        kind="color",
    )
    savefig_format = Enum(
        ["png", "ps", "pdf", "svg"],
        default_value="png",
        help=r"""png, ps, pdf, svg""",
    ).tag(config=True, kind="")
    savefig_bbox = Enum(
        ["tight", "standard"],
        default_value="standard",
        help=r""""tight" or "standard". "tight" is
    incompatible with pipe-based animation backends but will worked with temporary file based ones:
    e.g. setting animation.writer to ffmpeg will not work, use ffmpeg_file instead""",
    ).tag(config=True, kind="")
    savefig_pad_inches = Float(
        0.1,
        help=r'''Padding to be used when bbox is set to "tight"''',
    ).tag(config=True, kind="")
    savefig_jpeg_quality = Integer(
        95,
        help=r"""when a jpeg is saved, the default quality parameter.""",
    ).tag(config=True, kind="")
    savefig_directory = Unicode(
        "",
        help=r"""default directory in savefig dialog box, leave empty to always use current
                                working directory""",
    ).tag(config=True, kind="")
    savefig_transparent = Bool(
        False,
        help=r"""setting that controls whether figures are saved with a transparent
                               background by default""",
    ).tag(config=True, kind="")

    # -----------------
    # 13. Agg rendering
    # -----------------
    agg_path_chunksize = Integer(
        20000,
        help=r"""0 to disable; values in the range 10000 to 100000 can improve speed
                                 slightly and prevent an Agg rendering failure when plotting very large data sets,
                                 especially if they are very gappy. It may cause minor artifacts, though. A value of
                                 20000 is probably a good starting point.""",
    ).tag(config=True, kind="")
    path_simplify = Bool(
        True,
        help=r"""When True, simplify paths by removing "invisible" points to reduce file size
    and
                         increase rendering speed""",
    ).tag(config=True, kind="")
    path_simplify_threshold = Float(
        0.111111111111,
        help=r"""The threshold of similarity below which vertices will
    be removed in
                                      the simplification process""",
    ).tag(config=True, kind="")
    path_snap = Bool(
        True,
        help=r"""When True, rectilinear axis-aligned paths will be snapped to the nearest pixel
                    when certain criteria are met. When False, paths will never be snapped.""",
    ).tag(config=True, kind="")
    path_sketch = Unicode(
        "None",
        help=r"""May be none, or a 3-tuple of the form (scale, length, randomness). *scale*
    is the amplitude of the wiggle perpendicular to the line (in pixels). *length* is the length of
                          the wiggle along the line (in pixels). *randomness* is the factor by which the length is
                          randomly scaled.""",
    ).tag(config=True, kind="")

    # ==================================================================================
    # NON MATPLOTLIB OPTIONS
    # ==================================================================================
    #
    # The options below do NOT map to matplotlib.rcParams.
    #
    # They control:
    # - which plotting backend to use (matplotlib vs plotly)
    # - which plotting *method* to use for datasets (pen, image, stack, etc.)
    # - SpectroChemPy-specific plot behaviors
    #
    # These values are consumed by SpectroChemPy plotters,
    # not by Matplotlib itself.

    style = TraitUnion(
        trait_types=[Unicode(), List(), Tuple()],
        default_value="scpy",
        help="Matplotlib style(s) to apply",
    ).tag(config=True)

    stylesheets = Unicode(
        help="Directory where to look for local defined matplotlib styles when they are not in the "
        " standard location",
    ).tag(config=True, type="folder")

    use_plotly = Bool(
        False,
        help="Use Plotly instead of MatPlotLib for plotting (mode Matplotlib more suitable for "
        "printing publication ready figures)",
    ).tag(config=True)

    # - 2d
    # ------
    colorbar = Bool(False, help="Show color bar for 2D plots").tag(config=True)
    show_projections = Bool(False, help="Show all projections").tag(config=True)
    show_projection_x = Bool(False, help="Show projection along x").tag(config=True)
    show_projection_y = Bool(False, help="Show projection along y").tag(config=True)

    colormap = Unicode(
        "viridis",
        help="A matplotlib colormap name (case-insensitive)",
    ).tag(config=True)

    @validate("colormap")
    def _validate_colormap(self, proposal):
        # Validate and normalize colormap names early.
        # This avoids hard-to-debug errors deep inside Matplotlib.
        return _canonical_cmap_name(proposal["value"])

    max_lines_in_stack = Integer(
        1000,
        min=1,
        help="Maximum number of lines to plot in stack plots",
    ).tag(config=True)
    # -1d
    # ---_
    # antialias = Bool(True, help='Antialiasing')
    number_of_x_labels = Integer(5, min=3, help="Number of X labels").tag(config=True)
    number_of_y_labels = Integer(5, min=3, help="Number of Y labels").tag(config=True)
    number_of_z_labels = Integer(5, min=3, help="Number of Z labels").tag(config=True)
    number_of_contours = Integer(50, min=10, help="Number of contours").tag(config=True)
    contour_alpha = Float(
        1.00,
        min=0.0,
        max=1.0,
        help="Transparency of the contours",
    ).tag(config=True)
    contour_start = Float(
        0.05,
        min=0.001,
        help="Fraction of the maximum for starting contour levels",
    ).tag(config=True)
    antialiased = Bool(True, help="antialiased option for surface plot").tag(
        config=True,
    )
    rcount = Integer(50, help="rcount (steps in the row mode) for surface plot").tag(
        config=True,
    )
    ccount = Integer(50, help="ccount (steps in the column mode) for surface plot").tag(
        config=True,
    )

    # 3D view defaults
    axes3d_elev = Float(
        30.0,
        help="Default elevation angle (degrees) for 3D plots",
    ).tag(config=True)

    axes3d_azim = Float(
        45.0,
        help="Default azimuth angle (degrees) for 3D plots",
    ).tag(config=True)

    # Baseline region preferences
    baseline_region_color = Unicode(
        "#2ca02c",
        help="Color for baseline fitting region spans",
    ).tag(config=True)

    baseline_region_alpha = Float(
        0.5,
        min=0.0,
        max=1.0,
        help="Transparency for baseline region spans",
    ).tag(config=True)

    # Image aspect preference
    image_equal_aspect = Bool(
        True,
        help="Enforce equal physical aspect ratio for 2D image data when units are compatible",
    ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Lazy import to avoid loading matplotlib at class instantiation time
        # (but this is still called during init, so matplotlib WILL be imported
        # when PlotPreferences is instantiated - which is intentional as it's
        # part of the plotting system initialization)
        from matplotlib import pyplot as plt

        # Build internal maps of rcParams structure:
        # - groups:     e.g. "axes", "lines", "font"
        # - subgroups:  rarely used second-level keys
        # - members:    terminal rcParams names

        # This allows automatic mapping:
        #   lines_linewidth → lines.linewidth

        for key in plt.rcParams:
            lis = key.split(".")
            if len(lis) > 1:
                self._groups.add(lis.pop(0))
            if len(lis) > 1:
                self.subgroups.add(lis.pop(0))
            if len(lis) > 1:
                raise NotImplementedError
            self._members.add(lis[0])

    @property
    def available_styles(self):
        return available_styles()

    @property
    def members(self):
        return self._members

    @property
    def groups(self):
        return self._groups

    @property
    def subgroups(self):
        return self._subgroups

    def set_latex_font(self, family=None):
        """
        Return a dict of rcParams overrides for LaTeX font settings.

        This method does NOT mutate global rcParams. Instead, it returns
        a dictionary that can be used with mpl.rc_context() in plotting calls.

        Parameters
        ----------
        family : str, optional
            Font family to use. If None, uses self.font_family.

        Returns
        -------
        dict
            A dictionary of rcParams overrides for LaTeX font settings.
        """
        if family is None:
            family = self.font_family

        if family == "sans-serif":
            return {
                "text.usetex": False,
                "mathtext.fontset": "dejavusans",
                "mathtext.bf": "dejavusans:bold",
                "mathtext.cal": "cursive",
                "mathtext.default": "regular",
                "mathtext.rm": "dejavusans",
                "mathtext.it": "dejavusans:italic",
            }
        elif family == "serif":
            return {
                "text.usetex": False,
                "mathtext.fontset": "dejavuserif",
                "mathtext.bf": "dejavuserif:bold",
                "mathtext.cal": "cursive",
                "mathtext.default": "regular",
                "mathtext.rm": "dejavuserif",
                "mathtext.it": "dejavuserif:italic",
            }
        elif family == "cursive":
            return {
                "text.usetex": False,
                "mathtext.fontset": "custom",
                "mathtext.bf": "cursive:bold",
                "mathtext.cal": "cursive",
                "mathtext.default": "regular",
                "mathtext.rm": "cursive",
                "mathtext.it": "cursive:italic",
            }
        elif family == "monospace":
            return {
                "text.usetex": False,
                "mathtext.fontset": "custom",
                "mathtext.bf": "monospace:bold",
                "mathtext.cal": "cursive",
                "mathtext.default": "regular",
                "mathtext.rm": "monospace",
                "mathtext.it": "monospace:italic",
            }
        elif family == "fantasy":
            return {
                "text.usetex": False,
                "mathtext.fontset": "custom",
                "mathtext.bf": "Humor Sans:bold",
                "mathtext.cal": "cursive",
                "mathtext.default": "regular",
                "mathtext.rm": "Comic Sans MS",
                "mathtext.it": "Humor Sans:italic",
            }
        return {}

    # @observe("simplify")  # DISABLED: Remove automatic global rcParams mutation
    def _simplify_changed(self, change):
        # DISABLED: This observer updates rcParams immediately.
        # It bypasses the generic _anytrait_changed logic
        # because path.simplify has special semantics.
        # plt.rcParams["path.simplify"] = change.new
        # plt.rcParams["path.simplify_threshold"] = 1.0
        pass

    @default("stylesheets")
    def _get_stylesheets_default(self):
        # the spectra path in package data
        from spectrochempy.utils.packages import get_pkg_path

        value = get_pkg_path("data/stylesheets", "spectrochempy")
        if hasattr(value, "__fspath__"):  # Check if it's a path-like object
            value = str(value)
        return value

    @observe("style")
    def _style_changed(self, change):
        """
        Apply one or more matplotlib styles.

        The style trait may be:
        - a single string
        - a list / tuple of styles (applied in order)

        Each style is processed by _apply_style(),
        which parses mplstyle files and sets trait values.
        """
        changes = change.new
        if not isinstance(changes, list):
            changes = [changes]
        for _style in changes:
            try:
                if isinstance(_style, list | tuple):
                    for s in _style:
                        self._apply_style(s)
                else:
                    self._apply_style(_style)
            except Exception as e:
                raise e
        # Note: LaTeX font settings are now applied locally in plotting contexts
        # via rc_context, not via global rcParams mutation

    @staticmethod
    def _get_fontsize(fontsize):
        # Ensure Matplotlib is initialized (lazy & backend-safe)
        from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

        lazy_ensure_mpl_config()

        from contextlib import suppress

        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        if fontsize == "None":
            return str(matplotlib.rcParams["font.size"])

        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        t = ax.text(0.5, 0.5, "Text")

        with suppress(Exception):
            t.set_fontsize(fontsize)
            return str(round(t.get_fontsize(), 2))

        # 🔒 Fallback: ALWAYS return a non-None value
        try:
            return str(round(float(fontsize), 2))
        except (TypeError, ValueError):
            return str(fontsize)

    @staticmethod
    def _get_color(color):
        from matplotlib import pyplot as plt

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        c = [f"C{i}" for i in range(10)]
        if color in c:
            return f"{colors[c.index(color)]}"
        return f"{color}"

    def _coerce_style_value(self, name, value):
        """Convert mplstyle values to proper Python types, based on the *target trait type*."""

        # Already parsed
        if not isinstance(value, str):
            return value

        raw = value.strip()
        low = raw.lower()

        trait_name = name.replace(".", "_")
        trait = self.traits().get(trait_name)

        # --------------------------------------------------
        # Booleans
        # --------------------------------------------------
        if low == "true":
            return True
        if low == "false":
            return False

        # --------------------------------------------------
        # None handling (trait-aware)
        # --------------------------------------------------
        if low in ("none", "null"):
            # Enum traits always expect strings
            if isinstance(trait, Enum):
                return raw

            # Union or Unicode → keep string "None"
            if isinstance(trait, (TraitUnion, Unicode)):
                return raw

            # Otherwise allow real None
            return None

        # --------------------------------------------------
        # Tuple trait → parse comma-separated values
        # --------------------------------------------------
        if isinstance(trait, Tuple):
            try:
                return tuple(float(p.strip()) for p in raw.split(","))
            except Exception:
                return raw  # let traitlets raise a clean error

        # --------------------------------------------------
        # Numeric parsing (trait-aware)
        # --------------------------------------------------
        try:
            num = float(raw)
        except ValueError:
            return raw  # not a number → keep string

        # ---- Integer trait
        if isinstance(trait, Integer):
            if num.is_integer():
                return int(num)
            return num  # let traitlets complain

        # ---- Union(Int, Unicode)
        if isinstance(trait, TraitUnion):
            return raw

        # ---- Float trait
        if isinstance(trait, Float):
            return num

        # --------------------------------------------------
        # Fallback
        # --------------------------------------------------
        return raw

    def _coerce_for_trait(self, trait, raw, parsed):
        """
        Perform final coercion of style values for traitlets.

        This method exists to resolve the mismatch between:
        - how values are written in .mplstyle files (always strings)
        - how traitlets expect values (typed and validated)

        Design principles
        -----------------
        - `_coerce_style_value()` performs *semantic parsing*
          (numbers, booleans, tuples, "None", etc.)
        - `_coerce_for_trait()` performs *trait normalization*

        This separation is intentional and avoids:
        - double parsing
        - fragile isinstance chains
        - silent traitlets failures

        Parameters
        ----------
        trait : TraitType or None
            The target trait instance (e.g. Integer(), Unicode(), Tuple()).
            May be None if the style key does not map to a trait.
        raw : str
            Original string from the mplstyle file (unmodified).
        parsed : Any
            Output of `_coerce_style_value()`.

        Returns
        -------
        Any
            Value coerced to best match the target trait.
            Final validation is delegated to traitlets.
        """

        # --------------------------------------------------
        # Unicode traits
        # --------------------------------------------------
        # Intent:
        # Matplotlib rcParams frequently use sentinel *strings*
        # such as "None", "auto", "inherit", etc.
        #
        # Even if `_coerce_style_value()` parsed something numeric,
        # Unicode traits must *always* receive the original string.
        #
        # This guarantees:
        # - faithful round-tripping of mplstyle files
        # - compatibility with Matplotlib's string-based semantics
        if trait is not None and isinstance(trait, Unicode):
            return raw.strip()

        # --------------------------------------------------
        # Integer traits
        # --------------------------------------------------
        # Intent:
        # mplstyle files often contain floats for integer-valued rcParams
        # (e.g. "20000.0").
        #
        # If the parsed value is a float that represents an integer,
        # promote it to int explicitly.
        #
        # Otherwise, let traitlets raise a clean validation error.
        if trait is not None and isinstance(trait, Integer):
            if isinstance(parsed, float) and parsed.is_integer():
                return int(parsed)
            if isinstance(parsed, int):
                return parsed
            return parsed  # let traitlets complain

        # --------------------------------------------------
        # Tuple traits
        # --------------------------------------------------
        # Intent:
        # mplstyle syntax commonly encodes tuples as:
        #   "5.5, 3.5"
        # or
        #   "(5.5, 3.5)"
        #
        # This block normalizes both forms into a real Python tuple.
        #
        # If parsing fails, return the parsed value and let traitlets
        # produce an informative error.
        if trait is not None and isinstance(trait, Tuple):
            s = raw.strip()

            # allow either "(5.5, 3.5)" or "5.5, 3.5"
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]

            try:
                parts = [float(p.strip()) for p in s.split(",")]
                return tuple(parts)
            except Exception:
                return parsed  # let traitlets raise cleanly

        # --------------------------------------------------
        # Default fallback
        # --------------------------------------------------
        # Intent:
        # For all other trait types (Float, Bool, Enum, TraitUnion, etc.),
        # return the parsed value and let traitlets handle validation.
        #
        # This keeps coercion logic centralized and predictable.
        return parsed

    def _apply_style(self, _style):
        """
        Apply a single style to PlotPreferences.

        This method:
        - handles logical styles like "default"
        - resolves .mplstyle files
        - parses them safely
        - updates traitlets (NOT rcParams directly)
        """
        # IMPORTANT:
        # _apply_style must handle logical styles (e.g. "default")
        # BEFORE any filesystem access. Matplotlib does not ship
        # a default.mplstyle file. # Attempting to load it from disk will fail

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        from spectrochempy.utils.file import pathclean

        # --------------------------------------------------
        # Logical (non-file) Matplotlib styles
        # --------------------------------------------------
        if _style in (None, "", "none"):
            return

        # NOTE:
        # "default" is a logical Matplotlib style, NOT a .mplstyle file.
        # It must always be handled explicitly via rcdefaults(), including
        # when encountered inside other style sheets (e.g. scpy.mplstyle).
        if _style == "default":
            # plt.rcdefaults()  # DISABLED: No global rcParams mutation
            return

        # --------------------------------------------------
        # File-based styles (.mplstyle)
        # --------------------------------------------------
        stylesheets_path = self.stylesheets
        if hasattr(stylesheets_path, "__fspath__"):
            stylesheets_path = str(stylesheets_path)

        f = (pathclean(stylesheets_path) / _style).with_suffix(".mplstyle")

        if not f.exists():
            f = (
                pathclean(mpl.__file__).parent
                / "mpl-data"
                / "stylelib"
                / f"{_style}.mplstyle"
            )

        if not f.exists():
            raise FileNotFoundError(f"Matplotlib style '{_style}' not found")

        txt = f.read_text()
        pars = txt.split("\n")

        for lineno, line in enumerate(pars, start=1):
            # mplstyle files are line-based key:value pairs.
            # Comments start with '#'.
            # Values are strings and must be converted

            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            if ":" not in line:
                raise ValueError(
                    f"Invalid mplstyle syntax in '{f}', line {lineno}:\n"
                    f"  {line}\n"
                    "Expected format: key: value"
                )

            name, value = line.split(":", maxsplit=1)

            name = name.strip()
            value = value.split(" # ")[0].strip()

            # Font size handling
            if "size" in name and "figsize" not in name and "papersize" not in name:
                value = self._get_fontsize(value)

            # Color normalization
            elif name.endswith("color") and "force_" not in name:
                value = self._get_color(value)

            trait_name = name.replace(".", "_")

            raw_value = value
            parsed = self._coerce_style_value(name, raw_value)

            trait = self.traits().get(trait_name)
            coerced = self._coerce_for_trait(trait, raw_value, parsed)

            setattr(self, trait_name, coerced)

            # SpectroChemPy-only parameters
            if line.strip().startswith("##@"):
                name, value = line[3:].split(":", maxsplit=1)
                name = name.strip()
                value = value.strip()
                value = self._coerce_style_value(name, value)
                setattr(self, name, value)

    def to_rc_key(self, key):
        """
        Convert a trait name back to a matplotlib rcParams key.

        Example:
        -------
                axes_facecolor → axes.facecolor
        """
        rckey = ""
        lis = key.split("_")
        if len(lis) > 1 and lis[0] in self.groups:
            rckey += lis.pop(0)
            rckey += "."
        if len(lis) > 1 and lis[0] in self.subgroups:
            rckey += lis.pop(0)
            rckey += "."
        rckey += "_".join(lis)
        return rckey

    # @observe(All)  # DISABLED: Remove automatic global rcParams mutation

    def _anytrait_changed(self, change):
        """
        Handle trait changes - no longer applies to global matplotlib state.

        Preference changes are now handled locally in plotting contexts,
        not through global matplotlib state mutation.

        Note: LaTeX font settings are now applied via rc_context in plotting
        functions (plot1d.py, plot2d.py), not via trait observers.
        """
        # LaTeX font changes are handled locally in plotting contexts
        # via rc_context, not via global rcParams mutation
        super()._anytrait_changed(change)


# =============================================================================
# Summary for contributors
# =============================================================================
#
# - PlotPreferences is the authoritative source of plotting state
# - Matplotlib rcParams are derived, not authoritative
# - Styles are parsed, not blindly applied
# - Traitlets provide validation, observation, and reversibility
#
# This file is complex because it protects users from subtle,
# global plotting side effects.
