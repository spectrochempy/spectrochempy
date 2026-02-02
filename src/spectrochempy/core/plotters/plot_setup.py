# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================


def ensure_spectrochempy_plot_style():
    """
    Apply the SpectroChemPy Matplotlib plotting style.

    This is a HIGH-LEVEL function and must be called only after
    the SpectroChemPy application and preferences are initialized.

    Responsibilities:
    - Ensure Matplotlib is initialized (via ensure_mpl_setup)
    - Save the user's original rcParams (once)
    - Install SpectroChemPy Matplotlib assets (best-effort)
    - Apply the SpectroChemPy plotting style
    """

    # ------------------------------------------------------------------
    # Low-level Matplotlib init (safe, idempotent)
    # ------------------------------------------------------------------
    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

    ensure_mpl_setup()

    # ------------------------------------------------------------------
    # High-level imports (application is READY here)
    # ------------------------------------------------------------------
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from spectrochempy.application.application import debug_
    from spectrochempy.application.preferences import preferences
    from spectrochempy.core.plotters._mpl_assets import ensure_mpl_assets_installed

    # ------------------------------------------------------------------
    # Save user rcParams ONCE
    # ------------------------------------------------------------------
    if not hasattr(ensure_spectrochempy_plot_style, "_user_rcparams"):
        ensure_spectrochempy_plot_style._user_rcparams = mpl.rcParams.copy()
        debug_("SpectroChemPy: user rcParams saved")

    # ------------------------------------------------------------------
    # Install Matplotlib assets (stylesheets, fonts) – best effort
    # ------------------------------------------------------------------
    try:
        ensure_mpl_assets_installed()
        debug_("SpectroChemPy: matplotlib assets ensured")
    except Exception as exc:
        # Never fail plotting because of assets
        debug_(
            "SpectroChemPy: matplotlib asset installation failed "
            f"({exc.__class__.__name__}: {exc})"
        )

    # ------------------------------------------------------------------
    # Apply SpectroChemPy style (preferences may not be ready)
    # ------------------------------------------------------------------
    style = None
    try:
        plot_prefs = preferences.get("plot", None)
        if plot_prefs is not None:
            style = getattr(plot_prefs, "style", None)
    except Exception:
        style = None

    if style:
        try:
            plt.style.use(style)
            debug_(f"SpectroChemPy: matplotlib style '{style}' applied")
        except Exception as exc:
            debug_(
                f"SpectroChemPy: failed to apply matplotlib style '{style}' "
                f"({exc.__class__.__name__}: {exc})"
            )


def restore_user_rcparams():
    import matplotlib as mpl

    if hasattr(ensure_spectrochempy_plot_style, "_user_rcparams"):
        mpl.rcParams.update(ensure_spectrochempy_plot_style._user_rcparams)
