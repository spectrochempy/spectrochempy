# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path

import spectrochempy as scp


def test_plotpreferences(IR_dataset_2D):
    dataset = IR_dataset_2D
    prefs = scp.preferences  # we will use prefs instead of dataset.preference

    # Save original style to restore after test
    original_style = prefs.style
    try:
        prefs.figure.figsize = (6, 3)  # The default figsize is (6.8,4.4)
        prefs.colorbar = True  # This add a color bar on a side
        prefs.colormap = "magma"  # The default colormap is viridis
        prefs.axes.facecolor = (
            ".95"  # Make the graph background colored in a light gray
        )
        prefs.axes.grid = True
        prefs.reset()
        prefs.style = "grayscale"
        prefs.style = "ggplot"
        prefs.reset()
        prefs.style = "grayscale", "paper"  # This intentionally creates a tuple

        prefs.colormap = "magma"

        prefs.available_styles

        prefs.makestyle("scpy")
        prefs.makestyle()

        # %% [markdown]
        # **Example:**
        #

        # %%
        prefs.reset()
        prefs.colorbar = True
        prefs.colormap = "jet"
        prefs.font.family = "monospace"
        prefs.font.size = 14
        prefs.axes.labelcolor = "blue"
        prefs.axes.grid = True
        prefs.axes.grid_axis = "x"
    finally:
        # Always restore original style to prevent pollution
        prefs.style = original_style


def test_makestyle_skips_deprecated_matplotlib_rcparams(monkeypatch, tmp_path):
    prefs = scp.preferences
    monkeypatch.setattr(prefs, "stylesheets", str(tmp_path))

    stylename = prefs.makestyle("tmpstyle")

    assert stylename == "tmpstyle"

    stylesheet = Path(tmp_path) / "tmpstyle.mplstyle"
    text = stylesheet.read_text()

    assert "text.hinting_factor" not in text
    assert "text.kerning_factor" not in text


def test_makestyle_to_mpl_registers_style(monkeypatch, tmp_path, fake_mpl_dirs):
    from matplotlib import pyplot as plt

    prefs = scp.preferences
    monkeypatch.setattr(prefs, "stylesheets", str(tmp_path))
    configdir, _, _ = fake_mpl_dirs
    reloads = []
    monkeypatch.setattr(plt.style, "reload_library", lambda: reloads.append(True))

    prefs.makestyle("tmpstyle_mpl", to_mpl=True)

    stylesheet = configdir / "stylelib" / "tmpstyle_mpl.mplstyle"
    assert (
        stylesheet.read_text() == (Path(tmp_path) / "tmpstyle_mpl.mplstyle").read_text()
    )
    assert reloads


def test_makestyle_to_mpl_style_usable_by_name(monkeypatch, tmp_path, fake_mpl_dirs):
    import matplotlib as mpl
    import matplotlib.style
    from matplotlib import pyplot as plt

    prefs = scp.preferences
    monkeypatch.setattr(prefs, "stylesheets", str(tmp_path))
    configdir, _, _ = fake_mpl_dirs
    # Matplotlib >= 3.11 keeps the user library paths in matplotlib.style,
    # older versions in matplotlib.style.core.
    style_module = (
        matplotlib.style
        if hasattr(matplotlib.style, "USER_LIBRARY_PATHS")
        else matplotlib.style.core
    )
    monkeypatch.setattr(
        style_module, "USER_LIBRARY_PATHS", [str(configdir / "stylelib")]
    )

    try:
        with mpl.rc_context({"lines.linewidth": 7.5}):
            prefs.makestyle("tmpstyle_usable", to_mpl=True)

        assert "tmpstyle_usable" in plt.style.available
        with plt.style.context("tmpstyle_usable"):
            assert mpl.rcParams["lines.linewidth"] == 7.5
    finally:
        monkeypatch.undo()
        plt.style.reload_library()
