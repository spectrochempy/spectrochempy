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
