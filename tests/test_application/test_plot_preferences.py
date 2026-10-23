# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import json
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


def _new_app(config_dir):
    import logging

    from spectrochempy.application.application import SpectroChemPy

    app = SpectroChemPy(config_dir=config_dir, log_level=logging.WARNING)
    app._init_all_preferences()
    return app


def test_plot_preferences_are_saved_across_sessions(tmp_path):
    prefs = _new_app(tmp_path).plot_preferences
    prefs.colormap = "magma"
    prefs.lines_linewidth = 3.0

    saved = json.loads((tmp_path / "PlotPreferences.json").read_text())
    assert saved["PlotPreferences"]["colormap"] == "magma"
    assert saved["PlotPreferences"]["lines_linewidth"] == 3.0

    prefs = _new_app(tmp_path).plot_preferences
    assert prefs.colormap == "magma"
    assert prefs.lines_linewidth == 3.0


def test_plot_preferences_set_after_style_survive_reload(tmp_path):
    prefs = _new_app(tmp_path).plot_preferences
    prefs.style = "paper"
    prefs.lines_linewidth = 3.0
    prefs.font_size = 20.0

    prefs = _new_app(tmp_path).plot_preferences
    assert prefs.style == "paper"
    assert prefs.lines_linewidth == 3.0
    assert prefs.font_size == 20.0


def test_reset_restores_all_plot_preferences():
    from spectrochempy.application.application import app

    prefs = scp.preferences
    config_dir = Path(app.general_preferences.cfg.config_dir)
    prefs.colormap_sequential = "jet"
    prefs.colormap_categorical_small = "Set1"
    prefs.number_of_contours = 10

    prefs.reset()

    plot_prefs = app.plot_preferences
    for name in (
        "colormap_sequential",
        "colormap_categorical_small",
        "number_of_contours",
    ):
        assert getattr(prefs, name) == plot_prefs.traits()[name].default_value
    assert prefs.style == "scpy"
    assert not (config_dir / "PlotPreferences.json").exists()


def test_first_plot_keeps_saved_plot_preferences(tmp_path, monkeypatch):
    from spectrochempy.plotting.profile import PlotProfileManager

    prefs = _new_app(tmp_path).plot_preferences
    prefs.colormap = "magma"

    manager = PlotProfileManager()
    monkeypatch.setattr(manager, "_config_dir", tmp_path)
    manager.ensure_initialized()

    assert (tmp_path / "PlotPreferences.json").exists()
    assert _new_app(tmp_path).plot_preferences.colormap == "magma"
