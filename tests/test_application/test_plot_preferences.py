# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import json
from pathlib import Path

import pytest

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


def test_reset_restores_all_plot_preferences(tmp_path):
    from spectrochempy.application.application import app

    prefs = scp.preferences
    config_dir = Path(app.general_preferences.cfg.config_dir)
    prefs.style = "paper"
    prefs.colormap_sequential = "jet"
    prefs.colormap_categorical_small = "Set1"
    prefs.number_of_contours = 10

    prefs.reset()

    after_reset = app.plot_preferences.trait_values(config=True)
    assert not (config_dir / "PlotPreferences.json").exists()
    assert after_reset == _new_app(tmp_path).plot_preferences.trait_values(config=True)
    assert after_reset == _new_app(config_dir).plot_preferences.trait_values(
        config=True
    )


def test_first_plot_keeps_saved_plot_preferences(tmp_path, monkeypatch):
    from spectrochempy.plotting.profile import PlotProfileManager

    prefs = _new_app(tmp_path).plot_preferences
    prefs.colormap = "magma"

    manager = PlotProfileManager()
    monkeypatch.setattr(manager, "_config_dir", tmp_path)
    manager.ensure_initialized()

    assert (tmp_path / "PlotPreferences.json").exists()
    assert _new_app(tmp_path).plot_preferences.colormap == "magma"


def test_plot_preferences_reload_after_several_styles(tmp_path):
    prefs = _new_app(tmp_path).plot_preferences
    prefs.style = "ggplot"
    prefs.axes_facecolor = "yellow"
    prefs.style = ["paper", "talk"]
    expected = prefs.trait_values(config=True)

    prefs = _new_app(tmp_path).plot_preferences
    assert prefs.trait_values(config=True) == expected


def test_style_values_and_style_name_are_saved_in_two_writes(tmp_path, monkeypatch):
    prefs = _new_app(tmp_path).plot_preferences
    writes = []
    update = prefs.cfg.update
    monkeypatch.setattr(
        prefs.cfg, "update", lambda *args: writes.append(args) or update(*args)
    )

    prefs.style = "ggplot"

    assert len(writes) == 2


def test_failed_style_change_leaves_preferences_unchanged(tmp_path):
    prefs = _new_app(tmp_path).plot_preferences
    prefs.colormap = "magma"
    before = prefs.trait_values(config=True)
    saved = (tmp_path / "PlotPreferences.json").read_text()

    with pytest.raises(Exception):
        prefs.style = ["ggplot", "__not_a_style__"]

    assert prefs.trait_values(config=True) == before
    assert (tmp_path / "PlotPreferences.json").read_text() == saved
    assert _new_app(tmp_path).plot_preferences.trait_values(config=True) == before


def test_invalid_value_in_style_is_skipped_with_a_warning(tmp_path):
    prefs = _new_app(tmp_path).plot_preferences
    prefs.stylesheets = str(tmp_path)
    (tmp_path / "partly_invalid.mplstyle").write_text(
        "axes.titley : None\nlines.linewidth : 3\n"
    )

    with pytest.warns(UserWarning, match="axes.titley"):
        prefs.style = "partly_invalid"

    assert prefs.style == "partly_invalid"
    assert prefs.lines_linewidth == 3.0
