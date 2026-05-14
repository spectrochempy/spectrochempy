# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Tests for matplotlib assets installation (stylesheets only).

Note: Font installation has been removed. SpectroChemPy now relies on
Matplotlib's built-in fonts.
"""

from pathlib import Path

import pytest


@pytest.mark.usefixtures("fake_mpl_dirs")
def test_stylesheets_installed(monkeypatch, tmp_path):
    from spectrochempy.plotting import _mpl_assets

    stylesheets_src = Path(_mpl_assets.__file__).parent / "stylesheets"

    if not stylesheets_src.exists():
        pytest.skip("No stylesheets directory in plotting module")

    _mpl_assets.ensure_mpl_assets_installed()

    import matplotlib as mpl

    user_stylelib = Path(mpl.get_configdir()) / "stylelib"
    system_stylelib = Path(mpl.get_data_path()) / "stylelib"

    style_files = list(stylesheets_src.glob("*.mplstyle"))
    for src in style_files:
        assert (user_stylelib / src.name).exists() or (
            system_stylelib / src.name
        ).exists(), f"Stylesheet {src.name} not installed"


@pytest.mark.usefixtures("fake_mpl_dirs")
def test_stylesheets_idempotent(caplog):
    from spectrochempy.plotting._mpl_assets import ensure_mpl_assets_installed

    ensure_mpl_assets_installed()
    ensure_mpl_assets_installed()
