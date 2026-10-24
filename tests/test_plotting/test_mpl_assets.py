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

import hashlib
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


def test_obsolete_stylesheet_removed_from_user_stylelib_only(
    fake_mpl_dirs, monkeypatch
):
    from spectrochempy.plotting import _mpl_assets

    configdir, datadir, _ = fake_mpl_dirs
    shipped = b"axes3d.depthshade : True\n"
    monkeypatch.setattr(
        _mpl_assets,
        "_OBSOLETE_STYLESHEETS",
        {"mycustom.mplstyle": hashlib.sha256(shipped).hexdigest()},
    )

    user_copy = configdir / "stylelib" / "mycustom.mplstyle"
    system_copy = datadir / "stylelib" / "mycustom.mplstyle"
    for path in (user_copy, system_copy):
        path.parent.mkdir(parents=True)
        path.write_bytes(shipped)

    _mpl_assets.ensure_mpl_assets_installed()

    assert not user_copy.exists()
    assert system_copy.read_bytes() == shipped


def test_modified_obsolete_stylesheet_kept(fake_mpl_dirs, monkeypatch):
    from spectrochempy.plotting import _mpl_assets

    configdir, _, _ = fake_mpl_dirs
    monkeypatch.setattr(
        _mpl_assets,
        "_OBSOLETE_STYLESHEETS",
        {"mycustom.mplstyle": hashlib.sha256(b"shipped\n").hexdigest()},
    )

    user_copy = configdir / "stylelib" / "mycustom.mplstyle"
    user_copy.parent.mkdir(parents=True)
    user_copy.write_bytes(b"lines.linewidth : 2\n")

    _mpl_assets.ensure_mpl_assets_installed()

    assert user_copy.read_bytes() == b"lines.linewidth : 2\n"


def test_no_obsolete_stylesheet_shipped():
    from spectrochempy.plotting import _mpl_assets

    stylesheets_src = Path(_mpl_assets.__file__).parent / "stylesheets"
    for name in _mpl_assets._OBSOLETE_STYLESHEETS:
        assert not (stylesheets_src / name).exists()
