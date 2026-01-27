# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from pathlib import Path

import pytest


@pytest.mark.usefixtures("fake_mpl_dirs")
def test_stylesheets_installed(monkeypatch, tmp_path):
    from spectrochempy.core.plotters import _mpl_assets

    # Fake stylesheets
    styles_dir = Path(_mpl_assets.__file__).parent / "stylesheets"
    styles_dir.mkdir(exist_ok=True)

    for name in ("scpy", "sans", "paper"):
        (styles_dir / f"{name}.mplstyle").write_text("axes.titlesize: 10")

    _mpl_assets.ensure_mpl_assets_installed()

    import matplotlib as mpl

    user_stylelib = Path(mpl.get_configdir()) / "stylelib"
    system_stylelib = Path(mpl.get_data_path()) / "stylelib"

    for name in ("scpy", "sans", "paper"):
        assert (user_stylelib / f"{name}.mplstyle").exists()
        assert (system_stylelib / f"{name}.mplstyle").exists()


@pytest.mark.usefixtures("fake_mpl_dirs")
def test_stylesheets_idempotent(caplog):
    from spectrochempy.core.plotters._mpl_assets import ensure_mpl_assets_installed

    ensure_mpl_assets_installed()
    ensure_mpl_assets_installed()


@pytest.mark.usefixtures("fake_mpl_dirs")
def test_fonts_install_and_cache_cleanup(monkeypatch, tmp_path):
    from spectrochempy.core.plotters import _mpl_assets

    fonts_dir = Path(_mpl_assets.__file__).parent / "fonts"
    fonts_dir.mkdir(exist_ok=True)

    (fonts_dir / "testfont.ttf").write_bytes(b"fakefont")

    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / "font.cache"
    cache_file.write_text("old cache")

    _mpl_assets.ensure_mpl_assets_installed()

    assert not cache_file.exists()
