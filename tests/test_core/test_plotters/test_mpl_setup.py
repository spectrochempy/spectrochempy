# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import sys


def test_lazy_import(monkeypatch):
    sys.modules.pop("matplotlib", None)

    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

    ensure_mpl_setup()

    assert "matplotlib" in sys.modules


def test_setup_sets_initialized_flag():
    from spectrochempy.core.plotters import _mpl_setup

    _mpl_setup._MPL_INITIALIZED = False

    _mpl_setup.ensure_mpl_setup()

    assert _mpl_setup._MPL_INITIALIZED is True


def test_idempotent():
    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

    ensure_mpl_setup()
    ensure_mpl_setup()


def test_disable_via_env(monkeypatch):
    monkeypatch.setenv("SCPY_DISABLE_MPL_SETUP", "1")

    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

    ensure_mpl_setup()
