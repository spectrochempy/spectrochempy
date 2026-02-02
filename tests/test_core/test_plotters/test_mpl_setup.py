# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================


def test_ensure_mpl_setup_is_safe():
    """
    ensure_mpl_setup must not raise when matplotlib is importable
    in a normal Python environment.
    """
    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

    # Must not raise
    ensure_mpl_setup()


def test_idempotent():
    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

    ensure_mpl_setup()
    ensure_mpl_setup()


def test_disable_via_env(monkeypatch):
    monkeypatch.setenv("SCPY_DISABLE_MPL_SETUP", "1")

    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

    ensure_mpl_setup()
