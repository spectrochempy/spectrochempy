# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy import Script, info_


def test_script():
    Script("name", "print(2)")

    try:
        Script("0name", "print(3)")
    except Exception:
        info_("name not valid")
