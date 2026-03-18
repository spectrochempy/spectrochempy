# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy import Script, info_, INFO, WARNING, set_loglevel


def test_script():
    Script("name", "print(2)")

    try:
        Script("0name", "print(3)")
    except Exception:
        info_("name not valid")


def test_script_with_info_symbols():
    script = Script(
        "test_info", "set_loglevel(INFO)\ninfo_('INFO level set: %s' % INFO)"
    )
    script.execute()
    script.execute(locals())
