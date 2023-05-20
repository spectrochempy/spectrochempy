# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import logging

import spectrochempy as scp


def test_datadir():
    # test print a listing of the testdata directory
    from spectrochempy.application import DataDir

    print(DataDir().listing())
    # or simply
    print(DataDir())
    assert str(DataDir()).startswith("testdata")


def test_version():
    # test version
    assert len(scp.version.split(".")) >= 3


def test_log():
    # test log
    scp.set_loglevel("WARNING")
    scp.error_("an error!")
    scp.warning_("a warning!")
    scp.info_("an info!")
    scp.debug_("a DEBUG info!")

    assert scp.get_loglevel() == logging.WARNING
    # error and warning should be written in the handler[1]
    log_out = scp.app.log.handlers[1].stream.getvalue().rstrip()
    assert "ERROR | SpectroChemPyError: an error!" in log_out
    assert "WARNING | (UserWarning) a warning!" in log_out
    #  but also info as handler[1]  is always at level INFO.
    assert "an info!" in log_out
    assert (
        "DEBUG | a DEBUG info!" not in log_out
    ), " handler[1] is always at level INFO."

    scp.set_loglevel(logging.DEBUG)
    scp.debug_("a second DEBUG info!")

    assert scp.get_loglevel() == logging.DEBUG
    assert (
        "DEBUG | a second DEBUG info!" not in log_out
    ), " handler[1] is always at level INFO."


def test_magic_addscript(ip):
    if ip is None:
        scp.warning_("ip is None - pss this test ")
        return

    from spectrochempy.application import SpectroChemPyMagics

    ip.register_magics(SpectroChemPyMagics)

    ip.run_cell("from spectrochempy import *")

    assert "preferences" in ip.user_ns.keys()

    ip.run_cell("print(preferences.available_styles)", store_history=True)
    ip.run_cell("project = Project()", store_history=True)
    x = ip.run_line_magic("addscript", "-p project -o prefs -n preferences 2")

    print("x", x)
    assert x.strip() == "Script prefs created."

    # with cell contents
    x = ip.run_cell(
        "%%addscript -p project -o essai -n preferences\n"
        "print(preferences.available_styles)"
    )

    print("result\n", x.result)
    assert x.result.strip() == "Script essai created."
