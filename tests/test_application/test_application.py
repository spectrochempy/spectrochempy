# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import logging

import spectrochempy as scp
from spectrochempy.application import app


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
    log_out = app.log.handlers[1].stream.getvalue().rstrip()
    assert "ERROR | SpectroChemPyError: an error!" in log_out
    # assert "WARNING | (UserWarning) a warning!" in log_out   # for some
    # reason the WARNING is in the file when executed as a single test but in the suite of test
    # could not find why!
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
