# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import logging

from spectrochempy import (
    DEBUG,
    INFO,
    WARNING,
    debug_,
    error_,
    info_,
    set_loglevel,
    warning_,
)


def test_logger(caplog):
    logger = logging.getLogger("SpectroChemPy")
    logger.propagate = True
    caplog.set_level(DEBUG)

    # We can set the level using strings

    set_loglevel(WARNING)

    error_(Exception, "\n" + "*" * 80 + "\n")
    debug_("debug in WARNING level - should not appear")
    info_("info in WARNING level - should not appear")
    warning_("OK this is a Warning")
    error_(IndexError, "OK This is an Error")

    error_(NameError, "\n" + "*" * 80 + "\n")

    set_loglevel(INFO)

    debug_("debug in INFO level - should not appear on stdout")
    info_("OK - info in INFO level")
    warning_("OK this is a Warning")
    error_(Exception, "OK This is an Error")

    error_(Exception, "\n" + "*" * 80 + "\n")
