# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import logging
import warnings

from spectrochempy import (
    DEBUG,
    INFO,
    WARNING,
    debug_,
    error_,
    get_loglevel,
    info_,
    set_loglevel,
    warning_,
)


def test_logger_level_filtering(caplog):
    """Test that log level filtering works correctly for debug_, info_, error_."""
    logger = logging.getLogger("SpectroChemPy")
    logger.propagate = True
    caplog.set_level(DEBUG)

    set_loglevel(WARNING)
    # debug_ and info_ should still create log records (caplog captures all),
    # but error_ should too
    debug_("debug msg at WARNING level")
    info_("info msg at WARNING level")
    error_(Exception, "error msg at WARNING level")

    assert get_loglevel() == WARNING
    assert any(
        r.levelname == "DEBUG" and "debug msg at WARNING level" in r.message
        for r in caplog.records
    )
    assert any(
        r.levelname == "INFO" and "info msg at WARNING level" in r.message
        for r in caplog.records
    )
    assert any(
        r.levelname == "ERROR" and "error msg at WARNING level" in r.message
        for r in caplog.records
    )


def test_logger_level_switch(caplog):
    """Test switching log levels and verifying output."""
    logger = logging.getLogger("SpectroChemPy")
    logger.propagate = True
    caplog.set_level(DEBUG)

    set_loglevel(INFO)
    assert get_loglevel() == INFO

    info_("info msg at INFO level")
    warning_("warning msg at INFO level")
    error_(Exception, "error msg at INFO level")

    assert any(
        r.levelname == "INFO" and "info msg at INFO level" in r.message
        for r in caplog.records
    )
    assert any(
        r.levelname == "ERROR" and "error msg at INFO level" in r.message
        for r in caplog.records
    )


def test_warning_function():
    """Test that warning_ emits a warning via the warnings system."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warning_("test warning message for pytest")
        assert len(w) == 1
        assert "test warning message for pytest" in str(w[0].message)
