# -*- coding: utf-8 -*-
# flake8: noqa


import logging

from spectrochempy import (
    DEBUG,
    WARNING,
    INFO,
    error_,
    debug_,
    info_,
    warning_,
    set_loglevel,
)


def test_logger(caplog):
    logger = logging.getLogger("SpectroChemPy")
    logger.propagate = True
    caplog.set_level(DEBUG)

    # We can set the level using strings
    set_loglevel("DEBUG")
    assert logger.handlers[0].level == DEBUG
    assert logger.handlers[1].level == DEBUG

    set_loglevel(WARNING)
    assert logger.handlers[0].level == WARNING
    assert logger.handlers[1].level == DEBUG

    error_("\n" + "*" * 80 + "\n")
    debug_("debug in WARNING level - should not appear")
    info_("info in WARNING level - should not appear")
    warning_("OK this is a Warning")
    error_("OK This is an Error")

    error_("\n" + "*" * 80 + "\n")

    set_loglevel(INFO)
    assert logger.handlers[0].level == INFO
    assert logger.handlers[1].level == DEBUG

    debug_("debug in INFO level - should not appear on stdout")
    info_("OK - info in INFO level")
    warning_("OK this is a Warning")
    error_("OK This is an Error")

    error_("\n" + "*" * 80 + "\n")

    debug_("OK - debug in DEBUG level")
    info_("OK - info in DEBUG level")
    assert caplog.records[-1].levelname == "INFO"
    assert caplog.records[-1].message.endswith("OK - info in DEBUG level")
    warning_("OK this is a Warning")
    assert caplog.records[-1].levelname == "WARNING"
    assert caplog.records[-1].message.endswith("OK this is a Warning")
    error_("OK This is an Error")
    assert caplog.records[-1].levelname == "ERROR"
    assert caplog.records[-1].message.endswith("OK This is an Error")
