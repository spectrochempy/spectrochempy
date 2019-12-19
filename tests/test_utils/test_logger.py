# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import logging
from spectrochempy import WARNING, INFO, error_, debug_, info_, warning_, set_loglevel

def test_logger(caplog):

    logger = logging.getLogger('SpectroChemPy')
    logger.propagate = True
    caplog.set_level(logging.DEBUG)

    # We can set the level using strings
    set_loglevel("DEBUG")
    assert logger.level == logging.DEBUG

    set_loglevel(WARNING)
    assert logger.level == logging.WARNING

    error_('\n'+'*' * 80+'\n')
    debug_('debug in WARNING level - should not appear')
    info_('info in WARNING level - should not appear')
    warning_('OK this is a Warning')
    error_('OK This is an Error')

    error_('\n' + '*' * 80 + '\n')

    set_loglevel(INFO)
    assert logger.level == logging.INFO

    debug_('debug in INFO level - should not appear')
    info_('OK - info in INFO level')
    warning_('OK this is a Warning')
    error_('OK This is an Error')

    error_('\n' + '*' * 80 + '\n')

    set_loglevel('DEBUG')
    assert logger.level == logging.DEBUG

    debug_('OK - debug in DEBUG level')
    info_('OK - info in DEBUG level')
    assert caplog.records[-1].levelname == 'INFO'
    assert caplog.records[-1].message == 'OK - info in DEBUG level'
    warning_('OK this is a Warning')
    assert caplog.records[-1].levelname == 'WARNING'
    assert caplog.records[-1].message == 'OK this is a Warning'
    error_('OK This is an Error')
    assert caplog.records[-1].levelname == 'ERROR'
    assert caplog.records[-1].message == 'OK This is an Error'

