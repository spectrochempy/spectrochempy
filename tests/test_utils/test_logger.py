# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================


from spectrochempy.api import log, WARNING, INFO, preferences

def test_logger():

    log.error('----------')
    log.debug('test log output for debugging')
    log.info('ssssss')
    log.warning('aie aie aie')
    log.error('very bad')

    log.error('----------')
    preferences.log_level = INFO
    log.info('Changed to INFO')
    log.debug('test log output for debugging, after changing level')
    log.info('ssssss, after changing level')
    log.warning('aie aie aieafter changing level')
    log.error('very badafter changing level')

    log.error('----------')
    preferences.log_level = 'DEBUG'
    log.debug('test log output for debugging, after changing level')
    log.info('ssssss, after changing level')
    log.warning('aie aie aieafter changing level')
    log.error('very badafter changing level')
