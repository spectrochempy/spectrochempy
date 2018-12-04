# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================


from spectrochempy import log, WARNING, INFO, set_loglevel

def test_logger():

    log.error('----------')
    log.debug('test log output for debugging')
    log.info('ssssss')
    log.warning('aie aie aie')
    log.error('very bad')

    log.error('----------')
    set_loglevel(INFO)
    log.info('Changed to INFO')
    log.debug('test log output for debugging, after changing level')
    log.info('ssssss, after changing level')
    log.warning('aie aie aieafter changing level')
    log.error('very badafter changing level')

    log.error('----------')
    set_loglevel('DEBUG')
    log.debug('test log output for debugging, after changing level')
    log.info('ssssss, after changing level')
    log.warning('aie aie aieafter changing level')
    log.error('very badafter changing level')
