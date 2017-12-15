# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B) 
# See full LICENSE agreement in the root directory
# =============================================================================




from spectrochempy.application import app
plotoptions = app.plotoptions
log = app.log
options = app


from logging import WARNING

def test_logger():

    log.debug('test log output for debugging')
    log.info('ssssss')
    log.warning('aie aie aie')
    log.error('very bad')

    log.setLevel(WARNING)

    log.debug('test log output for debugging, after changing level')
    log.info('ssssssafter changing level')
    log.warning('aie aie aieafter changing level')
    log.error('very badafter changing level')