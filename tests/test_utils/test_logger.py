# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
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