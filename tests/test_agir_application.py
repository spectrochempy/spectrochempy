# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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



from spectrochempy.api import *

import os
import pytest

@pytest.fixture(scope="module")
def samples():
    _samples = {'P350':{'label':'$\mathrm{M_P}\,(623\,K)$'},
               'A350':{'label':'$\mathrm{M_A}\,(623\,K)$'},
               'B350':{'label':'$\mathrm{M_B}\,(623\,K)$'}}

    for key, sample in _samples.items():
        # our data are in our test `scpdata` directory.
        basename = os.path.join(scpdata,'agirdata/{}/FTIR/FTIR'.format(key))
        if os.path.exists(basename+'.scp'):
            #check if the scp file have already been saved
            filename = basename + '.scp'
            sample['IR'] = NDDataset.read( filename)
        else:
            # else read the original zip file
            filename = basename + '.zip'
            sample['IR'] = NDDataset.read_zip( filename, origin='omnic_export')
            # save
            sample['IR'].save(basename + '.scp')

    for key, sample in _samples.items():
        basename = os.path.join(scpdata, 'agirdata/{}/TGA/tg'.format(key))
        if os.path.exists(basename + '.scp'):
            # check if the scp file have already been saved
            filename = basename + '.scp'
            sample['TGA'] = NDDataset.read(filename)
        else:
            # else read the original csv file
            filename = basename + '.csv'
            ss = sample['TGA'] = NDDataset.read_csv(filename)
            # lets keep only data from something close to 0.
            s = sample['TGA'] = ss[-0.5:35.0]
            # for TGA, some information are missing.
            # we add them here
            s.x.units = 'hour'
            s.units = 'weight_percent'
            s.x.title = 'Time-on-stream'
            s.title = 'Mass change'
            # save
            sample['TGA'].save(basename + '.scp')

    return _samples


def test_slicing_agir(samples):


    # We will resize the data in the interesting region of wavenumbers

    for key in samples.keys():
        s = samples[key]['IR']

        # reduce to a useful windoww of wavenumbers
        W = (1290., 3990.)
        s = s[:, W[0]:W[1]]

        samples[key]['IR'] = s

    #options.log_level = DEBUG



