# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import os
from spectrochempy import *

import os

try:
    print('env', os.environ['CONDA_DEFAULT_ENV'])
except:
    pass
    #debug_('no conda env')
import pytest
from spectrochempy import general_preferences as prefs


# TODO: to revise with project!
@pytest.fixture(scope="module")
def test_samples():
    def _make_samples(force_original=False):
        _samples = {'P350': {'label': '$\mathrm{M_P}\,(623\,K)$'},
                    #'A350': {'label': '$\mathrm{M_A}\,(623\,K)$'},
                    #'B350': {'label': '$\mathrm{M_B}\,(623\,K)$'}
                    }

        for key, sample in _samples.items():
            # our data are in our test `datadir` directory.
            basename = os.path.join(prefs.datadir, f'agirdata/{key}/FTIR/FTIR')
            if os.path.exists(basename + '.scp') and not force_original:
                # check if the scp file have already been saved
                filename = basename + '.scp'
                sample['IR'] = NDDataset.read(filename)
            else:
                # else read the original zip file
                filename = basename + '.zip'
                sample['IR'] = NDDataset.read_zip(filename, origin='omnic_export')
                # save
                sample['IR'].save(basename + '.scp')

        for key, sample in _samples.items():
            basename = os.path.join(prefs.datadir, f'agirdata/{key}/TGA/tg')
            if os.path.exists(basename + '.scp') and not force_original:
                # check if the scp file have already been saved
                filename = basename + '.scp'
                sample['TGA'] = NDDataset.read(filename)
            else:
                # else read the original csv file
                filename = basename + '.csv'
                ss = sample['TGA'] = NDDataset.read_csv(filename, origin='tga_export')
                # lets keep only data from something close to 0.
                s = sample['TGA'] = ss[-0.5:35.0]
                # save
                sample['TGA'].save(basename + '.scp')

        return _samples

    return _make_samples


def test_slicing_agir(test_samples):
    samples = test_samples(force_original=True)

    # We will resize the data in the interesting region of wavenumbers

    for key in samples.keys():
        s = samples[key]['IR']

        # reduce to a useful windoww of wavenumbers
        W = (1290., 3990.)
        s = s[:, W[0]:W[1]]

        samples[key]['IR'] = s

    # set_loglevel(DEBUG)
