# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

from spectrochempy.api import *

import os
import pytest

@pytest.fixture(scope="module")
def samples():
    _samples = {'P350':{'label':'$\mathrm{M_P}\,(623\,K)$'},
               'A350':{'label':'$\mathrm{M_A}\,(623\,K)$'},
               'B350':{'label':'$\mathrm{M_B}\,(623\,K)$'}}
    return  _samples

def test_agir_application(samples):

    for key in samples.keys():
        basename = os.path.join(scpdata,'agirdata/{}/TGA/tg'.format(key))
        if os.path.exists(basename+'.scp'):
            os.remove(basename + '.scp')

        # else read the original csv file
        filename = basename + '.csv'
        ss = samples[key]['TGA'] = NDDataset.read_csv(filename)
        # lets keep only data from somrthing close to 0.
        s = samples[key]['TGA'] = ss[-0.5:60.0]
        # for TGA, some information are missing.
        # we add them here
        s.x.units = 'hour'
        s.units = 'weight_percent'
        s.x.title = 'Time on stream'
        s.title = 'Mass variation'

        # save
        samples[key]['TGA'].save(basename + '.scp')

        # load
        samples[key]['TGA1'] = NDDataset.read(basename + '.scp')
        assert samples[key]['TGA1'] == samples[key]['TGA']


def test_slicing_agir(samples):

    for key in samples.keys():
        # our data are in our test `scpdata` directory.
        basename = os.path.join(scpdata, 'agirdata/{}/FTIR/FTIR'.format(key))
        filename = basename + '.scp'
        samples[key]['IR'] = NDDataset.read(filename)

    for key in samples.keys():
        figure()  # this is necessary to prevent the plot of the spectra on the same figure
        s = samples[key]['IR']
        s.plot(kind='stack')
        label = samples[key]['label']
        title = 'IR spectra for sample {}'.format(label)
        s.ax.set_title(title, fontsize=16)

    # We will resize the data in the interesting region of wavenumbers

    for key in samples.keys():
        s = samples[key]['IR']

        # reduce to a useful windoww of wavenumbers
        W = (1290., 3990.)
        s = s[:, W[0]:W[1]]

        samples[key]['IR'] = s

    options.log_level = DEBUG

    figure(figsize=(9, 3))
    axes = subplots(nrow=1, ncol=3)

    for key in samples.keys():
        s = samples[key]['IR']
        s.axes = axes  # store the axes information in each dataset

    i = 1
    for key in samples.keys():
        s = samples[key]['IR']
        s.plot_stack(ax=i, colorbar=False)
        i += 1
        label = samples[key]['label']
        title = 'sample {}'.format(label)
        s.ax.set_title(title, fontsize=16)


