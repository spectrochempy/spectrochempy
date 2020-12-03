# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

from spectrochempy import NDDataset
from spectrochempy import general_preferences as prefs


# TODO: to revise with project!
def make_samples(force_original=False):
    _samples = {
            'P350': {
                    'label': r'$\mathrm{M_P}\,(623\,K)$'
                    },
            # 'A350': {'label': r'$\mathrm{M_A}\,(623\,K)$'},
            # 'B350': {'label': r'$\mathrm{M_B}\,(623\,K)$'}
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
            sample['IR'] = NDDataset.read_zip(filename, only=5, origin='omnic', merge=True)
            # save
            sample['IR'].save()

    for key, sample in _samples.items():
        basename = os.path.join(prefs.datadir, f'agirdata/{key}/TGA/tg')
        if os.path.exists(basename + '.scp') and not force_original:
            # check if the scp file have already been saved
            filename = basename + '.scp'
            sample['TGA'] = NDDataset.read(filename)
        else:
            # else read the original csv file
            filename = basename + '.csv'
            ss = sample['TGA'] = NDDataset.read_csv(filename, origin='tga')
            ss.squeeze(inplace=True)
            # lets keep only data from something close to 0.
            s = sample['TGA'] = ss[-0.5:35.0]
            # save
            s.save()

    return _samples


def test_slicing_agir():
    samples = make_samples(force_original=True)

    # We will resize the data in the interesting region of wavenumbers

    for key in samples.keys():
        s = samples[key]['IR']

        # reduce to a useful window of wavenumbers
        W = (1290., 3990.)
        s = s[:, W[0]:W[1]]

        samples[key]['IR'] = s

    assert samples['P350']['IR'].shape == (5, 2801)

    # set_loglevel(DEBUG)
