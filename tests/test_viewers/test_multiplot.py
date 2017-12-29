# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

from spectrochempy import *

def test_multiplot():

    source = NDDataset.read_omnic(
         os.path.join(datadir.path, 'irdata', 'NH4Y-activation.SPG'))[0:20]

    sources=[source, source*1.1, source*1.2, source*1.3]
    labels = ['sample {}'.format(label) for label in
              ["1", "2", "3", "4"]]
    multiplot(sources=sources, method='stack', labels=labels, nrow=2, ncol=2,
              figsize=(9, 5), style='sans',
              sharex=True, sharey=True, sharez=True)

    multiplot(sources=sources, method='image', labels=labels, nrow=2, ncol=2,
                    figsize=(9, 5), sharex=True, sharey=True, sharez=True)

    sources = [source * 1.2, source * 1.3,
               source, source * 1.1, source * 1.2, source * 1.3]
    labels = ['sample {}'.format(label) for label in
                                 ["1", "2", "3", "4", "5", "6"]]
    multiplot_map(sources=sources, labels=labels, nrow=2, ncol=3,
              figsize=(9, 5), sharex=False, sharey=False, sharez=True)

    multiplot_map(sources=sources, labels=labels, nrow=2, ncol=3,
              figsize=(9, 5), sharex=True, sharey=True, sharez=True)

    sources = [source * 1.2, source * 1.3, source, ]
    labels = ['sample {}'.format(label) for label in
              ["1", "2", "3"]]
    multiplot_stack(sources=sources, labels=labels, nrow=1, ncol=3,
                    figsize=(9, 5), sharex=True,
                    sharey=True, sharez=True)

    multiplot_stack(sources=sources, labels=labels, nrow=3, ncol=1,
                    figsize=(9, 5), sharex=True,
                    sharey=True, sharez=True)

    multiplot(method='lines', sources=[source[0], source[10]*1.1,
                                     source[19]*1.2, source[15]*1.3],
              nrow=2, ncol=2, figsize=(9, 5),
              labels=labels, sharex=True)

    show()







# =============================================================================
if __name__ == '__main__':
    pass
