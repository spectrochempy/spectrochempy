# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

from spectrochempy import masked, EFA


def test_EFA(IR_source_2D):

    ds = IR_source_2D.copy()

    print(ds)

    # columns masking
    ds[:, 1230.0:920.0] = masked  # do not forget to use float in slicing

    # difference spectra
    ds -= ds[-1]

    # row masking
    ds[10:12] = masked

    efa = EFA(ds)

    npc = 4
    c = efa.get_conc(npc, plot=True)





# =============================================================================
if __name__ == '__main__':
    pass
