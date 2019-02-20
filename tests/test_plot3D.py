# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


from spectrochempy import NDDataset, show

import os

def test_plot3D():
    data = NDDataset.read_matlab(os.path.join('matlabdata', 'als2004dataset.MAT'))

    X = data[0]


    X.plot_3D()

    show()

    pass






# =============================================================================
if __name__ == '__main__':
    pass
