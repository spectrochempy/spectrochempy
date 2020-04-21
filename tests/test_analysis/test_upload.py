# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================


from spectrochempy.core.readers.upload import upload_IRIS
from spectrochempy.core.analysis.pca import PCA
from spectrochempy.core import show
from spectrochempy.core import info_

def test_upload():

    ds = upload_IRIS()
    info_(ds)
    assert ds.shape == (150,4)
    assert repr(ds[0]) == "NDDataset: [float64] cm (shape: (y:1, x:4))"
    ds.plot_stack()

    pca = PCA(ds, centered=True)
    L, S = pca.reduce()
    L.plot_stack()
    pca.screeplot()
    pca.scoreplot(1,2, color_mapping='labels')
    show()

