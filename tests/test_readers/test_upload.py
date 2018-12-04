# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


from spectrochempy import upload_IRIS, PCA, show

def test_upload():

    ds = upload_IRIS()
    print(ds)
    assert repr(ds[0]) == "NDDataset: [   5.100,    3.500,    1.400,    " \
                        "0.200] cm"

    ds = upload_IRIS()
    print (ds)
    ds.plot_stack()

    pca = PCA(ds, centered=True)
    L, S = pca.transform()
    L.plot_stack()
    pca.screeplot()
    pca.scoreplot(1,2, color_mapping='labels')
    show()

