# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


from spectrochempy.core import upload_IRIS

def test_upload():

    ds = upload_IRIS()
    print(ds)
    assert repr(ds[0]) == "NDDataset: [   5.100,    3.500,    1.400,    " \
                        "0.200] cm"







# =============================================================================
if __name__ == '__main__':
    pass
