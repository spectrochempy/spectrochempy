# -*- coding: utf-8 -*-
# flake8: noqa


from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.testing import assert_dataset_almost_equal


def test_write_jcamp(IR_dataset_2D):

    X = IR_dataset_2D[:10, :50]

    # write
    f = X.write_jdx("nh4y-activation.jdx")

    f.unlink()
