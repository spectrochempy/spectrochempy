# -*- coding: utf-8 -*-
# flake8: noqa


from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.testing import assert_dataset_almost_equal


def test_read_write_jdx(IR_dataset_2D):
    X = IR_dataset_2D[:10, :100]

    # write
    f = X.write_jdx("nh4y-activation.jdx")

    # read
    Y = NDDataset.read_jdx(f)

    assert_dataset_almost_equal(X, Y, decimal=2)

    # delete
    f.unlink()

    # write
    f = X.write_jdx()
    assert f.stem == X.name

    Y = NDDataset.read_jcamp(f, name="xxx")
    assert Y.name == "xxx"

    f.unlink()
