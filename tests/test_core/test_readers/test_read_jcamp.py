# -*- coding: utf-8 -*-
# flake8: noqa


from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.testing import assert_dataset_almost_equal


def test_read_jdx(JDX_2D):

    # read
    Y = NDDataset.read_jdx({"some2Dspectra.jdx": JDX_2D.encode("utf8")})
    assert str(Y.coordset) == "CoordSet: [x:wavenumbers, y:acquisition timestamp (GMT)]"
    assert Y.shape == (3, 20)

    f = Y.write_jdx("2D.jdx", confirm=False)
    Y = NDDataset.read(f)
    assert str(Y.coordset) == "CoordSet: [x:wavenumbers, y:acquisition timestamp (GMT)]"
    assert Y.shape == (3, 20)
    assert Y.name == "IR_2D"

    f.unlink()
