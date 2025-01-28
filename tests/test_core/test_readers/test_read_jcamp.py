# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy.core.dataset.nddataset import NDDataset


def test_read_jcamp(JDX_2D):
    # read
    Y = NDDataset.read_jcamp({"some2Dspectra.jdx": JDX_2D.encode("utf8")})
    assert str(Y.coordset) == "CoordSet: [x:wavenumbers, y:acquisition timestamp (GMT)]"
    assert Y.shape == (3, 20)

    f = Y.write_jcamp("2D.jdx", confirm=False)
    Y = NDDataset.read(f)
    assert str(Y.coordset) == "CoordSet: [x:wavenumbers, y:acquisition timestamp (GMT)]"
    assert Y.shape == (3, 20)
    assert Y.name == "IR_2D"

    f.unlink()
