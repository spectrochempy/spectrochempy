# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np

from spectrochempy import read_jcamp, read, Coord, NDDataset


def test_read_jcamp(JDX_2D):
    # read
    Y = read_jcamp({"some2Dspectra.jdx": JDX_2D.encode("utf8")})
    assert str(Y.coordset) == "CoordSet: [x:wavenumbers, y:acquisition timestamp (GMT)]"
    assert Y.shape == (3, 20)

    f = Y.write_jcamp("2D.jdx", confirm=False)
    Y = read(f)
    assert str(Y.coordset) == "CoordSet: [x:wavenumbers, y:acquisition timestamp (GMT)]"
    assert Y.shape == (3, 20)
    assert Y.name == "IR_2D"

    f.unlink()


def test_write_jcamp_masked_values(tmp_path):
    # masked samples must be exported as JCAMP missing values ("?"), not as
    # their (stale) underlying data, and must be excluded from MAXY/MINY (#1132)
    nx = 30
    x = Coord(np.linspace(4000.0, 1000.0, nx), units="1/cm", title="wavenumber")
    y = Coord([0.0])
    data = np.linspace(0.1, 0.9, nx).reshape(1, nx)
    data[0, 10:20] = 999.0  # sentinel hidden under the mask
    ds = NDDataset(data, coordset=[y, x], units="absorbance", name="masked")
    mask = np.zeros((1, nx), dtype=bool)
    mask[0, 10:20] = True
    ds.mask = mask

    f = ds.write_jcamp(tmp_path / "masked.jdx", confirm=False)
    text = f.read_text()

    # the masked underlying value never leaks into the file
    assert "999" not in text
    # each masked point is written as the JCAMP missing marker
    assert text.count("? ") == 10
    # header extrema ignore the masked samples
    assert "##MAXY=0.900000" in text
    assert "##MINY=0.100000" in text

    # round-trip: the masked region reads back as NaN, the rest stays finite
    back = read_jcamp(f)
    arr = np.asarray(back.data, dtype=float).ravel()
    assert np.isnan(arr[10:20]).all()
    assert np.isfinite(np.concatenate([arr[:10], arr[20:]])).all()
