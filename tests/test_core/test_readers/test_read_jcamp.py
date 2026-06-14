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


def test_read_jcamp_transmittance_units():
    # a JCAMP-DX file declaring ##YUNITS=TRANSMITTANCE must come back with the
    # transmittance unit assigned, not just the title (#1080). Previously the
    # reader set the title but left units unset.
    jdx = """##TITLE=transmittance_test
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=TRANSMITTANCE
##FIRSTX=4000.0
##LASTX=3996.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 90.0 85.0 80.0 75.0
##END
"""
    ds = read_jcamp({"transmittance_test.jdx": jdx.encode("utf8")})
    assert ds.title == "transmittance"
    assert ds.units is not None
    assert ds.units == "transmittance"


def test_read_jcamp_header_value_with_equals():
    # A header value containing "=" (e.g. a sample id) must not break the
    # ``keyword=text`` split (#1150): the parser now splits on the first "="
    # only, instead of raising "too many values to unpack".
    jdx = """##TITLE=sample=A/2
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=3997.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 1.0 0.9 0.8 0.7
##END
"""
    ds = read_jcamp({"sample.jdx": jdx.encode("utf8")})
    assert ds.name == "sample=A/2"


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
