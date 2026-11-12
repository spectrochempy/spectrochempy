# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from datetime import datetime

import numpy as np
import pytest

from spectrochempy import read_jcamp, read, Coord, NDDataset
from spectrochempy.utils.datetimeutils import UTC


def test_read_jcamp(JDX_2D):
    # read
    Y = read_jcamp({"some2Dspectra.jdx": JDX_2D.encode("utf8")})
    assert str(Y.coordset) == "CoordSet: [x:wavenumbers, y:acquisition timestamp (GMT)]"
    assert Y.shape == (3, 20)
    assert Y._acquisition_date == datetime(2016, 7, 6, 19, 3, 14, tzinfo=UTC)

    f = Y.write_jcamp("2D.jdx", confirm=False)
    Y = read(f)
    assert str(Y.coordset) == "CoordSet: [x:wavenumbers, y:acquisition timestamp (GMT)]"
    assert Y.shape == (3, 20)
    assert Y.name == "IR_2D"

    f.unlink()


def test_read_jcamp_single_spectrum_sets_acquisition_date():
    jdx = """##TITLE=single_semantic
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##LONGDATE=2016/07/06
##TIME=19:03:14
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
    ds = read_jcamp({"single_semantic.jdx": jdx.encode("utf8")})
    assert ds.origin == ""
    assert ds._acquisition_date == datetime(2016, 7, 6, 19, 3, 14, tzinfo=UTC)
    assert ds.y.is_empty


def test_read_jcamp_single_spectrum_preserves_origin():
    jdx = """##TITLE=single_origin
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##ORIGIN=omnic
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
    ds = read_jcamp({"single_origin.jdx": jdx.encode("utf8")})
    assert ds.origin == "omnic"


def test_read_jcamp_linked_multi_origin_is_deterministic():
    jdx = """##TITLE=linked_origin
##JCAMP-DX=5.01
##DATA TYPE=LINK
##BLOCKS=2
##TITLE=spec_1
##ORIGIN=omnic
##LONGDATE=2016/07/06
##TIME=19:03:14
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=3997.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 1.0 0.9 0.8 0.7
##END=
##TITLE=spec_2
##ORIGIN=labspec
##LONGDATE=2016/07/06
##TIME=19:04:14
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=3997.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 0.7 0.6 0.5 0.4
##END=
"""
    ds = read_jcamp({"linked_origin.jdx": jdx.encode("utf8")})
    assert ds.origin == "labspec; omnic"


def test_read_jcamp_without_date_keeps_acquisition_date_empty():
    jdx = """##TITLE=no_date
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
    ds = read_jcamp({"no_date.jdx": jdx.encode("utf8")})
    assert ds.acquisition_date is None


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


def test_read_jcamp_arbitrary_yunits_leave_units_unset_and_title_default():
    jdx = """##TITLE=arbitrary_test
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ARBITRARY UNITS
##FIRSTX=4000.0
##LASTX=3996.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 90.0 85.0 80.0 75.0
##END
"""
    ds = read_jcamp({"arbitrary_test.jdx": jdx.encode("utf8")})
    assert ds.units is None
    assert ds.title == "<untitled>"


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
    xydata_section = text.split("##XYDATA=(X++(Y..Y))\n", maxsplit=1)[1].split(
        "##END",
        maxsplit=1,
    )[0]

    # Numeric leakage must be checked in the XYDATA payload itself: unrelated
    # metadata such as ##OWNER may legitimately contain "999" in some
    # environments and would otherwise cause a false-positive failure.
    assert "999" not in xydata_section
    # each masked point is written as the JCAMP missing marker
    assert xydata_section.count("? ") == 10
    # header extrema ignore the masked samples
    assert "##MAXY=0.900000" in text
    assert "##MINY=0.100000" in text

    # round-trip: the masked region reads back as NaN, the rest stays finite
    back = read_jcamp(f)
    arr = np.asarray(back.data, dtype=float).ravel()
    assert np.isnan(arr[10:20]).all()
    assert np.isfinite(np.concatenate([arr[:10], arr[20:]])).all()


_DIFDUP_IR = b"""##TITLE=Compressed IR
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=TRANSMITTANCE
##FIRSTX=20
##LASTX=28
##XFACTOR=2
##YFACTOR=0.01 $$ Scale the stored integers
##NPOINTS=5
##XYDATA=(X++(Y..Y))
10A0KT $$ Duplicate the +2 difference
12A4%j
14A3 $$ Final checkpoint
##END=
"""


@pytest.mark.parametrize("compressed", [False, True])
def test_jcamp_ir_scaled_anchors_and_comments(compressed):
    payload = _DIFDUP_IR
    if not compressed:
        payload = payload.split(b"10A0KT", 1)[0] + b"10 10 12 14 14 13\n##END=\n"
    ds = read_jcamp({"scaled.jdx": payload})
    np.testing.assert_allclose(ds.data, [[0.10, 0.12, 0.14, 0.14, 0.13]])
    np.testing.assert_array_equal(ds.x.data, [20, 22, 24, 26, 28])


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (b"12A4%j", b"12A5%j", "ordinate checkpoint"),
        (b"12A4%j", b"15A4%j", "abscissa"),
        (b"10A0KT", b"10A0KT999999", "exceeds NPOINTS"),
        (b"14A3 $$ Final checkpoint\n", b"", "final DIF checkpoint"),
    ],
)
def test_jcamp_ir_rejects_invalid_difdup(original, replacement, message):
    payload = _DIFDUP_IR.replace(original, replacement)
    with pytest.warns(UserWarning, match=message):
        assert read_jcamp({"invalid.jdx": payload}) is None


@pytest.mark.parametrize("anchor", [0, 1, 10])
def test_jcamp_ir_sqz_ordinate_that_looks_like_an_exponent(anchor):
    payload = (
        _DIFDUP_IR.split(b"10A0KT", 1)[0]
        .replace(b"##FIRSTX=20", f"##FIRSTX={2 * anchor}".encode())
        .replace(b"##LASTX=28", f"##LASTX={2 * anchor + 2}".encode())
        .replace(b"##NPOINTS=5", b"##NPOINTS=2")
        + f"{anchor}E1?\n##END=\n".encode()
    )
    ds = read_jcamp({"exponent.jdx": payload})
    assert ds.data[0, 0] == pytest.approx(0.51)
    assert np.isnan(ds.data[0, 1])


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (b"", "Missing JCAMP-DX XYDATA"),
        (b"##XYDATA=(X++(Y..Y))\n10 10\n", "does not match NPOINTS"),
    ],
)
def test_jcamp_ir_checks_count_before_allocating_the_axis(monkeypatch, table, message):
    original = np.linspace

    def linspace(start, stop, num=50, *args, **kwargs):
        assert num != 1_000_000_000
        return original(start, stop, num, *args, **kwargs)

    monkeypatch.setattr(np, "linspace", linspace)
    payload = _DIFDUP_IR.replace(b"##NPOINTS=5", b"##NPOINTS=1000000000")
    payload = payload.split(b"##XYDATA=", 1)[0] + table + b"##END=\n"
    with pytest.warns(UserWarning, match=message):
        assert read_jcamp({"truncated.jdx": payload}) is None
