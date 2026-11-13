import numpy as np
import pytest

import spectrochempy as scp

UVVIS = b"""##TITLE=UV-visible reference
##JCAMP-DX=5.01
##DATA TYPE=UV/VIS SPECTRUM
##XUNITS=Wavelength (nm)
##YUNITS=Logarithm epsilon
##XFACTOR=0.1
##YFACTOR=0.01
##FIRSTX=280.0001
##LASTX=220
##NPOINTS=4
##XYPOINTS=(XY..XY)
2800.001,175; 2650,200
2650 201 $$ repeat an explicitly sampled wavelength
2200,125
##END=
"""


@pytest.mark.parametrize("table", [b"XYDATA", b"XYPOINTS"])
def test_uvvis_explicit_grid(tmp_path, table):
    payload = UVVIS.replace(b"XYPOINTS", table)
    path = tmp_path / "uvvis.dx"
    path.write_bytes(payload)
    for spectrum in (scp.read_jcamp(path), scp.jcamp.read({path.name: payload})):
        assert spectrum.shape == (4,)
        np.testing.assert_allclose(
            spectrum.x.data, [280.0001, 265, 265, 220], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(spectrum.data, [1.75, 2, 2.01, 1.25])
        assert spectrum.x.units == scp.ur.nm
        assert spectrum.units is None
        assert spectrum.title == "Logarithm epsilon"
        assert spectrum.meta.jcamp_data_type == "UV/VIS SPECTRUM"


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        (b"NPOINTS=4", b"NPOINTS=5"),
        (b"LASTX=220", b"LASTX=221"),
        (b"2200,125", b"2200,125,999"),
        (b"(XY..XY)", b"(X++(Y..Y))"),
    ],
)
def test_uvvis_rejects_invalid_table(original, replacement):
    with pytest.warns(UserWarning, match="UV-visible JCAMP-DX"):
        assert (
            scp.read_jcamp({"invalid.dx": UVVIS.replace(original, replacement)}) is None
        )
