from pathlib import Path

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
        np.testing.assert_array_equal(spectrum.copy().x.data, spectrum.x.data)
        assert spectrum.x.units == scp.ur.nm
        assert spectrum.units is None
        assert spectrum.title == "Logarithm epsilon"
        assert spectrum.meta.jcamp_data_type == "UV/VIS SPECTRUM"


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        (b"##END=", b"##ENDNOTES=unexpected"),
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


@pytest.mark.parametrize("options", [{}, {"merge": True}])
def test_uvvis_files_keep_their_own_wavelengths(options):
    second = UVVIS.replace(b"280.0001", b"281.0001").replace(b"2800.001", b"2810.001")
    spectra = scp.read_jcamp({"first.dx": UVVIS, "second.dx": second}, **options)
    assert len(spectra) == 2
    assert [s.x.data[0] for s in spectra] == pytest.approx(
        [280.0001, 281.0001], rel=0, abs=1e-12
    )


@pytest.mark.parametrize("options", [{}, {"merge": False}])
def test_mixed_jcamp_files_keep_positional_names(options):
    ir_path = Path(__file__).with_name("jcamp_data") / "jtpolys.dx"
    spectra = scp.read_jcamp(
        {"first.dx": UVVIS, "second.dx": ir_path.read_bytes()},
        names=["ultraviolet", "infrared"],
        **options,
    )
    by_name = {s.name: s for s in spectra}
    assert by_name["ultraviolet"].x.units == scp.ur.nm
    assert by_name["infrared"].x.units == scp.ur.Unit("cm^-1")


def test_uvvis_native_roundtrip_preserves_wavelengths(tmp_path):
    spectrum = scp.read_jcamp({"sample.dx": UVVIS})
    path = spectrum.save_as(tmp_path / "sample.scp", confirm=False)
    restored = scp.NDDataset.load(path)
    np.testing.assert_array_equal(restored.x.data, spectrum.x.data)


def test_uvvis_transmittance_is_a_fraction():
    payload = UVVIS.replace(b"Logarithm epsilon", b"TRANSMITTANCE").replace(
        b"2800.001,175", b"2800.001,50"
    )
    spectrum = scp.read_jcamp({"transmittance.dx": payload})
    assert spectrum.units == scp.ur.absolute_transmittance
    assert spectrum.data[0] == 0.5
