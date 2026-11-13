import numpy as np
import pytest

import spectrochempy as scp

RAMAN = b"""##TITLE=Raman reference
##JCAMP-DX=5.01
##DATATYPE=RAMAN SPECTRUM
##XUNITS=1/CM
##YUNITS=ARBITRARY UNITS
##NPOINTS=4
##XFACTOR=0.1
##YFACTOR=2
##FIRSTX=780.0001
##LASTX=990
##XYDATA=(XY..XY)
7800.001,12; 7920,30
7920,31
9900,10
##END=
"""


@pytest.mark.parametrize("table", [b"XYDATA", b"XYPOINTS"])
def test_read_raman_explicit_pairs(tmp_path, table):
    payload = RAMAN.replace(b"XYDATA", table)
    path = tmp_path / "raman.dx"
    path.write_bytes(payload)
    for spectrum in (scp.read_jcamp(path), scp.jcamp.read({path.name: payload})):
        np.testing.assert_allclose(
            spectrum.x.data, [780.0001, 792, 792, 990], rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(spectrum.data, [24, 60, 62, 20])
        assert spectrum.x.units == scp.ur.Unit("cm^-1")
        assert spectrum.x.title == "Raman shift"
        assert spectrum.units is None
        assert spectrum.meta.jcamp_data_type == "RAMAN SPECTRUM"


def test_read_raman_keeps_separate_grids():
    second = RAMAN.replace(b"780.0001", b"781.0001").replace(b"7800.001", b"7810.001")
    spectra = scp.read_jcamp({"first.dx": RAMAN, "second.dx": second}, merge=True)
    assert sorted(s.x.data[0] for s in spectra) == pytest.approx(
        [780.0001, 781.0001], rel=0, abs=1e-12
    )


def test_read_raman_rejects_non_shift_units():
    payload = RAMAN.replace(b"1/CM", b"NANOMETERS")
    with pytest.warns(UserWarning, match="Unsupported Raman JCAMP-DX XUNITS"):
        assert scp.read_jcamp({"invalid.dx": payload}) is None


@pytest.mark.parametrize("metadata", [["Raman spectrum"], "RAMAN SPECTRUM"])
def test_importer_merges_native_files_with_raman_metadata(tmp_path, metadata):
    first = scp.NDDataset([1.0, 2.0], coordset=[scp.Coord([0, 1])])
    first.meta.jcamp_data_type = metadata
    second = scp.NDDataset([3.0, 4.0], coordset=[scp.Coord([0, 1])])
    paths = [
        first.dump(tmp_path / "first.scp"),
        second.dump(tmp_path / "second.scp"),
    ]
    merged = scp.read(*paths, merge=True)
    assert merged.shape == (2, 2)


def test_importer_merges_native_bytes_saved_from_raman(tmp_path):
    spectrum = scp.read_jcamp({"original.dx": RAMAN})
    paths = [
        scp.read_jcamp({"original.dx": RAMAN}).dump(tmp_path / name)
        for name in ("first.scp", "second.scp")
    ]
    merged = scp.read({path.name: path.read_bytes() for path in paths}, merge=True)
    assert merged.shape == (2, spectrum.size)
