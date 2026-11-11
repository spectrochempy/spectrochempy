"""Public JCAMP-DX routing tests with an optional NMR reader."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import spectrochempy as scp
from spectrochempy.plugins.deps import MissingPluginNamespaceError
from spectrochempy.utils.exceptions import SpectroChemPyError

NMR = b"""##TITLE=NMR specimen
##DATA TYPE=NMR SPECTRUM
##XYDATA=(X++(Y..Y))
0 1 2 3
##END=
"""
NMR_2D = b"""##TITLE=Real 2D spectrum
##JCAMP-DX=6.0
##DATA TYPE=nD NMR SPECTRUM
##DATA CLASS=NTUPLES
##NUMDIM=2
##NTUPLES=nD NMR SPECTRUM
##SYMBOL=F1,F2,Y
##VAR_DIM=2,3,6
##UNITS=PPM,HZ,ARBITRARY UNITS
##FIRST=1,10,2
##LAST=3,8,12
##FACTOR=1,1,2
##PAGE=F1=1
##DATA TABLE=(F2++(Y..Y)),PROFILE
10 1 2 3
##PAGE=F1=3
##FIRST=3,10,8
##DATA TABLE=(F2++(Y..Y)),PROFILE
10 4 5 6
##END NTUPLES=nD NMR SPECTRUM
##END=
"""
LINK = b"##TITLE=container\n##DATA TYPE=LINK\n##BLOCKS=1\n" + NMR + b"##END=\n"
IR = b"""##TITLE=IR specimen
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=3
##LASTX=1
##XFACTOR=1
##YFACTOR=1
##NPOINTS=3
##XYDATA=(X++(Y..Y))
3 1 2 3
##END=
"""


@pytest.mark.parametrize(
    "payload",
    [
        NMR,
        LINK,
        LINK.replace(b"\n", b"\r"),
        b"$$ comment \xe9\n" + LINK,
        LINK.replace(b"DATA TYPE=NMR", b"DATA-TYPE=NMR"),
        LINK.replace(b"DATA TYPE=LINK", b"DATA TYPE=\nLINK").replace(
            b"DATA TYPE=NMR", b"DATA TYPE=\nNMR"
        ),
        LINK.replace(b"##XYDATA=", b"##XYDATA\n##XYDATA="),
    ],
)
@pytest.mark.parametrize("as_content", [False, True])
def test_nmr_delegates_to_the_processed_interop_profile(
    monkeypatch, tmp_path, payload, as_content
):
    calls = []

    def read(path, **options):
        path = Path(path)
        calls.append((path, path.read_bytes(), options))
        return scp.NDDataset([1.0, 2.0, 3.0])

    monkeypatch.setitem(scp.__dict__, "nmr", SimpleNamespace(read=read))
    source = tmp_path / "sample.jdx"
    source.write_bytes(payload)
    argument = {source.name: payload} if as_content else source
    dataset = scp.read_jcamp(argument)
    assert len(calls) == 1
    path, content, options = calls[0]
    assert content == payload
    assert options == {"protocol": "interop", "profile": "jcampdx-spectrum-1d"}
    assert dataset.filename.name == source.name
    assert dataset.origin == "jcamp-nmr"
    if as_content:
        assert not path.exists()
    else:
        assert path == source


def test_ir_routing_and_explicit_selection():
    spectrum = scp.jcamp.read({"ir.dx": IR})
    assert spectrum.data.tolist() == [[1, 2, 3]]
    assert spectrum.x.units == scp.ur("cm^-1").units
    ambiguous = IR.replace(b"##DATA TYPE=INFRARED SPECTRUM\n", b"")
    with pytest.raises(SpectroChemPyError, match="jcamp_kind"):
        scp.read_jcamp({"ir.dx": ambiguous})
    assert scp.read_jcamp({"ir.dx": ambiguous}, jcamp_kind="ir").shape == (1, 3)


def test_nmr_2d_delegates_to_the_processed_interop_profile(monkeypatch):
    def read(path, **options):
        assert Path(path).read_bytes() == NMR_2D
        assert options == {"protocol": "interop", "profile": "jcampdx-spectrum-2d"}
        return scp.NDDataset([[2, 4, 6], [8, 10, 12]])

    monkeypatch.setitem(scp.__dict__, "nmr", SimpleNamespace(read=read))
    dataset = scp.jcamp.read({"plane.jdx": NMR_2D}, jcamp_kind="nmr")
    assert dataset.shape == (2, 3)
    assert dataset.filename.name == "plane.jdx"
    assert dataset.origin == "jcamp-nmr"


@pytest.mark.parametrize("payload", [NMR_2D, LINK.replace(NMR, NMR_2D)])
def test_nmr_2d_reads_values_and_axes_with_the_plugin(tmp_path, payload):
    pytest.importorskip("spectrochempy_nmr")
    interop = pytest.importorskip("nmrglue.interop")
    if not hasattr(interop, "read_jcampdx_2d"):
        pytest.skip("requires nmrglue-ng with JCAMP-DX 2D support")

    source = tmp_path / "plane.jdx"
    source.write_bytes(payload)
    dataset = scp.read_jcamp(source)
    assert dataset.data.tolist() == [[2, 4, 6], [8, 10, 12]]
    assert dataset.y.data.tolist() == [1, 3]
    assert dataset.x.data.tolist() == [10, 9, 8]
    assert dataset.y.units == scp.ur.ppm
    assert dataset.x.units == scp.ur.Hz


def test_nmr_2d_link_with_an_untyped_table_is_rejected():
    extra = IR.replace(b"##DATA TYPE=INFRARED SPECTRUM\n", b"")
    linked = LINK.replace(NMR, NMR_2D + extra)
    with pytest.raises(SpectroChemPyError, match="tables outside its spectrum"):
        scp.read_jcamp({"linked.jdx": linked})


@pytest.mark.parametrize("linked", [False, True])
def test_nmr_2d_with_a_nested_untyped_table_is_rejected(linked):
    child = b"##TITLE=untyped child\n##XYDATA=(X++(Y..Y))\n10 1 2 3\n##END=\n"
    nested = NMR_2D.replace(b"##END=\n", child + b"##END=\n")
    payload = LINK.replace(NMR, nested) if linked else nested
    with pytest.raises(SpectroChemPyError, match="tables outside its spectrum"):
        scp.read_jcamp({"nested.jdx": payload})


def test_mixed_link_is_rejected():
    mixed = LINK.replace(b"##BLOCKS=1", b"##BLOCKS=2")[:-7] + IR + b"##END=\n"
    with pytest.raises(SpectroChemPyError, match="conflicting data types"):
        scp.read_jcamp({"mixed.jdx": mixed})


@pytest.mark.parametrize("label", [b"XYDATA", b"X-Y-DATA", b"X_Y/DATA"])
def test_nmr_link_with_another_untyped_table_is_rejected(label):
    extra = IR.replace(b"##DATA TYPE=INFRARED SPECTRUM\n", b"").replace(
        b"XYDATA", label
    )
    mixed = LINK.replace(b"##BLOCKS=1", b"##BLOCKS=2")[:-7] + extra + b"##END=\n"
    with pytest.raises(SpectroChemPyError, match="exactly one spectral table"):
        scp.read_jcamp({"mixed.jdx": mixed})


def test_missing_nmr_plugin_message(monkeypatch):
    class MissingNamespace:
        @property
        def read(self):
            raise MissingPluginNamespaceError("missing")

    monkeypatch.setitem(scp.__dict__, "nmr", MissingNamespace())
    with pytest.raises(SpectroChemPyError, match=r"pip install 'spectrochempy\[nmr\]'"):
        scp.read_jcamp({"sample.jdx": LINK})


def test_explicit_kind_cannot_override_a_conflicting_declaration():
    with pytest.raises(SpectroChemPyError, match="conflicts"):
        scp.read_jcamp({"sample.jdx": LINK}, jcamp_kind="ir")


def test_nmr_fid_cannot_fall_back_to_ir():
    fid = NMR.replace(b"NMR SPECTRUM", b"NMR FID")
    with pytest.raises(
        SpectroChemPyError, match="processed 1D and real 2D NMR spectra only"
    ):
        scp.read_jcamp({"fid.jdx": fid}, jcamp_kind="ir")
