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


@pytest.mark.parametrize("payload", [NMR, LINK])
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


def test_mixed_link_is_rejected():
    mixed = LINK.replace(b"##BLOCKS=1", b"##BLOCKS=2")[:-7] + IR + b"##END=\n"
    with pytest.raises(SpectroChemPyError, match="conflicting data types"):
        scp.read_jcamp({"mixed.jdx": mixed})


def test_nmr_link_with_another_untyped_table_is_rejected():
    extra = IR.replace(b"##DATA TYPE=INFRARED SPECTRUM\n", b"")
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
    with pytest.raises(SpectroChemPyError, match="processed 1D NMR spectra only"):
        scp.read_jcamp({"fid.jdx": fid}, jcamp_kind="ir")
