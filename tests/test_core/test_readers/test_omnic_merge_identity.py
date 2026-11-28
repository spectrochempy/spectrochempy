"""Native OMNIC declarations constrain automatic assembly across containers."""

import io
import struct
import zipfile

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.objects import ScpObjectList
from tests.test_core.test_readers._omnic_synthetic import (
    synthetic_spa_with_optical_velocity,
)
from tests.test_core.test_readers._omnic_synthetic import synthetic_spg_members
from tests.test_core.test_readers._omnic_synthetic import synthetic_srs


def _content(kind, *, xcode=1, ycode=17):
    if kind == "spa":
        return synthetic_spa_with_optical_velocity(8.8617, xunits=xcode, yunits=ycode)
    if kind == "srs":
        return synthetic_srs(xcode=xcode, ycode=ycode)
    return synthetic_spg_members([{"xcode": xcode, "ycode": ycode}])


def _archive(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return stream.getvalue()


@pytest.mark.parametrize("kind", ["spa", "srs"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "field,codes,reason",
    [
        ("ycode", [16, 11], "signal quantities"),
        ("ycode", [15, 31], "signal quantities"),
        ("xcode", [1, 32], "spectral axis"),
    ],
)
def test_native_conflicts_remain_separate(kind, nested, field, codes, reason):
    members = {
        f"member{i}.{kind}": _content(kind, **{field: code})
        for i, code in enumerate(codes)
    }
    content = (
        {"outer.zip": _archive({"inner.zip": _archive(members)})} if nested else members
    )
    with pytest.warns(UserWarning, match=reason):
        result = scp.read(content, merge=True, sortbydate=False)
    assert isinstance(result, ScpObjectList)
    assert len(result) == 2
    for code in codes:
        reference = scp.read(
            {f"reference.{kind}": _content(kind, **{field: code})}, sortbydate=False
        )
        matches = [
            ds
            for ds in result
            if ds.title == reference.title and ds.x.title == reference.x.title
        ]
        assert len(matches) == 1
        np.testing.assert_array_equal(matches[0].data, reference.data)
        assert matches[0].units == reference.units
        np.testing.assert_array_equal(matches[0].x.data, reference.x.data)
    assert all("_import_measurement_identity" not in ds.__dict__ for ds in result)


@pytest.mark.parametrize("kind", ["spa", "srs"])
@pytest.mark.parametrize(
    "field,code", [("ycode", 16), ("ycode", 31), ("ycode", 20), ("xcode", 32)]
)
@pytest.mark.parametrize("merge", [False, True])
def test_matching_native_identities_preserve_assembly(kind, field, code, merge):
    content = _content(kind, **{field: code})
    result = scp.read(
        {f"a.{kind}": content, f"b.{kind}": content}, merge=merge, sortbydate=False
    )
    if merge:
        assert isinstance(result, scp.NDDataset)
        assert result.shape == (6 if kind == "srs" else 2, 2)
        assert "_import_measurement_identity" not in result.__dict__
    else:
        assert isinstance(result, ScpObjectList)
        assert len(result) == 2


@pytest.mark.parametrize("kinds", [("spa", "spg"), ("spa", "srs"), ("srs", "spg")])
@pytest.mark.parametrize(
    "field,codes,reason",
    [
        ("ycode", [16, 11], "signal quantities"),
        ("ycode", [15, 31], "signal quantities"),
        ("xcode", [1, 32], "spectral axis"),
    ],
)
def test_different_omnic_containers_share_known_constraints(
    kinds, field, codes, reason
):
    members = {
        f"member{i}.{kind}": _content(kind, **{field: code})
        for i, (kind, code) in enumerate(zip(kinds, codes, strict=True))
    }
    with pytest.warns(UserWarning, match=reason):
        result = scp.read(members, merge=True, sortbydate=False)
    assert isinstance(result, ScpObjectList)
    assert len(result) == 2
    assert sum(ds.shape[0] for ds in result) == (4 if "srs" in kinds else 2)


@pytest.mark.parametrize("kind", ["spa", "srs"])
def test_unknown_constraints_do_not_erase_known_conflicts_in_nested_zip(kind):
    members = {
        f"member{i}.{kind}": _content(kind, ycode=code)
        for i, code in enumerate([0, 15, 31])
    }
    with pytest.warns(UserWarning, match="signal quantities"):
        result = scp.read(
            {"outer.zip": _archive({"inner.zip": _archive(members)})},
            merge=True,
            sortbydate=False,
        )
    assert isinstance(result, ScpObjectList)
    assert len(result) == 2
    assert sum(ds.shape[0] for ds in result) == (9 if kind == "srs" else 3)
    assert all("_import_measurement_identity" not in ds.__dict__ for ds in result)


@pytest.mark.parametrize("conflict", [False, True])
def test_srs_background_identity_comes_from_background_header(conflict):
    members = {
        f"member{i}.srs": synthetic_srs(
            ycode=16 if i == 0 or conflict else 11,
            background_ycode=16 if i == 0 or not conflict else 11,
        )
        for i in range(2)
    }
    if conflict:
        with pytest.warns(UserWarning, match="signal quantities"):
            result = scp.read_srs(members, return_bg=True, merge=True, sortbydate=False)
        assert len(result) == 2
        assert {ds.title for ds in result} == {"transmittance", "reflectance"}
    else:
        result = scp.read_srs(members, return_bg=True, merge=True, sortbydate=False)
        assert isinstance(result, scp.NDDataset)
        assert result.shape == (2, 2)
        assert result.title == "transmittance"
        np.testing.assert_array_equal(result.data, [[8, 7], [8, 7]])


@pytest.mark.parametrize("payload", ["sample", "background"])
def test_associated_spa_interferograms_do_not_inherit_primary_spectral_identity(
    payload,
):
    members = {}
    for i, (xcode, ycode) in enumerate([(1, 16), (32, 11)]):
        content = bytearray(_content("spa", xcode=xcode, ycode=ycode))
        struct.pack_into("<H", content, 294, 4)
        for offset, key, position in [(336, 102, 800), (352, 103, 820)]:
            struct.pack_into("<BBII", content, offset, key, 0, position, 8)
            struct.pack_into("<ff", content, position, 5, 6)
        members[f"member{i}.spa"] = bytes(content)
    result = scp.read_spa(members, return_ifg=payload, merge=True, sortbydate=False)
    assert isinstance(result, scp.NDDataset)
    assert result.shape == (2, 2)
    assert result.title == "detector signal"
    assert result.x.title == "optical path difference"
    np.testing.assert_array_equal(result.data, [[5, 6], [5, 6]])
    assert "_import_measurement_identity" not in result.__dict__


@pytest.mark.parametrize("kind", ["spa", "srs"])
def test_disk_and_content_reads_have_same_output(tmp_path, kind):
    content = _content(kind)
    path = tmp_path / f"sample.{kind}"
    path.write_bytes(content)
    disk = scp.read(path, sortbydate=False)
    memory = scp.read({path.name: content}, sortbydate=False)
    np.testing.assert_array_equal(disk.data, memory.data)
    np.testing.assert_array_equal(disk.x.data, memory.x.data)
    assert disk.title == memory.title
    assert disk.units == memory.units


@pytest.mark.parametrize("ycode", [16, 20, 31])
def test_matching_constraints_allow_spa_spg_assembly(ycode):
    members = {
        "member.spa": _content("spa", ycode=ycode),
        "member.spg": _content("spg", ycode=ycode),
    }
    result = scp.read(members, merge=True, sortbydate=False)
    assert isinstance(result, scp.NDDataset)
    assert result.shape == (2, 2)
    assert "_import_measurement_identity" not in result.__dict__
