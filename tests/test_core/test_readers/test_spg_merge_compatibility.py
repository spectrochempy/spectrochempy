"""SPG assembly respects native member identities, not just shared units."""

import io
import struct
import zipfile
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.objects import ScpObjectList
from tests.test_core.test_readers._omnic_synthetic import (
    synthetic_spa_with_optical_velocity,
)
from tests.test_core.test_readers._omnic_synthetic import synthetic_spg_members


@pytest.mark.parametrize(
    "codes, field, message",
    [
        ([16, 11], "ycode", "signal quantities"),
        ([15, 31], "ycode", "signal quantities"),
        ([1, 32], "xcode", "spectral axis"),
    ],
)
def test_spg_rejects_known_member_conflicts_even_when_units_match(
    codes, field, message
):
    content = synthetic_spg_members([{field: value} for value in codes])
    with pytest.raises(ValueError, match=message):
        scp.read_spg({"mixed.spg": content}, sortbydate=False)


@pytest.mark.parametrize("merge", [None, False, True])
@pytest.mark.parametrize(
    "codes, field, message",
    [
        ([16, 11], "ycode", "signal quantities"),
        ([15, 31], "ycode", "signal quantities"),
        ([1, 32], "xcode", "spectral axis"),
    ],
)
def test_spg_split_keeps_native_identity_through_outer_assembly(
    merge, codes, field, message
):
    members = [
        {field: value, "data": [index + 10, index + 20]}
        for index, value in enumerate(codes)
    ]
    options = {} if merge is None else {"merge": merge}
    content = synthetic_spg_members(members)
    if merge is False:
        result = scp.read_spg(
            {"mixed.spg": content}, allow_inconsistent_x=True, **options
        )
    else:
        with pytest.warns(UserWarning, match=message):
            result = scp.read_spg(
                {"mixed.spg": content}, allow_inconsistent_x=True, **options
            )
    assert isinstance(result, ScpObjectList)
    assert len(result) == 2
    actual = sorted(tuple(ds.data[0]) for ds in result)
    assert actual == [(10, 20), (11, 21)]
    for member in members:
        reference = scp.read_spg({"one.spg": synthetic_spg_members([member])})
        selected = next(
            ds for ds in result if tuple(ds.data[0]) == tuple(member["data"])
        )
        assert selected.title == reference.title
        assert selected.units == reference.units
        assert selected.x.title == reference.x.title
        assert selected.x.units == reference.x.units
    assert all("_import_measurement_identity" not in ds.__dict__ for ds in result)


@pytest.mark.parametrize("field, value", [("ycode", 17), ("ycode", 16), ("xcode", 32)])
@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize(
    "options", [{}, {"allow_inconsistent_x": True, "merge": False}]
)
def test_spg_compatible_native_members_still_assemble_and_sort(
    field, value, count, options
):
    members = [
        {field: value, "data": [1, 2], "timestamp": 4_000_000_020, "label": "Raman"},
        {field: value, "data": [3, 4], "timestamp": 4_000_000_010, "label": "infrared"},
        {field: value, "data": [5, 6], "timestamp": 4_000_000_030, "label": "other"},
    ][:count]
    ds = scp.read_spg({"same.spg": synthetic_spg_members(members)}, **options)
    assert isinstance(ds, scp.NDDataset)
    ordered = sorted(members, key=lambda member: member["timestamp"])
    np.testing.assert_array_equal(ds.data, [member["data"] for member in ordered])
    assert ds.y.labels[:, 1].tolist() == [member["label"] for member in ordered]
    np.testing.assert_array_equal(
        ds.y.data, [label[0].timestamp() for label in ds.y.labels]
    )
    assert "_import_measurement_identity" not in ds.__dict__


def test_spg_unrecognized_codes_do_not_establish_a_conflict():
    members = [{"ycode": 254, "xcode": 254}, {"ycode": 255, "xcode": 255}]
    ds = scp.read_spg({"unknown.spg": synthetic_spg_members(members)}, sortbydate=False)
    assert isinstance(ds, scp.NDDataset)
    assert ds.shape == (2, 2)
    assert ds.title == "intensity"
    assert ds.x.title == "xaxis"


@pytest.mark.parametrize(
    "changed",
    [
        {"firstx": 3900},
        {"data": [3, 4, 5]},
        {"xcode": 3},
        {"ycode": 20},
    ],
)
def test_spg_existing_grid_and_unit_rejections_keep_the_split_escape_hatch(changed):
    second = {"data": [3, 4], **changed}
    content = synthetic_spg_members([{}, second])
    with pytest.raises(ValueError, match="Inconsistent data set"):
        scp.read_spg({"geometry.spg": content})
    result = scp.read_spg(
        {"geometry.spg": content}, allow_inconsistent_x=True, merge=False
    )
    assert len(result) == 2
    expected = synthetic_spg_members([second])
    reference = scp.read_spg({"reference.spg": expected})
    selected = next(
        ds
        for ds in result
        if ds.data.shape[1] == reference.shape[1]
        and tuple(ds.data[0]) == tuple(reference.data[0])
    )
    np.testing.assert_array_equal(selected.x.data, reference.x.data)
    assert selected.x.units == reference.x.units
    assert selected.units == reference.units


@pytest.mark.parametrize("route", ["files", "directory", "zip"])
def test_spg_known_constraints_survive_unknown_members_and_nested_imports(
    tmp_path, route
):
    first = synthetic_spg_members([{"ycode": 0}, {"ycode": 26}])
    second = synthetic_spg_members([{"ycode": 31}])
    if route == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("first.spg", first)
            archive.writestr("second.spg", second)
        source = {"nested.zip": buffer.getvalue()}
    else:
        (tmp_path / "first.spg").write_bytes(first)
        (tmp_path / "second.spg").write_bytes(second)
        source = (
            tmp_path
            if route == "directory"
            else [tmp_path / "first.spg", tmp_path / "second.spg"]
        )
    with pytest.warns(UserWarning, match="signal quantities"):
        result = scp.read(source, merge=True)
    assert len(result) == 2
    assert sorted(ds.shape for ds in result) == [(1, 2), (2, 2)]
    assert all("_import_measurement_identity" not in ds.__dict__ for ds in result)


@pytest.mark.parametrize("ycode", [16, 17])
def test_spg_native_signal_identity_agrees_with_named_spa_units(ycode):
    source = {
        "member.spg": synthetic_spg_members([{"ycode": ycode, "data": [20, 30]}]),
        "other.spa": synthetic_spa_with_optical_velocity(
            8.8617, timestamp=4_000_000_000
        ),
    }
    if ycode == 16:
        with pytest.warns(UserWarning, match="signal quantities"):
            result = scp.read(source, merge=True)
        assert len(result) == 2
        transmittance = next(ds for ds in result if ds.units == scp.ur.Unit("percent"))
        np.testing.assert_array_equal(transmittance.data, [[20, 30]])
    else:
        result = scp.read(source, merge=True)
        assert isinstance(result, scp.NDDataset)
        assert result.shape == (2, 2)


@pytest.mark.parametrize("sortbydate", [False, True])
def test_two_member_fixture_keeps_native_labels_timestamps_and_payloads_paired(
    sortbydate,
):
    path = Path(__file__).parent / "ressources" / "omnic" / "wodger.spg"
    content = path.read_bytes()
    count = struct.unpack_from("<H", content, 294)[0]
    records = [
        struct.unpack_from("<BBII", content, 304 + 16 * index) for index in range(count)
    ]
    captions = [position for key, _, position, _ in records if key == 107]
    payloads = [(position, length) for key, _, position, length in records if key == 3]
    epoch = datetime(1899, 12, 31, tzinfo=UTC)
    expected = []
    for caption, (payload, length) in zip(captions, payloads, strict=True):
        raw_time = struct.unpack_from("<I", content, caption + 256)[0]
        label = content[caption : caption + 256].split(b"\x00", 1)[0].decode("latin-1")
        row = struct.unpack_from(f"<{length // 4}f", content, payload)
        expected.append((epoch + timedelta(seconds=raw_time), label, row))
    if sortbydate:
        expected.sort(key=lambda member: member[0])
    ds = scp.read_spg(path, sortbydate=sortbydate)
    assert ds.shape[0] == len(expected) == 2
    for index, (stamp, label, row) in enumerate(expected):
        assert ds.y.labels[index, 0] == stamp
        assert ds.y.labels[index, 1] == label
        assert ds.y.data[index] == stamp.timestamp()
        np.testing.assert_array_equal(ds.data[index], row)
