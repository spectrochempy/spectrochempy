# ruff: noqa: S101

"""Carroucell preserves compatible subsets returned by the OMNIC importer."""

import struct
from datetime import timedelta
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

import spectrochempy as scp
from tests.test_core.test_readers._omnic_synthetic import (
    synthetic_spa_with_optical_velocity,
)


def write_scan(directory, holder, number, time, start=4000, units=17):
    path = directory / f"{holder}_{number:03d}.SPA"
    content = bytearray(
        synthetic_spa_with_optical_velocity(8.8617, timestamp=4_000_000_000 + time)
    )
    struct.pack_into("<ff", content, 416, start, start - 1)
    content[412] = units
    struct.pack_into(
        "<ff", content, 700, holder * 100 + number, holder * 100 + number + 0.5
    )
    path.write_bytes(content)
    return path


def split_holder(directory):
    return [
        write_scan(directory, 1, 1, 30),
        write_scan(directory, 1, 2, 10, start=3900),
        write_scan(directory, 1, 3, 20),
        write_scan(directory, 2, 1, 5),
        write_scan(directory, 2, 2, 40),
    ]


def test_incompatible_scans_form_flat_named_holder_groups(tmp_path):
    paths = split_holder(tmp_path)
    with pytest.warns(UserWarning, match="coordinates differ"):
        result = scp.carroucell.read(tmp_path, local_only=True)
    assert len(result) == 3
    assert [ds.name for ds in result] == ["1", "1", "2"]
    assert sorted(ds.shape for ds in result) == [(1, 2), (2, 2), (2, 2)]
    expected = {}
    for path in paths:
        ds = scp.read_spa(path, local_only=True)
        expected[tuple(ds.data[0])] = tuple(ds.x.data)
    actual = {tuple(row): tuple(ds.x.data) for ds in result for row in ds.data}
    assert actual == expected
    assert all(np.all(np.diff(ds.y.data) >= 0) for ds in result)


@pytest.mark.parametrize("spectra", [None, (1, 2)])
@pytest.mark.parametrize("clock_shift", [0, 7])
def test_temperatures_follow_each_partition_and_its_actual_scan_count(
    tmp_path, monkeypatch, spectra, clock_shift
):
    paths = split_holder(tmp_path)
    base = scp.read_spa(paths[0], local_only=True).y.labels[0][0] - timedelta(
        seconds=30
    )
    base += timedelta(seconds=clock_shift)
    times = [5, 10, 20, 30, 40]
    temperatures = [10, 20, 30, 50, 80]
    rows = [
        ((base + timedelta(seconds=t)).strftime("%d/%m/%y %H:%M:%S"), T)
        for t, T in zip(times, temperatures, strict=True)
    ]
    sheet = SimpleNamespace(
        nrows=9 + len(rows),
        cell=lambda row, col: SimpleNamespace(
            value=rows[row - 9][0 if col == 0 else 1]
        ),
    )
    book = SimpleNamespace(sheet_by_index=lambda index: sheet)
    reader = import_module("spectrochempy_carroucell.read_carroucell")
    monkeypatch.setattr(
        reader,
        "import_optional_dependency",
        lambda *args, **kwargs: SimpleNamespace(open_workbook=lambda path: book),
    )
    (tmp_path / "temperature.xls").touch()
    with pytest.warns(UserWarning, match="coordinates differ"):
        result = scp.carroucell.read(
            tmp_path, local_only=True, spectra=spectra, delta_clocks=clock_shift
        )
    expected = dict(zip(times, temperatures, strict=True))
    for ds in result:
        assert ds.y.units == scp.ur.Unit("s")
        scan_times = [
            (label[0] + timedelta(seconds=clock_shift) - base).total_seconds()
            for label in ds.y.labels
        ]
        np.testing.assert_allclose(
            ds.y.labels[:, -1].astype(float), [expected[t] for t in scan_times]
        )


def test_known_signal_quantities_remain_separate_in_one_holder(tmp_path):
    write_scan(tmp_path, 1, 1, 10)
    write_scan(tmp_path, 1, 2, 20, units=20)
    with pytest.warns(UserWarning, match="signal quantities differ"):
        result = scp.carroucell.read(tmp_path, local_only=True)
    assert len(result) == 2
    assert {ds.units for ds in result} == {
        scp.ur.Unit("absorbance"),
        scp.ur.Unit("Kubelka_Munk"),
    }
    assert [ds.name for ds in result] == ["1", "1"]


def test_compatible_holder_scans_still_form_one_dataset(tmp_path):
    write_scan(tmp_path, 1, 1, 30)
    write_scan(tmp_path, 1, 2, 10)
    result = scp.carroucell.read(tmp_path, local_only=True)
    assert isinstance(result, scp.NDDataset)
    assert result.name == "1"
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.data, [[102, 102.5], [101, 101.5]])


def test_single_scan_holders_still_assemble_when_compatible(tmp_path):
    write_scan(tmp_path, 1, 1, 30)
    write_scan(tmp_path, 2, 1, 10)
    result = scp.carroucell.read(tmp_path, local_only=True)
    assert isinstance(result, scp.NDDataset)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.data, [[201, 201.5], [101, 101.5]])


def test_explicit_merge_preserves_incompatible_spectral_partition(tmp_path):
    paths = split_holder(tmp_path)
    with pytest.warns(UserWarning, match="coordinates differ"):
        result = scp.carroucell.read(tmp_path, local_only=True, merge=True)
    assert len(result) >= 2
    assert sum(ds.shape[0] for ds in result) == 5
    expected = {
        tuple(scp.read_spa(p).data[0]): tuple(scp.read_spa(p).x.data) for p in paths
    }
    actual = {tuple(row): tuple(ds.x.data) for ds in result for row in ds.data}
    assert actual == expected


def test_last_holder_can_split_without_reordering_sample_holders(tmp_path):
    write_scan(tmp_path, 1, 1, 10)
    write_scan(tmp_path, 1, 2, 20)
    write_scan(tmp_path, 2, 1, 30)
    write_scan(tmp_path, 2, 2, 40, start=3900)
    write_scan(tmp_path, 2, 3, 50)
    with pytest.warns(UserWarning, match="coordinates differ"):
        result = scp.carroucell.read(tmp_path, local_only=True)
    assert [ds.name for ds in result] == ["1", "2", "2"]
    assert sum(ds.shape[0] for ds in result) == 5
    assert all("_import_measurement_identity" not in ds.__dict__ for ds in result)


@pytest.mark.parametrize("repeated_grid", [False, True])
def test_partitioned_holders_keep_their_groups_and_names(tmp_path, repeated_grid):
    expected = {}
    for holder in (1, 2):
        write_scan(tmp_path, holder, 1, 10, start=3900)
        write_scan(tmp_path, holder, 2, 20)
        if repeated_grid:
            write_scan(tmp_path, holder, 3, 30)
        expected[(str(holder), 3900)] = [holder * 100 + 1]
        expected[(str(holder), 4000)] = [holder * 100 + 2]
        if repeated_grid:
            expected[(str(holder), 4000)].append(holder * 100 + 3)
    with pytest.warns(UserWarning, match="coordinates differ"):
        result = scp.carroucell.read(tmp_path, local_only=True)
    assert [ds.name for ds in result] == ["1", "1", "2", "2"]
    actual = {(ds.name, ds.x.data[0]): ds.data[:, 0].tolist() for ds in result}
    assert actual == expected


def test_single_scan_holder_does_not_merge_with_a_multiscan_holder(tmp_path):
    write_scan(tmp_path, 1, 1, 10)
    write_scan(tmp_path, 2, 1, 20)
    write_scan(tmp_path, 2, 2, 30)
    result = scp.carroucell.read(tmp_path, local_only=True)
    assert [(ds.name, ds.shape) for ds in result] == [("1", (1, 2)), ("2", (2, 2))]
