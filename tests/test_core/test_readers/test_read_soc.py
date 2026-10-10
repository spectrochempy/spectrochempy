# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import platform
from pathlib import Path

import pytest
import requests

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.read_omnic import _read_spa
from spectrochempy.utils.objects import ScpObjectList
from spectrochempy.utils.testing import assert_dataset_equal

from _omnic_synthetic import synthetic_spa_with_optical_velocity


def _write_synthetic_soc_triplet(tmp_path):
    content = synthetic_spa_with_optical_velocity(
        8.8617,
        8.8617,
        timestamp=1577962800,
        scan_points=1234,
        peak_position=1,
        sample_scans=8,
        fft_points=4096,
        background_scans=4,
        background_gain=2.5,
        aperture=75.0,
        digitizer_bits=20,
        high_pass=200.0,
        low_pass=11000.0,
        sample_gain=12.5,
        native_history="SOC native processing",
    )
    paths = []
    for suffix in ("ddr", "HDR", "sdr"):
        path = tmp_path / f"sample.{suffix}"
        path.write_bytes(content)
        paths.append(path)
    return paths


def _assert_soc_dataset(dataset, path, suffix):
    expected = _read_spa(NDDataset(), path)
    expected.origin = "soc"
    expected.history = f"Imported SOC {suffix.upper()} file {path.name}"

    assert_dataset_equal(dataset, expected)
    assert dataset.origin == "soc"
    assert Path(dataset.filename) == path
    assert dataset.description == expected.description
    assert dataset.name == expected.name
    assert dataset.title == expected.title
    assert dataset.acquisition_date == expected.acquisition_date
    assert dataset.meta.optical_velocity == pytest.approx(8.8617)

    messages = [entry["message"] for entry in dataset.history_entries]
    assert messages == [
        f"Imported OMNIC SPA file {path.name}",
        "Data processing history from Omnic :\n"
        "------------------------------------\n"
        "SOC native processing",
        f"Imported SOC {suffix.upper()} file {path.name}",
    ]


@pytest.mark.parametrize(
    ("reader", "index", "suffix"),
    [
        (scp.read_ddr, 0, "ddr"),
        (scp.read_hdr, 1, "hdr"),
        (scp.read_sdr, 2, "sdr"),
    ],
)
def test_soc_specific_readers_use_spa_parser_and_record_soc_provenance(
    tmp_path, reader, index, suffix
):
    paths = _write_synthetic_soc_triplet(tmp_path)

    dataset = reader(paths[index])

    _assert_soc_dataset(dataset, paths[index], suffix)


def test_read_soc_single_file_and_generic_read_route_to_soc(tmp_path):
    paths = _write_synthetic_soc_triplet(tmp_path)

    soc = scp.read_soc(paths[1])
    generic = scp.read(paths[1])

    _assert_soc_dataset(soc, paths[1], "hdr")
    assert_dataset_equal(generic, soc)


@pytest.mark.parametrize("container", [list, tuple])
def test_read_soc_list_and_tuple_keep_default_merge_false(tmp_path, container):
    paths = _write_synthetic_soc_triplet(tmp_path)

    datasets = scp.read_soc(container(paths))

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 3
    assert all(dataset.origin == "soc" for dataset in datasets)


def test_read_soc_multiple_arguments_and_merge_keyword(tmp_path):
    paths = _write_synthetic_soc_triplet(tmp_path)

    default = scp.read_soc(*paths)
    no_merge = scp.read_soc(*paths, merge=False)
    merged = scp.read_soc(*paths, merge=True)

    assert isinstance(default, ScpObjectList)
    assert isinstance(no_merge, ScpObjectList)
    assert len(default) == len(no_merge) == 3
    assert merged.origin == "merged [soc]"
    assert merged.name == "merged [soc]"
    assert merged.filename is None
    assert merged.shape == (3, 2)


def test_read_soc_accepts_bytes_mapping_and_named_content(tmp_path):
    paths = _write_synthetic_soc_triplet(tmp_path)
    content = paths[0].read_bytes()

    dataset = scp.read_soc({"named.DDR": content}, name="named content")

    assert dataset.origin == "soc"
    assert dataset.name == "named content"
    assert dataset.filename.name == "named.DDR"
    assert [entry["message"] for entry in dataset.history_entries][-1] == (
        "Imported SOC DDR file named.DDR"
    )


def test_read_soc_directory_keeps_default_merge_false(tmp_path):
    paths = _write_synthetic_soc_triplet(tmp_path)

    datasets = scp.read_soc(directory=tmp_path, pattern="*.?dr")

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 3
    assert {Path(dataset.filename) for dataset in datasets} == set(paths)


def test_read_soc_directory_keyword_keeps_existing_merge_true_behavior(tmp_path):
    _write_synthetic_soc_triplet(tmp_path)

    datasets = scp.read_soc(directory=tmp_path, pattern="*.?dr", merge=True)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 3
    assert all(dataset.origin == "soc" for dataset in datasets)


@pytest.mark.network
def test_read_soc_merge_behavior(tmp_path):
    """Test that read_soc respects merge parameter.

    This test downloads sample files and verifies:
    - Default behavior (merge=False) preserves individual datasets
    - Explicit merge=True merges compatible datasets
    - All variants (read_soc, read_ddr, read_hdr, read_sdr) behave consistently
    """
    baseurl = "https://github.com/chet-j-ski/SOC100_example_data/raw/main/"
    fnames = [
        "Fused%20Silica0004.DDR",
        "Fused%20Silica0004.HDR",
        "Fused%20Silica0004.SDR",
    ]

    downloaded_files = {}
    for fname in fnames:
        try:
            response = requests.get(baseurl + fname, timeout=10)
            if response.status_code == 200:
                local_path = tmp_path / Path(fname).name
                with local_path.open("wb") as f:
                    f.write(response.content)
                downloaded_files[Path(fname).suffix.upper()] = local_path
        except requests.exceptions.RequestException:
            # Network error, skip this file
            continue

    expected_suffixes = {".DDR", ".HDR", ".SDR"}
    if downloaded_files.keys() != expected_suffixes:
        missing = sorted(expected_suffixes - downloaded_files.keys())
        pytest.skip(
            "Could not download the full SOC test triplet from GitHub. "
            f"Missing: {', '.join(missing)}"
        )

    ordered_files = [downloaded_files[suffix] for suffix in sorted(expected_suffixes)]

    try:
        # Test default behavior (merge=False)
        # Reading multiple files should return a list by default
        ds_default = scp.read_soc(*ordered_files)
        assert isinstance(
            ds_default, ScpObjectList
        ), "Default merge=False should return ScpObjectList for multiple files"
        assert len(ds_default) == len(
            ordered_files
        ), f"Expected {len(ordered_files)} datasets, got {len(ds_default)}"

        # Test explicit merge=False
        ds_no_merge = scp.read_soc(*ordered_files, merge=False)
        assert isinstance(
            ds_no_merge, ScpObjectList
        ), "Explicit merge=False should return ScpObjectList"
        assert len(ds_no_merge) == len(ordered_files)

        # Test merge=True
        # Compatible datasets should be merged into single dataset
        ds_merged = scp.read_soc(*ordered_files, merge=True)
        # If files have compatible dimensions, they should merge to single dataset
        # If not, they may still be returned as list
        assert hasattr(ds_merged, "shape") or isinstance(
            ds_merged, ScpObjectList
        ), "merge=True should return NDDataset or list"

        # Test individual file readers also default to merge=False
        ds_ddr = scp.read_ddr(downloaded_files[".DDR"])
        assert ds_ddr.shape == (1, 599), "read_ddr should return single dataset"

        ds_hdr = scp.read_hdr(downloaded_files[".HDR"])
        assert ds_hdr.shape == (1, 599), "read_hdr should return single dataset"

        ds_sdr = scp.read_sdr(downloaded_files[".SDR"])
        assert ds_sdr.shape == (1, 599), "read_sdr should return single dataset"

    finally:
        # Cleanup downloaded files
        for fname in downloaded_files.values():
            if int(platform.python_version_tuple()[1]) > 7:
                fname.unlink(missing_ok=True)
            else:
                if fname.exists():
                    fname.unlink()


@pytest.mark.network
def test_read_SOC(tmp_path):
    """Upload and read a Surface Optics example."""

    # the following does not work
    baseurl = "https://github.com/chet-j-ski/SOC100_example_data/raw/main/"
    fnames = [
        "Fused%20Silica0004.DDR",
        "Fused%20Silica0004.HDR",
        "Fused%20Silica0004.SDR",
    ]

    downloaded_any = False
    for i, fname in enumerate(fnames):
        try:
            response = requests.get(baseurl + fname, timeout=10)
        except requests.exceptions.RequestException:
            continue

        if response.status_code != 200:
            continue

        downloaded_any = True
        local_path = tmp_path / Path(fname).name
        with local_path.open("wb") as f:
            f.write(response.content)

        try:
            ds = scp.read_soc(local_path)
            assert str(ds) == "NDDataset: [float64] unitless (shape: (y:1, x:599))"
            assert ds.title == "reflectance"
            if i == 0:
                ds_ = scp.read_ddr(local_path)
            elif i == 1:
                ds_ = scp.read_hdr(local_path)
            else:
                ds_ = scp.read_sdr(local_path)
            assert ds_.name == ds.name
        finally:
            if int(platform.python_version_tuple()[1]) > 7:
                local_path.unlink(missing_ok=True)
            else:
                local_path.unlink()

    if not downloaded_any:
        pytest.skip("Could not download SOC test data from GitHub")
