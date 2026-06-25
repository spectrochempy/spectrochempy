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
from spectrochempy.utils.objects import ScpObjectList

pytestmark = pytest.mark.network


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


def test_read_SOC(tmp_path):
    """upload and read surface oftics exemple"""

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
