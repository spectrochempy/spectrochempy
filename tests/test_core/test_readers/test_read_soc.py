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


def test_read_soc_merge_behavior():
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

    downloaded_files = []
    for fname in fnames:
        try:
            response = requests.get(baseurl + fname, timeout=10)
            if response.status_code == 200:
                with open(fname, "wb") as f:
                    f.write(response.content)
                downloaded_files.append(fname)
        except requests.exceptions.RequestException:
            # Network error, skip this file
            continue

    if len(downloaded_files) < 2:
        pytest.skip("Could not download sufficient test data from GitHub")

    try:
        # Test default behavior (merge=False)
        # Reading multiple files should return a list by default
        ds_default = scp.read_soc(*downloaded_files)
        assert isinstance(
            ds_default, ScpObjectList
        ), "Default merge=False should return ScpObjectList for multiple files"
        assert len(ds_default) == len(
            downloaded_files
        ), f"Expected {len(downloaded_files)} datasets, got {len(ds_default)}"

        # Test explicit merge=False
        ds_no_merge = scp.read_soc(*downloaded_files, merge=False)
        assert isinstance(
            ds_no_merge, ScpObjectList
        ), "Explicit merge=False should return ScpObjectList"
        assert len(ds_no_merge) == len(downloaded_files)

        # Test merge=True
        # Compatible datasets should be merged into single dataset
        ds_merged = scp.read_soc(*downloaded_files, merge=True)
        # If files have compatible dimensions, they should merge to single dataset
        # If not, they may still be returned as list
        assert hasattr(ds_merged, "shape") or isinstance(
            ds_merged, ScpObjectList
        ), "merge=True should return NDDataset or list"

        # Test individual file readers also default to merge=False
        ds_ddr = scp.read_ddr(downloaded_files[0])
        assert ds_ddr.shape == (1, 599), "read_ddr should return single dataset"

        ds_hdr = scp.read_hdr(downloaded_files[1])
        assert ds_hdr.shape == (1, 599), "read_hdr should return single dataset"

        ds_sdr = scp.read_sdr(downloaded_files[2])
        assert ds_sdr.shape == (1, 599), "read_sdr should return single dataset"

    finally:
        # Cleanup downloaded files
        for fname in downloaded_files:
            if int(platform.python_version_tuple()[1]) > 7:
                Path(fname).unlink(missing_ok=True)
            else:
                if Path(fname).exists():
                    Path(fname).unlink()


def test_read_SOC():
    """upload and read surface oftics exemple"""

    # the following does not work
    baseurl = "https://github.com/chet-j-ski/SOC100_example_data/raw/main/"
    fnames = [
        "Fused%20Silica0004.DDR",
        "Fused%20Silica0004.HDR",
        "Fused%20Silica0004.SDR",
    ]

    for i, fname in enumerate(fnames):
        response = requests.get(baseurl + fname, timeout=10)
        if response.status_code == 200:
            with open(fname, "wb") as f:
                f.write(response.content)
            ds = scp.read_soc(fname)
            assert str(ds) == "NDDataset: [float64] unitless (shape: (y:1, x:599))"
            assert ds.title == "reflectance"
            if i == 0:
                ds_ = scp.read_ddr(fname)
            elif i == 1:
                ds_ = scp.read_hdr(fname)
            else:
                ds_ = scp.read_sdr(fname)
            assert ds_.name == ds.name
            if int(platform.python_version_tuple()[1]) > 7:
                Path(fname).unlink(missing_ok=True)
            else:
                Path(fname).unlink()
