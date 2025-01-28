# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import platform
from pathlib import Path

import requests

import spectrochempy as scp


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
        response = requests.get(baseurl + fname)
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
