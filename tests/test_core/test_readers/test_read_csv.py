# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs


def test_read_csv():
    """Test CSV reading with synthetic data (no external test data required)."""
    prefs.csv_delimiter = ","

    # Test reading a simple 2-column CSV (like omnic format)
    # Create synthetic omnic-like CSV: wavenumber, absorbance
    omnic_csv_content = """4000.0,0.5
4001.0,0.6
4002.0,0.7
4003.0,0.8
4004.0,0.9"""

    # Read via dict with bytes content (this is the reliable way)
    B = scp.read_csv(
        {"test_omnic.csv": omnic_csv_content.encode("utf-8")}, origin="omnic"
    )
    assert B.shape == (1, 5)
    assert B.origin == "omnic"
    assert B.units == "absorbance"
    assert B.title == "absorbance"
    assert str(B.x.units) == "cm⁻¹"

    # Test reading CSV with semicolon delimiter (like TGA format)
    tga_csv_content = """-16.13;7.496
-16.115;7.224
-16.101;7.027
-16.086;6.887"""

    A = scp.read_csv(
        {"test_tga.csv": tga_csv_content.encode("utf-8")},
        csv_delimiter=";",
        origin="tga",
    )
    assert A.shape == (1, 4)
    assert A.origin == "tga"
    assert A.units == "percent"
    assert A.x.units == "hour"
    assert A.x.title == "time-on-stream"
    assert A.title == "mass change"

    # Read CSV content via dict (bytes) - without origin
    C = scp.read_csv({"somename.csv": omnic_csv_content.encode("utf-8")})
    assert C.shape == (1, 5)

    # wrong origin parameters - should return None
    D = scp.read_csv(
        {"test_omnic.csv": omnic_csv_content.encode("utf-8")}, origin="opus"
    )
    assert not D
