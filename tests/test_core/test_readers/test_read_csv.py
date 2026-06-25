# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.utils.exceptions import UnsupportedOriginError


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

    # An unsupported origin should produce an actionable reader error.
    with pytest.raises(
        UnsupportedOriginError,
        match=(
            r"Cannot read CSV file 'test_omnic\.csv' with origin='opus'\.\n"
            r"Supported CSV origins are: 'omnic', 'tga'\.\n"
            r"Remove the origin argument or choose a supported origin\."
        ),
    ) as exc_info:
        scp.read_csv(
            {"test_omnic.csv": omnic_csv_content.encode("utf-8")},
            origin="opus",
        )
    assert isinstance(exc_info.value, NotImplementedError)

    with pytest.raises(UnsupportedOriginError, match="origin='vendor_omnic'"):
        scp.read_csv(
            {"test_omnic.csv": omnic_csv_content.encode("utf-8")},
            origin="vendor_omnic",
        )


def test_read_csv_skips_leading_comments_and_blank_lines():
    content = (
        "# collected by external script\n"
        "; exported manually\n"
        "\n"
        "time,intensity\n"
        "1,10\n"
        "2,20\n"
        "3,30\n"
    )

    dataset = scp.read_csv({"commented.csv": content.encode("utf-8")})

    assert dataset.shape == (1, 3)
    assert list(dataset.x.data) == [1.0, 2.0, 3.0]
    assert list(dataset.data.squeeze()) == [10.0, 20.0, 30.0]


def test_read_csv_accepts_simple_external_header():
    content = "wavenumber,absorbance\n4000,0.1\n3990,0.2\n3980,0.3\n"

    dataset = scp.read_csv({"header.csv": content.encode("utf-8")})

    assert dataset.shape == (1, 3)
    assert list(dataset.x.data) == [4000.0, 3990.0, 3980.0]
    assert list(dataset.data.squeeze()) == [0.1, 0.2, 0.3]


def test_read_csv_autodetects_tab_delimiter_for_simple_numeric_table():
    content = "x\tintensity\n1\t10\n2\t20\n3\t30\n"

    dataset = scp.read_csv({"tabbed.csv": content.encode("utf-8")})

    assert dataset.shape == (1, 3)
    assert list(dataset.x.data) == [1.0, 2.0, 3.0]
    assert list(dataset.data.squeeze()) == [10.0, 20.0, 30.0]
