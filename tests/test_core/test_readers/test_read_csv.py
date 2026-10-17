# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import datetime
import locale
import os
from pathlib import Path
import subprocess
import sys
import warnings

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


def test_read_csv_omnic_date_parses_independently_of_ambient_locale():
    """
    The OMNIC CSV acquisition date must parse regardless of the ambient locale.

    ``datetime.strptime`` resolves weekday and month names through the current
    locale, so the reader must not rely on an ``en_US`` locale being installed
    and set at import time. It should force an English date locale
    (``LC_TIME="C"``) around the date parsing only, leaving the process-wide
    locale untouched.

    A non-English ambient ``LC_TIME`` is required to make this test
    discriminating: in the ``"C"`` locale English weekday/month names parse
    anyway, so a missing or broken scoped locale would go undetected. A
    candidate locale is therefore kept only if ``strptime`` of an English date
    *fails* under it; the parse then succeeds solely because of the scoped
    ``LC_TIME="C"``. The locale is also checked immediately after each read,
    before the test restores it, so a leak from the reader is not masked.
    """
    content = b"4000.0,0.5\n4001.0,0.6\n"
    original = locale.setlocale(locale.LC_TIME)
    ambient = None
    for loc in (
        "fr_FR.UTF-8",
        "fr_FR.utf8",
        "fr_FR",
        "de_DE.UTF-8",
        "de_DE.utf8",
        "de_DE",
        "es_ES.UTF-8",
        "es_ES",
        "it_IT.UTF-8",
        "it_IT",
    ):
        try:
            locale.setlocale(locale.LC_TIME, loc)
        except locale.Error:
            continue
        try:
            datetime.datetime.strptime(
                "Wed Jul 06 21-00-38 2016", "%a %b %d %H-%M-%S %Y"
            )
        except ValueError:
            ambient = loc
            break
        locale.setlocale(locale.LC_TIME, original)
    if ambient is None:
        locale.setlocale(locale.LC_TIME, original)
        pytest.skip(
            "No non-English LC_TIME locale that rejects English weekday/month names."
        )

    try:
        en = scp.read_csv({"acq_Wed Jul 06 21-00-38 2016.csv": content}, origin="omnic")
        assert locale.setlocale(locale.LC_TIME) == ambient
        fr = scp.read_csv(
            {"acq_Mer Aout 06 21-00-38 2016.csv": content}, origin="omnic"
        )
        assert locale.setlocale(locale.LC_TIME) == ambient
    finally:
        locale.setlocale(locale.LC_TIME, original)

    expected_en = datetime.datetime(2016, 7, 6, 21, 0, 38)
    expected_fr = datetime.datetime(2016, 8, 6, 21, 0, 38)
    for dataset, expected in ((en, expected_en), (fr, expected_fr)):
        assert dataset.y.title == "acquisition timestamp (GMT)"
        assert str(dataset.y.units) == "s"
        assert float(dataset.y.data[0]) > 0
        assert dataset.y.labels[0][0] == expected


def test_read_csv_in_fresh_process_does_not_depend_on_en_us():
    """
    Simulate a fresh process where the ``en_US`` locale is unavailable.

    The reader used to force ``locale.setlocale(locale.LC_ALL, "en_US")`` at
    module import time, warning on systems without that locale. Importing the
    reader and reading a CSV in a fresh process must neither warn nor mutate
    the process-wide locale. A subprocess is used so the reader module is
    imported for the first time, making this independent of the test order.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [env.get("PYTHONPATH"), str(Path(__file__).parents[3] / "src")])
    )
    code = """\
import locale
import warnings

real_setlocale = locale.setlocale


def no_en_us(category, name=None):
    if name is not None and str(name).startswith("en_US"):
        raise locale.Error("en_US locale is not installed")
    return real_setlocale(category, name)


locale.setlocale = no_en_us

import spectrochempy as scp

before = real_setlocale(locale.LC_ALL)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    scp.read_csv({"plain.csv": b"4000.0,0.5\\n4001.0,0.6\\n"})

assert real_setlocale(locale.LC_ALL) == before
assert not any("Could not set locale" in str(w.message) for w in caught)

import sys

sys.exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
