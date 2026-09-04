# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.testing import assert_dataset_equal

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"
WODGER = Path(__file__).parent / "ressources" / "omnic" / "wodger.spg"

pytestmark = pytest.mark.data


@pytest.fixture
def _skip_if_no_testdata():
    if not IRDATA.exists():
        pytest.skip("test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")


def test_read_omnic_local_wodger():
    # It is also possible to use more specific reader function such as
    # `read_spg` , `read_spa` or `read_srs` - they are alias of the read_omnic function.
    nd1 = scp.read_omnic(WODGER)
    assert nd1.name == "wodger"

    # test read_omnic with byte spg content
    filename_wodger = "wodger.spg"
    with open(WODGER, "rb") as fil:
        content = fil.read()
    nd2 = scp.read_omnic({filename_wodger: content})
    assert nd1 == nd2
    assert nd1.origin == "omnic"
    assert nd1.acquisition_date is not None
    assert nd1.y.title == "acquisition timestamp (GMT)"
    assert str(nd1.y.units) == "s"


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_omnic():
    # Class method opening a dialog (but for test it is preset)
    nd1 = scp.read_omnic(IRDATA / "nh4y-activation.spg")
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:55, x:5549))"

    # API method
    nd2 = scp.read_omnic(IRDATA / "nh4y-activation.spg")
    assert nd1 == nd2

    # It is also possible to use more specific reader function such as
    # `read_spg` , `read_spa` or `read_srs` - they are alias of the read_omnic function.
    l2 = scp.read_spg(WODGER, "irdata/nh4y-activation.spg")
    assert len(l2) == 2

    # Test bytes contents for spa files
    filename = IRDATA / "subdir" / "7_CZ0-100_Pd_101.SPA"
    nds = scp.read_spa(filename)
    with open(IRDATA / "subdir" / filename, "rb") as fil:
        content = fil.read()
    nd = scp.read_spa({filename: content})
    assert_dataset_equal(nd, nds)

    nd = scp.read_spa(IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA")
    assert str(nd) == "NDDataset: [float64] a.u. (shape: (y:1, x:5549))"
    assert nd.origin == "omnic"

    nd2 = scp.read_omnic(IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA")
    assert nd2 == nd

    # test import sample IFG
    nd = scp.read_spa(IRDATA / "carroucell_samp" / "2-BaSO4_0.SPA", return_ifg="sample")
    assert str(nd) == "NDDataset: [float64] V (shape: (y:1, x:16384))"

    # test import background IFG
    nd = scp.read_spa(
        IRDATA / "carroucell_samp" / "2-BaSO4_0.SPA", return_ifg="background"
    )
    assert str(nd) == "NDDataset: [float64] V (shape: (y:1, x:16384))"

    # import IFG from file without IFG
    a = scp.read_spa(
        IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA", return_ifg="sample"
    )
    assert a is None

    # rapid_sca series
    a = scp.read_srs("irdata/omnic_series/rapid_scan.srs")
    assert str(a) == "NDDataset: [float64] V (shape: (y:643, x:4160))"

    # rapid_sca series, import bg
    a = scp.read_srs("irdata/omnic_series/rapid_scan.srs", return_bg=True)
    assert str(a) == "NDDataset: [float64] V (shape: (y:1, x:4160))"

    # GC Demo
    a = scp.read_srs("irdata/omnic_series/GC_Demo.srs")
    assert str(a) == "NDDataset: [float64] % (shape: (y:788, x:1738))"

    # high speed series
    a = scp.read_srs("irdata/omnic_series/high_speed.srs")
    assert str(a) == "NDDataset: [float64] a.u. (shape: (y:897, x:13898))"

    # high speed series, import bg
    a = scp.read_srs("irdata/omnic_series/high_speed.srs", return_bg=True)
    assert str(a) == "NDDataset: [float64] unitless (shape: (y:1, x:13898))"


def test_read_spg_history_appended():
    """Regression test for #1144: sort history should be appended, not overwrite
    the import history. The history setter appends string values — both entries
    are preserved."""
    nd = scp.read_spg(WODGER, sortbydate=True)
    # History is a list of timestamp-prefixed strings
    history_text = " ".join(nd.history)
    assert "Imported from spg file" in history_text
    assert "Sorted by date" in history_text


def test_return_ifg_validation(tmp_path):
    """Regression test for #1144: invalid return_ifg values must warn clearly.
    The Importer catches exceptions and re-emits them as warnings, so we check
    for the warning."""
    import warnings

    spa_file = tmp_path / "dummy.spa"
    spa_file.write_bytes(b"\x00" * 1024)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = scp.read_spa(spa_file, return_ifg="invalid")
    assert result is None
    assert len(w) >= 1
    assert any("Invalid return_ifg value" in str(warning.message) for warning in w)


def test_allow_inconsistent_x_parameter_documented():
    assert "allow_inconsistent_x" in scp.read_spg.__doc__
    assert "allow_inconsistent_x" in scp.read_omnic.__doc__
    assert "allow_inconsistent_x=True" in scp.read_omnic.__doc__


def test_decode_experiment_info_block():
    """_decode_experiment_info_block correctly decodes 0x79 blocks."""
    from spectrochempy.core.readers.read_omnic import _decode_experiment_info_block

    # Helper to build a 0x79 block
    def _build_block(*fields):
        payload = b"\x00".join(f.encode("utf-8") for f in fields) + b"\x00"
        header = bytes([0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00])
        block = header + payload
        if len(block) < 60:
            block += b"\x00" * (60 - len(block))
        return block

    # Normal 4-field block
    block = _build_block(
        r"C:\MYDOCU~1\omnic\Param\CARROU~2.EXP",
        "CARROU~2.EXP",
        "iS50 Main Sample",
        "Default experiment for iS50 Main Sample Compartment",
    )
    result = _decode_experiment_info_block(block)
    assert result is not None
    assert result["experiment_path"] == r"C:\MYDOCU~1\omnic\Param\CARROU~2.EXP"
    assert result["experiment_file"] == "CARROU~2.EXP"
    assert result["accessory_name"] == "iS50 Main Sample"
    assert (
        result["experiment_title"]
        == "Default experiment for iS50 Main Sample Compartment"
    )

    # Subtype 0x9e (System Status) -> None
    bad = bytearray(block)
    bad[0] = 0x9E
    assert _decode_experiment_info_block(bytes(bad)) is None

    # Too short -> None
    assert _decode_experiment_info_block(b"\x00" * 40) is None

    # Single field -> experiment_path only
    block = _build_block(r"C:\path\to\file.spa")
    result = _decode_experiment_info_block(block)
    assert result is not None
    assert result["experiment_path"] == r"C:\path\to\file.spa"
    assert "experiment_file" not in result
    assert "accessory_name" not in result
    assert "experiment_title" not in result

    # Three fields without title
    block = _build_block(r"C:\ATR\crystal.exp", "crystal.exp", "ATR Crystal")
    result = _decode_experiment_info_block(block)
    assert result is not None
    assert result["experiment_path"] == r"C:\ATR\crystal.exp"
    assert result["experiment_file"] == "crystal.exp"
    assert result["accessory_name"] == "ATR Crystal"
    assert "experiment_title" not in result

    # Unix-style path in field 0
    block = _build_block("/home/omnic/param/test.exp", "test.exp", "iS50 Sample")
    result = _decode_experiment_info_block(block)
    assert result is not None
    assert result["experiment_path"] == "/home/omnic/param/test.exp"
    assert result["experiment_file"] == "test.exp"
    assert result["accessory_name"] == "iS50 Sample"
    assert "experiment_title" not in result


@pytest.mark.skip(reason="Requires an SPG file with inconsistent x-axes (#863)")
def test_allow_inconsistent_x_with_real_file():
    """Exercise both return paths once a representative sample is available."""


def _srs_header(path):
    """Locate the SRS series header the way `read_srs` does and return its info."""
    from spectrochempy.core.readers.read_omnic import _read_header

    sub_rs = b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\x50\x43\x47"
    sub_tg = b"\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00"
    with open(path, "rb") as fid:
        bytestring = fid.read()
    sub = sub_rs if bytestring.find(sub_rs, 1) > 0 else sub_tg
    pos = bytestring.find(sub, 1)
    index = [pos]
    while pos != -1:
        pos = bytestring.find(sub, pos + 1)
        index.append(pos)
    pos_info_data = np.array(index[:-1])[0] + (-152)
    with open(path, "rb") as fid:
        return _read_header(fid, pos_info_data)


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_time_axis_anchored_at_time_min():
    """The SRS time axis must start from the series minimum/first time, not the
    (mislabeled) `firsty` field which is really the regular step.

    The public fixtures only differ from the step at sub-rounding precision
    (the axis is rounded to 3 decimals), so this test pins the correct anchored
    construction formula. The maintainer-only `series0001.srs` is the file where
    the bug is numerically visible.
    """
    path = IRDATA / "omnic_series" / "GC_Demo.srs"
    info = _srs_header(path)
    nd = scp.read_srs(path)

    y = nd.y.data
    assert len(y) == info["ny"]
    # Anchored at the series minimum (in minutes) and ending at `lasty`.
    expected = np.around(np.linspace(info["time_min"], info["lasty"], info["ny"]), 3)
    np.testing.assert_allclose(y, expected)
    assert y[0] == np.around(info["time_min"], 3)
    assert y[-1] == np.around(info["lasty"], 3)

    # Guard the field separation: `firsty` is the step, not the axis start, and
    # `time_min` is the same header field as `collection_length` kept in minutes.
    assert info["time_min"] == np.float32(info["collection_length"] / 60)
    assert info["time_min"] != info["firsty"]


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_labels_stop_at_record_boundary():
    """SRS spectrum labels must contain only the human-readable name and must not
    leak binary metadata or spectral bytes.

    Regression: the per-spectrum SRS record is 84 bytes but labels were read
    with a 256-byte window, so binary metadata and spectrum data were decoded
    into the label.
    """
    path = IRDATA / "omnic_series" / "GC_Demo.srs"
    info = _srs_header(path)
    nd = scp.read_srs(path)

    labels = nd.y.labels
    assert len(labels) == info["ny"] == 788
    assert labels[0] == "Linked spectrum at 0.025 min."
    assert labels[1] == "Linked spectrum at 0.051 min."
    # No label may contain non-text control bytes (the historical leak marker).
    for label in labels:
        for ch in label:
            assert not (ord(ch) < 32 and ch not in "\n\t"), label
    assert labels[-1].startswith("Linked spectrum at")


@pytest.mark.parametrize(
    "name",
    [
        "rapid_scan_reprocessed.srs",
        "GC_Demo.srs",
        "high_speed.srs",
        "TGA_demo.srs",
    ],
)
@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_spectral_descending_by_default(name):
    """Spectral SRS files must be exposed in the public descending-wavenumber
    convention (like `read_spa`) without any manual reversal.

    The raw SRS spectral array is stored ascending-wavenumber; `_read_srs`
    normalizes it so the X axis runs high -> low wavenumber with the intensity
    data matched to it.

    Regression: before the fix these files either required `reverse_x=True`
    (GC/TGA/high-speed) or were exposed ascending (`rapid_scan_reprocessed`).
    """
    nd = scp.read_srs(IRDATA / "omnic_series" / name)
    x = np.asarray(nd.x)
    # Spectral records carry real units (not None / data points).
    assert nd.x.units is not None
    assert x[0] > x[-1]  # descending wavenumber
    # The first sample is associated with the high wavenumber (X[0]).
    assert nd.shape[-1] == len(x)
    assert getattr(nd.meta, "interferogram", None) is None


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_spectral_default_equals_former_reverse_x():
    """The default read must now match what used to require `reverse_x=True`
    (the documented issue #858 workaround), for every affected file.

    Pinned endpoint values are the physical X/data association verified against
    the legacy `reverse_x=True` orientation, which in turn follows the same
    convention as the SPA-validated series storage.
    """
    cases = {
        "GC_Demo.srs": (99.8824, "high_wavenumber"),
        "high_speed.srs": (-0.0233, "high_wavenumber"),
    }
    for name, (expected_first, _) in cases.items():
        nd = scp.read_srs(IRDATA / "omnic_series" / name)
        data = np.asarray(nd.data)
        x = np.asarray(nd.x)
        # data[0] is the intensity at X[0] = high wavenumber (descending).
        assert np.isclose(data[0, 0], expected_first, atol=1e-3)
        assert x[0] > x[-1]


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_interferogram_not_reversed():
    """`rapid_scan.srs` interferograms must remain on the interferogram path and
    must not be treated as spectral data (no reversal, ascending OPD axis).
    """
    nd = scp.read_srs(IRDATA / "omnic_series" / "rapid_scan.srs")
    assert nd.meta.interferogram is True
    x = np.asarray(nd.x)
    # Interferogram axis is optical path difference, exposed ascending, not a
    # descending spectral wavenumber axis.
    assert nd.x.title == "optical path difference"
    assert x[0] < x[-1]
    assert nd.shape == (643, 4160)


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_background_spectral_descending():
    """Spectral backgrounds must be normalized per-record, using their own
    (possibly reversed) firstx/lastx endpoints, to the same descending X as the
    series.

    Regression: background headers store firstx/lastx reversed relative to the
    series header for the same grid; a single global header rule would expose
    them mis-oriented.
    """
    for name in ["GC_Demo.srs", "high_speed.srs"]:
        series = scp.read_srs(IRDATA / "omnic_series" / name)
        bg = scp.read_srs(IRDATA / "omnic_series" / name, return_bg=True)
        xbg = np.asarray(bg.x)
        # Background uses the same descending wavenumber grid as the series.
        assert xbg[0] > xbg[-1]
        np.testing.assert_allclose(xbg, np.asarray(series.x), rtol=1e-4)
        assert bg.shape == (1, series.shape[-1])


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_reverse_x_deprecated():
    """`reverse_x` is a deprecated no-op: passing it emits a DeprecationWarning
    and returns the same (correct, automatically normalized) result as the
    default, so users of the #858 workaround keep getting correct data without
    a double reversal.
    """
    path = IRDATA / "omnic_series" / "GC_Demo.srs"
    default = scp.read_srs(path)
    with pytest.warns(DeprecationWarning):
        legacy = scp.read_srs(path, reverse_x=True)
    np.testing.assert_allclose(np.asarray(legacy.data), np.asarray(default.data))
    np.testing.assert_allclose(np.asarray(legacy.x), np.asarray(default.x))


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_and_read_spa_share_descending_convention():
    """A corrected `read_srs` must expose spectral X with the same
    descending-wavenumber, data-matched convention as `read_spa`."""
    srs = scp.read_srs(IRDATA / "omnic_series" / "GC_Demo.srs")
    spa = scp.read_spa(IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA")
    assert np.asarray(srs.x)[0] > np.asarray(srs.x)[-1]
    assert np.asarray(spa.x)[0] > np.asarray(spa.x)[-1]
