# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import io
import struct
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.datetimeutils import UTC
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


def test_read_spg_experiment_info_uses_native_fixed_slots():
    dataset = scp.read_spg(WODGER, sortbydate=False)

    assert dataset.meta.omnic_experiment_path == (
        r"C:\MYDOCU~1\omnic\Param\VLADIM~1.EXP"
    )
    assert dataset.meta.omnic_experiment_title == "Transmission"
    assert dataset.meta.omnic_experiment_description == (
        "This is the default experiment file."
    )
    assert dataset.meta.omnic_accessory_name == "None"
    assert dataset.meta.omnic_experiment_file is None


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


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_spa_ifg_sample_spacing_factor1():
    """SPA sample interferogram: OPD step honors the native sample spacing.

    The header of ``carroucell_samp/2-BaSO4_0.SPA`` records a sample spacing
    of 1.0 together with a reference frequency of 15798.259765625 cm^-1, so
    the optical path difference step must be ``1 / (2 * nu)`` rather than the
    former (2x too large) ``1 / nu``.
    """
    from spectrochempy.core.units import ur

    nu = 15798.259765625  # reference frequency stored in the file (+80)
    nd = scp.read_spa(IRDATA / "carroucell_samp" / "2-BaSO4_0.SPA", return_ifg="sample")
    x = nd.x

    assert x.units == ur("mm")
    assert x.title == "optical path difference"
    assert nd.meta.sample_spacing == 1.0

    step = x._data[1] - x._data[0]
    expected_step = 1.0 / (2.0 * nu) * 10.0  # cm -> mm
    assert step == pytest.approx(expected_step, rel=1e-5)
    # make sure we are not back to the pre-fix (wrong) 1 / nu step
    legacy_step = 1.0 / nu * 10.0
    assert step != pytest.approx(legacy_step, rel=1e-3)

    # origin kept at the interferogram peak
    zpd = int(np.argmax(nd)[-1])
    assert x._data[zpd] == 0.0


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_spa_ifg_sample_spacing_factor2():
    """SPA interferogram with native sample spacing 2.0 keeps the 1/nu step.

    ``interferogram/interfero.SPA`` records a sample spacing of 2.0, so the
    step ``2 / (2 * nu) = 1 / nu`` must remain unchanged.
    """
    from spectrochempy.core.units import ur

    nu = 15798.259765625  # reference frequency stored in the file (+80)
    nd = scp.read_spa(IRDATA / "interferogram" / "interfero.SPA")
    x = nd.x

    assert x.units == ur("mm")
    assert x.title == "optical path difference"
    assert nd.meta.sample_spacing == 2.0

    step = x._data[1] - x._data[0]
    expected_step = 2.0 / (2.0 * nu) * 10.0  # = 1 / nu
    assert step == pytest.approx(expected_step, rel=1e-5)

    # origin kept at the interferogram peak
    zpd = int(np.argmax(nd)[-1])
    assert x._data[zpd] == 0.0

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


def _spa_key_table_bytes(records, suffix=b""):
    content = bytearray(304)
    struct.pack_into("<H", content, 294, len(records))
    for key, position, length in records:
        content.extend(struct.pack("<BBII", key, 0, position, length))
        content.extend(b"\x00" * 6)
    content.extend(suffix)
    return bytes(content)


def test_read_spa_key_table_uses_counted_records():
    from spectrochempy.core.readers.read_omnic import _read_spa_key_table

    records = _read_spa_key_table(
        io.BytesIO(
            _spa_key_table_bytes([(2, 400, 140), (3, 600, 8)], suffix=b"\x00" * 16)
        )
    )

    assert [(record.key, record.position, record.length) for record in records] == [
        (2, 400, 140),
        (3, 600, 8),
    ]


def test_read_spa_key_table_excludes_library_grid():
    from spectrochempy.core.readers.read_omnic import _read_spa_key_table

    grid = b"\x01\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8
    records = _read_spa_key_table(
        io.BytesIO(_spa_key_table_bytes([(2, 400, 140), (3, 600, 8)], suffix=grid * 3))
    )

    assert [record.key for record in records] == [2, 3]


def test_read_spa_key_table_preserves_repeated_keys():
    from spectrochempy.core.readers.read_omnic import _read_spa_key_table

    records = _read_spa_key_table(
        io.BytesIO(_spa_key_table_bytes([(4, 400, 5), (4, 500, 6)]))
    )

    assert [(record.key, record.position, record.length) for record in records] == [
        (4, 400, 5),
        (4, 500, 6),
    ]


def test_read_spa_key_table_rejects_truncated_table():
    from spectrochempy.core.readers.read_omnic import _read_spa_key_table

    content = bytearray(304)
    struct.pack_into("<H", content, 294, 1)
    content.extend(b"\x00" * 15)

    with pytest.raises(ValueError, match="truncated record table"):
        _read_spa_key_table(io.BytesIO(content))


def _synthetic_spa_with_optical_velocity(
    mirror,
    canonical=None,
    *,
    xunits=1,
    reference_frequency=15798.0,
    raman_frequency=0.0,
    timestamp=0,
    library=False,
    scan_points=0,
    peak_position=0,
    sample_scans=0,
    fft_points=0,
    trailing_geometry=0,
    background_scans=0,
    background_gain=0.0,
    aperture=0.0,
    digitizer_bits=0,
    high_pass=0.0,
    low_pass=0.0,
    sample_gain=0.0,
):
    """Build a minimal SPA with optional canonical optical-velocity metadata."""
    content = bytearray(768)
    content[:18] = b"Spectral Data File"
    content[30:42] = b"synthetic.spa"

    records = [(2, 400, 140)]
    if canonical is not None:
        records.append((106, 700, 56))
    if library:
        records.append((0x53, 0, 0))
    payload_position = 756 if canonical is not None else 700
    if library and canonical is None:
        payload_position = 700
    records.append((3, payload_position, 8))
    struct.pack_into("<H", content, 294, len(records))
    struct.pack_into("<I", content, 296, timestamp)
    for offset, (key, position, length) in zip((304, 320, 336), records):
        struct.pack_into("<BBII", content, offset, key, 0, position, length)
    if library:
        content[304 + 16 * len(records)] = 1

    header = 400
    struct.pack_into("<I", content, header + 4, 2)
    content[header + 8] = xunits
    content[header + 12] = 17
    struct.pack_into("<ff", content, header + 16, 4000.0, 3999.0)
    struct.pack_into("<I", content, header + 28, scan_points)
    struct.pack_into("<I", content, header + 32, peak_position)
    struct.pack_into("<I", content, header + 36, sample_scans)
    struct.pack_into("<I", content, header + 44, fft_points)
    struct.pack_into("<I", content, header + 48, trailing_geometry)
    struct.pack_into("<I", content, header + 52, background_scans)
    struct.pack_into("<f", content, header + 56, background_gain)
    struct.pack_into("<I", content, header + 68, 100)
    struct.pack_into("<f", content, header + 80, reference_frequency)
    struct.pack_into("<f", content, header + 84, 1.0)
    struct.pack_into("<f", content, header + 92, aperture)
    struct.pack_into("<f", content, header + 96, raman_frequency)
    struct.pack_into("<f", content, header + 188, mirror)
    if canonical is not None:
        struct.pack_into("<I", content, 700 + 16, digitizer_bits)
        struct.pack_into("<f", content, 700 + 20, high_pass)
        struct.pack_into("<f", content, 700 + 24, low_pass)
        struct.pack_into("<f", content, 700 + 44, sample_gain)
        struct.pack_into("<f", content, 700 + 48, canonical)
    struct.pack_into("<ff", content, payload_position, 1.0, 2.0)
    return bytes(content)


def test_spa_uses_canonical_optical_velocity(tmp_path):
    """The 0x6a value wins over the legacy 0x02 header mirror."""
    path = tmp_path / "canonical.spa"
    path.write_bytes(_synthetic_spa_with_optical_velocity(0.0, 8.8617))

    dataset = scp.read_spa(path)

    assert dataset.meta.optical_velocity == pytest.approx(8.8617)


def test_spa_preserves_matching_optical_velocity_layout(tmp_path):
    """The canonical and mirrored values remain unchanged when they agree."""
    path = tmp_path / "matching.spa"
    path.write_bytes(_synthetic_spa_with_optical_velocity(8.8617, 8.8617))

    dataset = scp.read_spa(path)

    assert dataset.meta.optical_velocity == pytest.approx(8.8617)


def test_spa_falls_back_to_mirror_without_canonical_parameters(tmp_path):
    """The legacy mirror remains supported when no 0x6a record is present."""
    path = tmp_path / "legacy.spa"
    path.write_bytes(_synthetic_spa_with_optical_velocity(8.8617))

    dataset = scp.read_spa(path)

    assert dataset.meta.optical_velocity == pytest.approx(8.8617)


def test_spa_exposes_mature_acquisition_metadata(tmp_path):
    path = tmp_path / "acquisition-metadata.spa"
    path.write_bytes(
        _synthetic_spa_with_optical_velocity(
            8.8617,
            8.8617,
            scan_points=1234,
            peak_position=512,
            sample_scans=8,
            fft_points=4096,
            trailing_geometry=2048,
            background_scans=4,
            background_gain=2.5,
            aperture=75.0,
            digitizer_bits=20,
            high_pass=200.0,
            low_pass=11000.0,
            sample_gain=12.5,
        )
    )

    dataset = scp.read_spa(path)

    assert dataset.meta.scan_points == 1234
    assert dataset.meta.interferogram_peak_position == 512
    assert dataset.meta.sample_scans == 8
    assert dataset.meta.background_scans == 4
    assert dataset.meta.fft_points == 4096
    assert dataset.meta.background_gain == pytest.approx(2.5)
    assert dataset.meta.aperture == pytest.approx(75.0)
    assert dataset.meta.digitizer_bits == 20
    assert dataset.meta.sample_gain == pytest.approx(12.5)
    assert dataset.meta.high_pass_filter == pytest.approx(200.0)
    assert dataset.meta.low_pass_filter == pytest.approx(11000.0)


def test_spa_raman_uses_excitation_and_preserves_reference_frequency(tmp_path):
    from spectrochempy.core.units import ur

    path = tmp_path / "raman.spa"
    path.write_bytes(
        _synthetic_spa_with_optical_velocity(
            8.8617,
            xunits=0x20,
            reference_frequency=15798.2,
            raman_frequency=9395.0,
        )
    )

    dataset = scp.read_spa(path)

    assert dataset.meta.laser_frequency.to(ur("cm^-1")).magnitude == pytest.approx(
        9395.0
    )
    assert dataset.meta.reference_frequency.to(ur("cm^-1")).magnitude == pytest.approx(
        15798.2
    )
    np.testing.assert_allclose(dataset.x.data, [4000.0, 3999.0])


def test_spa_library_timestamp_is_not_promoted(tmp_path):
    path = tmp_path / "library.spa"
    path.write_bytes(
        _synthetic_spa_with_optical_velocity(
            8.8617,
            timestamp=1577962800,
            library=True,
        )
    )

    dataset = scp.read_spa(path)

    assert dataset.acquisition_date is None
    assert dataset.y.title == "spectrum"
    assert dataset.y.labels[0, 0] is None


def test_allow_inconsistent_x_parameter_documented():
    assert "allow_inconsistent_x" in scp.read_spg.__doc__
    assert "return_ifg" not in scp.read_spg.__doc__
    assert "allow_inconsistent_x" in scp.read_omnic.__doc__
    assert "allow_inconsistent_x=True" in scp.read_omnic.__doc__
    assert (
        scp.read_spa.__doc__.count('return_ifg : {None, "sample", "background"}') == 1
    )
    assert "standalone data-points interferogram" in scp.read_spa.__doc__
    assert 'return_ifg="sample"' in scp.read_spa.__doc__
    assert "native sample-spacing" in scp.read_spa.__doc__


def test_decode_experiment_info_block():
    """Decode native fixed-slot subtype-0x79 Experiment Information blocks."""
    from spectrochempy.core.readers.read_omnic import _decode_experiment_info_block

    def _build_block(fields, size=700, subtype=0x79):
        block = bytearray(size)
        block[0] = subtype
        for offset, value in fields.items():
            encoded = value.encode("utf-8") + b"\x00"
            block[offset : offset + len(encoded)] = encoded
        return bytes(block)

    block = _build_block(
        {
            10: r"C:\MYDOCU~1\omnic\Param\CARROU~2.EXP",
            90: "CARROU~2.EXP",
            154: "Default experiment for iS50 Main Sample Compartment",
            413: "iS50 Main Sample",
        }
    )
    result = _decode_experiment_info_block(block)
    assert result is not None
    assert result["experiment_path"] == r"C:\MYDOCU~1\omnic\Param\CARROU~2.EXP"
    assert result["accessory_name"] == "iS50 Main Sample"
    assert result["experiment_title"] == "CARROU~2.EXP"
    assert (
        result["experiment_description"]
        == "Default experiment for iS50 Main Sample Compartment"
    )

    # Unsupported subtype is ignored.
    bad = bytearray(block)
    bad[0] = 0x9E
    assert _decode_experiment_info_block(bytes(bad)) is None

    # Empty middle slot does not compact later fixed fields.
    block = _build_block({10: "path", 154: "description", 413: "accessory"})
    result = _decode_experiment_info_block(block)
    assert result is not None
    assert result["experiment_path"] == "path"
    assert result["experiment_description"] == "description"
    assert result["accessory_name"] == "accessory"
    assert "experiment_title" not in result

    # Short blocks are safely bounded and return available fields only.
    result = _decode_experiment_info_block(_build_block({10: "short"}, size=40))
    assert result == {"experiment_path": "short"}


def _synthetic_spa_with_experiment_blocks(blocks):
    content = bytearray(5000)
    content[:18] = b"Spectral Data File"
    content[30:42] = b"synthetic.spa"
    records = [(2, 400, 140)]
    for index, block in enumerate(blocks):
        position = 700 + index * 800
        records.append((130, position, len(block)))
        content[position : position + len(block)] = block
    records.append((3, 4000, 8))
    struct.pack_into("<H", content, 294, len(records))
    for offset, (key, position, length) in zip(
        range(304, 304 + 16 * len(records), 16), records
    ):
        struct.pack_into("<BBII", content, offset, key, 0, position, length)
    header = 400
    struct.pack_into("<I", content, header + 4, 2)
    content[header + 8] = 1
    content[header + 12] = 17
    struct.pack_into("<ff", content, header + 16, 4000.0, 3999.0)
    struct.pack_into("<I", content, header + 68, 100)
    struct.pack_into("<f", content, header + 80, 15798.0)
    struct.pack_into("<f", content, header + 84, 1.0)
    struct.pack_into("<ff", content, 4000, 1.0, 2.0)
    return bytes(content)


def test_spa_uses_later_79_after_unsupported_82(tmp_path):
    path = tmp_path / "experiment-info.spa"
    unsupported = bytes([0x9D]) + b"\x00" * 699
    supported = bytearray(700)
    supported[0] = 0x79
    supported[10:18] = b"native\x00\x00"
    supported[90:99] = b"title\x00\x00\x00"
    path.write_bytes(
        _synthetic_spa_with_experiment_blocks([unsupported, bytes(supported)])
    )

    dataset = scp.read_spa(path)

    assert dataset.meta.omnic_experiment_path == "native"
    assert dataset.meta.omnic_experiment_title == "title"


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
    # `collection_length` is the total series time derived from the series last
    # time (+1006), not from the first time (+1002) kept in `time_min`.
    assert info["time_min"] != info["lasty"]
    assert info["time_min"] != info["firsty"]
    assert info["collection_length"] == np.float32(info["lasty"] * 60)


@pytest.mark.parametrize(
    "name",
    [
        "rapid_scan.srs",
        "rapid_scan_reprocessed.srs",
        "GC_Demo.srs",
        "high_speed.srs",
        "TGA_demo.srs",
    ],
)
@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_collection_length_is_total_series_time(name):
    """The SRS `meta.collection_length` must be the total series collection
    time — the series last time (+1006, minutes) converted to seconds —
    matching the OMNIC "Total collection time", not the first-time value
    (+1002) converted to seconds.

    Controlled SRS/TXT validation established +1006 as the OMNIC total
    collection time across the five public fixtures; this test pins the
    metadata value to `+1006 * 60` and keeps the time axis anchored at the
    series first time (+1002).
    """
    from spectrochempy.core.units import ur

    path = IRDATA / "omnic_series" / name
    info = _srs_header(path)
    nd = scp.read_srs(path)

    # Total series collection time, derived from the series last time (+1006).
    assert info["collection_length"] == np.float32(info["lasty"] * 60)
    assert nd.meta.collection_length.magnitude == np.float64(info["collection_length"])
    assert nd.meta.collection_length.units == ur("s")
    # ...and strictly larger than the historical (wrong) first-time value.
    assert nd.meta.collection_length.magnitude > np.float32(info["time_min"]) * 60

    # Y-axis start still comes from the series first time (+1002) and the axis
    # still ends at the series last time (+1006). The axis is rounded to 3
    # decimals (see `_read_srs`), hence the 0.5e-3 tolerance.
    assert nd.y.data[0] == pytest.approx(
        np.around(float(info["time_min"]), 3), abs=0.5e-3
    )
    assert nd.y.data[-1] == pytest.approx(
        np.around(float(info["lasty"]), 3), abs=0.5e-3
    )


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
def test_read_srs_interferogram_not_reversed(monkeypatch):
    """`rapid_scan.srs` interferograms must remain on the interferogram path and
    must not be treated as spectral data (no reversal, ascending OPD axis, data
    kept in raw storage order)."""
    from spectrochempy.core.readers import read_omnic

    nd = scp.read_srs(IRDATA / "omnic_series" / "rapid_scan.srs")
    assert nd.meta.interferogram is True
    x = np.asarray(nd.x)
    # Interferogram axis is optical path difference, exposed ascending, not a
    # descending spectral wavenumber axis.
    assert nd.x.title == "optical path difference"
    assert x[0] < x[-1]
    assert nd.shape == (643, 4160)

    # Data is kept in raw storage order (not spectral-normalized): reversing it
    # reproduces the same file read as if it were a (reversed) spectral record.
    real_read_header = read_omnic._read_header

    def _forced_spectral(fid, pos, is_first_spectrum=True):
        info = real_read_header(fid, pos, is_first_spectrum=is_first_spectrum)
        info["xtitle"] = "wavenumbers"
        info["xunits"] = "cm^-1"
        return info

    monkeypatch.setattr(read_omnic, "_read_header", _forced_spectral)
    spectral = read_omnic._read_srs(
        NDDataset(), IRDATA / "omnic_series" / "rapid_scan.srs"
    )
    np.testing.assert_allclose(np.asarray(nd.data)[:, ::-1], np.asarray(spectral.data))


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
    """`reverse_x` is a deprecated no-op: supplying the keyword (with any value,
    `True` or `False`) emits a `DeprecationWarning` and returns the same
    (correct, automatically normalized) result as the default, so users of the
    #858 workaround keep getting correct data without a double reversal."""
    path = IRDATA / "omnic_series" / "GC_Demo.srs"
    default = scp.read_srs(path)
    with pytest.warns(DeprecationWarning):
        legacy_true = scp.read_srs(path, reverse_x=True)
    with pytest.warns(DeprecationWarning):
        legacy_false = scp.read_srs(path, reverse_x=False)
    np.testing.assert_allclose(np.asarray(legacy_true.data), np.asarray(default.data))
    np.testing.assert_allclose(np.asarray(legacy_true.x), np.asarray(default.x))
    np.testing.assert_allclose(np.asarray(legacy_false.data), np.asarray(default.data))
    np.testing.assert_allclose(np.asarray(legacy_false.x), np.asarray(default.x))


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_no_reverse_x_keyword_no_deprecation():
    """Omitting the `reverse_x` keyword entirely must not emit a
    `DeprecationWarning`."""
    import warnings

    path = IRDATA / "omnic_series" / "GC_Demo.srs"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scp.read_srs(path)
    assert not any(issubclass(warning.category, DeprecationWarning) for warning in w)


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_and_read_spa_share_descending_convention():
    """A corrected `read_srs` must expose spectral X with the same
    descending-wavenumber, data-matched convention as `read_spa`."""
    srs = scp.read_srs(IRDATA / "omnic_series" / "GC_Demo.srs")
    spa = scp.read_spa(IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA")
    assert np.asarray(srs.x)[0] > np.asarray(srs.x)[-1]
    assert np.asarray(spa.x)[0] > np.asarray(spa.x)[-1]


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_unknown_xunits_not_interferogram(monkeypatch):
    """A record whose X-unit code is unknown (`xunits is None` but not an
    explicit data-points axis) must not be classified as an interferogram, must
    not be spectral-normalized, and must emit a warning instead of silently
    treating it as either an interferogram or a spectral record."""
    import warnings

    from spectrochempy.core.readers import read_omnic

    real_read_header = read_omnic._read_header

    def _forced(xtitle, xunits):
        def _wrap(fid, pos, is_first_spectrum=True):
            info = real_read_header(fid, pos, is_first_spectrum=is_first_spectrum)
            info["xtitle"] = xtitle
            info["xunits"] = xunits
            return info

        return _wrap

    path = IRDATA / "omnic_series" / "GC_Demo.srs"

    # Reference: the same file read as a spectral record (normalized to
    # descending wavenumber).
    monkeypatch.setattr(read_omnic, "_read_header", _forced("wavenumbers", "cm^-1"))
    spectral = read_omnic._read_srs(NDDataset(), path)

    # Unknown X-axis type: xunits is None but the axis is not data points.
    monkeypatch.setattr(read_omnic, "_read_header", _forced("xaxis", None))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        unknown = read_omnic._read_srs(NDDataset(), path)

    # An explicit warning is emitted about the unrecognized X axis.
    assert any("X axis is not recognized" in str(x.message) for x in w)
    # Not classified as an interferogram (no interferogram metadata).
    assert getattr(unknown.meta, "interferogram", None) is None
    assert unknown.x.title == "xaxis"
    # Left in raw storage orientation: the data is NOT spectral-normalized
    # (it is the reverse of the normalized spectral read).
    np.testing.assert_allclose(
        np.asarray(unknown.data)[:, ::-1], np.asarray(spectral.data)
    )


# ---------------------------------------------------------------------------
# SRS native series-level acquisition date and derived per-spectrum datetimes
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_skip_if_no_testdata")
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("rapid_scan.srs", datetime(2020, 11, 24, 10, 21, 29, tzinfo=UTC)),
        ("high_speed.srs", datetime(2023, 10, 22, 7, 2, 13, tzinfo=UTC)),
        ("TGA_demo.srs", datetime(2006, 2, 24, 18, 34, 57, tzinfo=UTC)),
    ],
)
def test_read_srs_native_acquisition_date(name, expected):
    """SRS series with a valid native timestamp expose the OMNIC `Collected`
    instant through the same `acquisition_date` convention as SPA: the stored
    value is a timezone-aware Python datetime in UTC that matches the native
    UInt32 decoding (OMNIC epoch 1899-12-31 UTC)."""
    nd = scp.read_srs(IRDATA / "omnic_series" / name)

    # Stored value: timezone-aware Python datetime in UTC.
    assert isinstance(nd._acquisition_date, datetime)
    assert nd._acquisition_date.tzinfo is not None
    assert nd._acquisition_date.utcoffset() == timedelta(0)
    assert nd._acquisition_date == expected

    # Public property: same instant, ISO-string rendering in the dataset
    # timezone (the SPA convention, which renders in `_timezone`).
    assert nd.acquisition_date is not None
    rendered = datetime.fromisoformat(nd.acquisition_date.replace(" ", "T"))
    assert rendered == expected.astimezone(rendered.tzinfo)


@pytest.mark.usefixtures("_skip_if_no_testdata")
@pytest.mark.parametrize("name", ["rapid_scan_reprocessed.srs", "GC_Demo.srs"])
def test_read_srs_absent_or_zero_timestamp_leaves_date_unset(name):
    """A zeroed native field (reprocessed RapidScan) or a variant without the
    field (GC: offset 296 holds unrelated bytes) must leave
    `dataset.acquisition_date` unset -- never the OMNIC epoch."""
    nd = scp.read_srs(IRDATA / "omnic_series" / name)

    assert nd._acquisition_date is None
    assert nd.acquisition_date is None


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_y_coordinate_invariant_with_datetime_labels():
    """Adding absolute datetime labels must not alter the numeric relative Y
    coordinate: same values, same minutes units, same endpoint anchors.

    The reader stores ``np.around(linspace(...), 3)`` and ``Coord.data``
    returns the display-linearized variant (sigdigits=4), which may differ
    by up to ~0.001 on interior points; endpoints are exact.
    """
    path = IRDATA / "omnic_series" / "TGA_demo.srs"
    info = _srs_header(path)
    nd = scp.read_srs(path)

    y = nd.y.data
    assert nd.y.units == "minute"
    assert len(y) == info["ny"]
    expected = np.around(np.linspace(info["time_min"], info["lasty"], info["ny"]), 3)
    np.testing.assert_allclose(y, expected, atol=1.5e-3)
    assert y[0] == pytest.approx(np.around(info["time_min"], 3), abs=0.5e-3)
    assert y[-1] == pytest.approx(np.around(info["lasty"], 3), abs=0.5e-3)


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_datetime_labels_follow_full_precision_native_formula():
    """Per-spectrum datetime labels are derived with

        datetime[i] = acquisition_date + timedelta(minutes=time_min + i*step)

    in full precision from the native +1002/+1010 series fields (not from the
    rounded 3-decimal Y coordinate), and their whole-second truncation matches
    OMNIC's exported SPA serialization.

    `TGA_demo.srs` is the public fixture of the controlled TGA series whose
    full 485-spectrum OMNIC SPA export established Model A (export oracle):
    the exported native SPA stamps for records 0/415/484 are 2006-02-24
    18:35:01 / 19:09:23 / 19:15:05 UTC.
    """
    path = IRDATA / "omnic_series" / "TGA_demo.srs"
    info = _srs_header(path)
    nd = scp.read_srs(path)

    collected = datetime(2006, 2, 24, 18, 34, 57, tzinfo=UTC)
    time_min = float(info["time_min"])
    step = float(info["firsty"])

    labels = nd.y.labels
    assert labels.shape == (info["ny"], 2)
    # Column 0: absolute datetime objects; column 1: unchanged spectrum names.
    assert all(isinstance(labels[i, 0], datetime) for i in (0, info["ny"] // 2, -1))
    assert all(isinstance(labels[i, 1], str) for i in (0, info["ny"] // 2, -1))
    assert labels[0, 1].startswith("Linked spectrum at")

    # Full-precision native formula, first/interior/last records.
    for i in (0, info["ny"] // 2, info["ny"] - 1):
        expected_dt = collected + timedelta(minutes=time_min + i * step)
        assert labels[i, 0] == expected_dt
        # Sub-second precision is preserved (no artificial whole-second
        # truncation mimicking the export serialization).
        assert labels[i, 0].microsecond == expected_dt.microsecond

    # Evidence-pinned whole-second exported SPA instants (records 0/415/484).
    exported = {
        0: datetime(2006, 2, 24, 18, 35, 1, tzinfo=UTC),
        415: datetime(2006, 2, 24, 19, 9, 23, tzinfo=UTC),
        484: datetime(2006, 2, 24, 19, 15, 5, tzinfo=UTC),
    }
    for i, stamp in exported.items():
        label = labels[i, 0]
        assert label.replace(microsecond=0) == stamp


@pytest.mark.usefixtures("_skip_if_no_testdata")
def test_read_srs_undated_labels_keep_single_column_names():
    """Variants without a native absolute anchor keep the current single-column
    Y labels exactly as before (spectrum names only, no datetime column)."""
    path = IRDATA / "omnic_series" / "GC_Demo.srs"
    info = _srs_header(path)
    nd = scp.read_srs(path)

    labels = nd.y.labels
    assert labels.shape == (info["ny"],)
    assert labels[0] == "Linked spectrum at 0.025 min."
    assert labels[1] == "Linked spectrum at 0.051 min."
    assert all(isinstance(label, str) for label in labels)
