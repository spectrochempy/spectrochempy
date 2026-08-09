# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import warnings
from datetime import datetime

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.units import ur
from spectrochempy.utils.datetimeutils import UTC


def _dataset_without_y():
    x = scp.Coord(np.linspace(4000.0, 1000.0, 6), units="cm^-1", title="wavenumber")
    ds = scp.NDDataset(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), coordset=[x])
    ds.name = "no_y"
    return ds


def _dataset_with_y():
    x = scp.Coord(np.linspace(4000.0, 1000.0, 6), units="cm^-1", title="wavenumber")
    y = scp.Coord([0.0])
    return scp.NDDataset(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(1, 6),
        coordset=[y, x],
        units="absorbance",
        name="valid_1d",
    )


def _single_spectrum_dataset(*, x_units="cm^-1", y_units="absorbance"):
    x = scp.Coord(np.linspace(4000.0, 1000.0, 6), units=x_units, title="x")
    y = scp.Coord([0.0])
    return scp.NDDataset(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(1, 6),
        coordset=[y, x],
        units=y_units,
        name="single_policy",
    )


def _complex_dataset_with_y():
    x = scp.Coord([0.0, 1.0])
    y = scp.Coord([0.0, 10.0])
    return scp.NDDataset(
        np.array([[1 + 2j, 2 + 1j], [3 + 0j, 4 + 1j]]),
        coordset=[y, x],
        name="complex",
    )


def _linked_dataset():
    x = scp.Coord([4000.0, 3999.0, 3998.0, 3997.0], units="cm^-1", title="wavenumber")
    y = scp.Coord([0.0, 1.0, 2.0])
    y.labels = np.array(
        [
            [datetime(2024, 1, 1, 12, 0, tzinfo=UTC), "spec_a"],
            [datetime(2024, 1, 1, 12, 1, tzinfo=UTC), "spec_b"],
            [datetime(2024, 1, 1, 12, 2, tzinfo=UTC), "spec_c"],
        ],
        dtype=object,
    )
    ds = scp.NDDataset(
        np.array(
            [
                [1.0, 5.0, np.nan, 2.0],
                [7.0, -3.0, 4.0, 6.0],
                [0.5, 8.0, -1.0, 9.0],
            ],
        ),
        coordset=[y, x],
        units="absorbance",
        name="linked_extrema",
    )
    ds.mask = np.array(
        [
            [False, False, False, False],
            [False, False, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    return ds


def _parse_jcamp_blocks(text):
    blocks = []
    current = None
    for line in text.splitlines():
        if line.startswith("##TITLE="):
            title = line.split("=", 1)[1]
            if current and current.get("xydata"):
                blocks.append(current)
            current = {"TITLE": title, "xydata": []}
            continue
        if current is None:
            continue
        if line.startswith("##") and "=" in line:
            key, value = line[2:].split("=", 1)
            current[key] = value
            continue
        if line and not line.startswith("##"):
            current["xydata"].append(line)
    if current and current.get("xydata"):
        blocks.append(current)
    return blocks


def test_write_jcamp_rejects_dataset_without_y_coordinate(tmp_path):
    dataset = _dataset_without_y()
    target = tmp_path / "no_y.jdx"
    original_filename = dataset.filename

    with pytest.raises(ValueError, match="`y` coordinate"):
        dataset.write_jcamp(target, confirm=False)

    assert not target.exists()
    assert dataset.filename == original_filename


def test_write_jcamp_rejects_dataset_without_y_keeps_existing_file(tmp_path):
    dataset = _dataset_without_y()
    target = tmp_path / "no_y.jdx"
    target.write_text("SENTINEL")
    original_filename = dataset.filename

    with pytest.raises(ValueError, match="`y` coordinate"):
        dataset.write_jcamp(target, confirm=False, overwrite=True)

    assert target.read_text() == "SENTINEL"
    assert dataset.filename == original_filename


def test_write_jcamp_rejects_complex_data(tmp_path):
    dataset = _complex_dataset_with_y()
    target = tmp_path / "complex.jdx"
    original_filename = dataset.filename

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=np.exceptions.ComplexWarning)
        with pytest.raises(TypeError, match="does not support complex"):
            dataset.write_jcamp(target, confirm=False)

    assert not target.exists()
    assert dataset.filename == original_filename


def test_write_jcamp_rejects_complex_data_keeps_existing_file(tmp_path):
    dataset = _complex_dataset_with_y()
    target = tmp_path / "complex.jdx"
    target.write_text("SENTINEL")
    original_filename = dataset.filename

    with pytest.raises(TypeError, match="does not support complex"):
        dataset.write_jcamp(target, confirm=False, overwrite=True)

    assert target.read_text() == "SENTINEL"
    assert dataset.filename == original_filename


def test_write_jcamp_generic_dispatch_applies_validation(tmp_path):
    no_y = _dataset_without_y()
    target = tmp_path / "dispatch.jdx"

    with pytest.raises(ValueError, match="`y` coordinate"):
        scp.write(no_y, target, confirm=False)

    assert not target.exists()

    complex_ds = _complex_dataset_with_y()
    target = tmp_path / "dispatch_complex.jdx"

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=np.exceptions.ComplexWarning)
        with pytest.raises(TypeError, match="does not support complex"):
            scp.write(complex_ds, target, confirm=False)

    assert not target.exists()


def test_write_jcamp_valid_dataset_round_trip(tmp_path):
    dataset = _dataset_with_y()

    path = dataset.write_jcamp(tmp_path / "valid.jdx", confirm=False)

    assert path.exists()
    assert path.stat().st_size > 0
    text = path.read_text()
    assert "##JCAMP-DX=5.01" in text
    assert "##DATA TYPE=INFRARED SPECTRUM" in text
    assert "##XYDATA=(X++(Y..Y))" in text
    assert "##FIRSTY=1.000000" in text
    assert "##LASTY=6.000000" in text
    assert "##MAXY=6.000000" in text
    assert "##MINY=1.000000" in text
    assert "##XUNITS=1/CM" in text
    assert "##YUNITS=ABSORBANCE" in text

    back = scp.read_jcamp(path)
    assert back.shape == (1, 6)
    assert np.allclose(back.data, dataset.data)


@pytest.mark.parametrize(
    ("x_units", "expected_token", "output_stem", "expected_readback_unit"),
    [
        ("cm^-1", "1/CM", "cm_inv", "cm^-1"),
        ("1/cm", "1/CM", "one_per_cm", "cm^-1"),
        ("um", "MICROMETERS", "micrometers", "um"),
        ("nm", "NANOMETERS", "nanometers", "nm"),
    ],
)
def test_write_jcamp_emits_truthful_xunits_for_exact_scale_inputs(
    tmp_path, x_units, expected_token, output_stem, expected_readback_unit
):
    dataset = _single_spectrum_dataset(x_units=x_units, y_units="absorbance")

    path = dataset.write_jcamp(tmp_path / f"{output_stem}.jdx", confirm=False)
    text = path.read_text()

    assert f"##XUNITS={expected_token}" in text
    back = scp.read_jcamp(path)
    assert np.allclose(back.x.data, dataset.x.data)
    assert back.x.units == ur.Unit(expected_readback_unit)


def test_write_jcamp_emits_arbitrary_units_for_unitless_x(tmp_path):
    dataset = _single_spectrum_dataset(x_units=None, y_units="absorbance")

    path = dataset.write_jcamp(tmp_path / "unitless_x.jdx", confirm=False)
    text = path.read_text()

    assert "##XUNITS=ARBITRARY UNITS" in text
    back = scp.read_jcamp(path)
    assert back.x.units is None
    assert back.x.title == "arbitrary unit"
    assert np.allclose(back.x.data, dataset.x.data)


@pytest.mark.parametrize(
    ("y_units", "expected_token", "expected_readback_unit", "expected_title"),
    [
        ("absorbance", "ABSORBANCE", "absorbance", "absorbance"),
        ("transmittance", "TRANSMITTANCE", "transmittance", "transmittance"),
    ],
)
def test_write_jcamp_emits_truthful_yunits_for_supported_inputs(
    tmp_path, y_units, expected_token, expected_readback_unit, expected_title
):
    dataset = _single_spectrum_dataset(x_units="cm^-1", y_units=y_units)

    path = dataset.write_jcamp(tmp_path / f"{expected_token}.jdx", confirm=False)
    text = path.read_text()

    assert f"##YUNITS={expected_token}" in text
    back = scp.read_jcamp(path)
    assert np.allclose(back.data, dataset.data)
    assert back.units == expected_readback_unit
    assert back.title == expected_title


def test_write_jcamp_emits_arbitrary_units_for_unitless_y(tmp_path):
    dataset = _single_spectrum_dataset(x_units="cm^-1", y_units=None)

    path = dataset.write_jcamp(tmp_path / "unitless_y.jdx", confirm=False)
    text = path.read_text()

    assert "##YUNITS=ARBITRARY UNITS" in text
    back = scp.read_jcamp(path)
    assert np.allclose(back.data, dataset.data)
    assert back.units is None
    assert back.title == "<untitled>"


@pytest.mark.parametrize(("x_units"), ["m^-1", "Hz", "dimensionless"])
def test_write_jcamp_rejects_unsupported_x_units_before_file_creation(
    tmp_path, x_units
):
    dataset = _single_spectrum_dataset(x_units=x_units, y_units="absorbance")
    target = tmp_path / "bad_x.jdx"
    original_filename = dataset.filename

    with pytest.raises(ValueError, match="exact numeric scale"):
        dataset.write_jcamp(target, confirm=False)

    assert not target.exists()
    assert dataset.filename == original_filename


@pytest.mark.parametrize(("x_units"), ["m^-1", "Hz", "dimensionless"])
def test_write_jcamp_rejects_unsupported_x_units_keeps_existing_file(tmp_path, x_units):
    dataset = _single_spectrum_dataset(x_units=x_units, y_units="absorbance")
    target = tmp_path / "bad_x_existing.jdx"
    target.write_text("SENTINEL")
    original_filename = dataset.filename

    with pytest.raises(ValueError, match="exact numeric scale"):
        dataset.write_jcamp(target, confirm=False, overwrite=True)

    assert target.read_text() == "SENTINEL"
    assert dataset.filename == original_filename


@pytest.mark.parametrize(
    ("y_units"), ["count", "dimensionless", "absolute_transmittance"]
)
def test_write_jcamp_rejects_unsupported_y_units_before_file_creation(
    tmp_path, y_units
):
    dataset = _single_spectrum_dataset(x_units="cm^-1", y_units=y_units)
    target = tmp_path / "bad_y.jdx"
    original_filename = dataset.filename

    with pytest.raises(ValueError, match="only supports y units"):
        dataset.write_jcamp(target, confirm=False)

    assert not target.exists()
    assert dataset.filename == original_filename


@pytest.mark.parametrize(
    ("y_units"), ["count", "dimensionless", "absolute_transmittance"]
)
def test_write_jcamp_rejects_unsupported_y_units_keeps_existing_file(tmp_path, y_units):
    dataset = _single_spectrum_dataset(x_units="cm^-1", y_units=y_units)
    target = tmp_path / "bad_y_existing.jdx"
    target.write_text("SENTINEL")
    original_filename = dataset.filename

    with pytest.raises(ValueError, match="only supports y units"):
        dataset.write_jcamp(target, confirm=False, overwrite=True)

    assert target.read_text() == "SENTINEL"
    assert dataset.filename == original_filename


def test_write_jcamp_generic_dispatch_applies_unit_policy(tmp_path):
    x_none_y_none = _single_spectrum_dataset(x_units=None, y_units=None)

    target = scp.write(x_none_y_none, tmp_path / "generic_policy.jdx", confirm=False)
    text = target.read_text()
    assert "##XUNITS=ARBITRARY UNITS" in text
    assert "##YUNITS=ARBITRARY UNITS" in text

    rejected = _single_spectrum_dataset(x_units="m^-1", y_units="absorbance")
    bad_target = tmp_path / "generic_bad.jdx"

    with pytest.raises(ValueError, match="exact numeric scale"):
        scp.write(rejected, bad_target, confirm=False)

    assert not bad_target.exists()


def test_write_jcamp_link_extrema_are_computed_per_spectrum(tmp_path):
    dataset = _linked_dataset()

    specialized = dataset.write_jcamp(
        tmp_path / "linked_specialized.jdx", confirm=False
    )
    generic = scp.write(dataset, tmp_path / "linked_generic.jdx", confirm=False)

    specialized_text = specialized.read_text()
    generic_text = generic.read_text()
    specialized_blocks = _parse_jcamp_blocks(specialized_text)
    generic_blocks = _parse_jcamp_blocks(generic_text)

    assert [block["TITLE"] for block in specialized_blocks] == [
        "spec_a",
        "spec_b",
        "spec_c",
    ]
    assert [block["TITLE"] for block in generic_blocks] == [
        "spec_a",
        "spec_b",
        "spec_c",
    ]

    expected = [
        {
            "FIRSTY": "1.000000",
            "LASTY": "2.000000",
            "MAXY": "5.000000",
            "MINY": "1.000000",
        },
        {
            "FIRSTY": "7.000000",
            "LASTY": "6.000000",
            "MAXY": "7.000000",
            "MINY": "-3.000000",
        },
        {
            "FIRSTY": "0.500000",
            "LASTY": "9.000000",
            "MAXY": "9.000000",
            "MINY": "-1.000000",
        },
    ]

    assert len(specialized_blocks) == 3
    assert len(generic_blocks) == 3

    for block, expected_values in zip(specialized_blocks, expected, strict=True):
        assert {key: block[key] for key in expected_values} == expected_values
        assert block["XUNITS"] == "1/CM"
        assert block["YUNITS"] == "ABSORBANCE"

    for block, expected_values in zip(generic_blocks, expected, strict=True):
        assert {key: block[key] for key in expected_values} == expected_values

    assert [block["FIRSTY"] for block in specialized_blocks] != ["1.000000"] * 3
    assert [block["LASTY"] for block in specialized_blocks] != ["2.000000"] * 3
    assert [block["MAXY"] for block in specialized_blocks] != ["9.000000"] * 3
    assert [block["MINY"] for block in specialized_blocks] != ["-3.000000"] * 3

    for specialized_block, generic_block in zip(
        specialized_blocks, generic_blocks, strict=True
    ):
        assert specialized_block["xydata"] == generic_block["xydata"]

    back = scp.read_jcamp(specialized)
    expected_data = np.array(
        [
            [1.0, 5.0, np.nan, 2.0],
            [7.0, -3.0, np.nan, 6.0],
            [0.5, 8.0, -1.0, 9.0],
        ],
    )
    np.testing.assert_allclose(
        np.nan_to_num(back.data, nan=0.0),
        np.nan_to_num(expected_data, nan=0.0),
    )
    assert np.array_equal(np.isnan(back.data), np.isnan(expected_data))


def test_write_jcamp_link_uses_same_unit_policy_as_singletons(tmp_path):
    x = scp.Coord([1.0, 2.0, 3.0, 4.0], units=None, title="x")
    y = scp.Coord([0.0, 1.0])
    y.labels = np.array(
        [
            [datetime(2024, 1, 1, 12, 0, tzinfo=UTC), "spec_a"],
            [datetime(2024, 1, 1, 12, 1, tzinfo=UTC), "spec_b"],
        ],
        dtype=object,
    )
    dataset = scp.NDDataset(
        np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]),
        coordset=[y, x],
        units=None,
        name="linked_policy",
    )

    path = dataset.write_jcamp(tmp_path / "linked_policy.jdx", confirm=False)
    blocks = _parse_jcamp_blocks(path.read_text())

    assert len(blocks) == 2
    for block in blocks:
        assert block["XUNITS"] == "ARBITRARY UNITS"
        assert block["YUNITS"] == "ARBITRARY UNITS"
