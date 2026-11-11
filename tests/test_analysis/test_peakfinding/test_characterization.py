import json

import numpy as np
import pytest

import spectrochempy as scp


@pytest.fixture
def spectrum():
    data = np.zeros((7, 11))
    data[2, 2:5] = [10, 6, 8]
    data[4, 7:10] = [9, 12, 3]
    data[3, 8] = 9
    data[5, 8] = 3
    return scp.NDDataset(
        data,
        units="V",
        coordset=[
            scp.Coord([8, 6, 4, 2, 0, -2, -4], units="Hz"),
            scp.Coord([0, 1, 2, 4, 5, 7, 9, 12, 15, 19, 22], units="Hz"),
        ],
    )


def detect(spectrum, height=5, dims=("y", "x")):
    return spectrum.find_peaks(dims=dims, height=height, as_result=True)


def characterize(spectrum, peaks, x=3):
    return scp.characterize_peaks_2d(
        spectrum, peaks, threshold=4 * scp.ur.V, radii={"y": 1, "x": x}
    )


def test_reference_and_provenance(spectrum):
    peaks = detect(spectrum)
    original = peaks.table.to_dict()
    original_data = spectrum.data.copy()
    result = characterize(spectrum, peaks, x=0.003 * scp.ur.kHz)
    assert isinstance(result, scp.PeakCharacterizationResult)
    assert [group["peak_indices"] for group in result.groups] == [(0, 1), (2,)]
    assert [row["region_label"] for row in result.rows] == [1, 1, 2]
    assert [row["sample_height"].magnitude for row in result.rows] == [10, 8, 12]
    assert result.rows[0]["width_status"] == {
        "y": "shared_region",
        "x": "shared_region",
    }
    assert result.rows[1]["widths"] == {"y": None, "x": None}
    widths = result.rows[2]["widths"]
    assert widths["y"].to("Hz").magnitude == pytest.approx(4)
    assert widths["x"].to("Hz").magnitude == pytest.approx(20 / 3)
    assert peaks.table.to_dict() == original
    np.testing.assert_array_equal(spectrum.data, original_data)
    assert result.history_entries[-1]["operation"] == "characterize_peaks_2d"
    assert (
        result.history_entries[-1]["parameters"]["radii"]["x"]["effective"]["value"]
        == 3
    )
    json.dumps(result.history_entries[-1]["parameters"])


def test_subradius_splits_shared_region(spectrum):
    result = characterize(spectrum, detect(spectrum, dims=("x", "y")), x=2.99)
    assert [group["peak_indices"] for group in result.groups] == [(0,), (1,), (2,)]
    assert result.rows[0]["width_status"]["x"] == "shared_region"


def test_mask_breaks_bridge_and_truncates_one_cut(spectrum):
    mask = np.zeros(spectrum.shape, dtype=bool)
    mask[2, 3] = True
    mask[4, 9] = True
    spectrum.mask = mask
    result = characterize(spectrum, detect(spectrum), x=20)
    assert len(result.groups) == 3
    assert result.rows[2]["width_status"] == {"y": "available", "x": "invalid_sample"}


def test_transitive_groups(spectrum):
    spectrum.data[2, 5:7] = [6, 9]
    result = characterize(spectrum, detect(spectrum), x=4)
    assert [group["peak_indices"] for group in result.groups] == [(0, 1, 2), (3,)]


def test_empty_and_stale_result(spectrum):
    empty = spectrum.characterize_peaks_2d(
        detect(spectrum, height=20), threshold=4, radii={"y": 1, "x": 3}
    )
    assert empty.rows == empty.groups == ()
    peaks = detect(spectrum)
    spectrum.data[2, 2] = 11
    with pytest.raises(ValueError, match="height does not match"):
        characterize(spectrum, peaks)


def test_threshold_and_radius_errors(spectrum):
    peaks = detect(spectrum)
    with pytest.raises(ValueError, match="below every"):
        spectrum.characterize_peaks_2d(peaks, threshold=8, radii={"y": 1, "x": 3})
    with pytest.raises(ValueError, match="strictly positive"):
        characterize(spectrum, peaks, x=0)


def test_native_height_change_below_float64_precision():
    data = np.zeros((3, 3), dtype=np.int64)
    data[1, 1] = 2**53 + 1
    spectrum = scp.NDDataset(
        data, dtype=data.dtype, coordset=[scp.Coord([0, 1, 2]), scp.Coord([0, 1, 2])]
    )
    peaks = detect(spectrum, dims=("x", "y"))
    kwargs = {"threshold": 4, "radii": {"x": 1, "y": 1}}
    result = spectrum.characterize_peaks_2d(peaks, **kwargs)
    assert result.rows[0]["sample_height"] == 2**53 + 1
    spectrum.data[1, 1] = 2**53
    with pytest.raises(ValueError, match="native height does not match"):
        spectrum.characterize_peaks_2d(peaks, **kwargs)


def test_radius_rejects_nonlinear_spectroscopy_conversion(spectrum):
    spectrum.x.units = "nm"
    peaks = detect(spectrum)
    with pytest.raises(ValueError, match="same dimensionality"):
        characterize(spectrum, peaks, x=100 * scp.ur("cm^-1"))


def test_radius_rejects_logarithmic_interval(spectrum):
    spectrum.x = scp.Coord(spectrum.x.data, units="percent")
    peaks = detect(spectrum)
    with pytest.raises(ValueError, match="non-logarithmic"):
        characterize(spectrum, peaks, x=scp.ur.Quantity(1, "dB"))


def test_affine_axis_uses_interval_units(spectrum):
    spectrum.x = scp.Coord(spectrum.x.data, units="degC")
    peaks = spectrum.find_peaks(
        dims=("y", "x"), height=5, as_result=True, use_coord=False
    )
    result = characterize(spectrum, peaks, x=scp.ur.Quantity(6, "delta_degF"))
    assert result.groups[0]["peak_indices"] == (0, 1)
    assert result.rows[2]["widths"]["x"].to("delta_degC").magnitude == pytest.approx(
        20 / 3
    )
