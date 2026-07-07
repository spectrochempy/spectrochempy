import os

import numpy as np
import pytest

from spectrochempy.analysis.peakfinding.peakfinding import PeakFindingResult
from spectrochempy.analysis.peakfinding.peakfinding import PeakTable
from spectrochempy.analysis.peakfinding.peakfinding import find_peaks
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur
from spectrochempy.utils import docutils as chd


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    os.environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_findpeaks_docstrings():
    import spectrochempy as scp

    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.peakfinding.peakfinding"
    chd.check_docstrings(
        module,
        obj=scp.find_peaks,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


@pytest.fixture
def simple_peaks_dataset():
    """Dataset with three clear gaussian peaks."""
    x = np.linspace(0, 10, 1000)
    peaks = [
        (2, 1, 0.2),  # (position, height, width)
        (5, 2, 0.3),
        (8, 1.5, 0.25),
    ]
    y = np.zeros_like(x)
    for pos, height, width in peaks:
        y += height * np.exp(-((x - pos) ** 2) / (2 * width**2))

    coord = Coord(x, title="x", units="cm⁻¹")
    return NDDataset(y, coordset=[coord], units="absorbance", title="Test Peaks")


@pytest.fixture
def noisy_peaks_dataset(simple_peaks_dataset):
    """Add noise to make peak detection more challenging."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, size=simple_peaks_dataset.size)
    return simple_peaks_dataset + noise


@pytest.fixture
def flat_peaks_dataset():
    """Dataset with flat-topped peaks."""
    x = np.linspace(0, 10, 1000)
    y = np.zeros_like(x)
    y[(x > 2) & (x < 2.5)] = 1.0  # flat peak
    y[(x > 5) & (x < 5.2)] = 2.0  # flat peak
    y[(x > 8) & (x < 8.3)] = 1.5  # flat peak

    coord = Coord(x, title="x", units="cm⁻¹")
    return NDDataset(y, coordset=[coord], units="absorbance", title="Flat Peaks")


def test_basic_peak_finding(simple_peaks_dataset):
    """Test basic peak finding with clear peaks."""
    peaks, properties = find_peaks(simple_peaks_dataset, height=0.5)

    assert len(peaks) == 3

    min_height = 0.5 * ur.absorbance
    assert all(h > min_height for h in properties["peak_heights"])

    # Check peak positions approximately match the known positions
    assert np.allclose(peaks.x.values, [2, 5, 8] * ur("cm^-1"), atol=0.1)


def test_noisy_peak_finding(noisy_peaks_dataset):
    """Test peak finding with noisy data."""
    peaks, properties = find_peaks(
        noisy_peaks_dataset, height=0.5, prominence=0.4, width=0.1
    )

    # Should still find the main peaks despite noise
    assert len(peaks) == 3
    assert all(h > 0.5 * ur.absorbance for h in properties["peak_heights"])


def test_flat_peak_detection(flat_peaks_dataset):
    """Test detection of flat-topped peaks."""
    peaks, properties = find_peaks(flat_peaks_dataset, plateau_size=0.1)

    assert len(peaks) == 3
    # Check if plateau sizes are detected
    assert all(size.m > 0 for size in properties["plateau_sizes"])


def test_no_peaks_case():
    """Test case where no peaks should be found."""
    x = np.linspace(0, 10, 100)
    y = np.zeros_like(x)  # Flat line
    dataset = NDDataset(y, coordset=[Coord(x, title="x")])

    peaks, properties = find_peaks(dataset, height=0.1)
    assert peaks is None
    assert properties is None


def test_no_peaks_result_case():
    """Structured result preserves the no-peak outcome without raising."""
    x = np.linspace(0, 10, 100)
    y = np.zeros_like(x)
    dataset = NDDataset(y, coordset=[Coord(x, title="x")])

    result = find_peaks(dataset, height=0.1, as_result=True)

    assert isinstance(result, PeakFindingResult)
    assert len(result) == 0
    assert result.peaks is None
    assert result.properties == {}
    assert result.to_dict() == []
    assert isinstance(result.table, PeakTable)
    assert len(result.table) == 0
    assert result.table.columns == ("index", "position", "height")
    assert result.table.to_dict() == []


def test_three_point_peak():
    """Test detection of three-point peaks with explicit properties request."""
    x = np.linspace(0, 10, 100)
    y = np.zeros_like(x)
    # Create a three-point peak
    y[49:52] = [0.5, 1.0, 0.5]  # Three-point peak
    dataset = NDDataset(y, coordset=[Coord(x, title="x")])

    # Request height property explicitly
    peaks, properties = find_peaks(dataset, height=0)

    assert len(peaks) == 1
    assert "peak_heights" in properties
    assert properties["peak_heights"][0] == 1.0

    # Test with multiple properties without using coordinates
    peaks, properties = find_peaks(
        dataset, height=0, width=1, prominence=0.1, use_coord=False
    )

    assert len(peaks) == 1
    assert all(key in properties for key in ["peak_heights", "widths", "prominences"])


def test_minimal_peak_properties():
    """Test peak finding with minimal configuration."""
    x = np.linspace(0, 10, 100)
    y = np.zeros_like(x)
    y[49:52] = [0.5, 1.0, 0.5]
    dataset = NDDataset(y, coordset=[Coord(x, title="x")])

    # Without specifying any properties
    peaks, properties = find_peaks(dataset)
    assert len(peaks) == 1
    # Properties should be empty but not None
    assert isinstance(properties, dict)
    assert len(properties) == 0


def test_as_result_keeps_peak_data_and_allows_unpacking(simple_peaks_dataset):
    """Opt-in structured result keeps the historical tuple data available."""
    result = find_peaks(simple_peaks_dataset, height=0.5, as_result=True)

    assert isinstance(result, PeakFindingResult)
    assert len(result) == 3

    peaks, properties = result
    assert peaks is result.peaks
    assert properties is result.properties
    assert np.allclose(peaks.x.values, [2, 5, 8] * ur("cm^-1"), atol=0.1)
    assert "peak_heights" in properties


def test_peak_finding_result_to_dict(simple_peaks_dataset):
    """Structured result exposes dependency-light row dictionaries."""
    result = find_peaks(simple_peaks_dataset, height=0.5, width=0.1, as_result=True)

    rows = result.to_dict()

    assert len(rows) == 3
    assert rows[0]["index"] == 0
    assert rows[0]["position"].units == ur("cm^-1")
    assert rows[0]["height"].units == ur.absorbance
    assert "peak_heights" in rows[0]
    assert "widths" in rows[0]


def test_peak_table_exposes_singular_column_names(simple_peaks_dataset):
    """PeakTable provides user-facing singular names while properties stay raw."""
    result = find_peaks(simple_peaks_dataset, height=0.5, width=0.1, as_result=True)

    table = result.table
    rows = table.to_dict()

    assert isinstance(table, PeakTable)
    assert repr(table) == "PeakTable(n_peaks=3)"
    assert len(table) == 3
    assert list(table) == rows
    assert "peak_height" in table.columns
    assert "width" in table.columns
    assert "peak_heights" not in table.columns
    assert "widths" not in table.columns
    assert rows[0]["position"].units == ur("cm^-1")
    assert rows[0]["height"].units == ur.absorbance
    assert rows[0]["peak_height"].units == ur.absorbance
    assert rows[0]["width"].units == ur("cm^-1")
    assert "peak_heights" in result.properties


def test_peak_table_sort_head_and_column_helpers(simple_peaks_dataset):
    """PeakTable offers notebook-friendly selection helpers."""
    result = find_peaks(simple_peaks_dataset, height=0.5, width=0.1, as_result=True)

    selected = result.table.top(2, by="height").sort_by(
        "position", reverse=True, unit="cm^-1"
    )

    assert isinstance(selected, PeakTable)
    assert len(selected) == 2
    positions = selected.column("position", unit="cm^-1", as_float=True)
    heights = selected.column("height", as_float=True)

    assert positions == sorted(positions, reverse=True)
    assert heights[0] >= 0.0
    assert heights[1] >= 0.0
    assert all(isinstance(value, float) for value in positions)
    assert all(isinstance(value, float) for value in heights)


def test_peak_table_helper_unknown_column(simple_peaks_dataset):
    """PeakTable helper methods fail clearly on unknown columns."""
    result = find_peaks(simple_peaks_dataset, height=0.5, width=0.1, as_result=True)

    with pytest.raises(KeyError, match="Unknown peak-table column"):
        result.table.sort_by("missing")

    with pytest.raises(KeyError, match="Unknown peak-table column"):
        result.table.column("missing")


def test_peak_finding_result_to_dict_single_peak(simple_peaks_dataset):
    """Single-peak coordinate values can be scalar quantities."""
    result = find_peaks(
        simple_peaks_dataset, height=(1.8, 2.2), prominence=0, as_result=True
    )

    rows = result.to_dict()

    assert len(rows) == 1
    assert rows[0]["position"].units == ur("cm^-1")
    assert rows[0]["height"].units == ur.absorbance


def test_peak_finding_result_to_csv(simple_peaks_dataset, tmp_path):
    """Structured result can be exported without pandas."""
    result = find_peaks(simple_peaks_dataset, height=0.5, width=0.1, as_result=True)
    path = tmp_path / "peaks.csv"

    written = result.to_csv(path)

    assert written == path
    text = path.read_text(encoding="utf-8")
    assert text.startswith("index,position,height")
    assert "peak_heights" in text
    assert "widths" in text


def test_peak_table_to_csv(simple_peaks_dataset, tmp_path):
    """PeakTable writes singular user-facing column names."""
    result = find_peaks(simple_peaks_dataset, height=0.5, width=0.1, as_result=True)
    path = tmp_path / "peak-table.csv"

    written = result.table.to_csv(path)

    assert written == path
    text = path.read_text(encoding="utf-8")
    assert text.startswith("index,position,height")
    assert "peak_height" in text
    assert "width" in text
    assert "peak_heights" not in text
    assert "widths" not in text


def test_peak_table_no_peak_csv_header(tmp_path):
    """Empty PeakTable writes a stable header-only CSV for batch workflows."""
    table = PeakTable(None, None)
    path = tmp_path / "empty-peak-table.csv"

    written = table.to_csv(path)

    assert written == path
    assert path.read_text(encoding="utf-8") == "index,position,height\n"


def test_single_points_peak():
    """Test behavior with single points for peak detection."""
    x = np.linspace(0, 10, 100)
    y = np.zeros_like(x)
    y[50] = 1.0  # Single point peak
    dataset = NDDataset(y, coordset=[Coord(x, title="x")])

    peaks, properties = find_peaks(dataset)
    assert len(peaks) == 1
    assert len(properties) == 0


def test_invalid_inputs():
    """Test error handling for invalid inputs."""
    # Test 2D dataset
    with pytest.raises(ValueError):
        data_2d = NDDataset(np.zeros((10, 10)))
        find_peaks(data_2d)

    # Test negative distance
    x = np.linspace(0, 10, 100)
    dataset = NDDataset(np.zeros_like(x), coordset=[Coord(x)])
    with pytest.raises(ValueError):
        find_peaks(dataset, distance=-1)


def test_peak_properties(simple_peaks_dataset):
    """Test various peak properties calculations."""
    peaks, properties = find_peaks(
        simple_peaks_dataset, height=0.5, width=0.1, prominence=0.4, distance=1.0
    )

    # Check all property keys exist
    expected_properties = {
        "peak_heights",
        "widths",
        "prominences",
        "left_bases",
        "right_bases",
    }
    assert all(key in properties for key in expected_properties)

    # Check property values are reasonable
    assert all(w.m > 0 for w in properties["widths"])
    assert all(p.m > 0 for p in properties["prominences"])


def test_distance_accepts_physical_units(simple_peaks_dataset):
    """Unit-aware distance matches existing numeric coordinate-space behavior."""
    numeric_peaks, _ = find_peaks(simple_peaks_dataset, height=0.5, distance=1.0)
    quantity_peaks, _ = find_peaks(
        simple_peaks_dataset, height=0.5, distance=1.0 * ur("cm^-1")
    )
    string_peaks, _ = find_peaks(simple_peaks_dataset, height=0.5, distance="1 cm^-1")

    assert np.allclose(quantity_peaks.x.values, numeric_peaks.x.values)
    assert np.allclose(string_peaks.x.values, numeric_peaks.x.values)


def test_width_accepts_physical_units(simple_peaks_dataset):
    """Unit-aware width matches existing numeric coordinate-space behavior."""
    numeric_peaks, numeric_props = find_peaks(
        simple_peaks_dataset, height=0.5, width=0.1, prominence=0.4
    )
    quantity_peaks, quantity_props = find_peaks(
        simple_peaks_dataset,
        height=0.5,
        width=0.1 * ur("cm^-1"),
        prominence=0.4,
    )
    string_peaks, string_props = find_peaks(
        simple_peaks_dataset, height=0.5, width="0.1 cm^-1", prominence=0.4
    )

    assert np.allclose(quantity_peaks.x.values, numeric_peaks.x.values)
    assert np.allclose(string_peaks.x.values, numeric_peaks.x.values)
    assert np.allclose(
        [width.m for width in quantity_props["widths"]],
        [width.m for width in numeric_props["widths"]],
    )
    assert np.allclose(
        [width.m for width in string_props["widths"]],
        [width.m for width in numeric_props["widths"]],
    )


def test_incompatible_peakfinding_units_raise_clear_error(simple_peaks_dataset):
    """Incompatible physical units should raise a clear coordinate-space error."""
    with pytest.raises(ValueError) as exc:
        find_peaks(simple_peaks_dataset, height=0.5, distance="1 s")

    message = str(exc.value)
    assert "peak-finding parameter `distance`" in message
    assert "dimension 'x'" in message
    assert "s" in message
    assert "cm" in message


def test_too_small_unit_aware_spacing_rejected(simple_peaks_dataset):
    """A positive physical spacing that rounds to zero points should be rejected."""
    with pytest.raises(
        ValueError, match="smaller than the coordinate sampling interval"
    ):
        find_peaks(simple_peaks_dataset, height=0.5, distance="0.001 cm^-1")


def test_window_length_interpolation(simple_peaks_dataset):
    """Test peak position interpolation with different window lengths."""
    # Test with different window lengths
    for window in [3, 5, 7]:
        peaks, _ = find_peaks(simple_peaks_dataset, window_length=window)
        # Positions should be similar regardless of window length
        assert np.allclose(peaks.x.values, [2, 5, 8] * ur("cm^-1"), atol=0.1)


def test_even_window_length_is_normalized_to_odd(simple_peaks_dataset):
    """Even interpolation windows should behave like the previous odd window."""
    odd_peaks, _ = find_peaks(simple_peaks_dataset, window_length=5)
    even_peaks, _ = find_peaks(simple_peaks_dataset, window_length=6)

    assert np.allclose(even_peaks.x.values, odd_peaks.x.values, atol=1e-8)


def test_peak_interpolation_near_dataset_edge_is_safe():
    """Quadratic refinement should not fail for peaks near the array border."""
    x = np.linspace(0.0, 10.0, 11)
    y = np.zeros_like(x)
    y[1] = 1.0
    dataset = NDDataset(y, coordset=[Coord(x, title="x", units="cm⁻¹")])

    peaks, properties = find_peaks(dataset, height=0.5, window_length=7)

    assert len(peaks) == 1
    assert float(peaks.x.values.magnitude) == pytest.approx(1.0, abs=0.5)
    assert float(properties["peak_heights"][0]) == pytest.approx(1.0, abs=1e-12)


def test_units_handling():
    """Test handling of units in coordinates and values."""
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    coord = Coord(x, title="x", units="cm⁻¹")
    dataset = NDDataset(y, coordset=[coord], units="absorbance")

    peaks, properties = find_peaks(dataset, height=0.5, width=0.5)

    # Check units are preserved
    assert peaks.units == dataset.units
    assert peaks.x.units == coord.units


def test_non_linear_coordinates():
    """Test peak finding with non-linear x coordinates."""
    x = np.exp(np.linspace(0, 2, 1000))  # Non-linear spacing
    y = np.sin(x)
    coord = Coord(x, title="x", units="cm⁻¹")
    dataset = NDDataset(y, coordset=[coord], units="absorbance")

    # Should work but issue a warning about non-linear coordinates
    with pytest.warns(UserWarning):
        peaks, _ = find_peaks(dataset)


def test_non_linear_coordinates_reject_unit_aware_spacing_constraints():
    """Physical spacing constraints should be rejected on non-linear coordinates."""
    x = np.exp(np.linspace(0, 2, 1000))
    y = np.sin(x)
    coord = Coord(x, title="x", units="cm⁻¹")
    dataset = NDDataset(y, coordset=[coord], units="absorbance")

    with pytest.raises(ValueError, match="require a linear coordinate axis"):
        find_peaks(dataset, height=0.5, distance="1 cm^-1")


def test_use_as_a_dataset_method(simple_peaks_dataset):
    """Test basic peak finding using a dataset method."""
    peaks, properties = simple_peaks_dataset.find_peaks(height=0.5)

    assert len(peaks) == 3

    min_height = 0.5 * ur.absorbance
    assert all(h > min_height for h in properties["peak_heights"])

    # Check peak positions approximately match the known positions
    assert np.allclose(peaks.x.values, [2, 5, 8] * ur("cm^-1"), atol=0.1)
