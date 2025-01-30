import os
import pytest
import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.analysis.peakfinding.peakfinding import find_peaks
from spectrochempy.core.units import ur
from spectrochempy.utils import docstrings as chd


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
    """Dataset with three clear gaussian peaks"""
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
    """Add noise to make peak detection more challenging"""
    noise = np.random.normal(0, 0.05, simple_peaks_dataset.size)
    return simple_peaks_dataset + noise


@pytest.fixture
def flat_peaks_dataset():
    """Dataset with flat-topped peaks"""
    x = np.linspace(0, 10, 1000)
    y = np.zeros_like(x)
    y[(x > 2) & (x < 2.5)] = 1.0  # flat peak
    y[(x > 5) & (x < 5.2)] = 2.0  # flat peak
    y[(x > 8) & (x < 8.3)] = 1.5  # flat peak

    coord = Coord(x, title="x", units="cm⁻¹")
    return NDDataset(y, coordset=[coord], units="absorbance", title="Flat Peaks")


def test_basic_peak_finding(simple_peaks_dataset):
    """Test basic peak finding with clear peaks"""
    peaks, properties = find_peaks(simple_peaks_dataset, height=0.5)

    assert len(peaks) == 3

    min_height = 0.5 * ur.absorbance
    assert all(h > min_height for h in properties["peak_heights"])

    # Check peak positions approximately match the known positions
    assert np.allclose(peaks.x.values, [2, 5, 8] * ur("cm^-1"), atol=0.1)


def test_noisy_peak_finding(noisy_peaks_dataset):
    """Test peak finding with noisy data"""
    peaks, properties = find_peaks(
        noisy_peaks_dataset, height=0.5, prominence=0.4, width=0.1
    )

    # Should still find the main peaks despite noise
    assert len(peaks) == 3
    assert all(h > 0.5 * ur.absorbance for h in properties["peak_heights"])


def test_flat_peak_detection(flat_peaks_dataset):
    """Test detection of flat-topped peaks"""
    peaks, properties = find_peaks(flat_peaks_dataset, plateau_size=0.1)

    assert len(peaks) == 3
    # Check if plateau sizes are detected
    assert all(size.m > 0 for size in properties["plateau_sizes"])


def test_no_peaks_case():
    """Test case where no peaks should be found"""
    x = np.linspace(0, 10, 100)
    y = np.zeros_like(x)  # Flat line
    dataset = NDDataset(y, coordset=[Coord(x, title="x")])

    peaks, properties = find_peaks(dataset, height=0.1)
    assert peaks is None
    assert properties is None


def test_three_point_peak():
    """Test detection of three-point peaks with explicit properties request"""
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
    """Test peak finding with minimal configuration"""
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


def test_single_points_peak():
    """Test behavior with single points for peak detection"""
    x = np.linspace(0, 10, 100)
    y = np.zeros_like(x)
    y[50] = 1.0  # Single point peak
    dataset = NDDataset(y, coordset=[Coord(x, title="x")])

    peaks, properties = find_peaks(dataset)
    assert len(peaks) == 1
    assert len(properties) == 0


def test_invalid_inputs():
    """Test error handling for invalid inputs"""
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
    """Test various peak properties calculations"""
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


def test_window_length_interpolation(simple_peaks_dataset):
    """Test peak position interpolation with different window lengths"""
    # Test with different window lengths
    for window in [3, 5, 7]:
        peaks, _ = find_peaks(simple_peaks_dataset, window_length=window)
        # Positions should be similar regardless of window length
        assert np.allclose(peaks.x.values, [2, 5, 8] * ur("cm^-1"), atol=0.1)


def test_units_handling():
    """Test handling of units in coordinates and values"""
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    coord = Coord(x, title="x", units="cm⁻¹")
    dataset = NDDataset(y, coordset=[coord], units="absorbance")

    peaks, properties = find_peaks(dataset, height=0.5, width=0.5)

    # Check units are preserved
    assert peaks.units == dataset.units
    assert peaks.x.units == coord.units


def test_non_linear_coordinates():
    """Test peak finding with non-linear x coordinates"""
    x = np.exp(np.linspace(0, 2, 1000))  # Non-linear spacing
    y = np.sin(x)
    coord = Coord(x, title="x", units="cm⁻¹")
    dataset = NDDataset(y, coordset=[coord], units="absorbance")

    # Should work but issue a warning about non-linear coordinates
    with pytest.warns(UserWarning):
        peaks, _ = find_peaks(dataset)


def test_use_as_a_dataset_method(simple_peaks_dataset):
    """Test basic peak finding using a dataset method"""
    peaks, properties = simple_peaks_dataset.find_peaks(height=0.5)

    assert len(peaks) == 3

    min_height = 0.5 * ur.absorbance
    assert all(h > min_height for h in properties["peak_heights"])

    # Check peak positions approximately match the known positions
    assert np.allclose(peaks.x.values, [2, 5, 8] * ur("cm^-1"), atol=0.1)
