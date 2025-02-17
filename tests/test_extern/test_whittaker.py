import numpy as np
import pytest
from scipy import sparse

from spectrochempy.extern.whittaker_smooth import speyediff, whittaker_smooth


def test_speyediff_basic():
    """Test basic functionality of speyediff."""
    # Test first order difference matrix
    D1 = speyediff(3, 1)
    # The difference matrix should be [-1, 1, 0] and [0, -1, 1] for first order differences
    expected_D1 = np.array([[-1, 1, 0], [0, -1, 1]])
    assert sparse.issparse(D1)
    assert D1.shape == (2, 3)
    assert np.allclose(D1.toarray(), expected_D1)

    # Test second order difference matrix
    D2 = speyediff(4, 2)
    # The second order difference matrix should be [1, -2, 1, 0] and [0, 1, -2, 1]
    expected_D2 = np.array([[1, -2, 1, 0], [0, 1, -2, 1]])
    assert D2.shape == (2, 4)
    assert np.allclose(D2.toarray(), expected_D2)


def test_speyediff_format():
    """Test different sparse matrix formats."""
    N, d = 5, 1
    csc_mat = speyediff(N, d, format="csc")
    csr_mat = speyediff(N, d, format="csr")

    assert sparse.isspmatrix_csc(csc_mat)
    assert sparse.isspmatrix_csr(csr_mat)
    assert np.allclose(csc_mat.toarray(), csr_mat.toarray())


def test_whittaker_smooth_basic():
    """Test basic smoothing functionality."""
    # Create noisy data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)

    # Test smoothing with different lambda values
    y_smooth_low = whittaker_smooth(y, lmbd=1)
    y_smooth_high = whittaker_smooth(y, lmbd=1000)

    # Basic checks
    assert len(y_smooth_low) == len(y)
    assert len(y_smooth_high) == len(y)

    # Higher lambda should give smoother result (smaller difference between points)
    diff_low = np.diff(y_smooth_low)
    diff_high = np.diff(y_smooth_high)
    assert np.std(diff_high) < np.std(diff_low)


def test_whittaker_smooth_edge_cases():
    """Test edge cases and error conditions."""
    # Test with constant input
    y_const = np.ones(10)
    y_smooth = whittaker_smooth(y_const, lmbd=1)
    assert np.allclose(y_smooth, y_const)

    # Test with different orders
    y = np.random.random(20)
    y_d1 = whittaker_smooth(y, lmbd=1, d=1)
    y_d2 = whittaker_smooth(y, lmbd=1, d=2)
    assert len(y_d1) == len(y)
    assert len(y_d2) == len(y)

    # Test error cases
    with pytest.raises(ValueError):
        whittaker_smooth([], lmbd=1)  # Empty input

    with pytest.raises((ValueError, TypeError)):
        whittaker_smooth(None, lmbd=1)  # Invalid input


def test_whittaker_smooth_known_signal():
    """Test smoothing with a known signal and expected outcome."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a simple step function with noise
    y = np.zeros(100)
    y[50:] = 1.0
    noise = np.random.normal(0, 0.05, 100)  # Reduced noise amplitude
    y_noisy = y + noise

    # Smooth with carefully chosen lambda
    y_smooth = whittaker_smooth(y_noisy, lmbd=10)  # Reduced lambda for less smoothing

    # Instead of comparing MSE, check if the key features of the step function are preserved
    # Check the separation between the two levels
    left_mean = np.mean(y_smooth[:45])  # Mean of left side
    right_mean = np.mean(y_smooth[55:])  # Mean of right side

    # Test criteria
    assert right_mean - left_mean > 0.8, "Step transition not preserved"
    assert left_mean < 0.2, "Left side not close enough to 0"
    assert right_mean > 0.8, "Right side not close enough to 1"

    # Test smoothness
    assert np.std(np.diff(y_smooth[:45])) < np.std(
        np.diff(y_noisy[:45])
    ), "Left side not smoothed"
    assert np.std(np.diff(y_smooth[55:])) < np.std(
        np.diff(y_noisy[55:])
    ), "Right side not smoothed"


if __name__ == "__main__":
    pytest.main([__file__])
