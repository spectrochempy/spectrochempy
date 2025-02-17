import pytest
import numpy as np
from pathlib import Path

from spectrochempy.extern.nmrglue import (
    unit_conversion,
    create_blank_udic,
    guess_shape,
    read_jcamp,
    read_binary,
    complexify_data,
    # read_pdata,
    scale_pdata,
)


# Test unit conversion class
def test_unit_conversion():
    """Test the unit conversion class."""
    # Create a unit conversion object
    uc = unit_conversion(size=1024, cplx=True, sw=10000.0, obs=600.0, car=4.7 * 600.0)

    # Test basic conversions
    assert np.isclose(uc.hz(0), 4.7 * 600.0 + 5000.0)  # First point in Hz
    assert np.isclose(uc.ppm(0), 4.7 + 5000.0 / 600.0)  # First point in ppm

    # Test string conversion
    assert isinstance(uc.f("700.5hz"), float)
    assert isinstance(uc.i("4.5ppm"), int)


# Test universal dictionary creation
def test_create_blank_udic():
    """Test creation of blank universal dictionary."""
    udic = create_blank_udic(ndim=2)

    assert udic["ndim"] == 2
    assert len(udic) == 3  # ndim + 2 dimensions
    assert all(key in udic[0] for key in ["sw", "complex", "obs", "car", "size"])
    assert udic[1]["encoding"] == "direct"  # Last dimension should be direct


# Test shape guessing
def test_guess_shape():
    """Test shape guessing from dictionary."""
    # Create a minimal dictionary for 1D data
    dic = {
        "acqus": {"TD": 1024, "BYTORDA": 0, "DTYPA": 0, "AQ_mod": 3, "PARMODE": 0},
        "FILE_SIZE": 4096,  # 1024 * 4 bytes
    }

    shape, cplex = guess_shape(dic)
    assert shape == (1024,)
    assert cplex is True


# Test binary file reading
def test_read_binary(tmp_path):
    """Test reading binary data files."""
    # Create test binary data
    data = np.arange(1024, dtype=np.int32)
    data_path = tmp_path / "test.bin"
    data.tofile(data_path)

    # Read data back
    dic, read_data = read_binary(data_path, shape=(1024,), cplex=False, big=False)

    assert isinstance(dic, dict)
    assert "FILE_SIZE" in dic
    np.testing.assert_array_equal(data, read_data)


# Test data processing functions
def test_complexify_data():
    """Test data complexification."""
    real = np.array([1, 2, 3, 4])
    imag = np.array([5, 6, 7, 8])
    data = np.array([1, 5, 2, 6, 3, 7, 4, 8])

    result = complexify_data(data)
    expected = real + 1j * imag

    np.testing.assert_array_equal(result, expected)


# Test JCAMP file reading
def test_read_jcamp(tmp_path):
    """Test reading JCAMP-DX format files."""
    # Create a minimal JCAMP file
    jcamp_content = """##TITLE= Test
##JCAMP-DX= 5.00
##DATA TYPE= Parameter Values
##ORIGIN= Bruker BioSpin GmbH
##OWNER= nmrsu
##$BYTORDA= 0
##$TD= 1024
##END=
"""
    jcamp_path = tmp_path / "test.jcamp"
    jcamp_path.write_text(jcamp_content)

    # Read the file
    dic = read_jcamp(jcamp_path)

    assert isinstance(dic, dict)
    assert dic["BYTORDA"] == 0
    assert dic["TD"] == 1024


# Test processed data reading and scaling
def test_scale_pdata():
    """Test scaling of processed data."""
    # Create test data and dictionary
    data = np.ones(1024)
    dic = {"procs": {"NC_proc": 2}}  # Scale factor of 2^-2 = 0.25

    scaled_data = scale_pdata(dic, data)
    np.testing.assert_array_equal(scaled_data, data / 0.25)

    scaled_data = scale_pdata(dic, data, reverse=True)
    np.testing.assert_array_equal(scaled_data, data * 0.25)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_read_pdata_dimensions(tmp_path, ndim):
    """Test reading processed data with different dimensions."""
    # Skip if no test data available
    pytest.skip("This test requires actual Bruker processed data files")

    # This test would need actual Bruker processed data files
    # The implementation would look like:
    """
    data_dir = tmp_path / f"{ndim}d_processed"
    data_dir.mkdir()

    # Create necessary files based on dimension
    if ndim == 1:
        files = ["1r"]
    elif ndim == 2:
        files = ["2rr"]
    else:
        files = ["3rrr"]

    # Create dummy processed data files
    for f in files:
        (data_dir / f).touch()

    dic, data = read_pdata(data_dir)
    assert data.ndim == ndim
    """


# Additional utility tests
def test_unit_conversion_scales():
    """Test unit conversion scale generation."""
    uc = unit_conversion(size=1024, cplx=True, sw=10000.0, obs=600.0, car=4.7 * 600.0)

    # Test different scale generations
    ppm_scale = uc.ppm_scale()
    hz_scale = uc.hz_scale()
    percent_scale = uc.percent_scale()

    assert len(ppm_scale) == 1024
    assert len(hz_scale) == 1024
    assert len(percent_scale) == 1024
    assert percent_scale[0] == 0.0
    assert percent_scale[-1] == 100.0


if __name__ == "__main__":
    pytest.main([__file__])
