import zipfile

import numpy as np
import pytest

from spectrochempy.utils.zip import ScpFile
from spectrochempy.utils.zip import make_zipfile


def test_make_zipfile(tmp_path):
    # Create a temporary file
    test_file = tmp_path / "test.zip"

    # Create a zip file using make_zipfile
    with make_zipfile(test_file, mode="w") as zf:
        zf.writestr("test.txt", "This is a test")

    # Verify the zip file was created
    assert test_file.exists()

    # Verify the contents of the zip file
    with zipfile.ZipFile(test_file, "r") as zf:
        assert "test.txt" in zf.namelist()
        assert zf.read("test.txt") == b"This is a test"


def test_scpfile(tmp_path):
    # Create a temporary zip file
    test_file = tmp_path / "test.scp"
    with make_zipfile(test_file, mode="w") as zf:
        # Correctly write a .npy file to the zip archive
        with zf.open("test.npy", "w") as f:
            np.save(f, np.array([1, 2, 3]))
        zf.writestr("test.json", '{"key": "value"}')

    # Test ScpFile
    with ScpFile(test_file) as scp:
        assert "test.npy" in scp
        assert "test.json" in scp
        assert np.array_equal(scp["test.npy"], np.array([1, 2, 3]))
        assert scp["test.json"] == {"key": "value"}


def test_scpfile_nonexistent_key(tmp_path):
    # Create a temporary zip file
    test_file = tmp_path / "test.scp"
    with make_zipfile(test_file, mode="w") as zf:
        zf.writestr("test.npy", np.array([1, 2, 3]).tobytes())

    # Test ScpFile
    with ScpFile(test_file) as scp, pytest.raises(KeyError):
        _ = scp["nonexistent.npy"]


if __name__ == "__main__":
    pytest.main()
