import zipfile

import numpy as np
import pytest

from spectrochempy.utils.exceptions import SpectroChemPyError
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


def test_scpfile_rejects_object_array_without_opt_in(tmp_path, monkeypatch):
    test_file = tmp_path / "legacy_object_array.scp"
    payload = np.array([{"safe": "payload"}], dtype=object)

    with make_zipfile(test_file, mode="w") as zf, zf.open("legacy.npy", "w") as f:
        np.save(f, payload, allow_pickle=True)

    def fail_read_array(*args, **kwargs):
        if kwargs.get("allow_pickle"):
            raise AssertionError("allow_pickle=True must not be used in safe mode")
        return np.array([0])

    monkeypatch.setattr("spectrochempy.utils.zip.read_array", fail_read_array)

    with ScpFile(test_file) as scp:
        assert np.array_equal(scp["legacy.npy"], np.array([0]))


def test_scpfile_object_array_requires_explicit_opt_in(tmp_path):
    test_file = tmp_path / "legacy_object_array.scp"
    payload = np.array([{"safe": "payload"}], dtype=object)

    with make_zipfile(test_file, mode="w") as zf, zf.open("legacy.npy", "w") as f:
        np.save(f, payload, allow_pickle=True)

    with ScpFile(test_file) as scp, pytest.raises(
        SpectroChemPyError,
        match="trusted legacy loading",
    ):
        _ = scp["legacy.npy"]

    with ScpFile(test_file, allow_unsafe_legacy=True) as scp:
        loaded = scp["legacy.npy"]

    assert loaded.dtype == object
    assert loaded[0]["safe"] == "payload"


if __name__ == "__main__":
    pytest.main()
