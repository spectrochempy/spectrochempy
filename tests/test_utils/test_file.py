# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Tests for file utilities.

This file contains tests for the file utility functions in spectrochempy.
Many tests use mocking to isolate functionality and avoid actual filesystem operations:

- @patch decorator: Temporarily replaces modules, classes or functions with mock objects
- MagicMock: A flexible mock object that records method calls and allows setting return values
- side_effect: A function or iterable that is called/used when the mock is called
- spec: Constrains the mock to only have the same attributes/methods as the specified object
"""

import io
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from spectrochempy.utils.file import _get_file_for_protocol
from spectrochempy.utils.file import _insensitive_case_glob
from spectrochempy.utils.file import _topspin_check_filename
from spectrochempy.utils.file import check_filename_to_open
from spectrochempy.utils.file import check_filename_to_save
from spectrochempy.utils.file import check_filenames
from spectrochempy.utils.file import find_or_create_spectrochempy_dir
from spectrochempy.utils.file import fromfile
from spectrochempy.utils.file import get_directory_name
from spectrochempy.utils.file import get_filenames
from spectrochempy.utils.file import get_repo_path
from spectrochempy.utils.file import is_editable_install
from spectrochempy.utils.file import pathclean
from spectrochempy.utils.file import patterns


class TestPathClean:
    def test_pathclean_string(self):
        """Test pathclean with a string path."""
        path = pathclean("test/path")
        assert isinstance(path, Path)
        assert str(path) == str(pathclean("test/path"))

    def test_pathclean_windows_path(self):
        """Test pathclean with Windows-style path."""
        path = pathclean("test\\path")
        assert isinstance(path, Path)
        assert str(path).replace("\\", "/") == "test/path"

    def test_pathclean_list(self):
        """Test pathclean with a list of paths."""
        paths = pathclean(["test/path1", "test/path2"])
        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)
        assert str(paths[0]) == str(pathclean("test/path1"))
        assert str(paths[1]) == str(pathclean("test/path2"))

    def test_pathclean_tuple(self):
        """Test pathclean with a tuple of paths."""
        paths = pathclean(("test/path1", "test/path2"))
        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)

    def test_pathclean_path_object(self):
        """Test pathclean with a Path object."""
        original_path = Path("test/path")
        path = pathclean(original_path)
        assert isinstance(path, Path)
        assert path == original_path

    def test_pathclean_none(self):
        """Test pathclean with None."""
        assert pathclean(None) is None


class TestInstallFunctions:
    """
    Tests for installation-related functions.

    These tests use mocking to simulate different installation configurations
    without needing actual different installations.
    """

    @patch("importlib.util.find_spec")
    def test_is_editable_install_true(self, mock_find_spec):
        """Test is_editable_install returns True when editable."""
        # Create a mock for the spec object returned by find_spec
        # and set its 'origin' attribute to simulate an editable install path
        mock_spec = MagicMock()
        mock_spec.origin = "path/to/package/src/module.py"
        mock_find_spec.return_value = mock_spec

        assert is_editable_install("package") is True

    @patch("importlib.util.find_spec")
    def test_is_editable_install_false(self, mock_find_spec):
        """Test is_editable_install returns False when not editable."""
        mock_spec = MagicMock()
        mock_spec.origin = "path/to/package/module.py"
        mock_find_spec.return_value = mock_spec

        assert is_editable_install("package") is False

    @patch("importlib.util.find_spec")
    def test_is_editable_install_none(self, mock_find_spec):
        """Test is_editable_install returns False when package not found."""
        mock_find_spec.return_value = None

        assert is_editable_install("nonexistent_package") is False

    @patch("spectrochempy.utils.file.is_editable_install")
    def test_get_repo_path_editable(self, mock_is_editable):
        """Test get_repo_path for editable install."""
        # Configure the is_editable_install mock to return True
        mock_is_editable.return_value = True

        # Mock the Path constructor to control the entire path chain
        # This allows testing without accessing the real file system
        with patch("pathlib.Path.__new__") as mock_path_new:
            mock_file_path = MagicMock()
            mock_path_new.return_value = mock_file_path

            # Set up a chain of parent attributes for traversing up the directory tree
            # Each parent is a separate mock to allow verification of which one is returned
            parent1 = MagicMock()
            parent2 = MagicMock()
            parent3 = MagicMock()
            parent4 = MagicMock()
            mock_file_path.parent = parent1
            parent1.parent = parent2
            parent2.parent = parent3
            parent3.parent = parent4

            result = get_repo_path()
            assert result == parent4  # For editable, should go up 4 levels

    @patch("spectrochempy.utils.file.is_editable_install")
    def test_get_repo_path_non_editable(self, mock_is_editable):
        """Test get_repo_path for non-editable install."""
        mock_is_editable.return_value = False

        # Updated mock approach for Path.parent chain
        with patch("pathlib.Path.__new__") as mock_path_new:
            mock_file_path = MagicMock()
            mock_path_new.return_value = mock_file_path

            # Mock the chain of .parent calls
            parent1 = MagicMock()
            parent2 = MagicMock()
            mock_file_path.parent = parent1
            parent1.parent = parent2

            result = get_repo_path()
            assert result == parent2


class TestFromFile:
    """
    Tests for the fromfile function.

    These tests create binary data with struct.pack and then use BytesIO
    to create file-like objects for testing without actual files.
    """

    def test_fromfile_uint8(self):
        """Test fromfile function with uint8 type."""
        data = bytes([1, 2, 3, 4])
        fid = io.BytesIO(data)
        result = fromfile(fid, "uint8", 4)
        assert np.array_equal(result, np.array([1, 2, 3, 4]))

    def test_fromfile_int16(self):
        """Test fromfile function with int16 type."""
        # Create bytes representation of int16 values
        data = struct.pack("hh", 1000, -1000)
        fid = io.BytesIO(data)
        result = fromfile(fid, "int16", 2)
        assert np.array_equal(result, np.array([1000, -1000]))

    def test_fromfile_float32(self):
        """Test fromfile function with float32 type."""
        data = struct.pack("ff", 1.5, 3.14)
        fid = io.BytesIO(data)
        result = fromfile(fid, "float32", 2)
        # Use np.allclose for floating point comparison instead of array_equal
        assert np.allclose(result, np.array([1.5, 3.14]))

    def test_fromfile_single_value(self):
        """Test fromfile function returning a single value."""
        data = struct.pack("B", 42)
        fid = io.BytesIO(data)
        result = fromfile(fid, "uint8", 1)
        assert result == 42


class TestPatternFunctions:
    """
    Tests for pattern-related functions.

    These tests verify string manipulation functions used for
    creating case-insensitive glob patterns.
    """

    def test_insensitive_case_glob(self):
        """Test _insensitive_case_glob function."""
        pattern = "Test.txt"
        result = _insensitive_case_glob(pattern)
        assert result == "[tT][eE][sS][tT].[tT][xX][tT]"

    def test_patterns_single_string(self):
        """Test patterns function with a single string."""
        result = patterns("*.txt")
        # Using this instead of literal "*.txt" since patterns converts to case-insensitive
        assert "*.[tT][xX][tT]" in result

    def test_patterns_list(self):
        """Test patterns function with a list."""
        result = patterns(["*.txt", "*.csv"])
        assert "*.[tT][xX][tT]" in result
        assert "*.[cC][sS][vV]" in result

    def test_patterns_complex(self):
        """Test patterns function with complex pattern."""
        result = patterns("*.txt and *.csv")
        assert "*.[tT][xX][tT]" in result
        assert "*.[cC][sS][vV]" in result

    def test_patterns_allcase_false(self):
        """Test patterns function with allcase=False."""
        result = patterns("*.TXT", allcase=False)
        assert result == ["*.TXT"]


class TestFileProtocol:
    """
    Tests for the _get_file_for_protocol function.

    These tests use a combination of temporary files and mocking
    to test different protocol handling scenarios.
    """

    def test_get_file_for_protocol_none(self):
        """Test _get_file_for_protocol with no protocol."""
        f = Path("test.txt")
        result = _get_file_for_protocol(f)
        assert result is None

    def test_get_file_for_protocol_all(self):
        """Test _get_file_for_protocol with ALL protocol."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            # Create a test file
            test_file = tmpdir / "test.txt"
            test_file.touch()

            # When using ALL protocol, it will find the test.txt file
            # So we need to check it's returning a Path, not None
            result = _get_file_for_protocol(tmpdir / "test", protocol="ALL")
            assert result is not None
            assert result.name == "test.txt"

    @patch("pathlib.Path.glob")
    def test_get_file_for_protocol_opus(self, mock_glob):
        """Test _get_file_for_protocol with opus protocol."""
        # Create a Path for the base file
        f = Path("test")

        # Mock the glob method to return a predetermined list of paths
        # This avoids needing actual files on the filesystem
        mock_glob.return_value = [Path("test.0")]

        result = _get_file_for_protocol(f, protocol="opus")
        assert result == f.parent / Path("test.0")


class TestCheckFilenames:
    """
    Tests for the check_filenames function.

    Uses a combination of temporary real files (to test file handling)
    and mocks for more complex scenarios.
    """

    def test_check_filenames_single_string(self):
        """Test check_filenames with a single string."""
        with tempfile.NamedTemporaryFile() as tmp:
            tmp_path = Path(tmp.name)
            result = check_filenames(tmp_path)
            assert isinstance(result, list)
            assert result[0] == tmp_path

    def test_check_filenames_list_of_strings(self):
        """Test check_filenames with a list of strings."""
        with (
            tempfile.NamedTemporaryFile() as tmp1,
            tempfile.NamedTemporaryFile() as tmp2,
        ):
            tmp_path1 = Path(tmp1.name)
            tmp_path2 = Path(tmp2.name)
            result = check_filenames([tmp_path1, tmp_path2])
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0] == tmp_path1
            assert result[1] == tmp_path2

    def test_check_filenames_with_bytes(self):
        """Test check_filenames with bytes content."""
        content = b"test content"
        result = check_filenames(content)
        assert isinstance(result, dict)
        assert result[Path("no_name_0")] == content

    def test_check_filenames_with_content_kwarg(self):
        """Test check_filenames with content keyword."""
        content = b"test content"
        result = check_filenames(content=content)
        assert isinstance(result, dict)
        assert result[Path("no_name")] == content

    @patch("spectrochempy.utils.file.get_filenames")
    def test_check_filenames_no_args(self, mock_get_filenames):
        """Test check_filenames with no args."""
        # Set the return value of the get_filenames function
        # This avoids the need to mock multiple layers of file system interactions
        mock_get_filenames.return_value = {"mock": "result"}
        result = check_filenames()
        assert result == {"mock": "result"}


class TestTopspinFilename:
    """
    Tests for the _topspin_check_filename function.

    These tests use complex mocking to simulate Topspin's directory structure
    and file detection logic without needing actual Topspin files.
    """

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_topspin_check_filename_iterdir(self, mock_glob, mock_exists):
        """Test _topspin_check_filename with iterdir=True."""
        # Set up a test path structure
        filename = Path("/path/to/topspin")
        ser_path = filename / "123" / "ser"

        # Use side_effect to make glob return different results depending on the input pattern
        # This simulates finding certain files only with specific glob patterns
        mock_glob.side_effect = lambda pattern: [ser_path] if "ser" in pattern else []
        mock_exists.return_value = True

        # Also need to mock patterns to control which patterns are used by the function
        with patch("spectrochempy.utils.file.patterns") as mock_patterns:
            mock_patterns.return_value = ["ser"]
            result = _topspin_check_filename(filename, iterdir=True)
            assert len(result) == 1
            assert result[0] == ser_path

    @patch("pathlib.Path.glob")
    def test_topspin_check_filename_with_expno(self, mock_glob):
        """Test _topspin_check_filename with expno specified."""
        # Set up test paths that simulate Topspin directory structure
        filename = Path("/path/to/topspin")
        expno_path = Path("/path/to/topspin/123")
        ser_path = Path("/path/to/topspin/123/ser")
        fid_path = Path("/path/to/topspin/123/fid")

        # Mock multiple parts of the Path interaction chain
        with (
            # Control what sorted returns when it sorts directory entries
            patch("builtins.sorted") as mock_sorted,
            # Control Path.exists behavior
            patch.object(Path, "exists") as mock_exists,
            # Control Path division (/ operator) to return specific paths
            patch("pathlib.Path.__truediv__", autospec=True) as mock_truediv,
        ):
            # Make sorted return our experiment number path
            mock_sorted.return_value = [expno_path]

            # Complex mocking of path division - different results based on inputs
            def truediv_side_effect(self, other):
                """Simulate path division (/) operator based on specific path components."""
                if str(self) == "/path/to/topspin" and other == "123":
                    return expno_path
                if str(self) == "/path/to/topspin/123" and other == "ser":
                    return ser_path
                if str(self) == "/path/to/topspin/123" and other == "fid":
                    return fid_path
                return Path(f"{self}/{other}")

            mock_truediv.side_effect = truediv_side_effect

            # Make exists() return True only for the fid path, False for ser
            def exists_side_effect(path=None):
                """Make exists() return true only for fid path, simulating only fid exists."""
                if path is None:  # Handle call with no args
                    return False
                return str(path) == str(fid_path)

            mock_exists.side_effect = exists_side_effect

            # Test the function with this complex mocking setup
            result = _topspin_check_filename(filename, expno="123")
            assert len(result) == 1
            assert result[0] == fid_path


class TestGetFilenames:
    """
    Tests for the get_filenames function.

    This function has complex interactions with the filesystem, so tests
    use extensive mocking to control its behavior.
    """

    @patch("spectrochempy.utils.file.pathclean")
    def test_get_filenames_single(self, mock_pathclean):
        """Test get_filenames with a single filename."""
        # Create a mock path with properties needed by get_filenames
        test_file = MagicMock(spec=Path)
        test_file.name = "test.txt"
        test_file.suffix = ".txt"
        test_file.is_dir.return_value = False

        # Create a parent directory mock that passes the is_dir check
        parent_dir = MagicMock(spec=Path)
        parent_dir.is_dir.return_value = True
        test_file.parent = parent_dir

        # Make pathclean return different values based on the input type
        # This handles both single path and list of paths cases
        mock_pathclean.side_effect = (
            lambda x: [test_file] if isinstance(x, list) else test_file
        )

        # We need to mock get_directory_name too, since it's called by get_filenames
        with patch("spectrochempy.utils.file.get_directory_name") as mock_get_dir:
            mock_get_dir.return_value = parent_dir

            # And finally mock exists to avoid file system checks
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True

                result = get_filenames("test.txt")
                assert isinstance(result, dict)
                assert ".txt" in result


class TestDirectoryFunctions:
    """
    Tests for directory-related functions.

    These tests mock filesystem operations to test directory
    creation and validation without changing the actual filesystem.
    """

    def test_find_or_create_spectrochempy_dir(self):
        """Test find_or_create_spectrochempy_dir function."""
        # Mock multiple Path methods at once using a context manager
        with (
            patch("pathlib.Path.mkdir") as mock_mkdir,  # Mock directory creation
            patch("pathlib.Path.is_file") as mock_is_file,  # Mock file check
            patch("pathlib.Path.home") as mock_home,  # Mock home directory
        ):
            # Set return values for the mocks
            mock_home.return_value = Path("/mock/home")
            mock_is_file.return_value = False

            result = find_or_create_spectrochempy_dir()
            assert result == Path("/mock/home/.spectrochempy")
            # Verify mkdir was called with the expected arguments
            mock_mkdir.assert_called_once_with(exist_ok=True)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="I do not have the OS to test on windows and it fails on github actions",
    )
    @patch("spectrochempy.utils.file.pathclean")
    @patch("spectrochempy.application.application.warning_")
    def test_get_directory_name_invalid(self, mock_warning, mock_pathclean):
        """
        Test get_directory_name with an invalid directory.

        This test demonstrates a complex mocking setup to force the function
        to raise an OSError by ensuring all directory checks fail.
        """
        # Create a proper input directory - not None or empty
        input_dir = "/mock/invalid/dir"

        # Use an actual Path object rather than a mock for this test
        # This avoids issues with mocking magic methods like __bool__
        mock_dir_path = Path(input_dir)

        # Set up pathclean to return our path
        mock_pathclean.return_value = mock_dir_path

        # Create working directory mock
        mock_working_dir = MagicMock()

        # Set up the combined path for (working_dir / directory)
        mock_combined1 = MagicMock()
        mock_combined1.is_dir.return_value = False
        mock_working_dir.__truediv__.return_value = mock_combined1

        # Create data directory mock
        mock_data_dir = MagicMock()

        # Set up the combined path for (data_dir / directory)
        mock_combined2 = MagicMock()
        mock_combined2.is_dir.return_value = False
        mock_data_dir.__truediv__.return_value = mock_combined2

        # Set up multiple patches at once using a context manager
        with (
            # Make all is_dir checks return False to force the error path
            patch("pathlib.Path.is_dir", return_value=False),
            # Control what Path.cwd() returns
            patch("pathlib.Path.cwd", return_value=mock_working_dir),
            # Mock preferences to control the datadir property
            patch("spectrochempy.application.preferences.preferences") as mock_prefs,
        ):
            # Set up the preferences.datadir property
            type(mock_prefs).datadir = mock_data_dir

            # Use pytest.raises to verify an exception is raised with the expected message
            with pytest.raises(
                OSError, match=f'"{pathclean(input_dir)}" is not a valid directory'
            ):
                get_directory_name(input_dir)


class TestFilenameChecks:
    """
    Tests for filename check functions.

    These tests verify the behavior of functions that validate and process
    filenames for opening and saving files.
    """

    @patch("spectrochempy.utils.file.pathclean")
    def test_check_filename_to_save_new(self, mock_pathclean):
        """Test check_filename_to_save with a new filename."""
        # Create a mock filename that doesn't exist
        mock_filename = MagicMock()
        mock_filename.exists.return_value = False
        mock_pathclean.return_value = mock_filename

        # Create a mock dataset with a name property
        mock_dataset = MagicMock()
        mock_dataset.name = "dataset"

        result = check_filename_to_save(mock_dataset, mock_filename)
        assert result == mock_filename

    @patch("spectrochempy.utils.file.pathclean")
    def test_check_filename_to_save_existing_overwrite(self, mock_pathclean):
        """Test check_filename_to_save with existing file and overwrite=True."""
        mock_filename = MagicMock()
        mock_filename.exists.return_value = True
        mock_pathclean.return_value = mock_filename

        mock_dataset = MagicMock()

        result = check_filename_to_save(mock_dataset, mock_filename, overwrite=True)
        assert result == mock_filename

    @patch("spectrochempy.utils.file.pathclean")
    def test_check_filename_to_save_existing_no_overwrite(self, mock_pathclean):
        """Test check_filename_to_save with existing file and overwrite=False."""
        mock_filename = MagicMock()
        mock_filename.exists.return_value = True
        mock_pathclean.return_value = mock_filename

        mock_dataset = MagicMock()

        with pytest.raises(FileExistsError):
            check_filename_to_save(mock_dataset, mock_filename)

    @patch("spectrochempy.utils.file.check_filenames")
    def test_check_filename_to_open(self, mock_check_filenames):
        """Test check_filename_to_open function."""
        mock_check_filenames.return_value = [Path("test.txt")]

        result = check_filename_to_open("test.txt")
        assert isinstance(result, dict)

    @patch("spectrochempy.utils.file.check_filenames")
    def test_check_filename_to_open_none(self, mock_check_filenames):
        """Test check_filename_to_open with None result."""
        mock_check_filenames.return_value = None

        result = check_filename_to_open()
        assert result is None

    @patch("spectrochempy.utils.file.check_filenames")
    def test_check_filename_to_open_dict(self, mock_check_filenames):
        """Test check_filename_to_open returning a dictionary."""
        mock_check_filenames.return_value = {"key": "value"}

        result = check_filename_to_open()
        assert result == {"key": "value"}
