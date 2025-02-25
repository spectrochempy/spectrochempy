# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import os
import pathlib
import pytest

import spectrochempy.utils.exceptions
from spectrochempy import NDDataset
from spectrochempy.core import preferences as prefs
from spectrochempy.core.readers.importer import (
    ALIAS,
    FILETYPES,
    Importer,
    _importer_method,
    read,
    read_dir,
)
from spectrochempy.utils.file import pathclean

try:
    from spectrochempy.core.common import dialogs
except ImportError:
    pytest.skip("dialogs not available with act", allow_module_level=True)

DATADIR = prefs.datadir

# --------------------------------------------------------------------------------------
# Helper functions and fixtures
# --------------------------------------------------------------------------------------


def dialog_cancel(*args, **kwargs):
    """Mock a dialog cancel action."""
    return None


def dialog_open(*args, **kwargs):
    """Mock dialog open with simulated file selection."""
    directory = kwargs.get("directory", None)
    if directory is None:
        directory = pathclean(DATADIR / "fakedir")

    if kwargs.get("filters") == "directory":
        return directory

    if not args and not kwargs.get("single"):
        return [DATADIR / "fakedir" / f"fake{i + 1}.fk" for i in range(2)]

    return [DATADIR / "fakedir" / f"fake{i + 1}.fk" for i in range(4)]


def directory_glob(*args, **kwargs):
    """Mock directory globbing."""
    res = [DATADIR / f"fakedir/fake{i + 1}.fk" for i in range(4)]
    res.append(DATADIR / "fakedir/emptyfake.fk")
    if len(args) > 1 and args[1].startswith("**/"):
        res.append(DATADIR / "fakedir/subdir/fakesub1.fk")
    return res


@_importer_method
def _read_fake(*args, **kwargs):
    """Fake read implementation for testing."""
    dataset, filename = args
    content = kwargs.get("content", False)

    if content:
        dataset = MockDatasetFactory.create(filename, origin="content")
    else:
        if os.path.exists(filename):
            if filename.stem == "otherfake":
                dataset = MockDatasetFactory.create(filename, size=6)
            elif filename.stem == "emptyfake":
                dataset = None
            else:
                dataset = MockDatasetFactory.create(filename)
        else:
            raise FileNotFoundError

    return dataset


def read_fake(*paths, **kwargs):
    """Wrapper for fake read function."""
    kwargs["filetypes"] = ["FAKE files (*fk, *.fk1, .fk2)"]
    kwargs["protocol"] = ["fake", ".fk", "fk1", "fk2"]
    kwargs["local_only"] = True
    importer = Importer()
    return importer(*paths, **kwargs)


@_importer_method
def _read_fk(*args, **kwargs):
    return Importer._read_fake(*args, **kwargs)


# Register fake reader
read_fk = read_fake
setattr(NDDataset, "read_fk", read_fk)

# --------------------------------------------------------------------------------------
# Mock Dataset Factory
# --------------------------------------------------------------------------------------


class MockDatasetFactory:
    """Factory to create mock datasets with consistent behaviors."""

    @staticmethod
    def create(filename, origin=None, size=3, implements="NDDataset"):
        """Create a mock dataset with specified properties."""
        ds = NDDataset([range(size)])
        ds.name = filename.stem

        # Set origin based on filename if not explicitly provided
        if origin is None:
            if "opus" in str(filename).lower():
                origin = "opus"
            elif "omnic" in str(filename).lower():
                origin = "omnic"
            else:
                origin = "unknown"

        ds.origin = origin
        # Ensure origin is preserved during operations
        ds._implements = lambda x=None: implements
        return ds

    @staticmethod
    def create_reader(origin=None, size=3):
        """Create a mock reader function that preserves origin."""

        def reader(filename, **kwargs):
            ds = MockDatasetFactory.create(filename, origin=origin, size=size)
            # Add any other metadata that needs to be preserved during merge
            ds.history = f"Created from {filename} with origin {origin}"
            return ds

        return reader


# --------------------------------------------------------------------------------------
# Test Classes
# --------------------------------------------------------------------------------------


class TestBasicImporter:
    """Test basic importer functionality."""

    def setup_method(self):
        """Setup common test environment."""
        # Setup test files
        self.fs = pytest.importorskip("pyfakefs").fake_filesystem.FakeFilesystem()
        self.fs.create_dir(DATADIR)
        self.fs.create_dir(DATADIR / "fakedir")

        # Create test file
        self.test_file = DATADIR / "fakedir/test.opus.fk"
        self.fs.create_file(self.test_file, contents=b"fake data")

        # Register protocol
        FILETYPES.append(("fake", "FAKE files (*.fk)"))
        ALIAS.append(("fk", "fake"))

    def test_file_not_found(self, fs):
        """Test behavior when file is not found."""
        self.setup_method()
        f = DATADIR / "fakedir/nonexistent.fk"
        with pytest.raises(FileNotFoundError):
            read_fake(f, local_only=True)

    def test_invalid_protocol(self, fs):
        """Test behavior with invalid protocol."""
        self.setup_method()
        with pytest.raises(spectrochempy.utils.exceptions.ProtocolError):
            read(self.test_file, protocol="wrongfake", local_only=True)

    def test_single_file_read(self, fs, monkeypatch):
        """Test reading a single file."""
        self.setup_method()

        # Mock file system access and reader
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        monkeypatch.setattr(
            NDDataset, "read_fk", MockDatasetFactory.create_reader(origin="opus")
        )

        # Test reading
        nd = read_fake(self.test_file, local_only=True)

        # Verify dataset properties
        assert isinstance(nd, NDDataset)
        assert nd.origin == "opus"  # Check origin is preserved
        assert nd.shape == (1, 3)  # Check shape
        assert nd.name == self.test_file.stem  # Check name preserved


class TestImporterMerging:
    """Test dataset merging behavior."""

    def setup_method(self):
        """Setup test files with different origins."""
        self.fs = pytest.importorskip("pyfakefs").fake_filesystem.FakeFilesystem()
        self.fs.create_dir(DATADIR)
        self.fs.create_dir(DATADIR / "fakedir")

        # Create opus test files
        self.opus_files = []
        for i in range(3):
            f = DATADIR / f"fakedir/opus{i}.fk"
            self.fs.create_file(f)
            self.opus_files.append(f)

        # Create omnic test files
        self.omnic_files = []
        for i in range(2):
            f = DATADIR / f"fakedir/omnic{i}.fk"
            self.fs.create_file(f)
            self.omnic_files.append(f)

        FILETYPES.append(("fake", "FAKE files (*.fk)"))
        ALIAS.append(("fk", "fake"))

    def test_merge_same_origin(self, monkeypatch, fs):
        """Test merging datasets with same origin."""
        self.setup_method()

        monkeypatch.setattr(os.path, "exists", lambda p: True)
        monkeypatch.setattr(
            NDDataset, "read_fk", MockDatasetFactory.create_reader(origin="opus")
        )

        datasets = read(self.opus_files, protocol="fake", local_only=True, merge=True)
        assert isinstance(datasets, NDDataset)
        assert datasets.shape == (3, 3)
        assert datasets.origin == "merged [opus]"

    def test_merge_different_origins(self, monkeypatch, fs):
        """Test handling datasets with different origins."""
        self.setup_method()

        monkeypatch.setattr(os.path, "exists", lambda p: True)
        monkeypatch.setattr(
            NDDataset,
            "read_fk",
            MockDatasetFactory.create_reader(),  # Uses filename-based origin
        )

        # Test without merging
        all_files = self.opus_files + self.omnic_files
        datasets = read(all_files, protocol="fake", local_only=True, merge=False)

        opus_datasets = [ds for ds in datasets if ds.origin == "opus"]
        omnic_datasets = [ds for ds in datasets if ds.origin == "omnic"]

        assert len(opus_datasets) == 3
        assert len(omnic_datasets) == 2
        assert all(ds.origin == "opus" for ds in opus_datasets)
        assert all(ds.origin == "omnic" for ds in omnic_datasets)

    def test_force_merge(self, monkeypatch, fs):
        """Test forcing merge of different origin datasets."""
        self.setup_method()

        monkeypatch.setattr(os.path, "exists", lambda p: True)
        monkeypatch.setattr(NDDataset, "read_fk", MockDatasetFactory.create_reader())

        datasets = read(
            self.opus_files + self.omnic_files,
            protocol="fake",
            merge=True,
            local_only=True,
        )

        assert isinstance(datasets, NDDataset)
        assert datasets.shape == (5, 3)
        assert datasets.origin == "merged [omnic, opus]"
        assert datasets.name == "merged [omnic, opus]"


class TestImporterDirectoryReading:
    """Test directory reading functionality."""

    def setup_method(self):
        """Setup common test environment."""
        FILETYPES.append(("fake", "FAKE files (*.fk)"))
        ALIAS.append(("fk", "fake"))

    def _setup_fake_files(self, fs):
        """Create fake test files in the filesystem."""
        # Create base directory
        fs.create_dir(DATADIR / "fakedir")

        # Create test files
        for i in range(4):
            f = DATADIR / "fakedir" / f"fake{i + 1}.fk"
            fs.create_file(f)

        # Create empty fake file
        fs.create_file(DATADIR / "fakedir/emptyfake.fk")

        # Create subdirectory with file
        fs.create_dir(DATADIR / "fakedir/subdir")
        fs.create_file(DATADIR / "fakedir/subdir/fakesub1.fk")

    def test_read_directory(self, fs, monkeypatch):
        """Test reading from directory."""
        # Setup mock filesystem with files
        self._setup_fake_files(fs)

        # Mock glob to return our fake files
        monkeypatch.setattr(pathlib.Path, "glob", directory_glob)

        # Mock os.path.exists to return True for our fake files
        def mock_exists(path):
            return str(path).endswith(".fk")

        monkeypatch.setattr(os.path, "exists", mock_exists)

        nd = read_dir(DATADIR / "fakedir", local_only=True)
        assert nd.shape == (4, 3)  # Expect 4 files, each with size 3

    def test_recursive_read(self, fs, monkeypatch):
        """Test recursive directory reading."""
        # Setup mock filesystem with files including subdirectory
        self._setup_fake_files(fs)

        # Mock glob and exists
        monkeypatch.setattr(pathlib.Path, "glob", directory_glob)

        def mock_exists(path):
            return str(path).endswith(".fk")

        monkeypatch.setattr(os.path, "exists", mock_exists)

        nd = read_dir(DATADIR / "fakedir", recursive=True, local_only=True)
        assert nd.shape == (5, 3)  # Expect 5 files including subdirectory

    def test_empty_directory(self, fs, monkeypatch):
        """Test reading from empty directory."""
        fs.create_dir(DATADIR / "emptydir")

        nd = read_dir(DATADIR / "emptydir", local_only=True)
        assert nd is None  # Should return None for empty directory


if __name__ == "__main__":
    pytest.main([__file__])
