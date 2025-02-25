from pathlib import Path
import os  # kept for os.walk
import pytest
from spectrochempy.extern.brukeropus.file import read_opus
from spectrochempy.extern.brukeropus.file import find_opus_files
from spectrochempy.extern.brukeropus import OPUSFile
from spectrochempy import preferences as prefs


def get_all_blocks(opusfile: OPUSFile) -> list:
    """Returns a list of all `FileBlock` in an `OPUSFile` instance."""
    blocks = (
        [opusfile.directory.block]
        + opusfile.special_blocks
        + opusfile.unknown_blocks
        + opusfile.unmatched_data_blocks
        + opusfile.unmatched_data_status_blocks
    )
    if hasattr(opusfile, "params"):
        blocks = blocks + opusfile.params.blocks
    if hasattr(opusfile, "rf_params"):
        blocks = blocks + opusfile.rf_params.blocks
    for d in opusfile.iter_all_data():
        blocks = blocks + d.blocks
    return blocks


def find_all_files(directory: str) -> list:
    """Recursively finds all files (regardless of filetype) in a directory."""
    filepaths = []
    directory = Path(directory)
    for root, _, filenames in os.walk(directory):
        root = Path(root)
        filepaths.extend(root / f for f in filenames)
    return filepaths


@pytest.fixture
def test_directory():
    DATADIR = prefs.datadir
    return DATADIR / "irdata" / "OPUS"


@pytest.fixture
def opus_data(test_directory):
    opus_files = find_opus_files(test_directory, recursive=True)
    data = [read_opus(f) for f in opus_files]
    for o in data:
        if o:  # Only add rel_path to valid OPUS files
            o.rel_path = "/" + str(Path(o.filepath).relative_to(test_directory))
    return data


def test_opus_file_detection(test_directory):
    """Test detection of OPUS files and non-OPUS files"""
    all_files = find_all_files(test_directory)
    opus_files = find_opus_files(test_directory, recursive=True)

    all_files = [f for f in all_files if f.name not in ["__index__", ".DS_Store"]]
    assert len(all_files) == 6, "No files found in test directory"
    assert len(opus_files) == 5, "No OPUS files found in test directory"

    # Test that files with OPUS extension are valid OPUS files
    opus_data = [read_opus(f) for f in opus_files]
    invalid_opus = [f for f, d in zip(opus_files, opus_data) if not d]
    assert (
        len(invalid_opus) == 0
    ), f"Found {len(invalid_opus)} files with OPUS extension that are not valid OPUS files"


def test_block_consistency(opus_data):
    """Test that parsed blocks match directory information"""
    valid_data = [d for d in opus_data if d]

    for o in valid_data:
        blocks = get_all_blocks(o)
        assert (
            len(blocks) == len(o.directory.block.data)
        ), f"Block count mismatch in {o.rel_path}: Found {len(blocks)}, expected {len(o.directory.block.data)}"
        assert (
            len(blocks) == o.directory.num_blocks
        ), f"Block count mismatch in {o.rel_path}: Found {len(blocks)}, header shows {o.directory.num_blocks}"


def test_no_redundant_blocks(opus_data):
    """Test that there are no redundant blocks in directory"""
    valid_data = [d for d in opus_data if d]

    for o in valid_data:
        assert len(o.directory.blocks) == 0, f"Found redundant blocks in {o.rel_path}"


def test_no_unknown_blocks(opus_data):
    """Test that there are no unknown block types"""
    valid_data = [d for d in opus_data if d]

    for o in valid_data:
        unknown = [b for b in o.unknown_blocks if b.type != (0, 0, 0, 0, 0, 0)]
        assert (
            len(unknown) == 0
        ), f"Found unknown blocks in {o.rel_path}: {[b.type for b in unknown]}"


def test_all_blocks_parsed(opus_data):
    """Test that all blocks were parsed successfully"""
    valid_data = [d for d in opus_data if d]

    for o in valid_data:
        blocks = get_all_blocks(o)
        blocks = [b for b in blocks if b.type != (0, 0, 0, 0, 0, 0)]
        unparsed = [b for b in blocks if b.parser is None or b.bytes != b""]
        assert len(unparsed) == 0, f"Found unparsed blocks in {o.rel_path}"


if __name__ == "__main__":
    pytest.main([__file__])
