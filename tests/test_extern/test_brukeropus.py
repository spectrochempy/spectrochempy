import os  # kept for os.walk
import struct
from pathlib import Path

import numpy as np
import pytest

from spectrochempy import preferences as prefs
from spectrochempy.extern.brukeropus import OPUSFile
from spectrochempy.extern.brukeropus.file import find_opus_files
from spectrochempy.extern.brukeropus.file import read_opus
from spectrochempy.extern.brukeropus.file.block import FileBlock
from spectrochempy.extern.brukeropus.file.constants import STRUCT_3D_INFO_BLOCK
from spectrochempy.extern.brukeropus.file.data import Data
from spectrochempy.extern.brukeropus.file.parse import parse_data_series
from spectrochempy.extern.brukeropus.file.report import Report

pytestmark = pytest.mark.data


def get_all_blocks(opusfile: OPUSFile) -> list:
    """Return a list of all `FileBlock` in an `OPUSFile` instance."""
    blocks = (
        [opusfile.directory.block]
        + (getattr(opusfile, "special_blocks", None) or [])
        + opusfile.unknown_blocks
        + opusfile.unmatched_data_blocks
        + opusfile.unmatched_data_status_blocks
        + getattr(opusfile, "parse_error_blocks", [])
    )
    if hasattr(opusfile, "params"):
        blocks = blocks + opusfile.params.blocks
    if hasattr(opusfile, "rf_params"):
        blocks = blocks + opusfile.rf_params.blocks
    for d in opusfile.iter_all_data():
        blocks = blocks + d.blocks
    for report in getattr(opusfile, "reports", []):
        blocks.append(report.block)
    if getattr(opusfile, "history", None):
        blocks = blocks + opusfile.history_blocks
    return [block for block in blocks if block is not None]


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
    directory = DATADIR / "irdata" / "OPUS"
    if not directory.exists():
        pytest.skip("OPUS testdata not available (set SCP_TEST_DATA_DOWNLOAD=1)")
    return directory


@pytest.fixture
def opus_data(test_directory):
    opus_files = find_opus_files(test_directory, recursive=True)
    data = [read_opus(f) for f in opus_files]
    for o in data:
        if o:  # Only add rel_path to valid OPUS files
            o.rel_path = "/" + str(Path(o.filepath).relative_to(test_directory))
    return data


def test_opus_file_detection(test_directory):
    """Test detection of OPUS files and non-OPUS files."""
    all_files = find_all_files(test_directory)
    opus_files = find_opus_files(test_directory, recursive=True)

    all_files = [f for f in all_files if f.name not in ["__index__", ".DS_Store"]]
    assert len(all_files) >= 5, "No files found in test directory"
    assert len(opus_files) >= 5, "No OPUS files found in test directory"

    # Test that files with OPUS extension are valid OPUS files
    opus_data = [read_opus(f) for f in opus_files]
    invalid_opus = [f for f, d in zip(opus_files, opus_data, strict=False) if not d]
    assert (
        len(invalid_opus) == 0
    ), f"Found {len(invalid_opus)} files with OPUS extension that are not valid OPUS files"


def test_block_consistency(opus_data):
    """Test that parsed blocks match directory information."""
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
    """Test that there are no redundant blocks in directory."""
    valid_data = [d for d in opus_data if d]

    for o in valid_data:
        assert len(o.directory.blocks) == 0, f"Found redundant blocks in {o.rel_path}"


def test_no_unknown_blocks(opus_data):
    """Test that there are no unknown block types."""
    valid_data = [d for d in opus_data if d]

    for o in valid_data:
        unknown = [b for b in o.unknown_blocks if b.type != (0, 0, 0, 0, 0, 0)]
        assert (
            len(unknown) == 0
        ), f"Found unknown blocks in {o.rel_path}: {[b.type for b in unknown]}"


def test_all_blocks_parsed(opus_data):
    """Test that all blocks were parsed successfully."""
    valid_data = [d for d in opus_data if d]

    for o in valid_data:
        blocks = get_all_blocks(o)
        blocks = [b for b in blocks if b.type != (0, 0, 0, 0, 0, 0)]
        unparsed = [b for b in blocks if b.parser is None or b.bytes != b""]
        assert len(unparsed) == 0, f"Found unparsed blocks in {o.rel_path}"


def test_data_block_classification_includes_compact_and_reports():
    compact = FileBlock(b"", (0, 1, 0, 1, 0, 4), 0, 0)
    report = FileBlock(b"", (0, 0, 0, 0, 0, 2), 0, 0)

    assert compact.is_data()
    assert not compact.is_data_series()
    assert compact.is_compact_data()
    assert not report.is_data()
    assert not report.is_data_series()
    assert report.is_report()


def test_parse_data_series_counts_decoded_records_not_header_count():
    info_size = struct.calcsize("<" + "".join(e["fmt"] for e in STRUCT_3D_INFO_BLOCK))
    data_size = 8
    offset = 32
    header = struct.pack("<6i", 0, 5, offset, data_size, info_size, 1)
    store_table = struct.pack("<2i", 1, 5)
    records = []
    for index in range(2):
        records.append(struct.pack("<2f", index + 1.0, index + 2.0))
        values = [0] * 8 + [0.0] * 4 + [0] * 2 + [0.0] * 7 + [0] * 2 + [0.0] * 2
        values[3] = 2
        values[14] = 1000.0
        values[15] = 2000.0
        values[18] = 1.0
        records.append(
            struct.pack("<" + "".join(e["fmt"] for e in STRUCT_3D_INFO_BLOCK), *values)
        )

    parsed = parse_data_series(header + store_table + b"".join(records))

    assert parsed["num_blocks"] == 5
    assert parsed["y"].shape == (2, 2)
    assert parsed["ert"].shape == (2,)
    assert parsed["store_table"] == [(1, 5)]


def test_empty_data_series_has_zero_decoded_records():
    info_size = struct.calcsize("<" + "".join(e["fmt"] for e in STRUCT_3D_INFO_BLOCK))
    block = struct.pack("<6i", 0, 0, 32, 8, info_size, 0)

    parsed = parse_data_series(block)

    assert parsed["y"].shape == (0, 2)
    assert parsed["ert"].size == 0


def test_compact_data_uses_the_trailing_spectrum_values():
    data_block = FileBlock(b"", (0, 1, 0, 1, 0, 4), 0, 0)
    data_block.data = np.array([99.0, 1.0, 2.0])
    status_block = FileBlock(b"", (0, 1, 1, 0, 0, 0), 0, 0)
    status_block.data = {
        "csf": 1.0,
        "dxu": "PNT",
        "fxv": 0.0,
        "lxv": 2.0,
        "npt": 2,
    }

    data = Data(data_block, status_block, key="sm_c", vel=0)

    assert data.y.tolist() == [1.0, 2.0]


def test_multichannel_data_key_is_distinct():
    block = FileBlock(b"", (0, 1, 0, 33, 0, 0), 0, 0)

    assert block.get_data_key() == "sm_2ch"


def test_report_with_missing_value_entry_is_retained():
    block = FileBlock(b"", (0, 0, 0, 0, 0, 2), 0, 0)
    block.data = {
        "header": {"tit": "Report", "e00": "Summary"},
        "info": {"e00": "Summary", "h00": "Image", "s00": "Value"},
        "data": [[1]],
        "subreports": [],
    }

    report = Report(block)

    assert report.properties == {"Image": None}
    assert report.table["Value"] == [1]


def test_truncated_report_is_recorded_as_parse_error():
    block = FileBlock(b"\x00", (0, 0, 0, 0, 0, 2), 1, 0)

    block.parse()

    assert block.data is None
    assert block.parse_error is not None


if __name__ == "__main__":
    pytest.main([__file__])
