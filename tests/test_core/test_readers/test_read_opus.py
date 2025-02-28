# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import pytest
from pathlib import Path

from spectrochempy import NDDataset
from spectrochempy import preferences as prefs
from spectrochempy.utils.testing import assert_dataset_equal

DATADIR = prefs.datadir
OPUSDATA = DATADIR / "irdata" / "OPUS"


class TestOpusBasic:
    """Test basic OPUS file reading functionality."""

    def test_single_file(self):
        """Test reading a single OPUS file."""
        dataset = NDDataset.read_opus(OPUSDATA / "test.0000")
        assert dataset.shape == (1, 2567)
        assert dataset[0, 2303.8694].data == pytest.approx(2.72740, rel=1e-5)
        assert dataset.units == "absorbance"

    def test_pathlib_input(self):
        """Test reading using pathlib.Path input."""
        p = Path(OPUSDATA / "test.0000")
        dataset = NDDataset.read_opus(p)
        assert dataset.shape == (1, 2567)
        assert dataset.units == "absorbance"

    def test_read_from_contents(self):
        """Test reading from file contents."""
        p = OPUSDATA / "test.0000"
        content = p.read_bytes()
        dataset = NDDataset.read_opus({p.name: content})
        assert dataset.name == p.stem
        assert dataset.shape == (1, 2567)


class TestOpusSpecialTypes:
    """Test reading different OPUS data types."""

    def test_sm_type(self):
        """Test reading with SM type specification."""
        dataset = NDDataset.read_opus(OPUSDATA / "test.0000", type="SM")
        assert dataset.shape == (1, 2567)
        assert dataset.units is None

    def test_rf_type(self):
        """Test reading reference (RF) type files."""
        ref = NDDataset.read_opus(OPUSDATA / "background.0", type="RF")
        assert ref.shape == (1, 4096)
        assert ref.units is None

        # Test automatic type inference for reference spectrum
        auto = NDDataset.read_opus(OPUSDATA / "background.0")
        assert_dataset_equal(ref, auto)

    def test_invalid_type(self):
        """Test behavior with invalid type specification."""
        with pytest.warns(UserWarning):
            NDDataset.read_opus(OPUSDATA / "test.0000", type="INVALID")


class TestOpusMetadata:
    """Test OPUS metadata handling."""

    def setup_method(self):
        """Setup test dataset with metadata."""
        self.dataset = NDDataset.read_opus(OPUSDATA / "test.0000")
        self.meta = self.dataset.meta

    def test_metadata_structure(self):
        """Test basic metadata structure."""
        assert "params" in self.meta
        assert "rf_params" in self.meta

    def test_params_content(self):
        """Test parameters metadata content."""
        params = self.meta["params"]
        assert params.name == "Sample/Result Parameters"
        assert "optical" in params

        assert params.optical.name == "Optical Parameters"
        assert "apf" in params["fourier_transform"]
        assert (
            params["Fourier Transform"].name == "Fourier Transform Parameters"
        )  # alternative way to get an item
        assert "hfq" in params["Fourier transform"]

    def test_block_metadata(self):
        """Test block-specific metadata."""
        optics = self.meta["params"]["optical"]
        assert "bms" in optics
        assert "acc" in optics
        assert optics["bms"].name == "Beamsplitter"
        assert optics["bms"].value == "KBr"

    def test_rf_params(self):
        """Test reference parameters metadata."""
        rf_params = self.meta["rf_params"]
        assert "acquisition" in rf_params
        assert "optical" in rf_params

        acquisition = rf_params["acquisition"]
        assert "aqm" in acquisition
        assert "del" in acquisition
        assert acquisition["AQM"].name == "Acquisition Mode"
        assert acquisition["DEL"].value == 0


class TestOpusMerging:
    """Test OPUS file merging behavior."""

    def test_single_origin_merge(self):
        """Test automatic merging of files with same origin."""
        # Reading multiple files with same origin
        datasets = NDDataset.read_opus(
            OPUSDATA / "test.0000", OPUSDATA / "test.0001", OPUSDATA / "test.0002"
        )

        # Should merge automatically when same origin and shape
        assert isinstance(datasets, NDDataset)
        assert datasets.shape == (3, 2567)
        assert datasets.origin == "opus-AB"

    def test_different_shapes_handling(self):
        """Test handling of files with different shapes."""
        datasets = NDDataset.read_opus(
            OPUSDATA / "test.0000",  # Shape (1, 2567)
            OPUSDATA / "background.0",  # Shape (1, 4096)
        )

        # Should keep separate due to different shapes
        assert len(datasets) == 2
        assert datasets[0].shape == (1, 2567)
        assert datasets[1].shape == (1, 4096)

    def test_merge_flag_behavior(self):
        """Test explicit merge flag behavior."""
        # Test merge=True
        merged = NDDataset.read_opus(
            OPUSDATA / "test.0000", OPUSDATA / "test.0001", merge=True
        )
        assert isinstance(merged, NDDataset)
        assert merged.shape == (2, 2567)

        # Test merge=False
        unmerged = NDDataset.read_opus(
            OPUSDATA / "test.0000", OPUSDATA / "test.0001", merge=False
        )
        assert len(unmerged) == 2
        assert all(ds.shape == (1, 2567) for ds in unmerged)

    def test_directory_reading(self):
        """Test reading and merging from directory."""
        # Test reading without merge
        datasets = NDDataset.read_opus(directory=OPUSDATA)
        assert len(datasets) > 0
        assert all(d.shape[1] in [2567, 4096] for d in datasets)

        # Test reading with merge
        grouped = NDDataset.read_opus(directory=OPUSDATA, merge=True)
        assert len(grouped) == 2  # Two groups by shape
        assert grouped[0].shape[1] == 2567
        assert grouped[1].shape[1] == 4096


if __name__ == "__main__":
    pytest.main([__file__])
