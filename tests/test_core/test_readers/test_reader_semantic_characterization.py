# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Semantic characterization baseline for high-value core readers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.read_labspec import _read_txt
from spectrochempy.utils.datetimeutils import UTC
from tests.test_core.test_readers._reader_semantic_helpers import (
    assert_coordinate_semantics,
)
from tests.test_core.test_readers._reader_semantic_helpers import (
    assert_dataset_identity,
)
from tests.test_core.test_readers._reader_semantic_helpers import (
    assert_dataset_provenance,
)
from tests.test_core.test_readers._reader_semantic_helpers import assert_history_present
from tests.test_core.test_readers._reader_semantic_helpers import assert_label_structure
from tests.test_core.test_readers._reader_semantic_helpers import (
    assert_meta_keys_present,
)

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"
OPUSDATA = DATADIR / "irdata" / "OPUS"
RAMANDIR = DATADIR / "ramandata" / "labspec"
WODGER = Path(__file__).parent / "ressources" / "omnic" / "wodger.spg"

pytestmark = pytest.mark.data


@pytest.fixture
def omnic_spg_dataset():
    return scp.read_omnic(WODGER)


@pytest.fixture
def omnic_spa_dataset():
    path = IRDATA / "subdir" / "20-50" / "7_CZ0-100_Pd_21.SPA"
    if not path.exists():
        pytest.skip("OMNIC SPA characterization data not available")
    return scp.read_spa(path)


@pytest.fixture
def omnic_srs_dataset():
    path = IRDATA / "omnic_series" / "rapid_scan.srs"
    if not path.exists():
        pytest.skip("OMNIC SRS characterization data not available")
    return scp.read_srs(path)


@pytest.fixture
def opus_single_dataset():
    path = OPUSDATA / "test.0000"
    if not path.exists():
        pytest.skip("OPUS characterization data not available")
    return scp.read_opus(path)


@pytest.fixture
def opus_assembled_dataset():
    path = OPUSDATA / "OPUS_assembled_file.0"
    if not path.exists():
        pytest.skip("assembled OPUS characterization data not available")
    return scp.read_opus(path)


@pytest.fixture
def jcamp_linked_dataset(JDX_2D):
    return scp.read_jcamp({"semantic_linked.jdx": JDX_2D.encode("utf8")})


@pytest.fixture
def jcamp_single_with_owner():
    content = """##TITLE=single_semantic
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##ORIGIN=omnic
##OWNER=reader-owner
##LONGDATE=2016/07/06
##TIME=19:03:14
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=3997.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 1.0 0.9 0.8 0.7
##END
"""
    return scp.read_jcamp({"single_semantic.jdx": content.encode("utf8")})


@pytest.fixture
def jcamp_single_without_date():
    content = """##TITLE=single_no_date
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=3997.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 1.0 0.9 0.8 0.7
##END
"""
    return scp.read_jcamp({"single_no_date.jdx": content.encode("utf8")})


@pytest.fixture
def jcamp_linked_multi_origin_dataset():
    content = """##TITLE=IR_multi_origin
##JCAMP-DX=5.01
##DATA TYPE=LINK
##BLOCKS=2
##TITLE=spec_1
##ORIGIN=omnic
##LONGDATE=2016/07/06
##TIME=19:03:14
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=3997.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 1.0 0.9 0.8 0.7
##END=
##TITLE=spec_2
##ORIGIN=labspec
##LONGDATE=2016/07/06
##TIME=19:04:14
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=3997.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 0.7 0.6 0.5 0.4
##END=
"""
    return scp.read_jcamp({"linked_multi_origin.jdx": content.encode("utf8")})


@pytest.fixture
def csv_omnic_dataset():
    content = "4000.0,0.5\n4001.0,0.6\n4002.0,0.7\n"
    return scp.read_csv(
        {"sample_Mon Aug 05 12-34-56 2024.csv": content.encode("utf-8")},
        origin="omnic",
    )


@pytest.fixture
def csv_generic_dataset():
    content = "1.0,10.0\n2.0,20.0\n3.0,30.0\n"
    return scp.read_csv({"generic.csv": content.encode("utf-8")})


@pytest.fixture
def labspec_synthetic_dataset():
    content = (
        "#Acq. time (s)=1\n"
        "#Dark correction=No\n"
        "#Acquired=01.01.2024 00:00:01\n"
        "#Accumulations=1\n"
        "#Comment=20\xb0C\n"
        "100\t1\n"
        "101\t2\n"
    ).encode("latin-1")
    return _read_txt(NDDataset(), Path("latin_labspec.txt"), content=content)


@pytest.fixture
def labspec_real_dataset():
    path = RAMANDIR / "Activation.txt"
    if not path.exists():
        pytest.skip("LabSpec characterization data not available")
    return scp.read_labspec(path)


class TestOmnicCharacterization:
    """Characterize current OMNIC semantic placement without changing it."""

    def test_spg_identity_provenance_coordinates_and_labels(self, omnic_spg_dataset):
        dataset = omnic_spg_dataset

        assert_dataset_identity(
            dataset,
            name="wodger",
            title="absorbance",
            units="absorbance",
        )
        assert_dataset_provenance(
            dataset,
            filename_name="wodger.spg",
            origin="omnic",
            description_contains="Omnic title: wodger.spg",
            acquisition_date_present=True,
        )
        assert_history_present(dataset, "Imported from spg file", "Sorted by date")

        x = assert_coordinate_semantics(dataset, "x", title="wavenumbers", units="cm⁻¹")
        y = assert_coordinate_semantics(
            dataset, "y", title="acquisition timestamp (GMT)", units="s"
        )
        labels = assert_label_structure(y, shape=(2, 2))
        assert isinstance(labels[0, 0], str)
        assert isinstance(labels[1, 0], datetime)
        assert x.labels is None

    def test_spa_uses_omnic_origin_and_label_rows(self, omnic_spa_dataset):
        dataset = omnic_spa_dataset

        assert_dataset_provenance(
            dataset,
            origin="omnic",
            acquisition_date_present=True,
        )
        assert_history_present(dataset, "Imported from spa file")
        assert_coordinate_semantics(
            dataset, "y", title="acquisition timestamp (GMT)", units="s"
        )
        labels = assert_label_structure(dataset.y, shape=(1, 2))
        assert isinstance(labels[0, 0], datetime)
        assert Path(labels[0, 1]).name == dataset.filename.name
        assert_meta_keys_present(
            dataset,
            "collection_length",
            "optical_velocity",
            "laser_frequency",
        )

    def test_srs_currently_sets_origin_and_history(self, omnic_srs_dataset):
        dataset = omnic_srs_dataset

        assert_dataset_provenance(
            dataset,
            origin="omnic",
            acquisition_date_present=False,
        )
        if dataset.history:
            history_text = " ".join(str(entry) for entry in dataset.history).lower()
            assert "srs file" in history_text
        assert_coordinate_semantics(dataset, "x")
        assert_coordinate_semantics(dataset, "y")
        if dataset.y.labels is not None:
            assert_label_structure(dataset.y)
        assert "laser_frequency" in dataset.meta
        assert "collection_length" in dataset.meta


class TestOpusCharacterization:
    """Characterize current OPUS semantic placement."""

    def test_single_file_identity_provenance_labels_and_meta(self, opus_single_dataset):
        dataset = opus_single_dataset

        assert_dataset_identity(
            dataset,
            name="test",
            title="absorbance",
            units="absorbance",
        )
        assert_dataset_provenance(
            dataset,
            filename_name="test.0000",
            origin="opus-AB",
            description_contains="opus files",
            acquisition_date_present=True,
        )
        assert_history_present(dataset, "import from opus files")

        x = assert_coordinate_semantics(dataset, "x", title="wavenumber")
        assert str(x.units) in {"cm^-1", "cm⁻¹"}
        y = assert_coordinate_semantics(
            dataset, "y", title="acquisition timestamp (GMT)", units="s"
        )
        labels = np.asarray(assert_label_structure(y))
        assert labels.size >= 1

        assert_meta_keys_present(dataset, "params", "rf_params", "other_data_types")
        assert dataset.meta.readonly

    def test_assembled_file_currently_uses_elapsed_time_without_point_labels(
        self, opus_assembled_dataset
    ):
        dataset = opus_assembled_dataset

        assert_dataset_provenance(
            dataset,
            origin="opus-AB",
            acquisition_date_present=True,
        )
        assert_history_present(dataset, "import from opus files")
        assert_coordinate_semantics(dataset, "x", title="wavenumber", units="cm⁻¹")
        y = assert_coordinate_semantics(dataset, "y", title="elapsed time", units="s")
        assert y.labels is None


class TestJcampCharacterization:
    """Characterize current JCAMP semantic placement and limitations."""

    def test_linked_jcamp_maps_title_origin_dates_labels_and_history(
        self, jcamp_linked_dataset
    ):
        dataset = jcamp_linked_dataset

        assert_dataset_identity(
            dataset,
            name="IR_2D",
            title="absorbance",
            units="absorbance",
        )
        assert_dataset_provenance(
            dataset,
            filename_name="semantic_linked.jdx",
            origin="omnic",
            description_contains="Dataset from jdx file: 'IR_2D'",
            acquisition_date_present=True,
        )
        assert dataset._acquisition_date == datetime(2016, 7, 6, 19, 3, 14, tzinfo=UTC)
        assert_history_present(dataset, "Imported from jdx file", "Sorted by date")
        assert_coordinate_semantics(dataset, "x", title="wavenumbers", units="cm⁻¹")
        y = assert_coordinate_semantics(
            dataset, "y", title="acquisition timestamp (GMT)", units="s"
        )
        labels = assert_label_structure(y, shape=(3, 2))
        assert isinstance(labels[0, 0], datetime)
        assert isinstance(labels[0, 1], str)

    def test_single_jcamp_preserves_origin_and_keeps_owner_unmapped(
        self, jcamp_single_with_owner
    ):
        dataset = jcamp_single_with_owner

        assert_dataset_identity(
            dataset,
            name="single_semantic",
            title="absorbance",
            units="absorbance",
        )
        assert_dataset_provenance(
            dataset,
            filename_name="single_semantic.jdx",
            origin="omnic",
            description_contains="Dataset from jdx file: 'single_semantic'",
            acquisition_date_present=True,
        )
        assert dataset._acquisition_date == datetime(2016, 7, 6, 19, 3, 14, tzinfo=UTC)
        assert dataset.author != "reader-owner"
        assert dataset.y.is_empty
        assert_history_present(dataset, "Imported from jdx file")

    def test_linked_jcamp_multi_origin_uses_deterministic_join(
        self, jcamp_linked_multi_origin_dataset
    ):
        dataset = jcamp_linked_multi_origin_dataset

        assert_dataset_provenance(
            dataset,
            filename_name="linked_multi_origin.jdx",
            origin="labspec; omnic",
            description_contains="Dataset from jdx file: 'IR_multi_origin'",
            acquisition_date_present=True,
        )
        assert_history_present(dataset, "Imported from jdx file", "Sorted by date")

    def test_single_jcamp_without_date_keeps_acquisition_date_empty(
        self, jcamp_single_without_date
    ):
        dataset = jcamp_single_without_date

        assert_dataset_provenance(
            dataset,
            filename_name="single_no_date.jdx",
            origin="",
            description_contains="Dataset from jdx file: 'single_no_date'",
            acquisition_date_present=False,
        )
        assert dataset.y.is_empty


class TestCsvCharacterization:
    """Characterize generic CSV and OMNIC CSV semantic placement."""

    def test_generic_csv_currently_uses_minimal_identity_and_provenance(
        self, csv_generic_dataset
    ):
        dataset = csv_generic_dataset

        assert_dataset_identity(dataset, name="generic")
        assert_dataset_provenance(
            dataset,
            filename_name="generic.csv",
            origin="",
            description_contains="read from .csv file",
            acquisition_date_present=False,
        )
        assert_history_present(dataset, "Read from .csv file")
        assert_coordinate_semantics(dataset, "x")
        assert_coordinate_semantics(dataset, "y")
        assert dataset.y.labels is None

    def test_omnic_csv_currently_places_dates_and_sample_name_in_y_labels(
        self, csv_omnic_dataset
    ):
        dataset = csv_omnic_dataset

        assert_dataset_identity(
            dataset,
            name="sample_Mon Aug 05 12-34-56 2024",
            title="absorbance",
            units="absorbance",
        )
        assert_dataset_provenance(
            dataset,
            filename_name="sample_Mon Aug 05 12-34-56 2024.csv",
            origin="omnic",
            description_contains="Dataset from .csv file",
            acquisition_date_present=False,
        )
        assert_history_present(dataset, "Read from .csv file", "Read from omnic")

        assert_coordinate_semantics(dataset, "x", title="wavenumbers", units="cm⁻¹")
        y = assert_coordinate_semantics(
            dataset, "y", title="acquisition timestamp (GMT)", units="s"
        )
        labels = assert_label_structure(y, shape=(1, 2))
        assert isinstance(labels[0, 0], datetime)
        assert labels[0, 1] == "sample"


class TestLabSpecCharacterization:
    """Characterize current LabSpec semantic placement."""

    def test_synthetic_labspec_uses_labspec_origin_description_and_meta(
        self, labspec_synthetic_dataset
    ):
        dataset = labspec_synthetic_dataset

        assert_dataset_identity(
            dataset,
            name="latin_labspec",
            title="Counts",
            units=None,
        )
        assert_dataset_provenance(
            dataset,
            filename_name="latin_labspec.txt",
            origin="labspec",
            description_contains="Spectrum acquisition : 2024-01-01 00:00:00",
            acquisition_date_present=True,
        )
        assert dataset._acquisition_date == datetime(2024, 1, 1, 0, 0, 0)
        assert_history_present(dataset, "Imported from LabSpec6 text file")

        assert_coordinate_semantics(dataset, "x", title="Raman shift", units="cm⁻¹")
        y = assert_coordinate_semantics(dataset, "y", title="Time", units="s")
        assert y.labels is None
        assert_meta_keys_present(dataset, "Comment", "Acquired", "Accumulations")

    def test_real_labspec_series_uses_labspec_origin_and_datetime_label_rows(
        self, labspec_real_dataset
    ):
        dataset = labspec_real_dataset

        assert_dataset_provenance(
            dataset,
            origin="labspec",
            acquisition_date_present=True,
        )
        assert_history_present(dataset, "Imported from LabSpec6 text file")
        assert_coordinate_semantics(dataset, "x", title="Raman shift", units="cm⁻¹")
        y = assert_coordinate_semantics(dataset, "y", title="Time", units="s")
        labels = assert_label_structure(y)
        assert isinstance(labels[0], datetime)
