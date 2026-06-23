# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import io
import warnings
import zipfile

import pytest

from spectrochempy import read_zip
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.utils.objects import ScpObjectList

DATADIR = prefs.datadir
AGIRDATA = DATADIR / "agirdata"


@pytest.fixture(autouse=True)
def _skip_if_no_testdata(request):
    if request.node.get_closest_marker("data") and not AGIRDATA.exists():
        pytest.skip("test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")


@pytest.mark.data
def test_read_zip():
    A = read_zip(
        "agirdata/P350/FTIR/FTIR.zip",
        origin="omnic",
        only=10,
        csv_delimiter=";",
        merge=True,
    )
    assert A.shape == (10, 2843)

    # Test bytes contents for ZIP files
    z = DATADIR / "agirdata" / "P350" / "FTIR" / "FTIR.zip"
    content2 = z.read_bytes()
    B = read_zip(
        {"name.zip": content2}, origin="omnic", only=10, csv_delimiter=";", merge=True
    )
    assert B.shape == (10, 2843)

    # Test read_zip with several contents
    C = read_zip(
        {"name1.zip": content2, "name2.zip": content2},
        origin="omnic",
        only=10,
        csv_delimiter=";",
        merge=True,
    )
    assert C.shape == (20, 2843)


def _make_synthetic_zip(mapping):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in mapping.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def test_read_zip_root_files_synthetic_csv_merge_false():
    content = _make_synthetic_zip(
        {
            "sample1.csv": "1,10\n2,20\n3,30\n",
            "sample2.csv": "1,11\n2,21\n3,31\n",
        }
    )

    datasets = read_zip({"root.zip": content}, merge=False)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 2
    assert sorted(ds.name for ds in datasets) == ["sample1", "sample2"]


def test_read_zip_single_top_level_directory_synthetic_csv_merge_false():
    content = _make_synthetic_zip(
        {
            "experiment/sample1.csv": "1,10\n2,20\n3,30\n",
            "experiment/sample2.csv": "1,11\n2,21\n3,31\n",
        }
    )

    datasets = read_zip({"single_dir.zip": content}, merge=False)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 2
    assert sorted(ds.name for ds in datasets) == ["sample1", "sample2"]


def test_read_zip_nested_directories_synthetic_csv_merge_false():
    content = _make_synthetic_zip(
        {
            "experiment1/sample1.csv": "1,10\n2,20\n3,30\n",
            "experiment1/sample2.csv": "1,11\n2,21\n3,31\n",
            "experiment2/sub/sample3.csv": "1,12\n2,22\n3,32\n",
            "experiment2/sub/sample4.csv": "1,13\n2,23\n3,33\n",
            "__MACOSX/ignored.csv": "1,999\n2,999\n",
            "experiment2/.DS_Store": "ignored",
        }
    )

    datasets = read_zip({"nested.zip": content}, merge=False)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 4
    assert sorted(ds.name for ds in datasets) == [
        "sample1",
        "sample2",
        "sample3",
        "sample4",
    ]


def test_read_zip_mixed_root_and_nested_files_are_all_discovered():
    content = _make_synthetic_zip(
        {
            "root.csv": "1,10\n2,20\n",
            "exp/nested.csv": "1,30\n2,40\n",
        }
    )

    datasets = read_zip({"mixed.zip": content}, merge=False)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 2
    assert sorted(ds.name for ds in datasets) == ["nested", "root"]


def test_read_zip_nested_only_preserves_discovery_limiting_semantics():
    content = _make_synthetic_zip(
        {
            "experiment1/sample1.csv": "1,10\n2,20\n3,30\n",
            "experiment1/sample2.csv": "1,11\n2,21\n3,31\n",
            "experiment2/sub/sample3.csv": "1,12\n2,22\n3,32\n",
            "experiment2/sub/sample4.csv": "1,13\n2,23\n3,33\n",
        }
    )

    datasets = read_zip({"nested_only.zip": content}, merge=False, only=2)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 2
    assert sorted(ds.name for ds in datasets) == ["sample1", "sample2"]


def test_read_zip_only_uses_archive_discovery_order_for_mixed_archives():
    content = _make_synthetic_zip(
        {
            "root.csv": "1,10\n2,20\n",
            "exp/nested.csv": "1,30\n2,40\n",
            "later.csv": "1,50\n2,60\n",
        }
    )

    datasets = read_zip({"mixed_only.zip": content}, merge=False, only=2)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 2
    assert sorted(ds.name for ds in datasets) == ["nested", "root"]


def test_read_zip_nested_identical_basenames_keep_current_naming_behavior():
    content = _make_synthetic_zip(
        {
            "experiment1/sample.csv": "1,10\n2,20\n",
            "experiment2/sample.csv": "1,30\n2,40\n",
        }
    )

    datasets = read_zip({"duplicate_names.zip": content}, merge=False)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 2
    assert [ds.name for ds in datasets] == ["sample", "sample"]


def test_read_zip_reports_ignored_unsupported_files_in_mixed_archive():
    content = _make_synthetic_zip(
        {
            "sample1.csv": "1,10\n2,20\n",
            "sample2.csv": "1,11\n2,21\n",
            "notes.txt": "notes",
            "calibration.dat": "binary-ish",
        }
    )

    with pytest.warns(UserWarning, match="ignored because no reader is available"):
        datasets = read_zip({"mixed_ignored.zip": content}, merge=False)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 2
    assert sorted(ds.name for ds in datasets) == ["sample1", "sample2"]


def test_read_zip_no_warning_when_all_files_are_supported():
    content = _make_synthetic_zip(
        {
            "sample1.csv": "1,10\n2,20\n",
            "sample2.csv": "1,11\n2,21\n",
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        datasets = read_zip({"all_supported.zip": content}, merge=False)

    assert isinstance(datasets, ScpObjectList)
    assert len(datasets) == 2
    assert [
        w for w in caught if "ignored because no reader is available" in str(w.message)
    ] == []


def test_read_zip_reports_ignored_files_in_nested_archives():
    content = _make_synthetic_zip(
        {
            "experiment/sample1.csv": "1,10\n2,20\n",
            "experiment/notes.txt": "notes",
            "calibration.dat": "binary-ish",
        }
    )

    with pytest.warns(UserWarning, match="notes.txt"):
        datasets = read_zip({"nested_ignored.zip": content}, merge=False)

    assert datasets.name == "sample1"
