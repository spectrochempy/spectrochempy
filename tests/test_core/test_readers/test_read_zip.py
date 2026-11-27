# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import io
import warnings
import zipfile
from contextlib import nullcontext

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy import read_zip
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.utils.objects import ScpObjectList
from tests.test_core.test_readers.test_read_spc import _make_mxy_spc, _make_xy_spc

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


@pytest.mark.parametrize("options", [{}, {"merge": False}, {"merge": True}])
@pytest.mark.parametrize("depth", [1, 2])
def test_zip_member_groups_are_flat_and_compatible_scans_still_merge(options, depth):
    inner = _make_synthetic_zip({"a.csv": "1,10\n2,20\n", "b.csv": "1,30\n2,40\n"})
    for _ in range(depth - 1):
        inner = _make_synthetic_zip({"deeper.zip": inner})
    content = _make_synthetic_zip({"inner.zip": inner, "c.csv": "1,50\n2,60\n"})
    result = read_zip({"outer.zip": content}, **options)
    datasets = [result] if isinstance(result, scp.NDDataset) else result
    assert all(isinstance(ds, scp.NDDataset) for ds in datasets)
    rows = sorted(tuple(row) for ds in datasets for row in np.atleast_2d(ds.data))
    assert rows == [(10, 20), (30, 40), (50, 60)]
    if options.get("merge") is False:
        assert sorted(ds.name for ds in datasets) == ["a", "b", "c"]
    else:
        assert len(datasets) == 1


@pytest.mark.parametrize("only, expected", [(1, ["a", "b"]), (2, ["a", "b", "c"])])
def test_zip_only_counts_members_rather_than_returned_groups(only, expected):
    inner = _make_synthetic_zip({"a.csv": "1,10\n2,20\n", "b.csv": "1,30\n2,40\n"})
    content = _make_synthetic_zip({"inner.zip": inner, "c.csv": "1,50\n2,60\n"})
    result = read_zip({"outer.zip": content}, merge=False, only=only)
    assert sorted(ds.name for ds in result) == expected


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("from_path", [False, True])
def test_zip_heterogeneous_spc_preserves_every_spectrum_and_its_axis(
    tmp_path, merge, from_path
):
    multi, _, _ = _make_mxy_spc(x_lists=[[1, 2], [3, 4]], y_lists=[[10, 20], [30, 40]])
    single = _make_xy_spc(
        x_values=np.array([1, 2], dtype="<f4"),
        y_values=np.array([50, 60], dtype="<f4"),
    )
    content = _make_synthetic_zip({"multi.spc": multi, "single.spc": single})
    if from_path:
        source = tmp_path / "spectra.zip"
        source.write_bytes(content)
    else:
        source = {"spectra.zip": content}
    expected_warning = (
        pytest.warns(UserWarning, match="coordinates differ")
        if merge
        else nullcontext()
    )
    with expected_warning:
        result = read_zip(source, merge=merge)
    assert isinstance(result, ScpObjectList)
    assert all(isinstance(ds, scp.NDDataset) for ds in result)
    actual = sorted(
        (tuple(ds.x.data), tuple(row))
        for ds in result
        for row in np.atleast_2d(ds.data)
    )
    assert actual == [((1, 2), (10, 20)), ((1, 2), (50, 60)), ((3, 4), (30, 40))]


def test_zip_nested_native_quantity_conflicts_remain_separate(tmp_path):
    members = {}
    for name, units, values in (
        ("abs", "absorbance", [0.1, 0.2]),
        ("trans", "transmittance", [20, 30]),
    ):
        ds = scp.NDDataset(values, units=units, coordset=[scp.Coord([1, 2])])
        path = ds.write(tmp_path / f"{name}.scp")
        members[f"{name}.scp"] = path.read_bytes()
    inner = _make_synthetic_zip(members)
    content = _make_synthetic_zip(
        {"inner.zip": inner, "another.scp": members["abs.scp"]}
    )
    with pytest.warns(UserWarning, match="signal quantities differ"):
        result = read_zip({"quantities.zip": content}, merge=True)
    assert isinstance(result, ScpObjectList)
    assert all(isinstance(ds, scp.NDDataset) for ds in result)
    trans = [ds for ds in result if ds.units == scp.ur.Unit("transmittance")]
    assert len(trans) == 1
    np.testing.assert_array_equal(trans[0].data, [20, 30])
    assert sum(np.atleast_2d(ds.data).shape[0] for ds in result) == 3
