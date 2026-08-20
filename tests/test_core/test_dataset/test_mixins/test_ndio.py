# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the ndplugin module"""

import base64
import copy
import json
import pickle
import zipfile
from datetime import UTC
from datetime import datetime

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.project.project import Project
from spectrochempy.utils.exceptions import SpectroChemPyError
from spectrochempy.utils.jsonutils import json_loads
from spectrochempy.utils.testing import assert_array_equal, assert_dataset_equal

# Basic
# --------------------------------------------------------------------------------------


def _rewrite_dataset_data_payload_as_legacy_pickle(filename):
    current = NDDataset.load(filename)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    js.pop("__format__", None)
    js.pop("__version__", None)
    js["data"] = {
        "__class__": "NUMPY_ARRAY",
        "base64": base64.b64encode(pickle.dumps(current.data)).decode(),
    }

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(member, json.dumps(js, indent=2))


def _rewrite_project_dataset_payload_as_legacy_pickle(filename):
    current = Project.load(filename)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    js.pop("__format__", None)
    js.pop("__version__", None)
    js["datasets"][0]["data"] = {
        "__class__": "NUMPY_ARRAY",
        "base64": base64.b64encode(
            pickle.dumps(np.array(current.datasets[0].data)),
        ).decode(),
    }

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(member, json.dumps(js, indent=2))


def _make_exact_history_entries():
    return [
        (
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
            "ENTRY_A: first line\nsecond line",
        ),
        (
            datetime(2024, 1, 2, 3, 5, 5, tzinfo=UTC),
            "ENTRY_B / punctuation?! µ identical payload",
        ),
        (
            datetime(2024, 1, 2, 3, 6, 5, tzinfo=UTC),
            "ENTRY_B / punctuation?! µ identical payload",
        ),
    ]


def _make_history_dataset(name="native_history"):
    ds = NDDataset(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        dims=["y", "x"],
        coordset=[
            Coord([0.0, 1.0], name="y", title="time", units="s"),
            Coord([10.0, 20.0, 30.0], name="x", title="wavenumber", units="cm^-1"),
        ],
        name=name,
        title="native history demo",
        description="native roundtrip",
        meta={"sample": "demo", "nested": {"state": "exact-history"}},
        mask=np.array([[False, True, False], [False, False, False]]),
    )
    return ds


def _assert_exact_history(restored, expected):
    assert restored.history == expected.history
    assert restored._history == expected._history


def _assert_preserved_dataset_roundtrip(restored, expected):
    assert_dataset_equal(restored, expected)
    _assert_exact_history(restored, expected)


def test_ndio_generic(ndataset_1d, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ir = ndataset_1d
    ir.name = "IR_1D"

    # save with a default filename derived from the dataset name
    f = ir.save_as(tmp_path / ir.name)
    assert ir.filename.name == f.name
    assert ir.directory == tmp_path

    # load back this  file : the full path f is given so no dialog is opened
    nd = NDDataset.load(f)
    assert_dataset_equal(nd, ir)

    # as it has been already saved,
    f = nd.save()
    assert nd.filename.name == "IR_1D.scp"

    # now save it with a new name
    f = ir.save_as(tmp_path / "essai")
    assert ir.filename.name == f.name

    # remove these files
    f.unlink()

    # save in a specified directory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    ir.save_as(subdir / "essai")  # save essai.scp
    assert ir.directory == subdir
    assert ir.filename.name == "essai.scp"
    (subdir / ir.filename.name).unlink()

    # save in the current directory
    f = ir.save_as(tmp_path / "essai")

    # try to load without extension specification (will first assume it is scp)
    dl = NDDataset.load("essai")
    # assert dl.directory == cwd
    assert_array_equal(dl.data, ir.data)
    f.unlink()


def test_ndio_2D(ndataset_2d, tmp_path):
    # test with a 2D

    ir2 = ndataset_2d.copy()
    f = ir2.save_as(tmp_path / "essai2D", confirm=False)
    assert ir2.directory == tmp_path
    with pytest.raises(FileNotFoundError):
        NDDataset.load("essai2D")
    nd = NDDataset.load(tmp_path / "essai2D")
    assert nd.directory == tmp_path
    f.unlink()


def test_filename_none_copy_and_deepcopy_preserve_explicit_none():
    ds = NDDataset([1.0, 2.0, 3.0], name="filename_none")
    ds.filename = None

    shallow = ds.copy()
    deep = copy.deepcopy(ds)

    assert ds.filename is None
    assert shallow.filename is None
    assert deep.filename is None


def test_filename_none_not_serialized_as_extra_schema_field(tmp_path):
    ds = NDDataset([1.0, 2.0, 3.0], name="schema_filename_none")
    ds.filename = None
    filename = ds.save_as(tmp_path / "schema_filename_none", confirm=False)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    assert "_filename_explicit_none" not in js
    assert "filename_explicit_none" not in js

    loaded = NDDataset.load(filename)
    assert loaded.filename == filename


def test_ndio_roundtrip_preserves_selected_non_first_default(tmp_path):
    ds = NDDataset([0.0, 1.0, 2.0])
    ds.x = CoordSet(Coord([10.0, 20.0, 30.0]), Coord([100.0, 200.0, 300.0]))
    ds.x.select(2)
    selected_data = ds.x.data.copy()
    filename = ds.save_as(tmp_path / "multicoord_default", confirm=False)

    loaded = NDDataset.load(filename)

    assert loaded.x.default == loaded.x["_2"]
    assert_array_equal(loaded.x.default.data, selected_data)
    assert_array_equal(loaded.x.data, selected_data)


def test_ndio_roundtrip_preserves_reference_lookup(tmp_path):
    c = Coord([100.0, 200.0, 300.0], name="x")
    ds = NDDataset([1.0, 2.0, 3.0], coordset=CoordSet(x=c, y="x"))
    filename = ds.save_as(tmp_path / "reference_coords", confirm=False)

    loaded = NDDataset.load(filename)

    assert loaded.coordset.references == ds.coordset.references
    assert loaded.coordset["y"] == "x"
    assert_array_equal(loaded.y.data, loaded.x.data)
    assert_array_equal(loaded.x.data, [100.0, 200.0, 300.0])


def test_ndio_load_without_default_field_keeps_legacy_behavior(tmp_path):
    ds = NDDataset([0.0, 1.0, 2.0])
    ds.x = CoordSet(Coord([10.0, 20.0, 30.0]), Coord([100.0, 200.0, 300.0]))
    ds.x.select(2)
    legacy_default_data = ds.x["_1"].data.copy()
    filename = ds.save_as(tmp_path / "legacy_default", confirm=False)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    js["coordset"]["coords"][0].pop("default_index", None)

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(member, json.dumps(js, indent=2))

    loaded = NDDataset.load(filename)

    assert loaded.x.default == loaded.x["_1"]
    assert_array_equal(loaded.x.data, legacy_default_data)


def test_ndio_load_ignores_legacy_roi_fields(tmp_path):
    ds = NDDataset([0.0, 1.0, 2.0], coordset=[Coord([10.0, 20.0, 30.0], title="x")])
    filename = ds.save_as(tmp_path / "legacy_roi", confirm=False)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    js["roi"] = [0.0, 1.0]
    js["coordset"]["coords"][0]["roi"] = [10.0, 20.0]

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(member, json.dumps(js, indent=2))

    loaded = NDDataset.load(filename)

    assert not hasattr(loaded, "roi")
    assert not hasattr(loaded.x, "roi")


def test_ndio_load_ignores_legacy_modeldata_field(tmp_path):
    ds = NDDataset([0.0, 1.0, 2.0])
    filename = ds.save_as(tmp_path / "legacy_modeldata", confirm=False)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    js["modeldata"] = [42.0, 42.0, 42.0]

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(member, json.dumps(js, indent=2))

    loaded = NDDataset.load(filename)


def test_ndio_load_requires_explicit_opt_in_for_legacy_scp(tmp_path, monkeypatch):
    ds = NDDataset([0.0, 1.0, 2.0], name="legacy_dataset")
    filename = ds.save_as(tmp_path / "legacy_dataset", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    def fail_pickle_loads(*args, **kwargs):
        raise AssertionError("pickle.loads must not run in safe mode")

    with monkeypatch.context() as m:
        m.setattr("spectrochempy.utils.jsonutils.pickle.loads", fail_pickle_loads)

        with pytest.raises(
            SpectroChemPyError,
            match="trusted legacy loading",
        ):
            NDDataset.load(filename)

    loaded = NDDataset.load(filename, allow_unsafe_legacy=True)
    assert_dataset_equal(loaded, ds)


def test_ndio_load_content_requires_explicit_opt_in(tmp_path):
    ds = NDDataset([0.0, 1.0, 2.0], name="legacy_content")
    filename = ds.save_as(tmp_path / "legacy_content", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)
    content = filename.read_bytes()

    with pytest.raises(
        SpectroChemPyError,
        match="allow_unsafe_legacy=True",
    ):
        NDDataset.load("legacy_content.scp", content=content)

    loaded = NDDataset.load(
        "legacy_content.scp",
        content=content,
        allow_unsafe_legacy=True,
    )
    assert_dataset_equal(loaded, ds)


@pytest.mark.parametrize("loader_name", ["load", "read"])
def test_native_load_aliases_require_explicit_opt_in(tmp_path, loader_name):
    ds = NDDataset([0.0, 1.0, 2.0], name="legacy_alias")
    filename = ds.save_as(tmp_path / "legacy_alias", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)
    loader = getattr(scp, loader_name)

    with pytest.raises(
        SpectroChemPyError,
        match="trusted legacy loading",
    ):
        loader(filename)

    loaded = loader(filename, allow_unsafe_legacy=True)
    assert_dataset_equal(loaded, ds)

    assert not hasattr(loaded, "modeldata")


def test_ndio_safe_roundtrip_uses_versioned_payload(tmp_path):
    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="safe_dataset")
    filename = ds.save_as(tmp_path / "safe_dataset", confirm=False)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    assert js["__format__"] == "scp"
    assert js["__version__"] == 2
    assert js["data"]["encoding"] == "raw-base64"

    loaded = NDDataset.load(filename)
    assert_dataset_equal(loaded, ds)


@pytest.mark.parametrize(
    "history_entries",
    [
        pytest.param([], id="empty"),
        pytest.param(_make_exact_history_entries()[:1], id="single"),
        pytest.param(_make_exact_history_entries(), id="multiple"),
    ],
)
def test_ndio_loads_roundtrip_preserves_exact_history_without_zip(history_entries):
    ds = _make_history_dataset()
    ds._history = list(history_entries)

    rebuilt = NDDataset.loads(json_loads(ds.dumps()))

    _assert_preserved_dataset_roundtrip(rebuilt, ds)


def test_ndio_loads_restores_serialized_history_instead_of_using_public_setter():
    ds = _make_history_dataset()
    ds._history = _make_exact_history_entries()
    serialized_history = list(ds.history)

    rebuilt = NDDataset.loads(json_loads(ds.dumps()))
    setter_target = NDDataset()
    setter_target.history = serialized_history

    _assert_exact_history(rebuilt, ds)
    assert setter_target.history != ds.history
    assert setter_target._history != ds._history


def test_ndio_dump_load_preserves_exact_history_across_repeated_cycles(tmp_path):
    source = _make_history_dataset(name="history_cycles")
    source._history = _make_exact_history_entries()
    current = source

    for cycle in range(3):
        filename = current.save_as(tmp_path / f"history_cycle_{cycle}", confirm=False)
        current = NDDataset.load(filename)
        _assert_preserved_dataset_roundtrip(current, source)


def test_ndio_save_load_allows_normal_history_append_after_restore(
    tmp_path, monkeypatch
):
    source = _make_history_dataset(name="history_append")
    source._history = _make_exact_history_entries()[:2]
    restored = NDDataset.load(
        source.save_as(tmp_path / "history_append", confirm=False)
    )

    appended_at = datetime(2024, 1, 2, 4, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "spectrochempy.core.dataset.nddataset.utcnow", lambda: appended_at
    )
    restored.history = "ENTRY_C appended after load"

    reloaded = NDDataset.load(
        restored.save_as(tmp_path / "history_append_again", confirm=False)
    )

    assert reloaded._history[:2] == source._history
    assert reloaded._history[2] == (appended_at, "ENTRY_C appended after load")
    assert reloaded.history[-1].endswith("> ENTRY_C appended after load")


def test_native_public_scp_surfaces_preserve_exact_history(tmp_path):
    function_ds = _make_history_dataset(name="function_surface")
    function_ds._history = _make_exact_history_entries()
    function_filename = scp.write(
        function_ds,
        tmp_path / "function_surface.scp",
        overwrite=True,
    )
    function_loaded = scp.load(function_filename)
    _assert_preserved_dataset_roundtrip(function_loaded, function_ds)

    method_ds = _make_history_dataset(name="method_surface")
    method_ds._history = _make_exact_history_entries()
    method_filename = method_ds.write(tmp_path / "method_surface.scp", overwrite=True)
    method_loaded = scp.read(method_filename)
    _assert_preserved_dataset_roundtrip(method_loaded, method_ds)


def test_ndio_load_without_history_field_keeps_dataset_loadable(tmp_path):
    ds = _make_history_dataset(name="missing_history")
    ds._history = _make_exact_history_entries()
    filename = ds.save_as(tmp_path / "missing_history", confirm=False)

    with zipfile.ZipFile(filename, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))

    js.pop("history", None)

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(member, json.dumps(js, indent=2))

    loaded = NDDataset.load(filename)

    assert loaded.history == []
    assert_array_equal(loaded.data, ds.data)
    assert_array_equal(loaded.mask, ds.mask)
    assert loaded.meta["nested"] == ds.meta["nested"]


# ---------------------------------------------------------------------------
# Legacy migration tests
# ---------------------------------------------------------------------------


def test_migrate_legacy_file_requires_trust_acknowledgement(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_trust")
    filename = ds.save_as(tmp_path / "migrate_trust", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    with pytest.raises(SpectroChemPyError, match="allow_unsafe_legacy=True"):
        migrate_legacy_file(filename)

    with pytest.warns(UserWarning, match="trusted sources"):
        migrated = migrate_legacy_file(filename, allow_unsafe_legacy=True)

    loaded = NDDataset.load(migrated)
    assert_dataset_equal(loaded, ds)


def test_migrate_legacy_file_converts_to_safe_format(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_basic")
    filename = ds.save_as(tmp_path / "migrate_basic", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    with pytest.warns(UserWarning):
        migrated = migrate_legacy_file(filename, allow_unsafe_legacy=True, verbose=True)

    loaded = NDDataset.load(migrated)
    assert_dataset_equal(loaded, ds)

    with zipfile.ZipFile(migrated, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))
    assert js["data"]["encoding"] == "raw-base64"
    assert js["__format__"] == "scp"
    assert js["__version__"] == 2


def test_migrate_legacy_file_safe_load_after_migration(tmp_path, monkeypatch):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_nopickle")
    filename = ds.save_as(tmp_path / "migrate_nopickle", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    with pytest.warns(UserWarning):
        migrated = migrate_legacy_file(filename, allow_unsafe_legacy=True)

    def fail_pickle_loads(*args, **kwargs):
        raise AssertionError("pickle.loads must not run during safe default load")

    with monkeypatch.context() as m:
        m.setattr("spectrochempy.utils.jsonutils.pickle.loads", fail_pickle_loads)
        loaded = NDDataset.load(migrated)

    assert_array_equal(loaded.data, ds.data)


def test_migrate_legacy_file_preserves_coords(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    x = Coord(np.linspace(4000, 400, 100), title="wavenumber")
    ds = NDDataset(
        np.random.default_rng(42).random((5, 100)),
        name="migrate_coords",
        coord=(Coord(np.arange(5)), x),
    )
    filename = ds.save_as(tmp_path / "migrate_coords", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    with pytest.warns(UserWarning):
        migrated = migrate_legacy_file(filename, allow_unsafe_legacy=True)
    loaded = NDDataset.load(migrated)

    assert_dataset_equal(loaded, ds)


def test_migrate_legacy_file_raises_for_already_safe(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_already_safe")
    filename = ds.save_as(tmp_path / "migrate_already_safe", confirm=False)

    with pytest.raises(SpectroChemPyError, match="already in safe format"):
        migrate_legacy_file(filename, allow_unsafe_legacy=True)


def test_migrate_legacy_file_raises_for_nonexistent_source():
    from spectrochempy.utils.persistence import migrate_legacy_file

    with pytest.raises(FileNotFoundError):
        migrate_legacy_file("/nonexistent/file.scp", allow_unsafe_legacy=True)


def test_migrate_legacy_file_raises_for_wrong_extension(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    f = tmp_path / "data.csv"
    f.write_text("a,b,c")

    with pytest.raises(ValueError, match="must have a .scp or .pscp extension"):
        migrate_legacy_file(f, allow_unsafe_legacy=True)


def test_migrate_legacy_file_raises_when_destination_exists(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_exists")
    filename = ds.save_as(tmp_path / "migrate_exists", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    dest = tmp_path / "migrate_exists_migrated.scp"
    dest.write_text("existing content")

    with pytest.raises(FileExistsError):
        migrate_legacy_file(filename, allow_unsafe_legacy=True)

    with pytest.warns(UserWarning):
        migrated = migrate_legacy_file(
            filename, destination=dest, allow_unsafe_legacy=True, overwrite=True
        )
    loaded = NDDataset.load(migrated)
    assert_array_equal(loaded.data, ds.data)


def test_migrate_legacy_file_raises_for_source_equals_destination(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_same")
    filename = ds.save_as(tmp_path / "migrate_same", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    with pytest.raises(ValueError, match="resolve to the same path"):
        migrate_legacy_file(filename, destination=filename, allow_unsafe_legacy=True)

    assert filename.exists()


def test_migrate_legacy_file_default_destination(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_default_dest")
    filename = ds.save_as(tmp_path / "migrate_default_dest", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    with pytest.warns(UserWarning):
        migrated = migrate_legacy_file(filename, allow_unsafe_legacy=True)

    assert migrated == tmp_path / "migrate_default_dest_migrated.scp"
    loaded = NDDataset.load(migrated)
    assert_dataset_equal(loaded, ds)


def test_migrate_legacy_file_preserves_existing_destination_on_failure(
    tmp_path, monkeypatch
):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_cleanup")
    filename = ds.save_as(tmp_path / "migrate_cleanup", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    dest = tmp_path / "migrate_cleanup_migrated.scp"
    dest.write_text("original content that must be preserved")

    def fail_dump(self, f, **kwargs):
        raise RuntimeError("dump failed")

    with monkeypatch.context() as m:
        m.setattr(NDDataset, "dump", fail_dump)
        with pytest.raises(RuntimeError, match="dump failed"):
            migrate_legacy_file(
                filename,
                destination=dest,
                allow_unsafe_legacy=True,
                overwrite=True,
            )

    # Original destination must survive the failure.
    assert dest.exists()
    assert dest.read_text() == "original content that must be preserved"


def test_migrate_legacy_file_pscp_roundtrip(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds1 = NDDataset(np.array([1.0, 2.0, 3.0]), name="pscp_child1")
    ds2 = NDDataset(np.array([4.0, 5.0, 6.0]), name="pscp_child2")
    proj = Project(name="migrate_pscp_project")
    proj.add_dataset(ds1)
    proj.add_dataset(ds2)

    filename = proj.save_as(tmp_path / "migrate_pscp_project", confirm=False)
    _rewrite_project_dataset_payload_as_legacy_pickle(filename)

    with pytest.warns(UserWarning):
        migrated = migrate_legacy_file(filename, allow_unsafe_legacy=True)

    loaded = Project.load(migrated)
    assert len(loaded.datasets) == 2
    assert_array_equal(loaded["pscp_child1"].data, ds1.data)
    assert_array_equal(loaded["pscp_child2"].data, ds2.data)

    with zipfile.ZipFile(migrated, "r") as zipf:
        member = zipf.namelist()[0]
        js = json.loads(zipf.read(member).decode("utf-8"))
    assert js["__format__"] == "pscp"
    assert js["__version__"] == 2


def test_migrate_legacy_file_atomic_replaces_existing(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_atomic")
    filename = ds.save_as(tmp_path / "migrate_atomic", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    dest = tmp_path / "migrate_atomic_migrated.scp"
    dest.write_text("old content")

    with pytest.warns(UserWarning):
        migrated = migrate_legacy_file(
            filename, destination=dest, allow_unsafe_legacy=True, overwrite=True
        )

    loaded = NDDataset.load(migrated)
    assert_dataset_equal(loaded, ds)


def test_migrate_legacy_file_rejects_cross_suffix(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_cross")
    filename = ds.save_as(tmp_path / "migrate_cross", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    with pytest.raises(ValueError, match="does not match source suffix"):
        migrate_legacy_file(
            filename, destination=tmp_path / "wrong.pscp", allow_unsafe_legacy=True
        )


def test_migrate_legacy_file_rejects_destination_without_suffix(tmp_path):
    from spectrochempy.utils.persistence import migrate_legacy_file

    ds = NDDataset(np.array([1.0, 2.0, 3.0]), name="migrate_nosuffix")
    filename = ds.save_as(tmp_path / "migrate_nosuffix", confirm=False)
    _rewrite_dataset_data_payload_as_legacy_pickle(filename)

    with pytest.raises(ValueError, match="must have a .scp or .pscp extension"):
        migrate_legacy_file(
            filename, destination=tmp_path / "output", allow_unsafe_legacy=True
        )


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
