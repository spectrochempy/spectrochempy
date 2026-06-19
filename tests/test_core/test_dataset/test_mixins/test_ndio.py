# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the ndplugin module"""

import base64
import json
import pickle
import zipfile

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.exceptions import SpectroChemPyError
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


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
