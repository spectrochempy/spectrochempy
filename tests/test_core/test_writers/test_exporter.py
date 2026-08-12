# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path

import pytest
from spectrochempy.utils import testing

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset


def test_write(mock_cwd, ndataset_1d):
    nd = ndataset_1d.copy()
    nd.name = "synthetic"

    # API write methods needs an instance of a NDDataset as the first argument
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'dataset'"
    ):
        scp.write()

    # the simplest way to save a dataset, is to use the function write with a filename as argument
    filename = nd.write("essai.scp")  # should not open a DIALOG
    assert filename == mock_cwd / "essai.scp"
    assert filename.exists()

    # try to write it again
    with pytest.raises(FileExistsError):
        nd.write("essai.scp")

    # write it again with overwrite
    filename = nd.write("essai.scp", overwrite=True)

    # Read the file and compare
    nd2 = NDDataset.load(filename)
    testing.assert_dataset_equal(nd2, nd)

    # we can also use the read method to read it
    nd3 = scp.read(filename)
    testing.assert_dataset_equal(nd3, nd)

    filename.unlink()

    # if the filename is omitted, write a file with the dataset name and the extension '.scp'
    filename = nd.write()
    assert filename is not None
    assert filename.stem == nd.name
    assert filename.suffix == ".scp"
    filename.unlink()


def test_excel_writer_entry_points_are_not_public(ndataset_1d):
    nd = ndataset_1d.copy()

    assert not hasattr(scp, "write_excel")
    assert not hasattr(scp, "write_xls")
    assert not hasattr(nd, "write_excel")
    assert not hasattr(nd, "write_xls")


def test_write_xls_uses_unsupported_format_path(ndataset_1d):
    nd = ndataset_1d.copy()

    with pytest.raises(ValueError, match=r"Unsupported export format `\.xls`"):
        nd.write("unsupported.xls", overwrite=True)


def test_generic_write_protocol_kwarg_warns_but_keeps_default_scp_save(
    mock_cwd, ndataset_1d
):
    nd = ndataset_1d.copy()
    nd.name = "synthetic"
    target = Path("synthetic.scp")
    if target.exists():
        target.unlink()

    with pytest.warns(DeprecationWarning, match="`protocol` is deprecated") as record:
        filename = scp.write(nd, protocol="csv")

    assert filename.stem == "synthetic"
    assert filename.suffix == ".scp"
    assert filename.exists()
    messages = [str(item.message) for item in record]
    assert all("0.13.0" not in message for message in messages)
    assert any("deprecation policy is satisfied" in message for message in messages)

    nd2 = NDDataset.load(filename)
    testing.assert_dataset_equal(nd2, nd)
    filename.unlink()


def test_generic_write_description_kwarg_warns_but_uses_dataset_description(
    tmp_path, ndataset_1d
):
    nd = ndataset_1d.copy()
    nd.name = "synthetic"
    nd.description = "source description"
    original_description = nd.description

    target = tmp_path / "synthetic.scp"

    with pytest.warns(
        DeprecationWarning, match="`description` is deprecated"
    ) as record:
        filename = scp.write(nd, target, description="custom", confirm=False)

    assert filename == target
    assert filename.exists()
    assert nd.description == original_description
    messages = [str(item.message) for item in record]
    assert all("0.13.0" not in message for message in messages)
    assert any("deprecation policy is satisfied" in message for message in messages)

    reloaded = NDDataset.load(filename)
    assert reloaded.description == "source description"


# EOF
