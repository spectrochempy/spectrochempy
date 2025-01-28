# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

import spectrochempy as scp


def test_write_csv(IR_dataset_2D):
    # 1D dataset without coords
    ds = scp.NDDataset([1, 2, 3])
    f = ds.write_csv("myfile.csv", confirm=False)
    assert f.name == "myfile.csv"
    f.unlink()

    f = ds.write_csv("myfile", confirm=False)
    assert f.name == "myfile.csv"
    f.unlink()

    # 1D dataset with coords
    ds = IR_dataset_2D[0]
    f = ds.write_csv("myfile.csv", confirm=False)
    assert f.name == "myfile.csv"
    f.unlink()

    # 2D dataset with coords
    ds = IR_dataset_2D
    with pytest.raises(NotImplementedError):
        f = ds.write_csv("myfile.csv", confirm=False)


def test_issue_706(IR_dataset_2D):
    ds = IR_dataset_2D[0]
    ds.write_csv("myfile.csv", confirm=False)
    f = ds.write_csv("../myfile.csv")
    assert f.name == "myfile.csv"
    assert scp.pathclean("../myfile.csv").exists()
    f.unlink()
