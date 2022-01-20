# import pytest

# import spectrochempy as scp


def test_write_netcdf(IR_dataset_2D):

    ds = IR_dataset_2D

    f = ds.write_netcdf("myfile.nc", confirm=False)
    print(f)
