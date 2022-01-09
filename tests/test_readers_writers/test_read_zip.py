# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path

from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset


def test_read_zip():
    datadir = prefs.datadir

    # with pytest.raises(NotImplementedError):
    #    NDDataset.read_zip('agirdata/P350/FTIR/FTIR.zip')

    A = NDDataset.read_zip(
        "agirdata/P350/FTIR/FTIR.zip",
        origin="omnic",
        only=10,
        csv_delimiter=";",
        merge=True,
    )
    assert A.shape == (10, 2843)

    # Test bytes contents for ZIP files
    z = Path(datadir) / "agirdata" / "P350" / "FTIR" / "FTIR.zip"
    content2 = z.read_bytes()
    B = NDDataset.read_zip(
        {"name.zip": content2}, origin="omnic", only=10, csv_delimiter=";", merge=True
    )
    assert B.shape == (10, 2843)

    # Test read_zip with several contents
    C = NDDataset.read_zip(
        {"name1.zip": content2, "name2.zip": content2},
        origin="omnic",
        only=10,
        csv_delimiter=";",
        merge=True,
    )
    assert C.shape == (2, 10, 2843)
