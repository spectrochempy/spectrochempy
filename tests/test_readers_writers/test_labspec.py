# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path

import spectrochempy as scp
from spectrochempy.utils import show


def test_read_labspec():

    ramandir = Path("ramandata")
    scp.info_(ramandir)

    A = scp.read_labspec("Activation.txt", directory=ramandir)

    B = scp.read(protocol="labspec", directory=ramandir)

    B = scp.read_dir(directory=ramandir / "subdir")
