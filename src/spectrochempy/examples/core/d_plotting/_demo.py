from os import environ

import numpy as np

import spectrochempy as scp


def load_ir_demo_dataset(datadir=None):
    test_file = environ.get("TEST_FILE")
    if test_file:
        dataset = scp.read(test_file)
        if dataset is not None:
            return dataset

    if datadir is not None:
        dataset = scp.read_omnic(datadir / "irdata" / "nh4y-activation.spg")
    else:
        dataset = scp.read("irdata/nh4y-activation.spg")
    if dataset is not None:
        return dataset

    x = scp.Coord(
        np.linspace(4000.0, 650.0, 256),
        title="wavenumber",
        units="cm^-1",
    )
    y = scp.Coord(np.linspace(0.0, 5.0, 16), title="time on stream", units="hour")
    xv = np.linspace(-1.0, 1.0, 256)
    yv = np.linspace(0.0, 1.0, 16)[:, None]
    data = np.exp(-(((xv + 0.35) / 0.12) ** 2)) * (1.0 + 0.5 * yv) + 0.7 * np.exp(
        -(((xv - 0.10) / 0.18) ** 2)
    ) * (1.2 - 0.4 * yv)
    return scp.NDDataset(data, coordset=[y, x], units="a.u.", title="absorbance")


def load_raman_demo_dataset():
    test_file = environ.get("TEST_FILE")
    if test_file:
        dataset = scp.read(test_file)
        if dataset is not None:
            return dataset

    dataset = scp.read("ramandata/labspec/serie190214-1.txt")
    if dataset is not None:
        return dataset

    x = scp.Coord(
        np.linspace(50.0, 1800.0, 256),
        title="raman shift",
        units="cm^-1",
    )
    y = scp.Coord(np.arange(10), title="sample")
    xv = np.linspace(-1.0, 1.0, 256)
    yv = np.linspace(0.0, 1.0, 10)[:, None]
    data = (
        np.exp(-(((xv + 0.20) / 0.10) ** 2)) * (1.0 + 0.5 * yv)
        + 0.5 * np.exp(-(((xv - 0.25) / 0.15) ** 2)) * (1.2 - 0.4 * yv)
        + 0.05 * yv
    )
    return scp.NDDataset(data, coordset=[y, x], units="a.u.", title="intensity")
