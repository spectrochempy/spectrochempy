# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp


@pytest.fixture()
def simplisma_dataset():
    n_observations = 20
    n_variables = 100

    t = np.linspace(0, 1, n_observations)
    c1 = np.exp(-((t - 0.3) ** 2) / 0.02)
    c2 = np.exp(-((t - 0.7) ** 2) / 0.02)
    C_true = np.column_stack([c1, c2])

    w = np.linspace(0, 100, n_variables)
    s1 = np.exp(-((w - 30) ** 2) / 20)
    s2 = np.exp(-((w - 70) ** 2) / 20)
    St_true = np.vstack([s1, s2])

    data = C_true @ St_true

    dataset = scp.NDDataset(
        data,
        coordset=[
            scp.Coord(t, title="time", units="hours"),
            scp.Coord(w, title="wavelength", units="nm"),
        ],
        title="synthetic SIMPLISMA mixture",
        units="absorbance",
    )
    return dataset


@pytest.fixture()
def fastica_dataset():
    n_observations = 100
    n_variables = 8
    n_components = 4

    rng = np.random.default_rng(42)

    t = np.linspace(0, 2 * np.pi, n_observations)
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t))
    s3 = rng.uniform(-1, 1, n_observations)
    s4 = rng.laplace(0, 1, n_observations)

    S_true = np.column_stack([s1, s2, s3, s4])

    rng_mix = np.random.default_rng(7)
    Mixing_true = rng_mix.uniform(-1, 1, (n_components, n_variables))

    data = S_true @ Mixing_true

    dataset = scp.NDDataset(
        data,
        coordset=[
            scp.Coord(t, title="time", units="s"),
            scp.Coord.arange(n_variables, title="feature"),
        ],
        title="synthetic ICA mixture",
        units="absorbance",
    )
    return dataset
