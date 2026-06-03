# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
from os import environ

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.utils import docutils as chd
from spectrochempy.utils import testing
from spectrochempy.utils.constants import MASKED


@pytest.fixture()
def efa_dataset():
    time = np.linspace(0.0, 1.0, 48)
    features = np.linspace(400.0, 700.0, 12)

    concentrations = np.column_stack(
        [
            np.exp(-0.5 * ((time - 0.35) / 0.12) ** 2),
            0.8 * np.exp(-0.5 * ((time - 0.68) / 0.14) ** 2),
        ]
    )
    spectra = np.vstack(
        [
            1.0 + 0.3 * np.cos(np.linspace(0.0, np.pi, features.size)),
            0.7 + 0.4 * np.sin(np.linspace(0.0, np.pi, features.size)),
        ]
    )
    data = concentrations @ spectra

    return scp.NDDataset(
        data=data,
        coordset=[
            scp.Coord(time, units="minutes", title="time"),
            scp.Coord(features, units="nm", title="wavelength"),
        ],
        title="synthetic EFA mixture",
        units="absorbance",
    )


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_EFA_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.efa"
    chd.check_docstrings(
        module,
        obj=scp.EFA,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "EX02", "ES01", "GL11", "GL08", "PR01"],
    )


@pytest.mark.data
def test_example():
    # Init the model
    model = scp.EFA()
    # Read an experimental 2D spectra (N x M )
    path = scp.preferences.datadir / "irdata" / "nh4y-activation.spg"
    if not path.exists():
        pytest.skip("IR test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")
    X = scp.read(path)
    # Fit the model
    model.fit(X)
    # Display components spectra (2 x M)
    model.n_components = 2
    _ = model.components.plot(title="Components")
    # Get the abstract concentration profile based on the FIFO EFA analysis
    c = model.transform()
    # Plot the transposed concentration matrix  (2 x N)
    _ = c.T.plot(title="Concentration")
    # scp.show()


def test_efa_fit_returns_forward_and_backward_ev(efa_dataset):
    efa = scp.EFA()
    result = efa.fit(efa_dataset)

    assert result is efa
    testing.assert_dataset_equal(efa.X, efa_dataset)
    assert efa.f_ev.shape == (efa_dataset.shape[0], efa_dataset.shape[1])
    assert efa.b_ev.shape == efa.f_ev.shape
    assert efa.f_ev.dims == ["y", "k"]
    assert efa.b_ev.dims == ["y", "k"]
    assert efa.f_ev.title == efa_dataset.title
    assert efa.b_ev.title == efa_dataset.title

    assert np.all(np.isfinite(efa.f_ev.data))
    assert np.all(np.isfinite(efa.b_ev.data))
    assert np.all(efa.f_ev.data >= 0.0)
    assert np.all(efa.b_ev.data >= 0.0)
    assert_allclose(efa.f_ev[-1, 2:].data, 0.0, atol=1.0e-12)
    assert_allclose(efa.b_ev[0, 2:].data, 0.0, atol=1.0e-12)
    assert efa.f_ev[-1, 1].data > efa.f_ev[-1, 2].data
    assert efa.b_ev[0, 1].data > efa.b_ev[0, 2].data


def test_efa_transform_uses_requested_components(efa_dataset):
    efa = scp.EFA(n_components=2)
    efa.fit(efa_dataset)

    concentrations = efa.transform()

    assert concentrations.shape == (efa_dataset.shape[0], 2)
    assert concentrations.dims == ["y", "k"]
    assert np.all(np.isfinite(concentrations.data))
    assert np.all(concentrations.data >= 0.0)
    testing.assert_coord_equal(concentrations.y, efa_dataset.y)


def test_efa_cutoff_clips_forward_and_backward_ev(efa_dataset):
    efa = scp.EFA()
    efa.fit(efa_dataset)
    raw_forward = efa.f_ev.copy()
    raw_backward = efa.b_ev.copy()

    cutoff = 1.0e-5
    efa.cutoff = cutoff

    assert np.min(efa.f_ev.data) >= cutoff
    assert np.min(efa.b_ev.data) >= cutoff
    assert np.max(raw_forward.data) > np.max(efa.f_ev.data[:, 2:])
    assert np.max(raw_backward.data) > np.max(efa.b_ev.data[:, 2:])


def test_efa_masked_data_uses_synthetic_dataset(efa_dataset):
    dataset = efa_dataset.copy()
    dataset[:, 3:5] = MASKED
    dataset[10:12] = MASKED

    efa = scp.EFA(n_components=2)
    efa.fit(dataset)
    concentrations = efa.transform()

    assert efa.X.shape == dataset.shape
    testing.assert_dataset_equal(efa.X, dataset)
    assert concentrations.shape == (dataset.shape[0], 2)
    assert concentrations.dims == ["y", "k"]
    assert np.all(np.isfinite(concentrations.data))
