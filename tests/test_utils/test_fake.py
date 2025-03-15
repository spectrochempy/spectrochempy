# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.fake import _make_concentrations_matrix
from spectrochempy.utils.fake import _make_spectra_matrix
from spectrochempy.utils.fake import generate_fake


def test_make_spectra_matrix():
    # Test with simple inputs
    MODEL = ("gaussian", "voigt")
    AMPL = (100.0, 70.0)
    POS = (6000.0, 4000.0)
    WIDTH = (6000.0, 1000.0)
    RATIO = (0.1, 0.5)
    ASYM = (0.0, 0.0)

    specs = _make_spectra_matrix(MODEL, AMPL, POS, WIDTH, RATIO, ASYM)

    # Verify returned object is an NDDataset
    assert isinstance(specs, NDDataset)
    # Check shape matches inputs
    assert specs.shape[0] == len(MODEL)
    assert specs.shape[1] == 4000  # based on default linspace
    # Check units
    assert str(specs.units) == "a.u."
    # Verify coordinates
    assert specs.x.size == 4000
    assert specs.x.title == "wavenumbers"
    assert str(specs.x.units) == "cm⁻¹"  # Using Unicode superscript


def test_make_concentrations_matrix():
    # Define simple concentration profiles
    def C1(t):
        return t * 0.05 + 0.01

    def C2(t):
        return np.sin(t / 2) + 1

    concs = _make_concentrations_matrix(C1, C2)

    # Verify returned object is an NDDataset
    assert isinstance(concs, NDDataset)
    # Check shape matches inputs
    assert concs.shape[0] == 2  # Two profiles
    assert concs.shape[1] == 50  # Default number of points
    # Check title
    assert concs.title == "concentration"
    # Verify coordinates
    assert concs.x.size == 50
    assert concs.x.title == "time"
    assert str(concs.x.units) == "h"  # Standard abbreviation for hour

    # Test normalization
    assert np.allclose(np.sum(concs.data, axis=0), 1.0)


def test_generate_fake():
    # Generate fake data
    d, specs, concs = generate_fake()

    # Verify the types
    assert all(isinstance(obj, NDDataset) for obj in (d, specs, concs))

    # Check shapes
    assert d.shape == (concs.shape[1], specs.shape[1])
    assert specs.shape[0] == 4  # 4 components from the function
    assert concs.shape[0] == 4  # 4 concentration profiles

    # Verify the dot product relation (allowing for noise)
    reconstructed = concs.T.dot(specs)
    assert reconstructed.shape == d.shape

    # Verify units are preserved
    assert d.x.title == "wavenumbers"
    assert str(d.x.units) == "cm⁻¹"  # Using Unicode superscript
    assert d.y.title == "time"

    # Check that noise was added (data shouldn't match perfectly)
    assert not np.array_equal(d.data, reconstructed.data)
