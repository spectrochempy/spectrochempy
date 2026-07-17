# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Physical metadata policy for MCR-ALS resolved factors."""

import numpy as np
import pytest

from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.core.dataset.nddataset import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur


def _standard_problem():
    C = np.array(
        [[1.0, 0.2], [0.7, 0.5], [0.3, 0.9], [0.1, 1.1]],
    )
    St = np.array(
        [[1.0, 0.8, 0.4, 0.1, 0.0], [0.0, 0.2, 0.6, 0.9, 1.0]],
    )
    time = Coord(np.arange(C.shape[0]), title="time", units="s")
    wavelength = Coord(np.arange(St.shape[1]), title="wavelength", units="nm")
    X = NDDataset(
        C @ St,
        coordset=(time, wavelength),
        title="absorbance",
        units=ur.absorbance,
        name="mixture",
    )
    return X, C, St


def _fit(X, guess, **kwargs):
    augmentation = kwargs.pop("augmentation", None)
    mcr = MCRALS(constraints=[], max_iter=3, tol=1.0e-10, **kwargs)
    return mcr.fit(X, guess, augmentation=augmentation)


def test_metadata_from_calibrated_st_guess():
    X, _, St = _standard_problem()
    St0 = NDDataset(
        St,
        title="molar absorptivity",
        units=ur.absorbance * ur.liter / ur.mole,
    )

    mcr = _fit(X, St0)

    assert mcr.St.title == "molar absorptivity"
    assert mcr.St.units == St0.units
    assert mcr.C.title == "concentration"
    assert mcr.C.units == X.units / St0.units
    np.testing.assert_array_equal(mcr.C.y.data, X.y.data)
    assert mcr.C.y.title == X.y.title
    assert mcr.C.y.units == X.y.units
    np.testing.assert_array_equal(mcr.St.x.data, X.x.data)
    assert mcr.St.x.title == X.x.title
    assert mcr.St.x.units == X.x.units


def test_metadata_from_calibrated_c_guess():
    X, C, _ = _standard_problem()
    C0 = NDDataset(C, title="amount concentration", units=ur.mole / ur.liter)

    mcr = _fit(X, C0)

    assert mcr.C.title == "amount concentration"
    assert mcr.C.units == C0.units
    assert mcr.St.title == X.title
    assert mcr.St.units == X.units / C0.units


def test_compatible_both_guesses_are_accepted():
    X, C, St = _standard_problem()
    C0 = NDDataset(C, title="amount concentration", units=ur.mole / ur.liter)
    St0 = NDDataset(
        St,
        title="molar absorptivity",
        units=ur.absorbance * ur.liter / ur.mole,
    )

    mcr = _fit(X, (C0, St0))

    assert mcr.C.units == C0.units
    assert mcr.St.units == St0.units


def test_incompatible_both_guesses_are_rejected():
    X, C, St = _standard_problem()
    C0 = NDDataset(C, title="amount concentration", units=ur.mole / ur.liter)
    St0 = NDDataset(St, title="time response", units=ur.second)

    with pytest.raises(ValueError, match="incompatible.*C0.*St0.*X"):
        _fit(X, (C0, St0))


def test_unitless_guess_uses_ambiguous_scale_fallback():
    X, C, _ = _standard_problem()
    C0 = NDDataset(C, title="unsupported guess title")

    mcr = _fit(X, C0)

    assert mcr.C.units is None
    assert mcr.St.units is None
    assert mcr.C.title == "relative concentration"
    assert mcr.St.title == X.title


def test_normalization_clears_calibrated_factor_units():
    X, C, _ = _standard_problem()
    C0 = NDDataset(C, title="amount concentration", units=ur.mole / ur.liter)
    mcr = MCRALS(
        normSpec="max",
        nonnegConc=[],
        nonnegSpec=[],
        unimodConc=[],
        max_iter=3,
        tol=1.0e-10,
    )

    mcr.fit(X, C0)

    assert mcr.C.units is None
    assert mcr.St.units is None
    assert mcr.C.title == "relative concentration"


def test_vertical_blocks_inherit_calibrated_c_metadata_and_coordinates():
    X, C, St = _standard_problem()
    split = 2
    X1 = X[:split].copy()
    X2 = X[split:].copy()
    C01 = NDDataset(C[:split], title="amount concentration", units=ur.mole / ur.liter)
    C02 = NDDataset(C[split:], title="amount concentration", units=ur.mole / ur.liter)

    mcr = _fit([X1, X2], [C01, C02], augmentation="vertical")
    blocks = mcr.C_blocks

    assert len(blocks) == 2
    for block, source in zip(blocks, (X1, X2), strict=True):
        assert isinstance(block, NDDataset)
        assert block.title == "amount concentration"
        assert block.units == C01.units
        np.testing.assert_array_equal(block.y.data, source.y.data)
        assert block.y.title == source.y.title
        assert block.y.units == source.y.units
    np.testing.assert_allclose(np.asarray(mcr.St.data), St, atol=1.0e-10)


def test_vertical_blocks_reject_inconsistent_calibrated_c_units():
    X, C, _ = _standard_problem()
    X1 = X[:2].copy()
    X2 = X[2:].copy()
    C01 = NDDataset(C[:2], title="concentration", units=ur.mole / ur.liter)
    C02 = NDDataset(C[2:], title="concentration", units=ur.second)

    with pytest.raises(ValueError, match="C0 blocks.*units"):
        _fit([X1, X2], [C01, C02], augmentation="vertical")


def test_horizontal_spectral_blocks_keep_block_physical_metadata():
    _, C, _ = _standard_problem()
    time = Coord(np.arange(C.shape[0]), title="temperature", units="K")
    uv_axis = Coord(np.arange(3), title="wavelength", units="nm")
    cd_axis = Coord(np.arange(4), title="wavelength", units="nm")
    St_uv = np.array([[1.0, 0.5, 0.1], [0.1, 0.5, 1.0]])
    St_cd = np.array([[0.4, 0.2, 0.1, 0.0], [0.0, 0.1, 0.3, 0.5]])
    X_uv = NDDataset(
        C @ St_uv,
        coordset=(time, uv_axis),
        title="absorbance",
        units=ur.absorbance,
        name="uv",
    )
    X_cd = NDDataset(
        C @ St_cd,
        coordset=(time, cd_axis),
        title="ellipticity",
        units=ur.volt,
        name="cd",
    )
    St0_uv = NDDataset(St_uv, units=ur.absorbance * ur.liter / ur.mole)
    St0_cd = NDDataset(St_cd, units=ur.volt * ur.liter / ur.mole)

    mcr = _fit(
        [X_uv, X_cd],
        [St0_uv, St0_cd],
        augmentation="horizontal",
    )
    blocks = mcr.St_blocks

    assert mcr.C.units == ur.mole / ur.liter
    assert mcr.St.units is None
    assert mcr.St.title == "concatenated spectral profiles"
    assert blocks[0].title == "absorbance"
    assert blocks[0].units == St0_uv.units
    assert blocks[1].title == "ellipticity"
    assert blocks[1].units == St0_cd.units
    np.testing.assert_array_equal(blocks[0].x.data, X_uv.x.data)
    np.testing.assert_array_equal(blocks[1].x.data, X_cd.x.data)
    before = np.asarray(mcr.St.data).copy()
    blocks[0].data[0, 0] = -999.0
    np.testing.assert_array_equal(np.asarray(mcr.St.data), before)


def test_metadata_policy_does_not_change_numerical_factors():
    X, C, _ = _standard_problem()
    plain = _fit(np.asarray(X.data), np.asarray(C))
    calibrated = _fit(
        X,
        NDDataset(C, title="amount concentration", units=ur.mole / ur.liter),
    )

    np.testing.assert_array_equal(np.asarray(calibrated.C.data), plain._outfit[0])
    np.testing.assert_array_equal(np.asarray(calibrated.St.data), plain._outfit[1])
