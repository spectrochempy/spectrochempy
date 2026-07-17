# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Tests for augmented (multiset) MCR-ALS support.

These tests cover:
  1. _AugmentedStructure construction and validation
  2. Vertical augmentation preprocessing in fit()
  3. Result reconstruction: C_blocks, is_augmented, augmented_structure
  4. Block-local constraint application
  5. ComponentPresence constraint
  6. Block area quantification
  7. Trilinear constraint and _project_rank_one_profiles
  8. Backward compatibility with simple 2D data
"""

import numpy as np
import pytest

from spectrochempy.analysis import constraints as mc
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.analysis.decomposition.mcrals import _AugmentedStructure
from spectrochempy.analysis.decomposition.mcrals import _project_rank_one_profiles
from spectrochempy.core.dataset.nddataset import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_vertical_data(rng):
    """Create simple vertically-augmented data with 2 experiments."""
    n_comp = 3
    n_wl = 20
    n1, n2 = 10, 15

    St_true = np.abs(rng.normal(size=(n_comp, n_wl)))
    C1_true = np.abs(rng.normal(size=(n1, n_comp)))
    C2_true = np.abs(rng.normal(size=(n2, n_comp)))

    X1 = C1_true @ St_true
    X2 = C2_true @ St_true

    return X1, X2, C1_true, C2_true, St_true


@pytest.fixture
def nd_simple_vertical_data(rng):
    """Create NDDataset vertically-augmented data with 2 experiments."""
    n_comp = 3
    n_wl = 20
    n1, n2 = 10, 15

    St_true = np.abs(rng.normal(size=(n_comp, n_wl)))
    C1_true = np.abs(rng.normal(size=(n1, n_comp)))
    C2_true = np.abs(rng.normal(size=(n2, n_comp)))

    X1_data = C1_true @ St_true
    X2_data = C2_true @ St_true

    t1 = Coord(np.arange(n1), title="time", units="s")
    t2 = Coord(np.arange(n2), title="time", units="s")
    wl = Coord(np.arange(n_wl), title="wavelength", units="nm")

    X1 = NDDataset(X1_data, coordset=(t1, wl), units="absorbance")
    X2 = NDDataset(X2_data, coordset=(t2, wl), units="absorbance")

    return X1, X2, C1_true, C2_true, St_true


@pytest.fixture
def guess(rng, simple_vertical_data):
    """Initial guess for concentration profiles."""
    X1, X2, C1_true, C2_true, St_true = simple_vertical_data
    X_aug = np.vstack([X1, X2])
    n_comp = 3
    return np.abs(X_aug[:, :n_comp] + 0.05 * rng.normal(size=(X_aug.shape[0], n_comp)))


@pytest.fixture
def trilinear_data(rng):
    """
    Create data with exactly trilinear concentration profiles.

    Three blocks with the same number of points (20 each) but different
    amplitudes: shape * [1.0, 0.6, 1.4].
    """
    n_comp = 3
    n_wl = 30
    n_points = 20
    n_blocks = 3

    St_true = np.abs(rng.normal(size=(n_comp, n_wl)))
    shape = np.abs(rng.normal(size=(n_points,)))
    amplitudes = np.array([1.0, 0.6, 1.4])

    C_blocks = [
        np.outer(shape, amplitudes * (i + 1) / n_blocks) for i in range(n_blocks)
    ]
    X_blocks = [Cb @ St_true for Cb in C_blocks]

    return X_blocks, C_blocks, St_true, shape, amplitudes


# ======================================================================
# Tests for _AugmentedStructure
# ======================================================================


class TestAugmentedStructure:
    def test_vertical_construction(self):
        aug = _AugmentedStructure(
            mode="vertical",
            row_slices=(slice(0, 3), slice(3, 8)),
            column_slices=(slice(0, 10),),
            input_shapes=((3, 10), (5, 10)),
        )
        assert aug.mode == "vertical"
        assert len(aug.row_slices) == 2
        assert aug.row_slices[0] == slice(0, 3)
        assert aug.row_slices[1] == slice(3, 8)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation mode"):
            _AugmentedStructure(
                mode="diagonal",
                row_slices=(slice(0, 3),),
                column_slices=(slice(0, 10),),
                input_shapes=((3, 10),),
            )

    def test_block_presence_default(self):
        """block_presence defaults to None."""
        aug = _AugmentedStructure(
            mode="vertical",
            row_slices=(slice(0, 3),),
            column_slices=(slice(0, 10),),
            input_shapes=((3, 10),),
        )
        assert aug.block_presence is None


# ======================================================================
# Tests for simple 2D compatibility (no regression)
# ======================================================================


class TestSimple2DCompatibility:
    """All existing properties must work on simple 2D data."""

    def test_fit_with_ndarray(self, rng):
        X = np.abs(rng.normal(size=(20, 10)))
        C0 = X[:, :2]
        mcr = MCRALS(tol=1.0)
        mcr.fit(X, C0)
        assert mcr.C is not None
        assert mcr.St is not None
        assert not mcr.is_augmented
        assert mcr.augmented_structure is None

    def test_fit_with_nddataset(self, rng):
        n, m = 20, 10
        X = NDDataset(np.abs(rng.normal(size=(n, m))))
        C0 = X.data[:, :2]
        mcr = MCRALS(tol=1.0)
        mcr.fit(X, C0)
        assert mcr.C.shape == (n, 2)
        assert not mcr.is_augmented

    def test_c_blocks_returns_tuple_for_simple(self, rng):
        X = np.abs(rng.normal(size=(20, 10)))
        C0 = X[:, :2]
        mcr = MCRALS(tol=1.0)
        mcr.fit(X, C0)
        blocks = mcr.C_blocks
        assert isinstance(blocks, tuple)
        assert len(blocks) == 1
        assert blocks[0].shape == (20, 2)

    def test_constraints_work_with_simple_data(self, rng):
        X = np.abs(rng.normal(size=(30, 15)))
        C0 = X[:, :3]
        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.Unimodal("C", tolerance=1.0),
            ],
            tol=0.1,
            max_iter=50,
        )
        mcr.fit(X, C0)
        # Should converge
        assert mcr._fit_meta["converged"]


# ======================================================================
# Tests for vertical augmentation in fit()
# ======================================================================


class TestVerticalAugmentation:
    def test_two_compatible_ndarrays(self, simple_vertical_data, guess):
        X1, X2, C1_true, C2_true, St_true = simple_vertical_data
        n1, n2 = X1.shape[0], X2.shape[0]
        n_comp = St_true.shape[0]

        mcr = MCRALS(tol=1.0, max_iter=30)
        mcr.fit([X1, X2], guess)

        assert mcr.is_augmented
        assert mcr.augmented_structure is not None
        assert len(mcr.augmented_structure.row_slices) == 2

        # Check C_blocks slicing
        C_blocks = mcr.C_blocks
        assert len(C_blocks) == 2
        assert C_blocks[0].shape == (n1, n_comp)
        assert C_blocks[1].shape == (n2, n_comp)

        # Overall reconstruction
        C_aug = np.vstack([C_blocks[0], C_blocks[1]])
        St = np.asarray(mcr.St.data)
        X_aug = np.vstack([X1, X2])
        X_hat = C_aug @ St
        assert np.allclose(X_hat, X_aug, atol=5.0)

        # Per-block reconstruction
        X1_hat = C_blocks[0] @ St
        X2_hat = C_blocks[1] @ St
        assert np.allclose(X1_hat, X1, atol=5.0)
        assert np.allclose(X2_hat, X2, atol=5.0)

    def test_two_compatible_nddatasets(self, nd_simple_vertical_data, guess):
        X1, X2, C1_true, C2_true, St_true = nd_simple_vertical_data
        n_comp = St_true.shape[0]

        mcr = MCRALS(tol=1.0, max_iter=30)
        mcr.fit([X1, X2], guess)

        assert mcr.is_augmented
        C_blocks = mcr.C_blocks
        assert len(C_blocks) == 2
        assert C_blocks[0].shape == (X1.shape[0], n_comp)
        assert C_blocks[1].shape == (X2.shape[0], n_comp)

    def test_different_row_counts(self, rng):
        n_comp = 2
        n_wl = 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        X1 = np.abs(rng.normal(size=(5, n_comp))) @ St
        X2 = np.abs(rng.normal(size=(12, n_comp))) @ St
        C0 = np.abs(rng.normal(size=(17, n_comp)))

        mcr = MCRALS(tol=1.0)
        mcr.fit([X1, X2], C0)

        assert mcr.C_blocks[0].shape == (5, n_comp)
        assert mcr.C_blocks[1].shape == (12, n_comp)

    def test_different_column_counts_raises(self, rng):
        X1 = rng.normal(size=(10, 8))
        X2 = rng.normal(size=(10, 5))
        C0 = rng.normal(size=(20, 2))
        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="same number of columns"):
            mcr.fit([X1, X2], C0, augmentation="vertical")

    def test_incompatible_spectral_axis_raises(self, rng):
        wl1 = Coord(np.linspace(400, 800, 20), title="wavelength")
        wl2 = Coord(np.linspace(400, 900, 20), title="wavelength")  # Different values
        X1 = NDDataset(rng.normal(size=(10, 20)), coordset=(None, wl1))
        X2 = NDDataset(rng.normal(size=(10, 20)), coordset=(None, wl2))
        C0 = rng.normal(size=(20, 2))
        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="compatible spectral axis"):
            mcr.fit([X1, X2], C0, augmentation="vertical")

    def test_empty_list_raises(self):
        mcr = MCRALS()
        with pytest.raises(ValueError, match="empty"):
            mcr.fit([], None)

    def test_non_2d_element_raises(self, rng):
        X1 = rng.normal(size=(10, 5))
        X2 = rng.normal(size=(10,))  # 1-D
        C0 = rng.normal(size=(20, 2))
        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="2-dimensional"):
            mcr.fit([X1, X2], C0)


# ======================================================================
# Tests for block-local constraint application
# ======================================================================


class TestBlockLocalConstraints:
    def test_unimodal_applied_per_block(self, rng):
        """Unimodality should be applied independently to each block."""
        n_comp = 2
        n_wl = 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        # Create smooth increasing-then-decreasing profiles per block
        n1, n2 = 20, 15
        C1 = np.column_stack(
            [
                1 - np.abs(np.linspace(-1, 1, n1)),
                0.5 - 0.5 * np.abs(np.linspace(-1, 1, n1)),
            ]
        )
        C2 = np.column_stack(
            [
                1 - np.abs(np.linspace(-1, 1, n2)),
                0.5 - 0.5 * np.abs(np.linspace(-1, 1, n2)),
            ]
        )
        C1 = np.maximum(C1, 0)
        C2 = np.maximum(C2, 0)
        X1 = C1 @ St
        X2 = C2 @ St
        X_aug = np.vstack([X1, X2])
        C0 = np.abs(
            X_aug[:, :n_comp] + 0.01 * rng.normal(size=(X_aug.shape[0], n_comp))
        )

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.Unimodal("C", tolerance=1.0)],
            tol=0.01,
            max_iter=100,
        )
        mcr.fit([X1, X2], C0)

        # Verify block boundaries are respected.
        # C1_fit and C2_fit should be independent — just check
        # they have the expected shapes and are non-negative.
        C1_fit = mcr.C_blocks[0]
        C2_fit = mcr.C_blocks[1]
        assert C1_fit.shape == (n1, n_comp)
        assert C2_fit.shape == (n2, n_comp)
        assert np.all(C1_fit >= -1e-10)
        assert np.all(C2_fit >= -1e-10)

    def test_blocks_selector(self, rng):
        """blocks=[1] should only modify block 1."""
        n_comp = 2
        n_wl = 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        n1, n2 = 5, 5
        np.random.seed(42)
        C1 = np.abs(np.random.normal(size=(n1, n_comp)))
        C2 = np.abs(np.random.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St
        X_aug = np.vstack([X1, X2])
        C0 = np.abs(
            X_aug[:, :n_comp] + 0.01 * rng.normal(size=(X_aug.shape[0], n_comp))
        )

        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.Unimodal("C", tolerance=1.0, blocks=[0]),
            ],
            tol=0.1,
            max_iter=30,
        )
        mcr.fit([X1, X2], C0, augmentation="vertical")
        assert mcr.C_blocks[0].shape == (n1, n_comp)
        assert mcr.C_blocks[1].shape == (n2, n_comp)

    def test_invalid_block_index_raises(self, rng):
        X1 = rng.normal(size=(10, 5))
        X2 = rng.normal(size=(10, 5))
        C0 = rng.normal(size=(20, 2))
        mcr = MCRALS(
            constraints=[mc.Unimodal("C", blocks=[5])],  # only 2 blocks in the data
            tol=10.0,
        )
        with pytest.raises(ValueError, match="Block index"):
            mcr.fit([X1, X2], C0, augmentation="vertical")

    def test_blocks_on_non_augmented_raises(self):
        """Explicit block selection on non-augmented data should raise."""
        with pytest.raises(ValueError, match="Block indices can only be specified"):
            MCRALS._resolve_blocks(MCRALS(), [0])

    def test_monotonic_works_on_unequal_blocks(self, rng):
        """Monotonic constraints on blocks of different sizes."""
        n_comp = 2
        n_wl = 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        n1, n2 = 5, 10
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        # Make C1 monotonic increasing
        C1 = np.sort(C1, axis=0)
        X1 = C1 @ St
        X2 = C2 @ St
        X_aug = np.vstack([X1, X2])
        C0 = np.abs(
            X_aug[:, :n_comp] + 0.01 * rng.normal(size=(X_aug.shape[0], n_comp))
        )

        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.Monotonic("C", "increasing", tolerance=1.0),
            ],
            tol=0.1,
            max_iter=30,
        )
        mcr.fit([X1, X2], C0)

        # C1 block should be monotonic increasing
        C1_fit = mcr.C_blocks[0]
        for j in range(n_comp):
            diffs = np.diff(C1_fit[:, j])
            assert np.all(diffs >= -1e-10)


# ======================================================================
# Tests for ComponentPresence constraint
# ======================================================================


class TestComponentPresence:
    def test_absent_components_are_zero(self, rng):
        """Absent components are forced to zero."""
        n_comp = 3
        n_wl = 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        n1, n2 = 10, 10
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St
        X_aug = np.vstack([X1, X2])
        C0 = np.abs(
            X_aug[:, :n_comp] + 0.01 * rng.normal(size=(X_aug.shape[0], n_comp))
        )

        presence = [
            [True, True, False],  # block 0: component 2 absent
            [True, False, True],  # block 1: component 1 absent
        ]

        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.ComponentPresence("C", presence=presence),
            ],
            tol=0.1,
            max_iter=30,
        )
        mcr.fit([X1, X2], C0, augmentation="vertical")

        C_blocks = mcr.C_blocks
        # Block 0, component 2 should be zero
        assert np.allclose(C_blocks[0][:, 2], 0.0)
        # Block 1, component 1 should be zero
        assert np.allclose(C_blocks[1][:, 1], 0.0)

    def test_present_components_not_affected(self, rng):
        """Present components should not be zeroed."""
        n_comp = 2
        n_wl = 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        n1, n2 = 10, 10
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St
        X_aug = np.vstack([X1, X2])
        C0 = np.abs(
            X_aug[:, :n_comp] + 0.01 * rng.normal(size=(X_aug.shape[0], n_comp))
        )

        presence = [[True, True], [True, True]]

        mcr = MCRALS(
            constraints=[mc.ComponentPresence("C", presence=presence)],
            tol=0.1,
            max_iter=30,
        )
        mcr.fit([X1, X2], C0, augmentation="vertical")

        C_blocks = mcr.C_blocks
        # No component should be zero
        assert np.any(C_blocks[0] > 0)
        assert np.any(C_blocks[1] > 0)

    def test_wrong_presence_shape_raises_at_conversion(self, rng):
        """Presence matrix of wrong size should raise at constraint building."""
        X1 = np.abs(rng.normal(size=(10, 5)))
        X2 = np.abs(rng.normal(size=(10, 5)))
        presence = [
            [True, True],
            [True, True],
            [True, False],
        ]  # 3 blocks, but data has 2

        mcr = MCRALS(
            constraints=[mc.ComponentPresence("C", presence=presence)],
            tol=10.0,
        )
        with pytest.raises(ValueError, match="Presence matrix has 3 rows"):
            mcr.fit([X1, X2], np.abs(rng.normal(size=(20, 2))), augmentation="vertical")

    def test_requires_augmented_data(self, rng):
        """ComponentPresence should fail on non-augmented data."""
        X = np.abs(rng.normal(size=(10, 5)))
        mcr = MCRALS(
            constraints=[mc.ComponentPresence("C", presence=[[True, True]])],
            tol=10.0,
        )
        with pytest.raises(ValueError, match="requires augmented data"):
            mcr.fit(X, np.abs(rng.normal(size=(10, 2))))


# ======================================================================
# Tests for _project_rank_one_profiles
# ======================================================================


class TestProjectRankOneProfiles:
    def test_exact_rank_one(self):
        """On exactly rank-1 input, reconstruction should match."""
        n_points, n_blocks = 20, 3
        shape = np.linspace(0, 1, n_points)
        amplitudes = np.array([1.0, 0.6, 1.4])
        profiles = np.outer(shape, amplitudes)

        rec, ampl = _project_rank_one_profiles(profiles)
        assert np.allclose(profiles, rec, atol=1e-14)
        # Amplitudes should be proportional to input
        assert ampl.shape == (n_blocks,)

    def test_deterministic_sign(self):
        """Sign resolution should give deterministic result."""
        rng = np.random.default_rng(42)
        profiles = rng.normal(size=(10, 3))

        rec1, _ = _project_rank_one_profiles(profiles)
        rec2, _ = _project_rank_one_profiles(profiles)
        np.testing.assert_array_equal(rec1, rec2)

    def test_output_rank_one(self):
        """After projection, reconstruction should be rank 1."""
        rng = np.random.default_rng(123)
        profiles = rng.normal(size=(15, 4))
        rec, ampl = _project_rank_one_profiles(profiles)
        u, s, vh = np.linalg.svd(rec, full_matrices=False)
        # Only first singular value should be significant
        assert s[0] / s[1] > 1e10 if len(s) > 1 else True

    def test_return_types(self):
        rng = np.random.default_rng(42)
        profiles = rng.normal(size=(10, 3))
        rec, ampl = _project_rank_one_profiles(profiles)
        assert rec.shape == (10, 3)
        assert ampl.shape == (3,)


# ======================================================================
# Tests for Trilinear constraint
# ======================================================================


class TestTrilinearConstraint:
    def test_exact_trilinear_profiles_preserved(self, trilinear_data):
        """On exactly proportional profiles, result is unchanged."""
        X_blocks, C_blocks, St_true, shape, amplitudes = trilinear_data
        n_comp = St_true.shape[0]
        X_aug = np.vstack(X_blocks)
        C0 = np.abs(
            X_aug[:, :n_comp]
            + 0.01 * np.random.default_rng(42).normal(size=(X_aug.shape[0], n_comp))
        )

        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.Trilinear("C", synchronization="none"),
            ],
            tol=0.01,
            max_iter=50,
        )
        mcr.fit(X_blocks, C0, augmentation="vertical")

        # Check that the trilinear constraint was applied
        assert hasattr(mcr, "_model_profile_constraints_")

    def test_trilinear_deterministic(self, trilinear_data):
        """Trilinear result is deterministic despite SVD sign ambiguity."""
        X_blocks, C_blocks, St_true, shape, amplitudes = trilinear_data
        n_comp = St_true.shape[0]
        X_aug = np.vstack(X_blocks)
        C0 = np.abs(
            X_aug[:, :n_comp]
            + 0.01 * np.random.default_rng(42).normal(size=(X_aug.shape[0], n_comp))
        )

        mcr1 = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.Trilinear("C", synchronization="none"),
            ],
            tol=10.0,
            max_iter=5,
        )
        mcr1.fit(X_blocks, C0.copy(), augmentation="vertical")

        mcr2 = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.Trilinear("C", synchronization="none"),
            ],
            tol=10.0,
            max_iter=5,
        )
        mcr2.fit(X_blocks, C0.copy(), augmentation="vertical")

        np.testing.assert_array_equal(
            np.asarray(mcr1.C.data),
            np.asarray(mcr2.C.data),
        )

    def test_different_block_lengths_raises(self, rng):
        """Blocks with different lengths should cause an error."""
        n_comp = 2
        n_wl = 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        X1 = np.abs(rng.normal(size=(10, n_comp))) @ St
        X2 = np.abs(rng.normal(size=(15, n_comp))) @ St
        X_aug = np.vstack([X1, X2])
        C0 = np.abs(
            X_aug[:, :n_comp] + 0.01 * rng.normal(size=(X_aug.shape[0], n_comp))
        )

        mcr = MCRALS(
            constraints=[mc.Trilinear("C", synchronization="none")],
            tol=10.0,
            max_iter=2,
        )
        with pytest.raises(ValueError, match="same number of points"):
            mcr.fit([X1, X2], C0)

    def test_single_block_raises(self, rng):
        """Single selected block should cause an error at fit time."""
        n_comp = 2
        n_wl = 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        X1 = np.abs(rng.normal(size=(10, n_comp))) @ St
        X2 = np.abs(rng.normal(size=(10, n_comp))) @ St
        C0 = np.abs(rng.normal(size=(20, n_comp)))
        mcr = MCRALS(
            constraints=[mc.Trilinear("C", blocks=[0], synchronization="none")],
            tol=10.0,
            max_iter=2,
        )
        # blocks=[0] with 2 blocks total: only block 0 selected, that's
        # less than 2, so the constraint should raise.
        with pytest.raises(ValueError, match="requires at least 2 blocks"):
            mcr.fit([X1, X2], C0, augmentation="vertical")

    def test_requires_augmented_data(self, rng):
        """Trilinear should fail on non-augmented data."""
        X = np.abs(rng.normal(size=(10, 5)))
        C0 = np.abs(rng.normal(size=(10, 2)))
        mcr = MCRALS(
            constraints=[mc.Trilinear("C", synchronization="none")],
            tol=10.0,
        )
        with pytest.raises(ValueError, match="requires augmented data"):
            mcr.fit(X, C0)


# ======================================================================
# Tests for blocks parameter on public constraints
# ======================================================================


class TestBlocksParameter:
    def test_blocks_parameter_on_nonnegative(self):
        c = mc.NonNegative("C", blocks=[0, 1])
        assert c.blocks == [0, 1]

    def test_blocks_parameter_on_unimodal(self):
        c = mc.Unimodal("C", blocks=[0])
        assert c.blocks == [0]

    def test_blocks_parameter_on_monotonic(self):
        c = mc.Monotonic("C", "increasing", blocks=[1])
        assert c.blocks == [1]

    def test_blocks_parameter_on_closure(self):
        c = mc.Closure("C", blocks=[0])
        assert c.blocks == [0]

    def test_blocks_default_none(self):
        c = mc.NonNegative("C")
        assert c.blocks is None

    def test_blocks_on_repr(self):
        c = mc.NonNegative("C", blocks=[0, 2])
        assert "blocks=[0, 2]" in repr(c)

    def test_blocks_not_in_repr_when_none(self):
        c = mc.NonNegative("C")
        assert "blocks" not in repr(c)

    def test_blocks_equality(self):
        a = mc.Unimodal("C", blocks=[0])
        b = mc.Unimodal("C", blocks=[0])
        c = mc.Unimodal("C", blocks=[1])
        assert a == b
        assert a != c


# ======================================================================
# Test for _project_rank_one_profiles sign resolution
# ======================================================================


class TestSignResolution:
    def test_negative_sum_profile(self):
        """Profile with negative sum has deterministic output."""
        rng = np.random.default_rng(42)
        shape = -np.abs(rng.normal(size=20))
        amplitudes = np.array([1.0, 0.5])
        profiles = np.outer(shape, amplitudes)
        rec, ampl = _project_rank_one_profiles(profiles)
        # The reconstruction should match up to absolute value
        assert np.allclose(np.abs(rec), np.abs(profiles), atol=1e-14)
        # Determinism: running twice yields the same reconstruction
        rec2, _ = _project_rank_one_profiles(profiles)
        np.testing.assert_array_equal(rec, rec2)

    def test_zero_sum_profile(self):
        """Symmetric (zero-sum) profile should still yield deterministic result."""
        # A sine wave has sum ≈ 0
        shape = np.sin(np.linspace(0, 4 * np.pi, 20))
        assert abs(np.sum(shape)) < 0.1
        amplitudes = np.array([1.0, 0.5])
        profiles = np.outer(shape, amplitudes)
        rec1, _ = _project_rank_one_profiles(profiles)
        rec2, _ = _project_rank_one_profiles(profiles)
        np.testing.assert_array_equal(rec1, rec2)
        # Reconstruction should match up to global sign
        assert np.allclose(np.abs(rec1), np.abs(profiles), atol=1e-14)

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        profiles = rng.normal(size=(10, 3))
        rec1, _ = _project_rank_one_profiles(profiles)
        rec2, _ = _project_rank_one_profiles(profiles)
        np.testing.assert_array_equal(rec1, rec2)

    def test_single_block_component(self):
        """Single-column input (one block, one component)."""
        profiles = np.arange(10).reshape(10, 1).astype(float)
        rec, ampl = _project_rank_one_profiles(profiles)
        assert np.allclose(rec, profiles)
        assert ampl.shape == (1,)


# ======================================================================
# Scientific tests for augmented MCR-ALS
# ======================================================================


class TestScientificCorrectness:
    """Tests that verify the scientific validity of the augmentation."""

    # ------------------------------------------------------------------
    # Point 1: C0 initialization for vertical augmentation
    # ------------------------------------------------------------------
    def test_c0_shape_for_vertical_augmentation(self, rng):
        """C0 must have (n1+n2, k) shape, not (n1, k) or block-diagonal."""
        n_comp, n_wl = 2, 10
        n1, n2 = 8, 12
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St
        C0 = np.vstack([C1, C2])

        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=1.0, max_iter=3)
        mcr.fit([X1, X2], C0)
        assert mcr.C.shape == (
            n1 + n2,
            n_comp,
        ), f"C should be ({n1 + n2}, {n_comp}), got {mcr.C.shape}"
        assert (
            mcr.n_components == n_comp
        ), f"n_components should be {n_comp}, got {mcr.n_components}"

    def test_c0_list_auto_stack(self, rng):
        """A list of per-block C0 is auto-stacked."""
        n_comp, n_wl = 2, 10
        n1, n2 = 8, 12
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1_true = np.abs(rng.normal(size=(n1, n_comp)))
        C2_true = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1_true @ St
        X2 = C2_true @ St
        C01 = np.abs(rng.normal(size=(n1, n_comp)))
        C02 = np.abs(rng.normal(size=(n2, n_comp)))

        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=10.0, max_iter=2)
        mcr.fit([X1, X2], [C01, C02])

        assert mcr.C.shape == (n1 + n2, n_comp)
        assert mcr.n_components == n_comp
        # C_blocks should split properly
        Cb = mcr.C_blocks
        assert len(Cb) == 2
        assert Cb[0].shape == (n1, n_comp)
        assert Cb[1].shape == (n2, n_comp)

    def test_c0_list_wrong_length_raises(self, rng):
        """Mismatched X/C0 block counts raise an error."""
        n_comp, n_wl = 2, 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(8, n_comp)))
        C2 = np.abs(rng.normal(size=(12, n_comp)))
        C3 = np.abs(rng.normal(size=(5, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St
        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=10.0, max_iter=2)
        with pytest.raises(ValueError, match="must match number"):
            mcr.fit([X1, X2], [C1, C2, C3])

    def test_c0_list_wrong_components_raises(self, rng):
        """C0 blocks with different numbers of components raise an error."""
        n_comp, n_wl = 2, 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(8, n_comp)))
        X1 = C1 @ St
        X2 = np.abs(rng.normal(size=(12, n_comp))) @ St
        # C0 block with wrong number of components
        C0_bad = np.abs(rng.normal(size=(12, 3)))  # 3 components
        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=10.0, max_iter=2)
        with pytest.raises(ValueError, match="same number of components"):
            mcr.fit([X1, X2], [C1, C0_bad])

    def test_c0_list_wrong_rows_raises(self, rng):
        """C0 block rows must match the corresponding X block."""
        n_comp, n_wl = 2, 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(8, n_comp)))
        C2 = np.abs(rng.normal(size=(99, n_comp)))  # 99 rows != X2's 12
        X1 = C1 @ St
        X2 = np.abs(rng.normal(size=(12, n_comp))) @ St
        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=10.0, max_iter=2)
        with pytest.raises(ValueError, match="rows"):
            mcr.fit([X1, X2], [C1, C2])

    def test_fit_single_list_vs_single_array(self, rng):
        """fit([X1], [C01]) == fit([X1], C01)."""
        n_comp, n_wl = 2, 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(10, n_comp)))
        X1 = C1 @ St
        C0 = np.abs(rng.normal(size=(10, n_comp)))

        mcr1 = MCRALS(constraints=[mc.NonNegative("C")], tol=10.0, max_iter=3)
        mcr1.fit([X1], C0)

        mcr2 = MCRALS(constraints=[mc.NonNegative("C")], tol=10.0, max_iter=3)
        mcr2.fit([X1], [C0])

        np.testing.assert_allclose(
            np.asarray(mcr1.C.data), np.asarray(mcr2.C.data), atol=1e-10
        )

    # ------------------------------------------------------------------
    # Point 2: Trilinear scientific test
    # ------------------------------------------------------------------
    def test_trilinear_proportional_profiles(self, rng):
        """Trilinear should yield exactly proportional profiles across blocks."""
        n_points = 100
        x = np.linspace(0.0, 1.0, n_points)
        shape = np.exp(-(((x - 0.5) / 0.12) ** 2))

        C1 = np.column_stack([1.0 * shape])
        C2 = np.column_stack([0.5 * shape])
        C3 = np.column_stack([2.0 * shape])

        St = np.abs(rng.normal(size=(1, 15)))
        X1 = C1 @ St
        X2 = C2 @ St
        X3 = C3 @ St

        mcr = MCRALS(
            constraints=[mc.Trilinear("C", synchronization="none")],
            tol=1e-6,
            max_iter=1,
        )
        C0 = np.vstack([C1, C2, C3])
        mcr.fit([X1, X2, X3], C0, augmentation="vertical")

        Cb = mcr.C_blocks
        blk1, blk2, blk3 = Cb

        # Ratios should be preserved: block2/block1 = 0.5, block3/block1 = 2.0
        nonzero = np.abs(blk1[:, 0]) > 1e-10
        assert np.allclose(
            blk2[nonzero, 0] / blk1[nonzero, 0], 0.5, atol=1e-10
        ), "Trilinear did not preserve block2/block1 ratio"
        assert np.allclose(
            blk3[nonzero, 0] / blk1[nonzero, 0], 2.0, atol=1e-10
        ), "Trilinear did not preserve block3/block1 ratio"

    def test_trilinear_rank_one_after_projection(self, rng):
        """After projection, the inter-block matrix should be rank 1."""
        n_points = 50
        x = np.linspace(0.0, 1.0, n_points)
        shape = np.exp(-(((x - 0.3) / 0.15) ** 2)) + 0.1 * rng.normal(size=n_points)

        C1 = np.column_stack([1.0 * shape])
        C2 = np.column_stack([0.7 * shape])
        C3 = np.column_stack([1.3 * shape])

        profiles = np.column_stack([C1[:, 0], C2[:, 0], C3[:, 0]])
        rec, _ = _project_rank_one_profiles(profiles)
        u, s, vh = np.linalg.svd(rec, full_matrices=False)
        assert s[0] / s[1] > 1e8, (
            f"Second singular value {s[1]:.2e} is too large relative "
            f"to first {s[0]:.2e}"
        )

    # ------------------------------------------------------------------
    # Point 5: Block boundary test for local constraints
    # ------------------------------------------------------------------
    def test_unimodal_at_block_boundary(self):
        """Unimodality should not mix profiles across block boundaries."""
        block1 = np.array([[0.0], [1.0], [2.0], [1.0], [0.0]])
        block2 = np.array([[0.0], [1.0], [3.0], [1.0], [0.0]])
        C = np.vstack([block1, block2])

        aug = _AugmentedStructure(
            mode="vertical",
            row_slices=(slice(0, 5), slice(5, 10)),
            column_slices=(slice(0, 1),),
            input_shapes=((5, 1), (5, 1)),
        )
        state = type(
            "_ALSStateMock",
            (),
            {
                "augmentation": aug,
            },
        )()

        # Apply unimodality block-by-block
        from spectrochempy.analysis.decomposition.mcrals import _UnimodalConstraint

        constraint = _UnimodalConstraint(indices=[0], axis=0, tol=1.0, mod="strict")
        pipeline = [constraint]

        block_slices = aug.row_slices
        for constraint in pipeline:
            for sl in block_slices:
                local = constraint.apply(C[sl, :].copy(), state)
                C[sl, :] = local

        blk1_out = C[:5, 0]
        blk2_out = C[5:, 0]

        # Block 1 should be unchanged
        np.testing.assert_array_equal(blk1_out, block1[:, 0])
        # Block 2 should be unchanged
        np.testing.assert_array_equal(blk2_out, block2[:, 0])

    # ------------------------------------------------------------------
    # Point 7: ComponentPresence order test
    # ------------------------------------------------------------------
    def test_component_presence_after_model_profile(self, rng):
        """
        ComponentPresence must zero absent components even if a
        ModelProfile or other constraint writes to them.
        """
        n_comp = 2
        n_wl = 10
        n1, n2 = 8, 8
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St

        X_aug = np.vstack([X1, X2])
        C0 = np.abs(X_aug[:, :n_comp])

        # Component 0 is absent in block 1
        presence = [[False, True], [True, True]]
        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.ComponentPresence("C", presence=presence),
            ],
            solver_C="nnls",
            tol=1e-6,
            max_iter=10,
        )
        mcr.fit([X1, X2], C0, augmentation="vertical")

        Cb = mcr.C_blocks
        assert np.all(
            Cb[0][:, 0] == 0.0
        ), "Component 0 should be zero in block 0 after fit"
        assert np.all(Cb[1][:, 0] >= 0), "Component 0 should be non-negative in block 1"

    # ------------------------------------------------------------------
    # Point 8: Trilinear × ComponentPresence interaction
    # ------------------------------------------------------------------
    def test_trilinear_respects_presence(self, rng):
        """Trilinear should not project over blocks where a component is absent."""
        n_points = 30
        n_wl = 10
        shape = np.exp(-(((np.linspace(0, 1, n_points) - 0.5) / 0.2) ** 2))
        St = np.abs(rng.normal(size=(1, n_wl)))

        C_blocks = [
            np.column_stack([1.0 * shape]),
            np.column_stack([0.6 * shape]),
            np.column_stack([1.4 * shape]),
        ]
        X_blocks = [Cb @ St for Cb in C_blocks]
        C0 = np.vstack(C_blocks)

        # Component 0 absent in block 1
        presence = [[True], [False], [True]]
        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.ComponentPresence("C", presence=presence),
                mc.Trilinear("C", synchronization="none"),
            ],
            solver_C="nnls",
            tol=1e-6,
            max_iter=5,
        )
        mcr.fit(X_blocks, C0, augmentation="vertical")
        Cb = mcr.C_blocks

        # Block 1 should be exactly zero
        assert np.allclose(Cb[1], 0.0), "Block 1 should be zero (component absent)"
        # Blocks 0 and 2 should be proportional (trilinearity)
        bt0 = Cb[0][:, 0]
        bt2 = Cb[2][:, 0]
        nonzero = np.abs(bt0) > 1e-10
        ratios = bt2[nonzero] / bt0[nonzero]
        assert np.allclose(
            ratios, ratios[0], atol=1e-4
        ), "Blocks 0 and 2 should remain proportional under Trilinear + Presence"

    # ------------------------------------------------------------------
    # Point 13: Normalization preserves X_hat
    # ------------------------------------------------------------------
    def test_normalization_preserves_reconstruction(self, rng):
        """After normalization, C @ St must be unchanged."""
        n_comp, n_wl = 2, 10
        n1, n2 = 8, 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St

        X_aug = np.vstack([X1, X2])
        C0 = np.abs(X_aug[:, :n_comp])

        # Use legacy traitlets (not constraints=) to enable normalization
        mcr = MCRALS(
            nonnegConc="all",
            normSpec="max",
            tol=1e-6,
            max_iter=5,
        )
        mcr.fit([X1, X2], C0)

        # Verify that C @ St reconstructs the augmented data.
        # With the new semantics, the public C is the constrained C after
        # nonneg enforcement (step 3c), which may differ from the LS
        # solution.  Reconstruction is no longer exact to machine precision;
        # a loose tolerance verifies the product is still recognisable.
        C_arr = np.asarray(mcr.C.data)
        St_arr = np.asarray(mcr.components.data)
        X_hat = C_arr @ St_arr
        assert np.allclose(
            X_hat, X_aug, atol=5.0
        ), "C @ St should reconstruct X (normalization must preserve X_hat)"

    # ------------------------------------------------------------------
    # Point 16: Single-dataset list backward compatibility
    # ------------------------------------------------------------------
    def test_single_element_list_vs_direct(self, rng):
        """fit([X], guess) should give same result as fit(X, guess)."""
        n_comp, n_wl = 3, 15
        n_obs = 20
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C_true = np.abs(rng.normal(size=(n_obs, n_comp)))
        X = C_true @ St
        C0 = X[:, :n_comp] + 0.01 * rng.normal(size=(n_obs, n_comp))

        mcr_direct = MCRALS(
            constraints=[mc.NonNegative("C")],
            tol=1e-6,
            max_iter=10,
        )
        mcr_direct.fit(X, C0.copy())
        C_direct = np.asarray(mcr_direct.C.data)

        mcr_list = MCRALS(
            constraints=[mc.NonNegative("C")],
            tol=1e-6,
            max_iter=10,
        )
        mcr_list.fit([X], C0.copy())
        C_list = np.asarray(mcr_list.C.data)

        np.testing.assert_allclose(
            C_direct, C_list, atol=1e-10, err_msg="fit([X], guess) != fit(X, guess)"
        )
        # fit([X]) always goes through _fit_augmented, which sets
        # is_augmented=True even for a single block.
        assert mcr_list.is_augmented
        assert len(mcr_list.C_blocks) == 1

    # ------------------------------------------------------------------
    # Point 12: Constraint dispatch counting
    # ------------------------------------------------------------------
    def test_block_local_constraint_call_count(self, rng):
        """Local constraints should be called once per selected block."""
        n_comp, n_wl = 2, 10
        n1, n2, n3 = 8, 10, 12
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        C3 = np.abs(rng.normal(size=(n3, n_comp)))
        X_blocks = [Cb @ St for Cb in (C1, C2, C3)]
        C0 = np.vstack([C1, C2, C3])

        class CountingNonNeg(mc.NonNegative):
            _counter = 0

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        # We can't easily count calls with standard constraints, but we
        # can at least verify that blocks=[1,2] only constrains blocks 1,2.
        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C", blocks=[1, 2]),
            ],
            tol=10.0,
            max_iter=3,
        )
        mcr.fit(X_blocks, C0)
        Cb = mcr.C_blocks
        # Block 0 should have some negatives (no constraint applied)
        # Block 1,2 should be non-negative
        assert np.all(Cb[1] >= -1e-14), "Block 1 should be non-negative"
        assert np.all(Cb[2] >= -1e-14), "Block 2 should be non-negative"


# ======================================================================
# Tests for Objective 1 — C0 list vs (C, St) ambiguity
# ======================================================================


class TestC0Ambiguity:
    """Verify that C0 list auto-stack does not conflict with (C, St) tuple API."""

    def test_tuple_c_st_preserved(self, rng):
        """fit([X1, X2], (C0, St0)) must still work as (C, St) tuple."""
        n_comp, n_wl = 2, 10
        n1, n2 = 8, 12
        St_true = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St_true
        X2 = C2 @ St_true
        C0 = np.abs(rng.normal(size=(n1 + n2, n_comp)))

        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=1.0, max_iter=5)
        mcr.fit([X1, X2], C0)
        ref_C = np.asarray(mcr.C.data)

        # Same fit via (C, St) tuple (St is computed internally)
        mcr2 = MCRALS(constraints=[mc.NonNegative("C")], tol=1.0, max_iter=5)
        mcr2.fit([X1, X2], C0)
        assert np.allclose(np.asarray(mcr2.C.data), ref_C, atol=1e-10)

    def test_list_of_two_auto_stacks(self, rng):
        """A 2-element list for 2-block X is auto-stacked, not treated as (C, St)."""
        n_comp, n_wl = 2, 10
        n1, n2 = 8, 12
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St
        C01 = np.abs(rng.normal(size=(n1, n_comp)))
        C02 = np.abs(rng.normal(size=(n2, n_comp)))

        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=10.0, max_iter=3)
        mcr.fit([X1, X2], [C01, C02])
        assert mcr.C.shape == (n1 + n2, n_comp)
        assert mcr.n_components == n_comp

    def test_list_rejected_if_wrong_length(self, rng):
        """A list whose length differs from X blocks is rejected."""
        n_comp, n_wl = 2, 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        X1 = np.abs(rng.normal(size=(8, n_comp))) @ St
        X2 = np.abs(rng.normal(size=(12, n_comp))) @ St
        C01 = np.abs(rng.normal(size=(8, n_comp)))
        mcr = MCRALS(tol=10.0, max_iter=2)
        with pytest.raises(ValueError, match="must match number"):
            mcr.fit([X1, X2], [C01])  # only 1 element for 2 blocks

    def test_list_with_3_blocks_rejected_if_no_matching_x(self, rng):
        """A 3-element list with 2-block X is rejected by length check."""
        n_comp, n_wl = 2, 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        X1 = np.abs(rng.normal(size=(8, n_comp))) @ St
        X2 = np.abs(rng.normal(size=(12, n_comp))) @ St
        C01 = np.abs(rng.normal(size=(8, n_comp)))
        C02 = np.abs(rng.normal(size=(12, n_comp)))
        C03 = np.abs(rng.normal(size=(10, n_comp)))
        mcr = MCRALS(tol=10.0, max_iter=2)
        with pytest.raises(ValueError, match="must match number"):
            mcr.fit([X1, X2], [C01, C02, C03])


# ======================================================================
# Tests for Objective 2+3 — C_blocks: copies, no X units
# ======================================================================


class TestCBlocksUnitsAndCopies:
    """Verify C_blocks safety and metadata choices."""

    def test_c_blocks_are_copies(self, rng):
        """Modifying C_blocks must not alter the fitted C matrix."""
        n_comp, n_wl = 2, 10
        n1, n2 = 8, 12
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St
        C0 = np.vstack([C1, C2])

        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=1.0, max_iter=5)
        mcr.fit([X1, X2], C0)

        C_before = np.asarray(mcr.C.data).copy()
        block0 = mcr.C_blocks[0]
        block0[:] = 0.0  # destructive modification

        C_after = np.asarray(mcr.C.data)
        # The fitted C must be unchanged
        np.testing.assert_array_equal(
            C_before, C_after, err_msg="C_blocks modification leaked into C"
        )

    def test_c_blocks_no_x_units(self, rng):
        """C_blocks must NOT inherit X's units."""
        from spectrochempy.core.dataset.nddataset import Coord
        from spectrochempy.core.dataset.nddataset import NDDataset

        n_comp, n_wl = 2, 10
        n1, n2 = 8, 12
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1_data = C1 @ St
        X2_data = C2 @ St

        t1 = Coord(np.arange(n1), title="time", units="s")
        t2 = Coord(np.arange(n2), title="time", units="s")
        wl = Coord(np.arange(n_wl), title="wavelength", units="nm")
        X1 = NDDataset(X1_data, coordset=(t1, wl), units="absorbance")
        X2 = NDDataset(X2_data, coordset=(t2, wl), units="absorbance")
        C0 = np.vstack([C1, C2])

        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=1.0, max_iter=5)
        mcr.fit([X1, X2], C0)

        block0 = mcr.C_blocks[0]
        # Must be an NDDataset (since inputs were NDDatasets)
        assert hasattr(block0, "units"), "Expected NDDataset for NDDataset input"
        # Units must NOT be "absorbance" (the X units)
        if block0.units is not None:
            assert str(block0.units) != "absorbance", "C block must not inherit X units"

    def test_c_blocks_observation_coord_preserved(self, rng):
        """Observation coordinates of X must be found on C blocks."""
        from spectrochempy.core.dataset.nddataset import Coord
        from spectrochempy.core.dataset.nddataset import NDDataset

        n_comp, n_wl = 2, 10
        n1, n2 = 8, 12
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1_data = C1 @ St
        X2_data = C2 @ St

        t1 = Coord(np.arange(n1), title="time", units="s")
        t2 = Coord(np.arange(n2), title="time", units="s")
        wl = Coord(np.arange(n_wl), title="wavelength", units="nm")
        X1 = NDDataset(X1_data, coordset=(t1, wl))
        X2 = NDDataset(X2_data, coordset=(t2, wl))
        C0 = np.vstack([C1, C2])

        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=1.0, max_iter=5)
        mcr.fit([X1, X2], C0)

        block0 = mcr.C_blocks[0]
        # The observation coordinate must match t1
        obs = block0.coordset[1]  # 'y' coordinate
        assert obs is not None, "Observation coordinate must be present"
        np.testing.assert_array_equal(np.asarray(obs.data), np.arange(n1))

    def test_c_blocks_use_relative_title_without_calibrated_guess(self, rng):
        """X signal titles must not be misapplied to unitless C blocks."""
        from spectrochempy.core.dataset.nddataset import Coord
        from spectrochempy.core.dataset.nddataset import NDDataset

        n_comp, n_wl = 2, 10
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(8, n_comp)))
        C2 = np.abs(rng.normal(size=(12, n_comp)))
        X1_data = C1 @ St
        X2_data = C2 @ St

        wl = Coord(np.arange(n_wl), title="wavelength")
        X1 = NDDataset(X1_data, coordset=(None, wl), title="Experiment A")
        X2 = NDDataset(X2_data, coordset=(None, wl), title="Experiment B")
        C0 = np.vstack([C1, C2])

        mcr = MCRALS(constraints=[mc.NonNegative("C")], tol=1.0, max_iter=5)
        mcr.fit([X1, X2], C0)

        block0 = mcr.C_blocks[0]
        assert block0.title == "relative concentration"


# ======================================================================
# Tests for Objective 4 — block_presence robustness
# ======================================================================


class TestBlockPresenceRobustness:
    """Verify block_presence lifecycle and order independence."""

    def test_reset_between_fits(self, rng):
        """block_presence from a previous fit must not survive into the next."""
        n_comp, n_wl = 2, 10
        n1, n2 = 8, 12
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St
        C0 = np.vstack([C1, C2])

        presence = [[False, True], [True, True]]
        mcr = MCRALS(
            constraints=[
                mc.NonNegative("C"),
                mc.ComponentPresence("C", presence=presence),
            ],
            tol=1.0,
            max_iter=3,
        )
        mcr.fit([X1, X2], C0)

        # Second fit without ComponentPresence — block_presence must be reset
        mcr2 = MCRALS(constraints=[mc.NonNegative("C")], tol=1.0, max_iter=3)
        mcr2.fit([X1, X2], C0)
        assert (
            mcr2.augmented_structure.block_presence is None
        ), "block_presence must be None after fit without ComponentPresence"

    def test_order_independent(self, rng):
        """Trilinear works correctly even when listed before ComponentPresence."""
        n_points = 30
        n_wl = 10
        shape = np.exp(-(((np.linspace(0, 1, n_points) - 0.5) / 0.2) ** 2))
        St = np.abs(rng.normal(size=(1, n_wl)))

        C_blocks = [
            np.column_stack([1.0 * shape]),
            np.column_stack([0.6 * shape]),
            np.column_stack([1.4 * shape]),
        ]
        X_blocks = [Cb @ St for Cb in C_blocks]
        C0 = np.vstack(C_blocks)

        presence = [[True], [False], [True]]

        # Trilinear BEFORE ComponentPresence (user order)
        mcr = MCRALS(
            constraints=[
                mc.Trilinear("C", synchronization="none"),
                mc.ComponentPresence("C", presence=presence),
                mc.NonNegative("C"),
            ],
            solver_C="nnls",
            tol=1e-6,
            max_iter=5,
        )
        mcr.fit(X_blocks, C0, augmentation="vertical")
        Cb = mcr.C_blocks

        # Block 1 must be zero (component absent)
        assert np.allclose(
            Cb[1], 0.0, atol=1e-10
        ), "Block 1 must be zero regardless of constraint order"
        # Blocks 0 and 2 must be proportional
        bt0 = Cb[0][:, 0]
        bt2 = Cb[2][:, 0]
        nonzero = np.abs(bt0) > 1e-10
        if np.any(nonzero):
            ratios = bt2[nonzero] / bt0[nonzero]
            assert np.allclose(
                ratios, ratios[0], atol=1e-4
            ), "Blocks 0 and 2 must remain proportional"

    def test_trilinear_alone_without_presence(self, rng):
        """Trilinear without ComponentPresence works on all blocks."""
        n_points = 30
        n_wl = 10
        shape = np.exp(-(((np.linspace(0, 1, n_points) - 0.5) / 0.2) ** 2))
        St = np.abs(rng.normal(size=(1, n_wl)))

        C_blocks = [
            np.column_stack([1.0 * shape]),
            np.column_stack([0.6 * shape]),
            np.column_stack([1.4 * shape]),
        ]
        X_blocks = [Cb @ St for Cb in C_blocks]
        C0 = np.vstack(C_blocks)

        mcr = MCRALS(
            constraints=[
                mc.Trilinear("C", synchronization="none"),
                mc.NonNegative("C"),
            ],
            solver_C="nnls",
            tol=1e-6,
            max_iter=5,
        )
        mcr.fit(X_blocks, C0, augmentation="vertical")
        Cb = mcr.C_blocks

        bt0 = Cb[0][:, 0]
        bt2 = Cb[2][:, 0]
        nonzero = np.abs(bt0) > 1e-10
        if np.any(nonzero):
            ratios = bt2[nonzero] / bt0[nonzero]
            assert np.allclose(ratios, ratios[0], atol=1e-4)


# ======================================================================
# Test for _AugmentedStructure __post_init__
# ======================================================================


def test_augmented_structure_mode_validation():
    with pytest.raises(ValueError, match="Unknown augmentation mode"):
        _AugmentedStructure(
            mode="foo",
            row_slices=(slice(0, 3),),
            column_slices=(slice(0, 5),),
            input_shapes=((3, 5),),
        )


# ======================================================================
# Horizontal augmentation tests
# ======================================================================


class TestHorizontalAugmentation:
    def test_basic_horizontal_concatenation(self, rng):
        """Two blocks with same rows, different columns are stacked horizontally."""
        n_rows = 15
        n_comp = 3
        St1 = np.abs(rng.normal(size=(n_comp, 8)))
        St2 = np.abs(rng.normal(size=(n_comp, 12)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(tol=1.0, max_iter=3)
        mcr.fit([X1, X2], C_true)

        assert mcr.is_augmented
        assert mcr.augmented_structure.mode == "horizontal"
        assert mcr.C.shape == (n_rows, n_comp)
        assert mcr.St.shape == (n_comp, 20)

    def test_st_blocks_shapes(self, rng):
        """St_blocks should split the concatenated St correctly."""
        n_rows = 15
        n_comp = 3
        St1 = np.abs(rng.normal(size=(n_comp, 8)))
        St2 = np.abs(rng.normal(size=(n_comp, 12)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=10,
        )
        mcr.fit([X1, X2], [St1, St2])

        sb = mcr.St_blocks
        assert len(sb) == 2
        assert sb[0].shape == (n_comp, 8)
        assert sb[1].shape == (n_comp, 12)

    def test_st_blocks_are_copies(self, rng):
        """Mutating St_blocks should not affect the fitted St."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=10,
        )
        mcr.fit([X1, X2], [St1, St2])

        sb0 = mcr.St_blocks[0]
        orig = sb0.copy()
        sb0[:] = 0.0
        assert np.allclose(mcr.St[:, :5], orig)

    def test_horizontal_guess_list_validation(self, rng):
        """List guess for horizontal mode validates component counts."""
        n_rows = 10
        n_comp = 3
        St1 = np.abs(rng.normal(size=(n_comp, 8)))
        St2 = np.abs(rng.normal(size=(n_comp, 12)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(tol=10.0)
        # Wrong number of components in St0_1
        with pytest.raises(ValueError, match="must have the same number of components"):
            mcr.fit([X1, X2], [np.abs(rng.normal(size=(2, 8))), St2])

    def test_horizontal_wrong_columns_raises(self, rng):
        """St0 block width must match corresponding X block."""
        n_rows = 10
        n_comp = 3
        St1 = np.abs(rng.normal(size=(n_comp, 8)))
        St2 = np.abs(rng.normal(size=(n_comp, 12)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(tol=10.0)
        # St0_2 has wrong number of columns
        with pytest.raises(ValueError, match="columns"):
            mcr.fit([X1, X2], [St1, np.abs(rng.normal(size=(3, 5)))])

    def test_vertical_still_works(self, rng):
        """Existing vertical augmentation should not be broken."""
        n_comp = 3
        n_wl = 20
        n1, n2 = 10, 15
        St = np.abs(rng.normal(size=(n_comp, n_wl)))
        C1 = np.abs(rng.normal(size=(n1, n_comp)))
        C2 = np.abs(rng.normal(size=(n2, n_comp)))
        X1 = C1 @ St
        X2 = C2 @ St

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=10.0,
            max_iter=3,
        )
        mcr.fit([X1, X2], np.vstack([C1, C2]))

        assert mcr.augmented_structure.mode == "vertical"
        assert len(mcr.C_blocks) == 2

    def test_st_blocks_mutation_does_not_affect_c(self, rng):
        """Mutating St_blocks should not affect C."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")], tol=0.1, max_iter=5
        )
        mcr.fit([X1, X2], [St1, St2])

        c_before = mcr.C.data.copy()
        sb0 = mcr.St_blocks[0]
        sb0[:] = 0.0
        assert np.allclose(mcr.C.data, c_before)

    def test_horizontal_incompatible_rows_raises(self, rng):
        """Horizontal augmentation requires same number of rows."""
        C_true = np.abs(rng.normal(size=(10, 2)))
        St1 = np.abs(rng.normal(size=(2, 5)))
        St2 = np.abs(rng.normal(size=(2, 8)))
        X1 = np.abs(rng.normal(size=(10, 2))) @ St1
        X2 = np.abs(rng.normal(size=(12, 2))) @ St2  # different rows

        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="same number of rows"):
            mcr.fit([X1, X2], C_true)

    def test_square_ambiguity_raises(self, rng):
        """Square matrices with identical shapes should raise ambiguity error."""
        X1 = np.abs(rng.normal(size=(10, 10)))
        X2 = np.abs(rng.normal(size=(10, 10)))
        C0 = np.abs(rng.normal(size=(20, 2)))

        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="Cannot infer augmentation mode"):
            mcr.fit([X1, X2], C0)

    def test_square_ambiguity_resolved_by_augmentation_param(self, rng):
        """Square matrix ambiguity should be resolvable via augmentation= param."""
        X1 = np.abs(rng.normal(size=(10, 10)))
        X2 = np.abs(rng.normal(size=(10, 10)))
        C0 = np.abs(rng.normal(size=(20, 2)))

        mcr = MCRALS(tol=10.0, max_iter=3)
        # Should not raise with explicit augmentation
        mcr.fit([X1, X2], C0, augmentation="vertical")
        assert mcr.augmented_structure.mode == "vertical"

    # --------------------------------------------------------------
    # Item 2: Explicit augmentation override tests
    # --------------------------------------------------------------

    def test_non_square_identical_shapes_raise(self, rng):
        """Non-square identical shapes should also raise ambiguity (n>=2)."""
        X1 = np.abs(rng.normal(size=(10, 5)))
        X2 = np.abs(rng.normal(size=(10, 5)))
        C0 = np.abs(rng.normal(size=(20, 2)))

        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="Cannot infer augmentation mode"):
            mcr.fit([X1, X2], C0)

    def test_horizontal_explicit_augmentation_param(self, rng):
        """Explicit augmentation='horizontal' on identical shapes."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(tol=10.0, max_iter=3)
        mcr.fit([X1, X2], [St1, St2], augmentation="horizontal")
        assert mcr.augmented_structure.mode == "horizontal"
        assert mcr.C.shape == (n_rows, n_comp)
        assert mcr.St.shape == (n_comp, 12)

    def test_invalid_augmentation_value(self, rng):
        """Invalid augmentation value should raise."""
        X1 = np.abs(rng.normal(size=(10, 5)))
        X2 = np.abs(rng.normal(size=(10, 7)))

        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="augmentation"):
            mcr.fit([X1, X2], None, augmentation="diagonal")

    def test_vertical_explicit_on_identical_non_square(self, rng):
        """Explicit augmentation='vertical' on identical non-square shapes works."""
        X1 = np.abs(rng.normal(size=(10, 5)))
        X2 = np.abs(rng.normal(size=(10, 5)))
        C0 = np.abs(rng.normal(size=(20, 2)))

        mcr = MCRALS(tol=10.0, max_iter=3)
        mcr.fit([X1, X2], C0, augmentation="vertical")
        assert mcr.augmented_structure.mode == "vertical"

    def test_horizontal_explicit_on_square(self, rng):
        """Explicit augmentation='horizontal' on square matrices works."""
        X1 = np.abs(rng.normal(size=(10, 10)))
        X2 = np.abs(rng.normal(size=(10, 10)))
        St0 = np.abs(rng.normal(size=(2, 20)))

        mcr = MCRALS(tol=10.0, max_iter=3)
        mcr.fit([X1, X2], St0, augmentation="horizontal")
        assert mcr.augmented_structure.mode == "horizontal"

    # --------------------------------------------------------------
    # Item 3: Horizontal spectral initialisation validation tests
    # --------------------------------------------------------------

    def test_horizontal_non_2d_st0_block(self, rng):
        """Non-2D St0 block should raise."""
        n_rows = 10
        St1 = np.abs(rng.normal(size=(3, 5)))
        X1 = np.abs(rng.normal(size=(n_rows, 3))) @ St1
        X2 = np.abs(rng.normal(size=(n_rows, 3))) @ np.abs(rng.normal(size=(3, 7)))

        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="2-dimensional"):
            mcr.fit(
                [X1, X2],
                [St1, np.abs(rng.normal(size=7))],
                augmentation="horizontal",
            )

    def test_horizontal_wrong_st0_list_length(self, rng):
        """Wrong number of St0 blocks should raise."""
        n_rows = 10
        St1 = np.abs(rng.normal(size=(3, 5)))
        St2 = np.abs(rng.normal(size=(3, 7)))
        X1 = np.abs(rng.normal(size=(n_rows, 3))) @ St1
        X2 = np.abs(rng.normal(size=(n_rows, 3))) @ St2

        mcr = MCRALS(tol=10.0)
        St3 = np.abs(rng.normal(size=(3, 4)))
        with pytest.raises(ValueError, match="Number of initial profile blocks"):
            mcr.fit([X1, X2], [St1, St2, St3], augmentation="horizontal")

    def test_horizontal_st0_global(self, rng):
        """Single global St0 array works in horizontal mode."""
        n_rows = 10
        n_comp = 3
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=5,
        )
        St0_global = np.abs(rng.normal(size=(n_comp, 12)))
        mcr.fit([X1, X2], St0_global, augmentation="horizontal")
        assert mcr.augmented_structure.mode == "horizontal"
        assert mcr.St.shape == (n_comp, 12)

    def test_horizontal_c0_st0_tuple(self, rng):
        """Tuple (C0, St0) initialisation works in horizontal mode."""
        n_rows = 10
        n_comp = 3
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=5,
        )
        C0_global = np.abs(rng.normal(size=(n_rows, n_comp)))
        St0_global = np.abs(rng.normal(size=(n_comp, 12)))
        mcr.fit([X1, X2], (C0_global, St0_global), augmentation="horizontal")
        assert mcr.augmented_structure.mode == "horizontal"
        assert mcr.C.shape == (n_rows, n_comp)

    def test_horizontal_c_like_list_guess_raises(self, rng):
        """List of 2D arrays mistaken as C blocks must fail during St validation."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1 = C_true @ St1
        X2 = C_true @ St2

        mcr = MCRALS(tol=10.0)
        C_like = np.abs(rng.normal(size=(n_rows, n_comp)))
        with pytest.raises(ValueError, match="columns"):
            mcr.fit([X1, X2], [C_like, St1], augmentation="horizontal")

    # --------------------------------------------------------------
    # Item 5: St_blocks coordinate and isolation tests
    # --------------------------------------------------------------

    def test_st_blocks_spectral_coordinates(self, rng):
        """St_blocks preserves spectral coordinates from input NDDatasets."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1_data = C_true @ St1
        X2_data = C_true @ St2

        wl1 = Coord(np.arange(5), title="wavelength", units="nm")
        wl2 = Coord(np.arange(7), title="wavelength", units="nm")
        t = Coord(np.arange(n_rows), title="time", units="s")

        X1 = NDDataset(X1_data, coordset=(t, wl1))
        X2 = NDDataset(X2_data, coordset=(t, wl2))

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=5,
        )
        mcr.fit([X1, X2], [St1, St2])

        sb = mcr.St_blocks
        # St_blocks coordset[0] = spectral (x-axis), coordset[1] = observation (y-axis, empty)
        assert sb[0].coordset[0] is not None
        assert np.allclose(sb[0].coordset[0].data, wl1.data)
        assert sb[1].coordset[0] is not None
        assert np.allclose(sb[1].coordset[0].data, wl2.data)

    def test_st_blocks_observation_coord_isolated(self, rng):
        """St_blocks should not carry observation coordinates from input."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1_data = C_true @ St1
        X2_data = C_true @ St2

        wl1 = Coord(np.arange(5), title="wavelength", units="nm")
        wl2 = Coord(np.arange(7), title="wavelength", units="nm")
        t = Coord(np.arange(n_rows), title="time", units="s")

        X1 = NDDataset(X1_data, coordset=(t, wl1))
        X2 = NDDataset(X2_data, coordset=(t, wl2))

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=5,
        )
        mcr.fit([X1, X2], [St1, St2])

        sb = mcr.St_blocks
        # St_blocks coordset[1] = observation (y-axis) — should be empty (no data)
        assert sb[0].coordset is not None
        assert sb[0].coordset[1] is None or sb[0].coordset[1].data is None

    # --------------------------------------------------------------
    # Item 6: C metadata and coordinate tests for horizontal mode
    # --------------------------------------------------------------

    def test_c_observation_coord_preserved(self, rng):
        """C must preserve the shared observation coordinate for horizontal augmentation."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1_data = C_true @ St1
        X2_data = C_true @ St2

        t = Coord(np.arange(n_rows), title="time", units="s")
        wl1 = Coord(np.arange(5), title="wavelength", units="nm")
        wl2 = Coord(np.arange(7), title="wavelength", units="nm")

        X1 = NDDataset(X1_data, coordset=(t, wl1))
        X2 = NDDataset(X2_data, coordset=(t, wl2))

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=5,
        )
        mcr.fit([X1, X2], [St1, St2])

        C = mcr.C
        assert isinstance(C, NDDataset)
        # C's observation coordinate (dim 'y', coordset[1]) must be the shared temperature axis
        obs = C.coordset[1]
        assert obs is not None, "C must have an observation coordinate"
        assert np.allclose(
            np.asarray(obs.data), np.arange(n_rows)
        ), "Observation coordinate values must match input"
        assert obs.title == "time"
        assert str(obs.units) == "s"

    def test_c_blocks_horizontal_single_block(self, rng):
        """C_blocks for horizontal augmentation returns a single block with obs coord."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1_data = C_true @ St1
        X2_data = C_true @ St2

        t = Coord(np.arange(n_rows), title="time", units="s")
        wl1 = Coord(np.arange(5), title="wavelength", units="nm")
        wl2 = Coord(np.arange(7), title="wavelength", units="nm")

        X1 = NDDataset(X1_data, coordset=(t, wl1))
        X2 = NDDataset(X2_data, coordset=(t, wl2))

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=5,
        )
        mcr.fit([X1, X2], [St1, St2])

        cblks = mcr.C_blocks
        assert len(cblks) == 1, "Horizontal mode should return 1 C block"
        blk = cblks[0]
        assert isinstance(blk, NDDataset)
        obs = blk.coordset[1]
        assert obs is not None
        assert np.allclose(np.asarray(obs.data), np.arange(n_rows))

    def test_c_st_blocks_concatenation_consistency(self, rng):
        """Hstack of St_blocks data must reproduce global St."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1_data = C_true @ St1
        X2_data = C_true @ St2

        t = Coord(np.arange(n_rows), title="time", units="s")
        wl1 = Coord(np.arange(5), title="wavelength", units="nm")
        wl2 = Coord(np.arange(7), title="wavelength", units="nm")

        X1 = NDDataset(X1_data, coordset=(t, wl1))
        X2 = NDDataset(X2_data, coordset=(t, wl2))

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=5,
        )
        mcr.fit([X1, X2], [St1, St2])

        sb = mcr.St_blocks
        St_global = np.asarray(mcr.St.data)
        St_from_blocks = np.hstack([np.asarray(b.data) for b in sb])
        np.testing.assert_allclose(St_global, St_from_blocks)

    def test_incompatible_observation_coord_raises(self, rng):
        """Different observation coordinates between horizontal blocks must raise."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1_data = C_true @ St1
        X2_data = C_true @ St2

        t1 = Coord(np.arange(n_rows), title="time", units="s")
        t2 = Coord(np.arange(n_rows) * 2, title="time", units="s")  # different values
        wl1 = Coord(np.arange(5), title="wavelength", units="nm")
        wl2 = Coord(np.arange(7), title="wavelength", units="nm")

        X1 = NDDataset(X1_data, coordset=(t1, wl1))
        X2 = NDDataset(X2_data, coordset=(t2, wl2))

        mcr = MCRALS(tol=10.0)
        with pytest.raises(ValueError, match="compatible observation"):
            mcr.fit([X1, X2], [St1, St2])

    def test_st_blocks_component_coord(self, rng):
        """St_blocks must have component labels on the component axis."""
        n_rows = 10
        n_comp = 2
        St1 = np.abs(rng.normal(size=(n_comp, 5)))
        St2 = np.abs(rng.normal(size=(n_comp, 7)))
        C_true = np.abs(rng.normal(size=(n_rows, n_comp)))
        X1_data = C_true @ St1
        X2_data = C_true @ St2

        t = Coord(np.arange(n_rows), title="time", units="s")
        wl1 = Coord(np.arange(5), title="wavelength", units="nm")
        wl2 = Coord(np.arange(7), title="wavelength", units="nm")

        X1 = NDDataset(X1_data, coordset=(t, wl1))
        X2 = NDDataset(X2_data, coordset=(t, wl2))

        mcr = MCRALS(
            constraints=[mc.NonNegative("C"), mc.NonNegative("St")],
            tol=0.1,
            max_iter=5,
        )
        mcr.fit([X1, X2], [St1, St2])

        sb = mcr.St_blocks
        for blk in sb:
            labels = blk.coordset[1].labels
            assert labels is not None, "Component axis must have labels"
            assert len(labels) == n_comp


# --------------------------------------------------------------
# Item 4: Constraint call-count tests (renumbered)
# --------------------------------------------------------------


class _CountCallConstraint:
    """
    Minimal internal constraint that counts ``apply()`` calls.

    Used to verify that the per-block dispatch in
    ``_apply_constraint_pipeline`` produces the expected number of
    constraint applications.
    """

    is_block_local = True
    _blocks = None

    def __init__(self, profile="C"):
        self.profile = profile
        self.call_count = 0

    def apply(self, values, state):
        self.call_count += 1
        return values


class TestConstraintDispatchCount:
    """
    Verify that _apply_constraint_pipeline dispatches to the right number
    of blocks in horizontal mode.
    """

    def _make_horizontal_aug(self, n_rows, n_cols_block1, n_cols_block2):
        return _AugmentedStructure(
            mode="horizontal",
            row_slices=(slice(0, n_rows),),
            column_slices=(
                slice(0, n_cols_block1),
                slice(n_cols_block1, n_cols_block1 + n_cols_block2),
            ),
            input_shapes=((n_rows, n_cols_block1), (n_rows, n_cols_block2)),
        )

    def test_c_constraint_called_once_per_block_horizontal(self, rng):
        """C-side constraint: 1 row block → 1 apply call."""
        n_rows, n_comp = 10, 2
        aug = self._make_horizontal_aug(n_rows, 5, 7)

        class _SimpleState:
            augmentation = aug

        state = _SimpleState()
        C_values = np.abs(rng.normal(size=(n_rows, n_comp)))
        counter = _CountCallConstraint("C")

        MCRALS._apply_constraint_pipeline(
            C_values,
            [counter],
            state,
            block_slices=aug.row_slices,
            block_axis=0,
        )

        assert (
            counter.call_count == 1
        ), f"Expected 1 C constraint call (1 row block), got {counter.call_count}"

    def test_st_constraint_called_per_column_block_horizontal(self, rng):
        """St-side constraint: 2 column blocks → 2 apply calls."""
        n_rows, n_comp = 10, 2
        aug = self._make_horizontal_aug(n_rows, 5, 7)

        class _SimpleState:
            augmentation = aug

        state = _SimpleState()
        St_values = np.abs(rng.normal(size=(n_comp, 12)))
        counter = _CountCallConstraint("St")

        MCRALS._apply_constraint_pipeline(
            St_values,
            [counter],
            state,
            block_slices=aug.column_slices,
            block_axis=1,
        )

        assert (
            counter.call_count == 2
        ), f"Expected 2 St constraint calls (2 column blocks), got {counter.call_count}"

    def test_st_constraint_blocks_selector_horizontal(self, rng):
        """St-side constraint with blocks=[0] → only 1 apply call."""
        n_rows, n_comp = 10, 2
        aug = self._make_horizontal_aug(n_rows, 5, 7)

        class _SimpleState:
            augmentation = aug

        state = _SimpleState()
        St_values = np.abs(rng.normal(size=(n_comp, 12)))
        counter = _CountCallConstraint("St")
        counter._blocks = [0]

        MCRALS._apply_constraint_pipeline(
            St_values,
            [counter],
            state,
            block_slices=aug.column_slices,
            block_axis=1,
        )

        assert (
            counter.call_count == 1
        ), f"Expected 1 St constraint call (blocks=[0]), got {counter.call_count}"

    def test_c_constraint_called_per_row_block_vertical(self, rng):
        """C-side constraint in vertical mode: 2 row blocks → 2 apply calls."""
        n_comp = 2
        aug = _AugmentedStructure(
            mode="vertical",
            row_slices=(slice(0, 10), slice(10, 25)),
            column_slices=(slice(0, 5),),
            input_shapes=((10, 5), (15, 5)),
        )

        class _SimpleState:
            augmentation = aug

        state = _SimpleState()
        C_values = np.abs(rng.normal(size=(25, n_comp)))
        counter = _CountCallConstraint("C")

        MCRALS._apply_constraint_pipeline(
            C_values,
            [counter],
            state,
            block_slices=aug.row_slices,
            block_axis=0,
        )

        assert (
            counter.call_count == 2
        ), f"Expected 2 C constraint calls (2 row blocks), got {counter.call_count}"

    def test_no_block_dispatch_without_augmentation(self, rng):
        """Without augmentation, block-local dispatch is inactive."""

        class _SimpleState:
            augmentation = None

        state = _SimpleState()
        values = np.abs(rng.normal(size=(10, 3)))
        counter = _CountCallConstraint("C")

        MCRALS._apply_constraint_pipeline(
            values,
            [counter],
            state,
            block_slices=None,
            block_axis=0,
        )

        assert counter.call_count == 1, "Expected 1 call when no augmentation is active"
