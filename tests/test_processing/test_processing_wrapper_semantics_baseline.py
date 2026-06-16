"""
Characterization tests for processing-wrapper semantics on NDDataset.

This suite characterizes CURRENT behavior of processing wrappers
(smooth, savgol, whittaker, basc, detrend, asls, denoise).
It does NOT validate a desired future policy.

Coverage:
    - Return type, shape, dims, CoordSet, units, masks
    - Metadata (title, name, author, description, origin, meta)
    - History, ROI, identity, provenance

Two distinct assembly patterns emerge:
    Group A (Filter/PCA-based: smooth, savgol, whittaker, denoise):
        name appended, no modeldata attribute, roi recomputed,
        history rewritten
    Group B (Baseline-based: basc, detrend, asls):
        name preserved, no modeldata attribute, roi preserved,
        history appended
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset


def _call_filter(ds, method):
    """Dispatch a filter method with appropriate kwargs."""
    kwargs = {}
    if method in ("smooth", "savgol"):
        kwargs["size"] = 3
    elif method == "whittaker":
        kwargs["lamb"] = 1.0
    fn = getattr(ds, method)
    return fn(**kwargs)


# ======================================================================================
# FIXTURES
# ======================================================================================


@pytest.fixture
def ds():
    """
    Semantic-rich 2D dataset for processing wrapper characterization.

    - dims: ['y', 'x'] (5, 7)
    - CoordSet with titles, units
    - Full metadata
    """
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(np.linspace(4000.0, 1000.0, 7), title="wavenumber", units="cm^-1")
    ds = NDDataset(
        np.arange(35.0, dtype="float64").reshape(5, 7),
        coordset=[y, x],
        title="ds_title",
        name="ds_name",
    )
    ds.author = "test_author"
    ds.description = "test description"
    ds.origin = "test_origin"
    ds.meta.project = "test_project"
    ds.roi = [0.0, 10.0]
    ds.history = ["original entry"]
    return ds


@pytest.fixture
def ds_masked():
    """2D dataset with a masked value at [0, 0]."""
    arr = np.ma.MaskedArray(
        np.arange(35.0).reshape(5, 7), mask=np.zeros((5, 7), dtype=bool)
    )
    arr[0, 0] = np.ma.masked
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(np.linspace(4000.0, 1000.0, 7), title="wn", units="cm^-1")
    return NDDataset(arr, coordset=[y, x])


@pytest.fixture
def ds_1d():
    """Simple 1D dataset with CoordSet (needed for baseline wrappers)."""
    x = Coord(np.linspace(0.0, 10.0, 3))
    return NDDataset(np.array([10.0, 20.0, 30.0]), coordset=[x])


@pytest.fixture
def ds_unitful():
    """2D dataset with data units."""
    y = Coord(np.linspace(0.0, 60.0, 5), title="time", units="s")
    x = Coord(np.linspace(4000.0, 1000.0, 7))
    return NDDataset(np.random.rand(5, 7), coordset=[y, x], units="m")


# ======================================================================================
# GROUP A: FILTER-BASED WRAPPERS (smooth, savgol, whittaker)
# ======================================================================================


class TestFilterWrappers:
    """
    Characterize Filter-based wrappers: smooth, savgol, whittaker.

    Observation: these all rewrite history, append _Filter.transform
    to name, expose no modeldata attribute, and recompute roi from data range.
    """

    @pytest.mark.parametrize("method", ["smooth", "savgol", "whittaker"])
    def test_return_type(self, ds, method):
        r = _call_filter(ds, method)
        assert isinstance(r, NDDataset)

    @pytest.mark.parametrize("method", ["smooth", "savgol", "whittaker"])
    def test_shape_preserved(self, ds, method):
        r = _call_filter(ds, method)
        assert r.shape == (5, 7)
        assert r.dims == ["y", "x"]

    @pytest.mark.parametrize("method", ["smooth", "savgol", "whittaker"])
    def test_coordset_preserved(self, ds, method):
        r = _call_filter(ds, method)
        assert r.y.size == 5
        assert r.x.size == 7
        assert str(r.y.units) == "s"
        assert str(r.x.units) == "cm⁻¹"
        assert r.y.title == "time"
        assert r.x.title == "wavenumber"

    @pytest.mark.parametrize("method", ["smooth", "savgol", "whittaker"])
    def test_title_preserved(self, ds, method):
        r = _call_filter(ds, method)
        assert r.title == "ds_title"

    @pytest.mark.parametrize("method", ["smooth", "savgol", "whittaker"])
    def test_name_appended(self, ds, method):
        """Notable: name is appended with '_Filter.transform'."""
        r = _call_filter(ds, method)
        assert r.name == "ds_name_Filter.transform"

    def test_author_preserved(self, ds):
        r = ds.smooth(size=3)
        assert r.author == "test_author"

    def test_description_preserved(self, ds):
        r = ds.smooth(size=3)
        assert r.description == "test description"

    def test_origin_preserved(self, ds):
        r = ds.smooth(size=3)
        assert r.origin == "test_origin"

    def test_meta_preserved(self, ds):
        r = ds.smooth(size=3)
        assert r.meta.project == "test_project"

    def test_meta_deep_copied(self, ds):
        r = ds.smooth(size=3)
        r.meta.project = "modified"
        assert ds.meta.project == "test_project"

    @pytest.mark.parametrize("method", ["smooth", "savgol", "whittaker"])
    def test_history_rewritten(self, ds, method):
        """
        Notable: history is REWRITTEN, not appended.
        Original entries are lost.
        """
        r = _call_filter(ds, method)
        assert len(r.history) == 1
        assert "Created using method Filter.transform" in r.history[0]
        assert "original" not in r.history[0].lower()

    @pytest.mark.parametrize("method", ["smooth", "savgol", "whittaker"])
    def test_modeldata_dropped(self, ds, method):
        """Notable: wrappers no longer expose a modeldata attribute."""
        r = _call_filter(ds, method)
        assert not hasattr(r, "modeldata")

    @pytest.mark.parametrize("method", ["smooth", "savgol", "whittaker"])
    def test_roi_recomputed(self, ds, method):
        """Notable: roi is recomputed from data range, not preserved."""
        r = _call_filter(ds, method)
        assert r.roi != [0.0, 10.0]
        # roi reflects data min/max after processing
        assert len(r.roi) == 2


# ======================================================================================
# GROUP B: BASELINE-BASED WRAPPERS (basc, detrend, asls)
# ======================================================================================


class TestBaselineWrappers:
    """
    Characterize Baseline-based wrappers: basc, detrend, asls.

    Observation: these all append history, preserve name unchanged,
    expose no modeldata attribute, and preserve roi unchanged.
    """

    @pytest.mark.parametrize("method", ["basc", "detrend", "asls"])
    def test_return_type(self, ds, method):
        r = getattr(ds, method)()
        assert isinstance(r, NDDataset)

    @pytest.mark.parametrize("method", ["basc", "detrend", "asls"])
    def test_shape_preserved(self, ds, method):
        r = getattr(ds, method)()
        assert r.shape == (5, 7)
        assert r.dims == ["y", "x"]

    @pytest.mark.parametrize("method", ["basc", "detrend", "asls"])
    def test_coordset_preserved(self, ds, method):
        r = getattr(ds, method)()
        assert r.y.size == 5
        assert r.x.size == 7

    @pytest.mark.parametrize("method", ["basc", "detrend", "asls"])
    def test_title_preserved(self, ds, method):
        r = getattr(ds, method)()
        assert r.title == "ds_title"

    @pytest.mark.parametrize("method", ["basc", "detrend", "asls"])
    def test_name_preserved(self, ds, method):
        """Notable: name is preserved unchanged (unlike Filter wrappers)."""
        r = getattr(ds, method)()
        assert r.name == "ds_name"

    def test_author_preserved(self, ds):
        r = ds.basc()
        assert r.author == "test_author"

    def test_description_preserved(self, ds):
        r = ds.basc()
        assert r.description == "test description"

    def test_origin_preserved(self, ds):
        r = ds.basc()
        assert r.origin == "test_origin"

    def test_meta_preserved(self, ds):
        r = ds.basc()
        assert r.meta.project == "test_project"

    def test_meta_deep_copied(self, ds):
        r = ds.basc()
        r.meta.project = "modified"
        assert ds.meta.project == "test_project"

    @pytest.mark.parametrize("method", ["basc", "detrend", "asls"])
    def test_history_appended(self, ds, method):
        """
        Notable: history is APPENDED (unlike Filter wrappers
        which rewrite it).
        """
        r = getattr(ds, method)()
        assert len(r.history) == 2
        assert "original entry" in r.history[0].lower()
        assert "Binary operation" in r.history[1]

    @pytest.mark.parametrize("method", ["basc", "detrend", "asls"])
    def test_modeldata_preserved(self, ds, method):
        """Notable: wrappers no longer expose a modeldata attribute."""
        r = getattr(ds, method)()
        assert not hasattr(r, "modeldata")

    @pytest.mark.parametrize("method", ["basc", "detrend", "asls"])
    def test_roi_preserved(self, ds, method):
        """
        Notable: roi is preserved unchanged (unlike Filter wrappers
        which recompute it from data range).
        """
        r = getattr(ds, method)()
        assert r.roi == [0.0, 10.0]


# ======================================================================================
# GROUP C: PCA-BASED WRAPPER (denoise)
# ======================================================================================


class TestPcaWrapper:
    """
    Characterize PCA-based denoise wrapper.

    Observation: Follows Group A pattern (name appended, no modeldata
    attribute, roi recomputed, history rewritten) but with different
    method suffix (_PCA.inverse_transform).
    """

    def test_return_type(self, ds):
        r = ds.denoise(ratio=99.0)
        assert isinstance(r, NDDataset)

    def test_shape_preserved(self, ds):
        r = ds.denoise(ratio=99.0)
        assert r.shape == (5, 7)
        assert r.dims == ["y", "x"]

    def test_coordset_preserved(self, ds):
        r = ds.denoise(ratio=99.0)
        assert r.y.size == 5
        assert r.x.size == 7

    def test_title_preserved(self, ds):
        r = ds.denoise(ratio=99.0)
        assert r.title == "ds_title"

    def test_name_appended(self, ds):
        """Notable: name is appended with '_PCA.inverse_transform'."""
        r = ds.denoise(ratio=99.0)
        assert r.name == "ds_name_PCA.inverse_transform"

    def test_author_from_system(self, ds):
        """
        Notable: denoise overrides author with system hostname,
        unlike Filter wrappers which preserve it.
        """
        r = ds.denoise(ratio=99.0)
        assert r.author != "test_author"
        assert "@" in r.author  # likely "user@hostname" format

    def test_origin_preserved(self, ds):
        r = ds.denoise(ratio=99.0)
        assert r.origin == "test_origin"

    def test_meta_preserved(self, ds):
        r = ds.denoise(ratio=99.0)
        assert r.meta.project == "test_project"

    def test_history_rewritten(self, ds):
        """Notable: history rewritten (Group A pattern)."""
        r = ds.denoise(ratio=99.0)
        assert len(r.history) == 1
        assert "Created using method PCA.inverse_transform" in r.history[0]

    def test_modeldata_dropped(self, ds):
        """Notable: wrappers no longer expose a modeldata attribute."""
        r = ds.denoise(ratio=99.0)
        assert not hasattr(r, "modeldata")

    def test_roi_recomputed(self, ds):
        """Notable: roi recomputed from data (Group A pattern)."""
        r = ds.denoise(ratio=99.0)
        assert r.roi != [0.0, 10.0]


# ======================================================================================
# SHARED BEHAVIOR (all wrappers)
# ======================================================================================


class TestSharedBehavior:
    """Characterize behavior common to all processing wrappers."""

    def test_units_preserved(self, ds_unitful):
        r = ds_unitful.smooth(size=3)
        assert str(r.units) == "m"

    def test_basc_units_preserved(self, ds_unitful):
        r = ds_unitful.basc()
        assert str(r.units) == "m"

    def test_mask_after_smooth(self, ds_masked):
        r = ds_masked.smooth(size=3)
        assert not np.any(r.mask)

    def test_mask_after_basc(self, ds_masked):
        r = ds_masked.basc()
        assert np.any(r.mask)

    def test_1d_smooth(self, ds_1d):
        r = ds_1d.smooth(size=3)
        assert r.shape == (3,)

    def test_1d_basc(self, ds_1d):
        r = ds_1d.basc()
        assert r.shape == (3,)

    def test_1d_denoise_shape(self, ds_1d):
        ds_2d = NDDataset(np.random.rand(5, 7))
        r = ds_2d.denoise(ratio=99.0)
        assert r.shape == (5, 7)

    def test_dims_unchanged(self, ds):
        for method in ["smooth", "basc"]:
            r = _call_filter(ds, method) if method == "smooth" else ds.basc()
            assert r.dims == ["y", "x"], f"dims changed for {method}"


# ======================================================================================
# IDENTITY AND PROVENANCE
# ======================================================================================


class TestIdentityProvenance:
    """
    Characterize identity and provenance through processing wrappers.

    Observations:
    - Group A (Filter/PCA): identity partially preserved (title, author,
      origin, meta survive), but name is modified and history rewritten,
      suggesting a derived or transformed identity.
    - Group B (Baseline): identity preserved (all fields survive),
      history appended, consistent with same-object identity.
    """

    def test_filter_identity_partial(self, ds):
        r = ds.smooth(size=3)
        assert r.title == ds.title
        assert r.author == ds.author
        assert r.origin == ds.origin
        assert r.name != ds.name  # appended suffix

    def test_baseline_identity_preserved(self, ds):
        r = ds.basc()
        assert r.title == ds.title
        assert r.name == ds.name
        assert r.author == ds.author
        assert r.origin == ds.origin

    def test_filter_provenance_rewritten(self, ds):
        r = ds.smooth(size=3)
        assert len(r.history) == 1
        assert r.history[0] != ds.history[0]

    def test_baseline_provenance_extended(self, ds):
        r = ds.basc()
        assert len(r.history) == 2
        assert "original entry" in r.history[0].lower()
