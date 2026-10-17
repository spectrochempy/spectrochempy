"""
Policy tests for the processing-assembly single-source metadata policy (RFC).

Codifies the accepted RFC ``processing-assembly-metadata-policy.md`` Q1/Q2
for the Filter family (smooth, savgol, savgol_filter, whittaker):

- ``name`` is preserved;
- ``history`` appends exactly one generated entry per complete public
  operation, retaining all prior entries;
- the behavior is identical across entry forms (functional wrapper,
  NDDataset method, configurable ``Filter(...).transform``);
- Savitzky-Golay derivatives preserve identity and append history (DQ1);
- ``denoise`` / ``inverse_transform`` and Category C analysis outputs keep
  their own derived-output policies.

Prior structured entries are compared exactly; new entries are checked by
count and message without fixing their generated timestamps.
"""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.decomposition.pca import PCA
from spectrochempy.core.dataset.nddataset import NDDataset
from tests.test_core.test_dataset._semantic_dataset_helpers import (
    make_semantic_2d_dataset,
)

_OP_KWARGS = {
    "smooth": {"size": 3},
    "savgol": {"size": 3},
    "savgol_filter": {"size": 3},
    "whittaker": {"lamb": 1.0},
}

_OP_METHOD = {
    "smooth": "avg",
    "savgol": "savgol",
    "savgol_filter": "savgol",
    "whittaker": "whittaker",
}

_FILTER_ENTRY = "Created using method Filter.transform"


def _apply(dataset, op, form, **kwargs):
    if form == "function":
        return getattr(scp, op)(dataset, **kwargs)
    if form == "method":
        return getattr(dataset, op)(**kwargs)
    if form == "transform":
        return scp.Filter(method=_OP_METHOD[op], **kwargs).transform(dataset)
    raise ValueError(f"Unknown form: {form}")


@pytest.fixture
def ds():
    return make_semantic_2d_dataset(
        title="ds_title",
        name="ds_name",
        history="original entry",
    )


# ======================================================================================
# Category A: name preserved, history appended (one entry per public operation)
# ======================================================================================


class TestCategoryANameAndHistory:
    @pytest.mark.parametrize("op", ["smooth", "savgol", "savgol_filter", "whittaker"])
    @pytest.mark.parametrize("form", ["function", "method", "transform"])
    def test_name_preserved(self, ds, op, form):
        result = _apply(ds, op, form, **_OP_KWARGS[op])
        assert result.name == "ds_name"

    @pytest.mark.parametrize("op", ["smooth", "savgol", "savgol_filter", "whittaker"])
    @pytest.mark.parametrize("form", ["function", "method", "transform"])
    def test_history_appended_single_entry(self, ds, op, form):
        result = _apply(ds, op, form, **_OP_KWARGS[op])
        assert len(result.history) == len(ds.history) + 1
        assert result.history[0] == ds.history[0]
        assert _FILTER_ENTRY in result.history[-1]

    def test_prior_entries_retained_in_order(self, ds):
        ds.history = "second entry"
        result = ds.smooth(size=3)
        assert len(result.history) == 3
        assert "original entry" in result.history[0].lower()
        assert "second entry" in result.history[1].lower()
        assert _FILTER_ENTRY in result.history[2]

    def test_entry_forms_equivalent(self, ds):
        for form in ("function", "method", "transform"):
            result = _apply(ds, "savgol", form, size=3)
            assert result.name == "ds_name"
            assert len(result.history) == 2
            assert result.history[0] == ds.history[0]
            assert _FILTER_ENTRY in result.history[-1]

    def test_savgol_filter_no_double_entry(self, ds):
        result = scp.savgol_filter(ds, size=3)
        assert len(result.history) == len(ds.history) + 1


# ======================================================================================
# Category A: 1D inputs append exactly one entry (no internal squeeze entry)
# ======================================================================================


class TestCategoryA1D:
    def test_1d_single_appended_entry(self):
        ds1 = NDDataset(np.array([1.0, 2.0, 3.0]), name="ds1")
        ds1.history = "original entry"
        result = ds1.smooth(size=3)
        assert result.name == "ds1"
        assert result.shape == (3,)
        assert len(result.history) == len(ds1.history) + 1
        assert "data squeezed" not in " ".join(result.history).lower()


class TestDerivativeIdentityAndHistory:
    @pytest.mark.parametrize("ndim", [1, 2])
    @pytest.mark.parametrize("deriv", [1, 2])
    @pytest.mark.parametrize(
        "entry",
        [
            "savgol_function",
            "savgol_method",
            "alias_function",
            "alias_method",
            "differentiate_function",
            "differentiate_method",
            "transform",
        ],
    )
    def test_derivative_preserves_identity_and_appends_once(
        self, ds, ndim, deriv, entry
    ):
        ds.units = "absorbance"
        source = ds[0].squeeze() if ndim == 1 else ds
        source.annotate("prepared for differentiation")
        before = source.copy()
        kwargs = {"size": 5, "order": 2, "deriv": deriv}

        if entry == "savgol_function":
            result = scp.savgol(source, **kwargs)
        elif entry == "savgol_method":
            result = source.savgol(**kwargs)
        elif entry == "alias_function":
            result = scp.savgol_filter(source, **kwargs)
        elif entry == "alias_method":
            result = source.savgol_filter(**kwargs)
        elif entry == "differentiate_function":
            result = scp.differentiate(source, derivative_order=deriv, size=5)
        elif entry == "differentiate_method":
            result = source.differentiate(derivative_order=deriv, size=5)
        else:
            result = scp.Filter(method="savgol", **kwargs).transform(dataset=source)

        assert result.name == source.name
        assert result.history_entries[:-1] == before.history_entries
        assert len(result.history_entries) == len(before.history_entries) + 1
        assert result.history_entries[-1]["message"] == _FILTER_ENTRY
        assert result.shape == source.shape
        assert result.dims == source.dims
        assert result.coordset == source.coordset
        assert result.units == source.units / source.x.units**deriv
        ordinal = "1st" if deriv == 1 else "2nd"
        assert result.title == f"{source.title} ({ordinal} derivative)"
        result.annotate("result annotation")
        assert source.history_entries == before.history_entries
        assert source.name == before.name
        assert source.title == before.title
        assert source.units == before.units
        np.testing.assert_array_equal(source.data, before.data)

    def test_repeated_derivatives_keep_prior_operations(self, ds):
        first = ds.savgol(size=3, order=2, deriv=1)
        second = first.differentiate(size=3)
        assert first.name == second.name == ds.name
        assert second.history_entries[:-1] == first.history_entries
        assert len(second.history_entries) == len(ds.history_entries) + 2

    def test_reused_filter_preserves_each_source_lineage(self, ds):
        processor = scp.Filter(method="savgol", size=3, order=2, deriv=1)
        first = processor.transform(ds)
        other = ds.copy()
        other.name = "other_source"
        other.annotate("second source preparation")
        processor.deriv = 0
        second = processor.transform(other)
        assert first.name == ds.name
        assert first.history_entries[:-1] == ds.history_entries
        assert second.name == other.name
        assert second.history_entries[:-1] == other.history_entries
        assert len(second.history_entries) == len(other.history_entries) + 1
        assert second.title == other.title

    def test_masked_derivative_preserves_lineage(self, ds):
        ds[0] = scp.MASKED
        before = ds.history_entries
        result = ds.differentiate(size=3)
        assert result.name == ds.name
        assert result.history_entries[:-1] == before
        np.testing.assert_array_equal(result.mask, ds.mask)
        assert ds.history_entries == before


class TestCategoryAExclusions:
    def test_denoise_unchanged(self, ds):
        result = ds.denoise(ratio=99.0)
        assert result.name == "ds_name_PCA.reconstruction"
        assert len(result.history) == 1

    def test_pca_scores_unchanged(self, ds):
        pca = PCA(n_components=2)
        scores = pca.fit_transform(ds)
        assert scores.name == "ds_name_PCA.scores"
        assert len(scores.history) == 1
