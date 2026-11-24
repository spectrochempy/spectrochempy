"""Processing copies may contain protected instrument metadata."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.decorators import transfer_source_context


@pytest.mark.parametrize("locked_target", [False, True])
@pytest.mark.parametrize("locked_source", [False, True])
def test_context_transfer_merges_protected_metadata_without_unlocking_source(
    locked_target,
    locked_source,
):
    source = scp.NDDataset([1.0, 2.0])
    source.meta.acquisition = {"receiver": {"gain": 12}}
    source.meta.readonly = locked_source
    source.meta.acquisition.set_readonly(True)
    target = source.copy()
    target.meta.readonly = False
    target.meta.processing = {"method": "test"}
    target.meta.processing.set_readonly(True)
    target.meta.readonly = locked_target
    target.meta.acquisition.set_readonly(True)
    target.meta.processing.set_readonly(True)

    transfer_source_context(source, target)

    assert target.meta.acquisition == source.meta.acquisition
    assert target.meta.processing == {"method": "test"}
    assert target.meta.readonly is locked_target
    with pytest.raises(ValueError, match="read-only"):
        target.meta.processing["method"] = "changed"
    assert source.meta.readonly is locked_source
    with pytest.raises(ValueError, match="read-only"):
        source.meta.acquisition["receiver"]["gain"] = 99
    target.meta.readonly = False
    target.meta.acquisition["receiver"]["gain"] = 99
    assert source.meta.acquisition["receiver"]["gain"] == 12


@pytest.mark.parametrize("model", ["polynomial", "distribution"])
@pytest.mark.parametrize("rows", [False, True])
def test_baseline_corrects_dataset_with_readonly_instrument_metadata(model, rows):
    x = np.linspace(0, 10, 1024)
    values = 1 + 0.02 * x + np.exp(-(((x - 5) / 0.05) ** 2))
    source = scp.NDDataset(
        np.stack([values, 2 * values]) if rows else values,
        coordset=[scp.Coord([0, 1]), scp.Coord.linspace(0, 10, 1024)]
        if rows
        else [scp.Coord.linspace(0, 10, 1024)],
        units="a.u.",
    )
    source.meta.acquisition = {"receiver": {"gain": 12}}
    source.meta.readonly = True
    original = source.data.copy()
    processor = scp.Baseline(model=model).fit(source)
    result = processor.corrected
    assert result.shape == source.shape
    assert result.units == source.units
    assert np.isfinite(result.data).all()
    np.testing.assert_array_equal(result.x.data, source.x.data)
    np.testing.assert_array_equal(source.data, original)
    assert source.meta.readonly
    assert result.meta.acquisition == source.meta.acquisition
    result.meta.readonly = False
    result.meta.acquisition["receiver"]["gain"] = 99
    assert source.meta.acquisition["receiver"]["gain"] == 12


@pytest.mark.parametrize("locked", [False, True])
def test_new_context_fields_follow_target_lock_policy(locked):
    source = scp.NDDataset([1.0, 2.0])
    source.meta.new = {"nested": {"gain": 12}}
    source.meta.readonly = True
    target = scp.NDDataset([3.0, 4.0])
    target.meta.readonly = locked
    transfer_source_context(source, target)
    if locked:
        with pytest.raises(ValueError, match="read-only"):
            target.meta.new["nested"]["gain"] = 99
    else:
        target.meta.new["nested"]["gain"] = 99
    assert source.meta.new["nested"]["gain"] == 12
