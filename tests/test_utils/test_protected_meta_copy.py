"""Copying metadata must not temporarily unlock the original graph."""

import copy

import pytest

from spectrochempy.utils.meta import Meta
from spectrochempy.utils.objects import ReadOnlyDict


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy, lambda meta: meta.copy()])
def test_meta_copy_preserves_independently_locked_subtree(copier):
    source = Meta(acquisition={"receiver": {"gain": [12]}})
    source.acquisition.set_readonly(True)
    result = copier(source)
    assert not source.readonly
    assert not result.readonly
    for metadata in (source, result):
        with pytest.raises(ValueError, match="read-only"):
            metadata.acquisition["receiver"]["gain"] = [99]
    result.acquisition.set_readonly(False)
    result.acquisition["receiver"]["gain"][0] = 99
    assert source.acquisition["receiver"]["gain"] == [12]


def test_failed_deepcopy_leaves_source_subtree_locked():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("copy failed")

    source = Meta(acquisition={"receiver": {"gain": 12}}, payload=Uncopyable())
    source.acquisition.set_readonly(True)
    with pytest.raises(RuntimeError, match="copy failed"):
        copy.deepcopy(source)
    with pytest.raises(ValueError, match="read-only"):
        source.acquisition["receiver"]["gain"] = 99


def test_readonly_dict_deepcopy_preserves_graph_aliases_and_cycles():
    source = ReadOnlyDict(values=[1])
    dict.__setitem__(source, "alias", source["values"])
    source["self"] = source
    source._readonly = True
    result = copy.deepcopy(source)
    assert result is not source
    assert result["self"] is result
    assert result["alias"] is result["values"]
    assert result["values"] is not source["values"]
    with pytest.raises(ValueError, match="read-only"):
        result["new"] = 1
