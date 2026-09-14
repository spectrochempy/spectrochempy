import json
from datetime import UTC
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.provenance import ProvenanceValidationError
from spectrochempy.provenance._normalization import normalize_value


class StableChoice(IntEnum):
    FIRST = 1


def test_normalizes_supported_recursive_values_with_sorted_mapping_keys():
    normalized = normalize_value(
        {
            "tuple": (np.int64(2), np.float32(3.5)),
            "none": None,
            "flag": True,
            "choice": StableChoice.FIRST,
        }
    )

    assert list(normalized) == ["choice", "flag", "none", "tuple"]
    assert normalized["tuple"] == [2, 3.5]
    assert normalized["choice"] == {
        "type": "enum",
        "enum": f"{__name__}.StableChoice",
        "member": "FIRST",
        "value": 1,
    }
    json.dumps(normalized, allow_nan=False)


@pytest.mark.parametrize(
    ("value", "tag"),
    [
        (float("nan"), "nan"),
        (float("inf"), "+infinity"),
        (float("-inf"), "-infinity"),
    ],
)
def test_non_finite_floats_use_valid_tagged_json(value, tag):
    assert normalize_value(value) == {"type": "float", "value": tag}


def test_paths_are_reduced_to_basename_with_visible_redaction():
    normalized = normalize_value(Path("/private/experiment/sample.spc"))
    assert normalized == {
        "type": "path",
        "value": "sample.spc",
        "redaction": "basename",
    }
    assert "/private" not in json.dumps(normalized)


def test_datetime_is_normalized_to_utc_rfc3339():
    value = datetime(2026, 9, 14, 10, 30, tzinfo=UTC)
    assert normalize_value(value) == {
        "type": "datetime",
        "value": "2026-09-14T10:30:00Z",
    }


def test_units_and_quantities_have_bounded_semantic_values():
    assert normalize_value(scp.ur.cm) == {"type": "unit", "value": "cm"}
    assert normalize_value(2.5 * scp.ur.cm) == {
        "type": "quantity",
        "magnitude": 2.5,
        "unit": "cm",
    }


def test_coordinate_is_reduced_to_metadata_and_ordered_range():
    coordinate = scp.Coord(
        [3.0, 2.0, 1.0],
        title="wavenumber",
        units="cm^-1",
        labels=["a", "b", "c"],
    )

    assert normalize_value(coordinate) == {
        "type": "coordinate",
        "dimension": "x",
        "title": "wavenumber",
        "unit": "cm^-1",
        "size": 3,
        "range": [3.0, 1.0],
        "labels": {"status": "omitted", "reason": "privacy_default"},
    }


def test_unordered_collection_has_deterministic_order():
    first = normalize_value({"b", "a", "c"})
    second = normalize_value({"c", "b", "a"})
    assert first == second == {"type": "set", "values": ["a", "b", "c"]}


@pytest.mark.parametrize("factory", [set, frozenset])
def test_unordered_collection_size_limit_is_consistent(factory):
    value = factory(range(257))

    assert normalize_value(value) == {
        "status": "omitted",
        "reason": "collection_size_limit",
        "size": 257,
    }
    with pytest.raises(ProvenanceValidationError, match="collection is too large"):
        normalize_value(value, strict=True)


def test_small_numpy_array_is_bounded_json_data():
    normalized = normalize_value(np.array([[1, 2], [3, 4]], dtype=np.int16))
    assert normalized == {
        "type": "numpy.ndarray",
        "dtype": "int16",
        "shape": [2, 2],
        "values": [[1, 2], [3, 4]],
    }


def test_large_numpy_array_is_omitted_or_rejected():
    value = np.arange(65)
    assert normalize_value(value)["status"] == "omitted"
    with pytest.raises(ProvenanceValidationError, match="64 elements"):
        normalize_value(value, strict=True)


def test_masked_array_is_summarized_without_ambiguous_null_values():
    value = np.ma.array([1.0, 2.0], mask=[False, True])
    normalized = normalize_value(value)

    assert normalized == {
        "status": "omitted",
        "reason": "masked_array",
        "type": "numpy.ma.MaskedArray",
        "dtype": "float64",
        "shape": [2],
        "size": 2,
        "masked": 1,
    }
    with pytest.raises(ProvenanceValidationError, match="Masked array"):
        normalize_value(value, strict=True)


def test_unsupported_value_never_uses_repr():
    class Unsafe:
        def __repr__(self):
            raise AssertionError("repr must not be called")

    normalized = normalize_value(Unsafe())
    assert normalized["status"] == "unsupported"
    assert normalized["type"].endswith(".Unsafe")
    with pytest.raises(ProvenanceValidationError, match="Unsupported"):
        normalize_value(Unsafe(), strict=True)


def test_non_string_mapping_keys_are_explicitly_unsupported():
    assert normalize_value({1: "value"}) == {
        "status": "unsupported",
        "reason": "non_string_mapping_key",
    }
    with pytest.raises(ProvenanceValidationError, match="keys must be strings"):
        normalize_value({1: "value"}, strict=True)


def test_cyclic_collection_is_explicitly_unsupported():
    value = []
    value.append(value)
    normalized = normalize_value(value)
    assert normalized == [
        {
            "status": "unsupported",
            "reason": "cyclic_value",
            "type": "builtins.list",
        }
    ]
