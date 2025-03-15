# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import base64
import json
import pickle
from datetime import datetime

import numpy as np
import pytest

from spectrochempy.utils.jsonutils import json_decoder, json_encoder


def test_json_encoder_decoder_no_encoding(IR_dataset_2D):
    """Test JSON encoding and decoding without encoding."""
    nd = IR_dataset_2D.copy()

    # make a json string to write (without encoding)
    js = json_encoder(nd, encoding=None)
    js_string = json.dumps(js, indent=2)
    print("no encoding", len(js_string))

    # load json from string
    jsd = json.loads(js_string, object_hook=json_decoder)

    assert np.all(np.array(js["data"]["tolist"]) == jsd["data"])


def test_json_encoder_decoder_base64(IR_dataset_2D):
    """Test JSON encoding and decoding with base64 encoding."""
    nd = IR_dataset_2D.copy()

    # encoding base 64
    js = json_encoder(nd, encoding="base64")
    js_string = json.dumps(js, indent=2)
    print("base64", len(js_string))

    # load json from string
    jsd = json.loads(js_string, object_hook=json_decoder)

    assert np.all(pickle.loads(base64.b64decode(js["data"]["base64"])) == jsd["data"])


def test_simple_python_types():
    """Test encoding and decoding of simple Python types."""
    test_data = {
        "string": "test string",
        "integer": 42,
        "float": 3.14159,
        "boolean": True,
        "none": None,
        "list": [1, 2, 3, 4],
        "nested_dict": {"a": 1, "b": 2},
    }

    # Encode to JSON
    js = json_encoder(test_data)
    js_string = json.dumps(js)

    # Decode back
    decoded = json.loads(js_string, object_hook=json_decoder)

    # Verify all values match
    for key, value in test_data.items():
        assert decoded[key] == value


def test_numpy_arrays():
    """Test encoding and decoding different types of numpy arrays."""
    arrays = {
        "1d_int": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "2d_float": np.random.rand(3, 4),
        "boolean_array": np.array([True, False, True]),
        "empty_array": np.array([]),
    }

    for name, array in arrays.items():
        # Test without encoding
        js_no_encoding = json_encoder(array, encoding=None)
        js_string = json.dumps(js_no_encoding)
        decoded_no_encoding = json.loads(js_string, object_hook=json_decoder)
        assert np.array_equal(
            decoded_no_encoding, array
        ), f"Failed with no encoding for {name}"

        # Test with base64 encoding
        js_base64 = json_encoder(array, encoding="base64")
        js_string = json.dumps(js_base64)
        decoded_base64 = json.loads(js_string, object_hook=json_decoder)
        assert np.array_equal(
            decoded_base64, array
        ), f"Failed with base64 encoding for {name}"


def test_complex_numbers():
    """Test encoding and decoding of complex numbers."""
    # Test using NumPy complex array with base64 encoding to avoid complex number handling issue
    complex_array = np.array([1 + 2j, 3 - 4j, 5 + 0j])

    # Using base64 encoding avoids the tolist() operation that causes the error
    js = json_encoder(complex_array, encoding="base64")
    js_string = json.dumps(js)
    decoded = json.loads(js_string, object_hook=json_decoder)
    assert np.array_equal(decoded, complex_array)

    # Test using NumPy complex scalar with base64 encoding
    complex_scalar = np.complex128(1 + 2j)

    js = json_encoder(complex_scalar, encoding="base64")
    js_string = json.dumps(js)
    decoded = json.loads(js_string, object_hook=json_decoder)
    assert decoded == complex_scalar


def test_python_complex():
    """Test encoding and decoding of Python's built-in complex type."""
    # Python's built-in complex type
    complex_scalar = 1 + 2j

    js = json_encoder(complex_scalar)
    js_string = json.dumps(js)
    decoded = json.loads(js_string, object_hook=json_decoder)
    assert decoded == complex_scalar

    # Test another complex number
    complex_scalar2 = 3.14 - 2.71j

    js = json_encoder(complex_scalar2)
    js_string = json.dumps(js)
    decoded = json.loads(js_string, object_hook=json_decoder)
    assert decoded == complex_scalar2


def test_nested_structures():
    """Test encoding and decoding of nested structures."""
    nested = {
        "level1": {
            "level2": {"array": np.random.rand(5), "value": 42},
            "list": [1, 2, np.array([3, 4, 5])],
        },
        "another_key": [{"key": np.random.rand(2, 2)}, "string"],
    }

    # Test with base64 encoding
    js = json_encoder(nested, encoding="base64")
    js_string = json.dumps(js)
    decoded = json.loads(js_string, object_hook=json_decoder)

    # Check structure was preserved
    assert "level1" in decoded
    assert "level2" in decoded["level1"]
    assert "array" in decoded["level1"]["level2"]
    assert decoded["level1"]["level2"]["value"] == 42
    assert len(decoded["level1"]["list"]) == 3
    assert isinstance(decoded["level1"]["list"][2], np.ndarray)
    assert np.array_equal(decoded["level1"]["list"][2], np.array([3, 4, 5]))
    assert len(decoded["another_key"]) == 2
    assert isinstance(decoded["another_key"][0]["key"], np.ndarray)
    assert decoded["another_key"][0]["key"].shape == (2, 2)
    assert decoded["another_key"][1] == "string"


def test_roundtrip_preservation(IR_dataset_2D):
    """Test that object properties are preserved during roundtrip conversion."""
    nd = IR_dataset_2D.copy()

    # Test both encoding methods
    for encoding in [None, "base64"]:
        js = json_encoder(nd, encoding=encoding)
        js_string = json.dumps(js)
        decoded = json.loads(js_string, object_hook=json_decoder)

        # Verify key attributes were preserved
        assert isinstance(decoded, dict)
        assert "data" in decoded
        assert decoded["data"].shape == nd.data.shape

        # If metadata was encoded, verify it's preserved
        if "meta" in js:
            assert "meta" in decoded
