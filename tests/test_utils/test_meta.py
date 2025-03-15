# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import copy
import json
from spectrochempy.utils.meta import Meta
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import raises
from spectrochempy.utils.jsonutils import json_decoder, json_encoder

# ======================================================================================================================
# Basic functionality tests
# ======================================================================================================================


def test_init():
    """Test Meta initialization with various parameters."""
    # Test empty initialization
    meta = Meta()
    assert len(meta) == 0
    assert meta.readonly is False

    # Test attribute setting and getting
    meta.td = [200, 400]
    assert meta.td[0] == 200
    assert meta.si is None

    # Test dictionary-style access
    meta["si"] = "a string"
    assert isinstance(meta.si, str)
    assert meta.si.startswith("a")
    assert meta["si"] == "a string"

    # Test initialization with data
    meta = Meta(td=[100, 200], si=1024, name="test")
    assert meta.td == [100, 200]
    assert meta.si == 1024
    assert meta.name == "test"
    assert meta.readonly is False

    # Test initialization with parent
    parent = object()
    meta = Meta(parent=parent)
    assert meta.parent is parent

    # Test initialization with nested dictionary
    meta = Meta(nested={"key": "value"})
    assert meta.nested["key"] == "value"

    # Test initialization with nested Meta
    nested_meta = Meta(key="value")
    meta = Meta(nested=nested_meta)
    assert isinstance(meta.nested, Meta)
    assert meta.nested.key == "value"


def test_instance():
    """Test Meta instance checking."""
    meta = Meta()
    assert isinstance(meta, Meta)
    assert Meta._implements() == "Meta"
    assert Meta._implements("Meta") is True
    assert Meta._implements("NotMeta") is False


def test_attribute_access():
    """Test attribute access for Meta objects."""
    meta = Meta()
    meta.int_attr = 10
    meta.float_attr = 10.5
    meta.str_attr = "test"
    meta.list_attr = [1, 2, 3]
    meta.dict_attr = {"key": "value"}
    assert meta.int_attr == 10
    assert meta.float_attr == 10.5
    assert meta.str_attr == "test"
    assert meta.list_attr == [1, 2, 3]
    assert meta.dict_attr == {"key": "value"}

    # Test non-existent attribute
    assert meta.non_existent_attr is None

    # Test get method with default
    assert meta.get("non_existent_attr", "default") == "default"
    assert meta.get("int_attr", "default") == 10


def test_invalid_key():
    """Test invalid key handling."""
    meta = Meta()
    meta.readonly = False  # this is accepted
    with raises(KeyError):
        meta["readonly"] = True  # this not because readonly is reserved
    with raises(KeyError):
        meta["_data"] = True  # this not because _xxx type attributes are private

    # Test access to special attributes
    assert meta.__wrapped__ is False  # __wrapped__ returns False, not raises
    with raises(AttributeError):
        meta._private_attr  # This should raise AttributeError
    with raises(AttributeError):
        meta._ipython_thing  # Should raise AttributeError


def test_properties():
    """Test property access."""
    meta = Meta(td=[100, 200], si=1024)
    # Test data property
    assert meta.data == meta._data
    assert isinstance(meta.data, dict)

    # Test readonly property
    assert meta.readonly is False
    meta.readonly = True
    assert meta.readonly is True


# ======================================================================================================================
# Data operations tests
# ======================================================================================================================


def test_equal():
    """Test equality operations."""
    meta1 = Meta()
    meta2 = Meta()
    assert meta1 == meta2

    # Test equality with non-empty Meta objects
    meta1.td = [100, 200]
    meta2.td = [100, 200]
    assert meta1 == meta2

    # Test inequality
    meta2.td = [200, 300]
    assert meta1 != meta2

    # Test equality with dictionary
    meta1 = Meta(td=[100, 200], si=1024)
    dict1 = {"td": [100, 200], "si": 1024}
    assert meta1 == dict1

    # Test inequality with non-dict/Meta object
    assert meta1 != "string"


def test_iterator():
    """Test iterator functionality."""
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 2048
    meta.ls = 3
    meta.ns = 1024
    assert sorted([val for val in meta]) == ["ls", "ns", "si", "td"]


def test_dict_methods():
    """Test dictionary-like methods."""
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 1024
    meta.ls = 3

    # Test keys
    assert sorted(meta.keys()) == ["ls", "si", "td"]

    # Test values
    values = meta.values()
    assert 3 in values
    assert 1024 in values
    assert [200, 400] in values

    # Test items
    items = meta.items()
    assert ("ls", 3) in items
    assert ("si", 1024) in items
    assert ("td", [200, 400]) in items

    # Test to_dict
    data_dict = meta.to_dict()
    assert isinstance(data_dict, dict)
    assert data_dict["ls"] == 3
    assert data_dict["si"] == 1024
    assert data_dict["td"] == [200, 400]


def test_update():
    """Test update method."""
    meta1 = Meta(td=[200, 400], si=2048)
    meta2 = Meta(ls=3, ns=1024)
    meta1.update(meta2)
    assert meta1.ls == 3
    assert meta1.ns == 1024

    # Test update with dictionary
    meta1.update({"new_key": "new_value"})
    assert meta1.new_key == "new_value"

    # Test update with nested structures
    meta1 = Meta(nested={"key": "value"})
    meta1.update({"nested": {"new_key": "new_value"}})
    assert meta1.nested["key"] == "value"
    assert meta1.nested["new_key"] == "new_value"


# ======================================================================================================================
# Copy tests
# ======================================================================================================================


def test_copy():
    """Test copy functionality."""
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 2048
    meta.ls = 3
    meta.ns = 1024

    # Test direct assignment
    meta2 = meta
    assert meta2 is meta

    # Test copy method
    meta2 = meta.copy()
    assert meta2 is not meta
    assert sorted([val for val in meta2]) == ["ls", "ns", "si", "td"]

    # Test with quantity units
    si = 2048 * ur.s
    meta.si = si
    meta3 = meta.copy()
    meta3.si = si / 2.0
    assert meta3 is not meta


def test_deepcopy():
    """Test deepcopy functionality."""
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 2048
    meta.ls = 3
    meta.ns = 1024
    meta.nested = {"key": "value"}

    meta2 = copy.deepcopy(meta)
    assert meta2 is not meta
    assert meta2 == meta
    assert meta2.nested is not meta.nested


def test_copy_readonly():
    """Test copy with readonly state."""
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 2048
    meta.xe = {"key": "value"}
    meta.nested = Meta(other_key="other_value")
    meta.readonly = True

    meta_copy = meta.copy()
    assert meta_copy is not meta
    assert meta_copy.td == [200, 400]
    assert meta_copy.si == 2048
    assert meta_copy.xe == {"key": "value"}
    assert meta_copy.nested == Meta(other_key="other_value")
    assert meta_copy.readonly is True


# ======================================================================================================================
# Nested structure tests
# ======================================================================================================================


def test_nested_meta():
    """Test nested Meta objects."""
    meta = Meta()
    nested_meta = Meta(key="value")
    meta.nested = nested_meta
    second_nested_meta = Meta(key2="value2")
    meta.nested.nested2 = second_nested_meta

    assert isinstance(meta.nested, Meta)
    assert meta.nested.key == "value"
    assert isinstance(meta.nested.nested2, Meta)
    assert meta.nested.nested2.key2 == "value2"

    # Test read-only with nested Meta
    meta.readonly = True
    with raises(ValueError):
        meta.nested.key = "new value"
    assert meta.nested.key == "value"
    assert meta.nested.readonly is True
    assert meta.nested.nested2.readonly is True
    with raises(ValueError):
        meta.nested.nested2.key2 = "new value2"

    # Test updating nested Meta
    meta.readonly = False
    meta.nested.key = "new value"
    assert meta.nested.key == "new value"

    # Test nested Meta read-only state
    nested_meta.readonly = True
    with raises(ValueError):
        nested_meta.key = "another value"
    assert nested_meta.key == "new value"


def test_nested_meta_update():
    """Test updating nested Meta objects."""
    meta = Meta()
    nested_meta = Meta(key="value")
    meta.nested = nested_meta

    # Update with another Meta object
    update_meta = Meta(nested=Meta(key="new value"))
    meta.update(update_meta)
    assert meta.nested.key == "new value"

    # Update with a dictionary
    meta2 = Meta()
    nested_meta = {"key": "value"}
    meta2.nested = nested_meta

    meta2.update({"nested": {"key": "updated value"}})
    assert meta2.nested["key"] == "updated value"


def test_readonly():
    """Test read-only functionality."""
    meta = Meta()
    meta.chaine = "a string"
    assert meta.chaine == "a string"
    meta.readonly = True
    with raises(ValueError):
        meta.chaine = "a modified string"
    assert meta.chaine != "a modified string"

    # Test read-only with nested dictionary
    meta = Meta(nested={"key": "value"})
    meta.readonly = True
    with raises(ValueError):
        meta.nested["key"] = "new value"
    assert meta.nested["key"] == "value"

    # Test read-only with nested Meta object
    meta = Meta(nested=Meta(key="value"))
    meta.readonly = True
    with raises(ValueError):
        meta.nested.key = "new value"
    assert meta.nested["key"] == "value"


# ======================================================================================================================
# Dimension operation tests
# ======================================================================================================================


def test_swap():
    """Test swap method."""
    meta = Meta()
    meta.td = [200, 400, 500]
    meta.xe = [30, 40, 80]
    meta.si = 2048

    # Test in-place swap
    meta.swap(1, 2)
    assert meta.td == [200, 500, 400]
    assert meta.xe == [30, 80, 40]
    assert meta.si == 2048

    # Test non-inplace swap
    result = meta.swap(0, 1, inplace=False)
    assert result is not None
    assert result.td == [500, 200, 400]
    assert meta.td == [200, 500, 400]  # Original unchanged


def test_permute():
    """Test permute method."""
    meta = Meta()
    meta.td = [200, 400, 500]
    meta.xe = [30, 40, 80]
    meta.si = 2048

    # Test in-place permute
    p = (2, 0, 1)
    meta.permute(*p)
    assert meta.td == [500, 200, 400]
    assert meta.xe == [80, 30, 40]
    assert meta.si == 2048

    # Test non-inplace permute
    result = meta.permute(1, 2, 0, inplace=False)
    assert result is not None
    assert result.td == [200, 400, 500]
    assert meta.td == [500, 200, 400]  # Original unchanged


# ======================================================================================================================
# Serialization tests
# ======================================================================================================================


def test_str_representation():
    """Test string representation."""
    meta = Meta(td=[200, 400], si=2048, name="test_meta")
    str_repr = str(meta)
    assert isinstance(str_repr, str)
    # Check if it's valid JSON
    json_dict = json.loads(str_repr)
    # Metadata is stored under the 'data' key
    assert "data" in json_dict
    assert "td" in json_dict["data"]
    assert "si" in json_dict["data"]
    assert json_dict["data"]["td"] == [200, 400]
    assert json_dict["data"]["si"] == 2048
    assert json_dict["name"] == "test_meta"


def test_html_representation():
    """Test HTML representation."""
    meta = Meta(td=[200, 400], si=2048, name="test_meta")
    html_repr = meta._repr_html_()
    assert isinstance(html_repr, str)
    assert html_repr == meta.html()
    assert "test_meta" in html_repr
    assert "scp-output" in html_repr


def test_json_serialization():
    """Test JSON serialization of Meta objects."""
    meta = Meta(
        nested=Meta(key="value"), td=[200, 400], si=2048, dicnested={"key": "value"}
    )
    json_str = json_encoder(meta)
    meta2 = json_decoder(json_str)
    assert meta == meta2


# ======================================================================================================================
# PreferencesSet tests (derived class)
# ======================================================================================================================

from spectrochempy.application.preferences import PreferencesSet


def test_preferences_set():
    """Test PreferencesSet functionality."""
    prefs = PreferencesSet()

    # Test getitem/setitem
    prefs["font_family"] = "serif"
    assert prefs["font_family"] == "serif"
    prefs["font_family"] = "sans-serif"
    assert prefs["font_family"] == "sans-serif"

    # Test reset
    prefs.reset()
    assert prefs["font_family"] == prefs.traits()["font_family"].default_value

    # Test helpers (just for coverage, no assertions needed)
    prefs.list_all()
    prefs.help("font_family")

    # Test style creation
    stylename = prefs.makestyle("mydefault")
    assert stylename == "mydefault"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
