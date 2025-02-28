# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from spectrochempy.core.dataset.baseobjects.meta import Meta
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import raises
from spectrochempy.utils.jsonutils import json_decoder, json_encoder
import copy


def test_init():
    meta = Meta()
    meta.td = [200, 400]
    assert meta.td[0] == 200
    assert meta.si is None
    meta["si"] = "a string"
    assert isinstance(meta.si, str)
    assert meta.si.startswith("a")
    # Test initialization with data
    meta = Meta(td=[100, 200], si=1024, name="test")
    assert meta.td == [100, 200]
    assert meta.si == 1024
    assert meta.name == "test"
    assert meta.readonly is False


def test_instance():
    meta = Meta()
    assert isinstance(meta, Meta)


def test_equal():
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


def test_readonly():
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


def test_invalid_key():
    meta = Meta()
    meta.readonly = False  # this is accepted`
    with raises(KeyError):
        meta["readonly"] = True  # this not because readonly is reserved
    with raises(KeyError):
        meta["_data"] = True  # this not because _xxx type attributes are private


def test_get_inexistent():
    meta = Meta()
    assert meta.existepas is None
    # Test default value
    assert meta.get("existepas", "default") == "default"


def test_get_keys_items():
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 1024
    assert list(meta.keys()) == ["si", "td"]
    assert list(meta.items()) == [("si", 1024), ("td", [200, 400])]
    # Test with more complex metadata
    meta.nested = {"key": "value"}
    assert list(meta.keys()) == ["nested", "si", "td"]
    assert list(meta.items()) == [
        ("nested", {"key": "value"}),
        ("si", 1024),
        ("td", [200, 400]),
    ]


def test_iterator():
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 2048
    meta.ls = 3
    meta.ns = 1024
    assert sorted([val for val in meta]) == ["ls", "ns", "si", "td"]


def test_copy():
    meta = Meta()
    meta.td = [200, 400]
    meta.si = 2048
    meta.ls = 3
    meta.ns = 1024

    meta2 = meta
    assert meta2 is meta

    meta2 = meta.copy()
    assert meta2 is not meta
    assert sorted([val for val in meta2]) == ["ls", "ns", "si", "td"]

    # bug with quantity

    si = 2048 * ur.s
    meta.si = si

    meta3 = meta.copy()
    meta3.si = si / 2.0

    assert meta3 is not meta


def test_deepcopy():
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


def test_swap():
    meta = Meta()
    meta.td = [200, 400, 500]
    meta.xe = [30, 40, 80]
    meta.si = 2048
    meta.swap(1, 2)
    assert meta.td == [200, 500, 400]
    assert meta.xe == [30, 80, 40]
    assert meta.si == 2048


def test_permute():
    meta = Meta()
    meta.td = [200, 400, 500]
    meta.xe = [30, 40, 80]
    meta.si = 2048

    p = (2, 0, 1)
    meta.permute(*p)
    assert meta.td == [500, 200, 400]
    assert meta.xe == [80, 30, 40]
    assert meta.si == 2048


def test_update():
    meta1 = Meta(td=[200, 400], si=2048)
    meta2 = Meta(ls=3, ns=1024)
    meta1.update(meta2)
    assert meta1.ls == 3
    assert meta1.ns == 1024
    # Test update with dictionary
    meta1.update({"new_key": "new_value"})
    assert meta1.new_key == "new_value"


def test_attribute_access():
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


def test_nested_meta():
    """Test nested Meta objects."""
    meta = Meta()
    nested_meta = Meta(key="value")
    meta.nested = nested_meta
    secound_nested_meta = Meta(key2="value2")
    meta.nested.nested2 = secound_nested_meta

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
    assert (
        meta2.nested["key"] == "updated value"
    )  # here we need to getitem as it is not  a meta object


def test_nested_meta_copy():
    """Test copying nested Meta objects."""
    meta = Meta()
    nested_meta = Meta(key="value")
    meta.nested = nested_meta

    # Shallow copy
    meta_copy = meta.copy()
    assert meta_copy is not meta
    assert meta_copy.nested is not meta.nested
    assert meta_copy.nested.key == "value"

    # Deep copy
    meta_deepcopy = copy.deepcopy(meta)
    assert meta_deepcopy is not meta
    assert meta_deepcopy.nested is not meta.nested
    assert meta_deepcopy.nested.key == "value"


def test_json_serialization():
    """Test JSON serialization of nested Meta objects."""
    meta = Meta(
        nested=Meta(key="value"), td=[200, 400], si=2048, dicnested={"key": "value"}
    )
    json_str = json_encoder(meta)
    meta2 = json_decoder(json_str)
    assert meta == meta2


# test derived class PreferencesSet
from spectrochempy.core.dataset.arraymixins.ndplot import PreferencesSet


def test_preferences_set_getitem():
    prefs = PreferencesSet()
    prefs["font_family"] = "serif"
    assert prefs["font_family"] == "serif"


def test_preferences_set_setitem():
    prefs = PreferencesSet()
    prefs["font_family"] = "serif"
    prefs["font_family"] = "sans-serif"
    assert prefs["font_family"] == "sans-serif"


def test_preferences_set_reset():
    prefs = PreferencesSet()
    prefs["font_family"] = "serif"
    prefs.reset()
    assert prefs["font_family"] == prefs.traits()["font_family"].default_value


def test_preferences_set_all():
    prefs = PreferencesSet()
    prefs.all()


def test_preferences_set_help():
    prefs = PreferencesSet()
    prefs.help("font_family")


def test_preferences_set_makestyle():
    prefs = PreferencesSet()
    stylename = prefs.makestyle("mydefault")
    assert stylename == "mydefault"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
