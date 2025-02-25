import pytest

from spectrochempy.utils.objects import Adict, ReadOnlyDict


class TestAdict:
    """Test suite for Adict class."""

    def test_basic_operations(self):
        """Test basic dictionary operations."""
        d = Adict(a=1, b=2)

        # Test creation and access
        assert d.a == 1
        assert d["b"] == 2

        # Test setting attributes
        d.c = 3
        assert d["c"] == 3
        d["d"] = 4
        assert d.d == 4

    def test_error_handling(self):
        """Test error conditions."""
        d = Adict()

        with pytest.raises(AttributeError):
            _ = d.nonexistent

        with pytest.raises(KeyError):
            _ = d["nonexistent"]

    def test_nested_structures(self):
        """Test nested dictionary handling."""
        d = Adict(a={"b": 1}, c={"d": {"e": 2}})

        # Test nested access and conversion
        assert d.a.b == 1
        assert d.c.d.e == 2
        assert isinstance(d.a, Adict)
        assert isinstance(d.c.d, Adict)

        # Test nested modification
        d.a.f = 3
        assert d.a.f == 3

    def test_update_behavior(self):
        """Test update functionality."""
        d = Adict(a=1)

        # Test simple update
        d.update({"b": 2})
        assert d.b == 2

        # Test nested update
        d.update({"c": {"d": 3}})
        assert d.c.d == 3
        assert isinstance(d.c, Adict)


class TestReadOnlyDict:
    """Test suite for ReadOnlyDict class."""

    def test_basic_operations(self):
        """Test basic dictionary operations."""
        d = ReadOnlyDict({"a": 1, "b": {"c": 2}})

        # Test initial state
        assert not d._readonly
        assert isinstance(d["b"], ReadOnlyDict)

        # Test modifications
        d["new"] = 3
        assert d["new"] == 3
        d["b"]["d"] = 4
        assert d["b"]["d"] == 4

    def test_readonly_protection(self):
        """Test protection mechanisms when readonly."""
        d = ReadOnlyDict({"a": 1, "b": {"c": 2}})
        d.set_readonly(True)

        with pytest.raises(ValueError):
            d["new"] = 3
        with pytest.raises(ValueError):
            d["b"]["new"] = 3
        with pytest.raises(ValueError):
            d.update({"new": 3})

    def test_nested_structures(self):
        """Test nested structure handling."""
        d = ReadOnlyDict({"dict": {"a": 1}, "nested": {"c": {"d": 3}}})

        # Test structure conversion
        assert isinstance(d["dict"], ReadOnlyDict)
        assert isinstance(d["nested"]["c"], ReadOnlyDict)

        # Test nested protection
        d.set_readonly(True)
        with pytest.raises(ValueError):
            d["nested"]["c"]["new"] = 4

    def test_update_behavior(self):
        """Test update functionality."""
        d = ReadOnlyDict({"a": 1, "b": {"c": 2}})

        # Test nested updates
        d.update({"d": {"e": 3}})
        assert isinstance(d["d"], ReadOnlyDict)
        d["b"].update({"f": 4})
        assert d["b"]["f"] == 4

        # Test readonly protection
        d.set_readonly(True)
        with pytest.raises(ValueError):
            d.update({"g": 5})
        with pytest.raises(ValueError):
            d["b"].update({"h": 6})

    def test_dict_operations(self):
        """Test standard dictionary operations."""
        d = ReadOnlyDict({"a": 1, "b": 2})

        # Test operations before readonly
        d_copy = d.copy()
        assert isinstance(d_copy, ReadOnlyDict)

        d.clear()
        assert len(d) == 0

        d.update({"c": 3})
        value = d.pop("c")
        assert value == 3

        # Test readonly protection
        d.update({"x": 1})
        d.set_readonly(True)
        with pytest.raises(ValueError):
            d.clear()
        with pytest.raises(ValueError):
            d.pop("x")

    def test_type_conversion(self):
        """Test type conversion handling."""
        d = ReadOnlyDict(
            {
                "list": [1, {"a": 2}],
                "tuple": (3, {"b": 4}),
                "dict": {"c": {"d": 7}},
            }
        )

        # Test conversions
        assert isinstance(d["list"][1], ReadOnlyDict)
        assert isinstance(d["tuple"][1], ReadOnlyDict)
        assert isinstance(d["dict"]["c"], ReadOnlyDict)

        # Test readonly protection
        d.set_readonly(True)
        with pytest.raises(ValueError):
            d["list"][1]["new"] = 8

    def test_meta_handling(self):
        """Test Meta object handling."""

        class MockMeta:
            _implements = lambda self, x: x == "Meta"

            def __init__(self):
                self.readonly = False

        meta = MockMeta()
        d = ReadOnlyDict({"meta": meta})

        # Test Meta behavior
        d.set_readonly(True)
        assert meta.readonly

        with pytest.raises(ValueError):
            d["meta"] = MockMeta()


if __name__ == "__main__":
    pytest.main([__file__])
