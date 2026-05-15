# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Tests for PluginRegistry."""

from spectrochempy.plugins.registry import PluginRegistry
from spectrochempy.plugins.registry import registry as global_registry


def test_independent_instances():
    """PluginRegistry() now creates independent instances."""
    r1 = PluginRegistry()
    r2 = PluginRegistry()
    assert r1 is not r2
    assert r1 is not global_registry


def test_global_registry_is_instance():
    """The module-level registry is a PluginRegistry instance."""
    assert isinstance(global_registry, PluginRegistry)


def test_register_reader():
    registry = PluginRegistry()

    def dummy_reader(path):
        ...

    registry.register_reader("test", dummy_reader, description="test reader")
    info = registry.get_reader("test")
    assert info is not None
    assert info["func"] is dummy_reader
    assert info["description"] == "test reader"


def test_available_readers():
    registry = PluginRegistry()

    def dummy_reader(path):
        ...

    registry.register_reader("test", dummy_reader)
    assert "test" in registry.available_readers


def test_register_processor():
    registry = PluginRegistry()

    def dummy_processor(data):
        ...

    registry.register_processor("test_proc", dummy_processor)
    info = registry.get_processor("test_proc")
    assert info is not None
    assert info["func"] is dummy_processor


def test_register_writer():
    registry = PluginRegistry()

    def dummy_writer(data, path):
        ...

    registry.register_writer("test_writer", dummy_writer)
    info = registry.get_writer("test_writer")
    assert info is not None
    assert info["func"] is dummy_writer


def test_dtype_handler():
    registry = PluginRegistry()
    handler = object()
    registry.register_dtype_handler("quaternion", handler)
    assert registry.has_dtype_handler("quaternion")
    assert registry.get_dtype_handler("quaternion") is handler


def test_unit_context():
    registry = PluginRegistry()

    def setup():
        ...

    registry.register_unit_context("nmr", setup)
    assert registry.get_unit_context("nmr") is setup


def test_register_plugin():
    registry = PluginRegistry()
    plugin = object()
    registry.register_plugin("test_plugin", plugin)
    assert registry.get_plugin("test_plugin") is plugin
    assert "test_plugin" in registry.available_plugins


def test_filetype():
    registry = PluginRegistry()
    info = {"ext": ".xyz", "reader": "test"}
    registry.register_filetype("xyz", info)
    assert registry.get_filetype("xyz") == info
    assert "xyz" in registry.available_filetypes


def test_clear():
    """clear() removes all entries from the registry."""
    registry = PluginRegistry()

    def dummy():
        ...

    registry.register_reader("r1", dummy)
    registry.register_writer("w1", dummy)
    registry.register_processor("p1", dummy)
    registry.register_plugin("p1", dummy)
    registry.register_filetype("ext", {})
    registry.register_dtype_handler("d1", dummy)
    registry.register_unit_context("u1", dummy)

    registry.clear()

    assert registry.available_readers == {}
    assert registry.available_writers == {}
    assert registry.available_processors == {}
    assert registry.available_plugins == {}
    assert registry.available_filetypes == {}
    assert registry.get_reader("r1") is None
    assert registry.get_dtype_handler("d1") is None


def test_registries_isolated():
    """Two independent registries do not share state."""
    r1 = PluginRegistry()
    r2 = PluginRegistry()

    def dummy():
        ...

    r1.register_reader("shared", dummy)
    assert r2.get_reader("shared") is None
