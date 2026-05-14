# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from spectrochempy.plugins.registry import PluginRegistry, registry


def test_registry_singleton():
    r1 = PluginRegistry()
    r2 = PluginRegistry()
    assert r1 is r2


def test_register_reader():
    def dummy_reader(path): ...

    registry.register_reader("test", dummy_reader, description="test reader")
    info = registry.get_reader("test")
    assert info is not None
    assert info["func"] is dummy_reader
    assert info["description"] == "test reader"


def test_available_readers():
    assert "test" in registry.available_readers


def test_register_processor():
    def dummy_processor(data): ...

    registry.register_processor("test_proc", dummy_processor)
    info = registry.get_processor("test_proc")
    assert info is not None
    assert info["func"] is dummy_processor


def test_register_writer():
    def dummy_writer(data, path): ...

    registry.register_writer("test_writer", dummy_writer)
    info = registry.get_writer("test_writer")
    assert info is not None
    assert info["func"] is dummy_writer


def test_dtype_handler():
    handler = object()
    registry.register_dtype_handler("quaternion", handler)
    assert registry.has_dtype_handler("quaternion")
    assert registry.get_dtype_handler("quaternion") is handler


def test_unit_context():
    def setup(): ...

    registry.register_unit_context("nmr", setup)
    assert registry.get_unit_context("nmr") is setup


def test_register_plugin():
    plugin = object()
    registry.register_plugin("test_plugin", plugin)
    assert registry.get_plugin("test_plugin") is plugin
    assert "test_plugin" in registry.available_plugins


def test_filetype():
    info = {"ext": ".xyz", "reader": "test"}
    registry.register_filetype("xyz", info)
    assert registry.get_filetype("xyz") == info
    assert "xyz" in registry.available_filetypes
