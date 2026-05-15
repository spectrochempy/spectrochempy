# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Tests for PluginRegistry and specialised registries."""

from spectrochempy.plugins.registries import IORegistry
from spectrochempy.plugins.registries import MetadataRegistry
from spectrochempy.plugins.registries import ProcessingRegistry
from spectrochempy.plugins.registries import VisualizationRegistry
from spectrochempy.plugins.registry import PluginRegistry
from spectrochempy.plugins.registry import registry as global_registry

# ------------------------------------------------------------------
# PluginRegistry — composite structure
# ------------------------------------------------------------------


def test_composite_structure():
    """PluginRegistry composes all specialised registries."""
    registry = PluginRegistry()
    assert isinstance(registry.io, IORegistry)
    assert isinstance(registry.processing, ProcessingRegistry)
    assert isinstance(registry.visualization, VisualizationRegistry)
    assert isinstance(registry.metadata, MetadataRegistry)


def test_independent_instances():
    """PluginRegistry() now creates independent instances."""
    r1 = PluginRegistry()
    r2 = PluginRegistry()
    assert r1 is not r2
    assert r1 is not global_registry


def test_global_registry_is_instance():
    """The module-level registry is a PluginRegistry instance."""
    assert isinstance(global_registry, PluginRegistry)


# ------------------------------------------------------------------
# PluginRegistry — backward-compatible forwarding
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# PluginRegistry — forwarding targets sub-registries
# ------------------------------------------------------------------


def test_register_reader_forwards_to_io():
    registry = PluginRegistry()

    def dummy(path):
        ...

    registry.register_reader("fwd", dummy)
    assert registry.io.get_reader("fwd") is not None


def test_register_writer_forwards_to_io():
    registry = PluginRegistry()

    def dummy(data, path):
        ...

    registry.register_writer("fwd", dummy)
    assert registry.io.get_writer("fwd") is not None


def test_register_processor_forwards_to_processing():
    registry = PluginRegistry()

    def dummy(data):
        ...

    registry.register_processor("fwd", dummy)
    assert registry.processing.get_processor("fwd") is not None


def test_register_plugin_forwards_to_metadata():
    registry = PluginRegistry()
    registry.register_plugin("fwd", "plugin")
    assert registry.metadata.get_plugin("fwd") == "plugin"


def test_dtype_handler_forwards_to_processing():
    registry = PluginRegistry()
    registry.register_dtype_handler("fwd", object())
    assert registry.processing.has_dtype_handler("fwd")


# ------------------------------------------------------------------
# IORegistry — isolated tests
# ------------------------------------------------------------------


def test_io_registry_independent():
    io = IORegistry()

    def dummy(path):
        ...

    io.register_reader("r", dummy)
    io.register_writer("w", dummy)
    io.register_filetype("ext", {})

    assert "r" in io.available_readers
    assert "w" in io.available_writers
    assert "ext" in io.available_filetypes

    io.clear()
    assert io.available_readers == {}
    assert io.available_writers == {}
    assert io.available_filetypes == {}


# ------------------------------------------------------------------
# ProcessingRegistry — isolated tests
# ------------------------------------------------------------------


def test_processing_registry_independent():
    pr = ProcessingRegistry()

    def dummy(data):
        ...

    def setup():
        ...

    pr.register_processor("p", dummy)
    pr.register_unit_context("ctx", setup)
    pr.register_dtype_handler("dtype", object())

    assert "p" in pr.available_processors
    assert pr.get_unit_context("ctx") is setup
    assert pr.has_dtype_handler("dtype")

    pr.clear()
    assert pr.available_processors == {}
    assert pr.get_unit_context("ctx") is None
    assert not pr.has_dtype_handler("dtype")


# ------------------------------------------------------------------
# VisualizationRegistry — isolated tests
# ------------------------------------------------------------------


def test_visualization_registry():
    vr = VisualizationRegistry()

    def plot():
        ...

    vr.register_visualizer("myplot", plot)
    assert vr.get_visualizer("myplot") is not None
    assert "myplot" in vr.available_visualizers

    vr.clear()
    assert vr.available_visualizers == {}


# ------------------------------------------------------------------
# MetadataRegistry — isolated tests
# ------------------------------------------------------------------


def test_metadata_registry():
    mr = MetadataRegistry()
    mr.register_plugin("p1", "descriptor")
    assert mr.get_plugin("p1") == "descriptor"
    assert "p1" in mr.available_plugins

    mr.clear()
    assert mr.available_plugins == {}


# ------------------------------------------------------------------
# Specialised registries don't leak across PluginRegistry instances
# ------------------------------------------------------------------


def test_sub_registries_are_independent_instances():
    r1 = PluginRegistry()
    r2 = PluginRegistry()

    assert r1.io is not r2.io
    assert r1.processing is not r2.processing
    assert r1.visualization is not r2.visualization
    assert r1.metadata is not r2.metadata


# ------------------------------------------------------------------
# Contribution dataclasses
# ------------------------------------------------------------------


def test_reader_contribution():
    from spectrochempy.plugins.contributions import ReaderContribution

    def dummy(path):
        ...

    c = ReaderContribution(
        name="test", func=dummy, description="desc", extensions=[".ext"]
    )
    assert c.name == "test"
    assert c.func is dummy
    assert c.description == "desc"
    assert c.extensions == [".ext"]


def test_reader_contribution_defaults():
    from spectrochempy.plugins.contributions import ReaderContribution

    def dummy(path):
        ...

    c = ReaderContribution(name="test", func=dummy)
    assert c.description == ""
    assert c.extensions is None


def test_writer_contribution():
    from spectrochempy.plugins.contributions import WriterContribution

    def dummy(data, path):
        ...

    c = WriterContribution(name="test", func=dummy, description="desc")
    assert c.name == "test"
    assert c.func is dummy
    assert c.description == "desc"


def test_processor_contribution():
    from spectrochempy.plugins.contributions import ProcessorContribution

    def dummy(data):
        ...

    c = ProcessorContribution(name="test", func=dummy, description="desc")
    assert c.name == "test"
    assert c.func is dummy
    assert c.description == "desc"


def test_reader_from_dict():
    from spectrochempy.plugins.contributions import reader_from_dict

    def dummy(path):
        ...

    c = reader_from_dict(
        {"name": "r", "func": dummy, "description": "d", "extensions": [".x"]}
    )
    assert c.name == "r"
    assert c.func is dummy
    assert c.extensions == [".x"]


def test_writer_from_dict():
    from spectrochempy.plugins.contributions import writer_from_dict

    def dummy(data, path):
        ...

    c = writer_from_dict({"name": "w", "func": dummy})
    assert c.name == "w"
    assert c.func is dummy
    assert c.description == ""


def test_processor_from_dict():
    from spectrochempy.plugins.contributions import processor_from_dict

    def dummy(data):
        ...

    c = processor_from_dict({"name": "p", "func": dummy, "description": "proc"})
    assert c.name == "p"
    assert c.func is dummy
    assert c.description == "proc"
